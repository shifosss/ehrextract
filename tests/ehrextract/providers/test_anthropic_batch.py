"""Tests for AnthropicProvider.generate_batch via mocked httpx transport.

The handler simulates the Message Batches lifecycle: inline create -> status
polls -> results JSONL stream. The SDK's results() re-retrieves the batch and
follows its results_url, so the ended-batch JSON must carry a same-host URL.
Results are served OUT of input order to prove custom_id matching.
"""

import json

import httpx
import pytest
from anthropic import Anthropic

import ehrextract.providers as providers
from ehrextract.providers import AnthropicProvider, GenerationConfig, ProviderResponse

_RESULTS_URL = "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"


def _message_json(tool_input: dict) -> dict:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": [{"type": "tool_use", "id": "tu_1", "name": "extract", "input": tool_input}],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 80, "output_tokens": 25},
    }


def _succeeded_line(i: int, tool_input: dict) -> str:
    return json.dumps({
        "custom_id": f"req-{i}",
        "result": {"type": "succeeded", "message": _message_json(tool_input)},
    })


def _errored_line(i: int) -> str:
    return json.dumps({
        "custom_id": f"req-{i}",
        "result": {"type": "errored",
                   "error": {"type": "error",
                             "error": {"type": "invalid_request_error", "message": "bad"}}},
    })


class _BatchServer:
    def __init__(self, result_lines, polls_before_ended=1, create_status=200):
        self.result_lines = result_lines
        self.polls_before_ended = polls_before_ended
        self.create_status = create_status
        self.creates: list[dict] = []
        self.polls = 0

    def _batch_json(self, status: str) -> dict:
        return {
            "id": "msgbatch_1", "type": "message_batch", "processing_status": status,
            "request_counts": {"processing": 0, "succeeded": 0, "errored": 0,
                               "canceled": 0, "expired": 0},
            "created_at": "2026-06-12T00:00:00Z", "expires_at": "2026-06-13T00:00:00Z",
            "results_url": _RESULTS_URL if status == "ended" else None,
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/messages/batches":
            if self.create_status != 200:
                return httpx.Response(self.create_status,
                                      json={"type": "error",
                                            "error": {"type": "not_found_error", "message": "nope"}})
            self.creates.append(json.loads(request.content))
            return httpx.Response(200, json=self._batch_json("in_progress"))
        if method == "GET" and path == "/v1/messages/batches/msgbatch_1":
            self.polls += 1
            status = "ended" if self.polls >= self.polls_before_ended else "in_progress"
            return httpx.Response(200, json=self._batch_json(status))
        if method == "GET" and path == "/v1/messages/batches/msgbatch_1/results":
            return httpx.Response(200, text="\n".join(self.result_lines),
                                  headers={"content-type": "application/binary"})
        raise AssertionError(f"unexpected request {method} {path}")


def _provider(server: _BatchServer) -> AnthropicProvider:
    p = AnthropicProvider(model="claude-haiku-4-5", api_key="k")
    transport = httpx.MockTransport(server.handler)
    p._client = Anthropic(api_key="k", http_client=httpx.Client(transport=transport))
    return p


_SCHEMA = {
    "type": "object",
    "properties": {"smoker": {"type": "string", "enum": ["Y", "N"]}},
    "required": ["smoker"],
    "additionalProperties": False,
}
_MSGS = [[{"role": "system", "content": "SYS"}, {"role": "user", "content": f"note {i}"}]
         for i in range(2)]


def test_supports_batch_is_true():
    p = AnthropicProvider(model="m", api_key="k")
    assert p.supports_batch is True


def test_batch_create_payload_uses_forced_tool_use(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer([_succeeded_line(0, {"smoker": "Y"}),
                           _succeeded_line(1, {"smoker": "N"})])
    p = _provider(server)
    p.generate_batch(_MSGS, GenerationConfig(max_new_tokens=64), json_schema=_SCHEMA)
    requests = server.creates[0]["requests"]
    assert [r["custom_id"] for r in requests] == ["req-0", "req-1"]
    for r in requests:
        assert r["params"]["tool_choice"] == {"type": "tool", "name": "extract"}
        assert r["params"]["tools"][0]["input_schema"] == _SCHEMA
        assert r["params"]["system"] == "SYS"
        assert r["params"]["max_tokens"] == 64


def test_batch_polls_until_ended_then_matches_out_of_order_results(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(providers.time, "sleep", sleeps.append)
    server = _BatchServer(
        [_succeeded_line(1, {"smoker": "N"}), _succeeded_line(0, {"smoker": "Y"})],
        polls_before_ended=2,
    )
    p = _provider(server)
    results = p.generate_batch(_MSGS, GenerationConfig(), json_schema=_SCHEMA)
    assert len(sleeps) == 2
    assert [json.loads(r.text) for r in results] == [{"smoker": "Y"}, {"smoker": "N"}]
    assert results[0].usage == {"input_tokens": 80, "output_tokens": 25}
    assert results[0].finish_reason == "tool_use"


def test_batch_errored_result_becomes_exception_slot(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer([_succeeded_line(0, {"smoker": "Y"}), _errored_line(1)])
    p = _provider(server)
    results = p.generate_batch(_MSGS, GenerationConfig(), json_schema=_SCHEMA)
    assert isinstance(results[0], ProviderResponse)
    assert isinstance(results[1], Exception)
    assert "req-1 errored" in str(results[1])


def test_batch_anomalous_custom_id_skipped_with_warning(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer([
        _succeeded_line(999, {"smoker": "Y"}),  # index far past the 2 inputs
        _succeeded_line(0, {"smoker": "Y"}),
    ])
    p = _provider(server)
    with caplog.at_level(logging.WARNING, logger="ehrextract.providers"):
        results = p.generate_batch(_MSGS, GenerationConfig(), json_schema=_SCHEMA)
    assert json.loads(results[0].text) == {"smoker": "Y"}
    assert isinstance(results[1], Exception) and "no result" in str(results[1])
    assert any("unknown custom_id" in r.getMessage() for r in caplog.records)


def test_batch_without_json_schema_raises():
    p = AnthropicProvider(model="m", api_key="k")
    with pytest.raises(RuntimeError, match="json_schema"):
        p.generate_batch(_MSGS, GenerationConfig())


def test_batch_create_404_raises(monkeypatch):
    """Fail-loudly pin for endpoints without batch support."""
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer([], create_status=404)
    p = _provider(server)
    with pytest.raises(Exception, match="not_found|404|nope"):
        p.generate_batch(_MSGS, GenerationConfig(), json_schema=_SCHEMA)
