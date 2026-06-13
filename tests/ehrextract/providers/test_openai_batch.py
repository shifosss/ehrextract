"""Tests for OpenAIProvider.generate_batch via mocked httpx transport.

The handler simulates the full Batch lifecycle: file upload -> batch create
-> status polls -> output/error file retrieval -> file deletion. Results are
served OUT of input order to prove custom_id matching.
"""

import json
import logging

import httpx
import pytest
from openai import OpenAI

import ehrextract.providers as providers
from ehrextract.providers import GenerationConfig, OpenAIProvider, ProviderResponse


def _completion_body(text: str) -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _output_line(i: int, text: str) -> str:
    return json.dumps({
        "id": f"resp-{i}", "custom_id": f"req-{i}",
        "response": {"status_code": 200, "request_id": "r", "body": _completion_body(text)},
        "error": None,
    })


def _error_line(i: int, message: str) -> str:
    return json.dumps({
        "id": f"resp-{i}", "custom_id": f"req-{i}",
        "response": None,
        "error": {"code": "server_error", "message": message},
    })


class _BatchServer:
    """Routes the openai SDK's batch HTTP calls; configurable terminal state."""

    def __init__(self, output_lines=None, error_lines=None,
                 terminal_status="completed", polls_before_terminal=1, delete_status=200):
        self.output_lines = output_lines
        self.error_lines = error_lines
        self.terminal_status = terminal_status
        self.polls_before_terminal = polls_before_terminal
        self.delete_status = delete_status
        self.uploads: list[bytes] = []
        self.batch_creates: list[dict] = []
        self.deletes: list[str] = []
        self.polls = 0

    def _batch_json(self, status: str) -> dict:
        return {
            "id": "batch_1", "object": "batch", "endpoint": "/v1/chat/completions",
            "input_file_id": "file-in", "completion_window": "24h", "status": status,
            "created_at": 1,
            "output_file_id": "file-out" if self.output_lines is not None else None,
            "error_file_id": "file-err" if self.error_lines is not None else None,
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path == "/v1/files":
            self.uploads.append(request.content)
            return httpx.Response(200, json={
                "id": "file-in", "object": "file", "bytes": 1, "created_at": 1,
                "filename": "ehrextract_batch.jsonl", "purpose": "batch",
            })
        if method == "POST" and path == "/v1/batches":
            self.batch_creates.append(json.loads(request.content))
            return httpx.Response(200, json=self._batch_json("validating"))
        if method == "GET" and path == "/v1/batches/batch_1":
            self.polls += 1
            if self.polls < self.polls_before_terminal:
                return httpx.Response(200, json=self._batch_json("in_progress"))
            return httpx.Response(200, json=self._batch_json(self.terminal_status))
        if method == "GET" and path == "/v1/files/file-out/content":
            return httpx.Response(200, text="\n".join(self.output_lines))
        if method == "GET" and path == "/v1/files/file-err/content":
            return httpx.Response(200, text="\n".join(self.error_lines))
        if method == "DELETE" and path.startswith("/v1/files/"):
            fid = path.rsplit("/", 1)[1]
            self.deletes.append(fid)
            return httpx.Response(self.delete_status,
                                  json={"id": fid, "object": "file", "deleted": True})
        raise AssertionError(f"unexpected request {method} {path}")


def _provider(server: _BatchServer) -> OpenAIProvider:
    p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    transport = httpx.MockTransport(server.handler)
    p._client = OpenAI(api_key="sk-test", http_client=httpx.Client(transport=transport))
    return p


_MSGS = [[{"role": "system", "content": "SYS"}, {"role": "user", "content": f"note {i}"}]
         for i in range(3)]


def test_supports_batch_is_true():
    p = OpenAIProvider(model="m", api_key="sk")
    assert p.supports_batch is True


def test_batch_lifecycle_roundtrip(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(providers.time, "sleep", sleeps.append)
    # Output served OUT of input order to prove custom_id matching.
    server = _BatchServer(
        output_lines=[_output_line(2, "C"), _output_line(0, "A"), _output_line(1, "B")],
        polls_before_terminal=2,
    )
    p = _provider(server)
    results = p.generate_batch(_MSGS, GenerationConfig(max_new_tokens=64))

    upload = server.uploads[0]
    for i in range(3):
        assert f'"custom_id": "req-{i}"'.encode() in upload
    assert b'"url": "/v1/chat/completions"' in upload
    assert b'"max_tokens": 64' in upload
    assert server.batch_creates[0]["input_file_id"] == "file-in"
    assert server.batch_creates[0]["completion_window"] == "24h"

    assert [r.text for r in results] == ["A", "B", "C"]  # input order restored
    assert all(isinstance(r, ProviderResponse) for r in results)
    assert results[0].usage == {"input_tokens": 10, "output_tokens": 5}
    assert len(sleeps) == 2  # validating -> in_progress -> completed


def test_batch_error_file_rows_become_exception_slots(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer(
        output_lines=[_output_line(0, "A"), _output_line(1, "B")],
        error_lines=[_error_line(2, "boom")],
    )
    p = _provider(server)
    results = p.generate_batch(_MSGS, GenerationConfig())
    assert isinstance(results[0], ProviderResponse)
    assert isinstance(results[1], ProviderResponse)
    assert isinstance(results[2], Exception)
    assert "req-2" in str(results[2])
    assert "boom" in str(results[2])


def test_batch_missing_custom_id_becomes_exception_slot(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer(output_lines=[_output_line(0, "A")])
    p = _provider(server)
    results = p.generate_batch(_MSGS[:2], GenerationConfig())
    assert isinstance(results[0], ProviderResponse)
    assert isinstance(results[1], Exception)
    assert "no result" in str(results[1])


@pytest.mark.parametrize("status", ["failed", "expired"])
def test_batch_terminal_failure_raises_with_batch_id(monkeypatch, status):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer(output_lines=[], terminal_status=status)
    p = _provider(server)
    with pytest.raises(RuntimeError, match="batch_1"):
        p.generate_batch(_MSGS, GenerationConfig())


def test_batch_files_deleted_after_retrieval(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer(
        output_lines=[_output_line(i, t) for i, t in enumerate("ABC")],
        error_lines=[],
    )
    p = _provider(server)
    p.generate_batch(_MSGS, GenerationConfig())
    assert set(server.deletes) == {"file-in", "file-out", "file-err"}


def test_batch_file_deletion_failure_only_warns(monkeypatch, caplog):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer(
        output_lines=[_output_line(i, t) for i, t in enumerate("ABC")],
        delete_status=500,
    )
    p = _provider(server)
    with caplog.at_level(logging.WARNING, logger="ehrextract.providers"):
        results = p.generate_batch(_MSGS, GenerationConfig())
    assert all(isinstance(r, ProviderResponse) for r in results)  # run still succeeds
    assert any("could not delete" in r.getMessage() for r in caplog.records)


def test_batch_anomalous_custom_id_skipped_with_warning(monkeypatch, caplog):
    """An out-of-range or garbage custom_id from the API must not crash the
    retrieval of a paid batch; the unmatched slot keeps its no-result error."""
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    server = _BatchServer(output_lines=[
        _output_line(999, "ghost"),  # index far past the 3 inputs
        json.dumps({"id": "x", "custom_id": "garbage",
                    "response": {"status_code": 200, "request_id": "r",
                                 "body": _completion_body("?")},
                    "error": None}),
        _output_line(0, "A"),
    ])
    p = _provider(server)
    with caplog.at_level(logging.WARNING, logger="ehrextract.providers"):
        results = p.generate_batch(_MSGS, GenerationConfig())
    assert results[0].text == "A"  # the legitimate row still maps
    assert isinstance(results[1], Exception) and "no result" in str(results[1])
    assert isinstance(results[2], Exception) and "no result" in str(results[2])
    warned = [r.getMessage() for r in caplog.records if "unknown custom_id" in r.getMessage()]
    assert len(warned) == 2


def test_poll_sleeps_with_interval(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(providers.time, "sleep", sleeps.append)
    server = _BatchServer(
        output_lines=[_output_line(i, t) for i, t in enumerate("ABC")],
        polls_before_terminal=3,
    )
    p = _provider(server)
    p.generate_batch(_MSGS, GenerationConfig())
    assert len(sleeps) == 3
    assert all(s >= providers._OPENAI_BATCH_POLL_SECONDS for s in sleeps)
