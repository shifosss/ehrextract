"""Repair-loop tests (v0.3 D4). All response->row bindings run at max_concurrency=1."""

import json

import pytest

import ehrextract.pipeline as pipeline
from ehrextract.pipeline import Extractor
from ehrextract.providers import ProviderResponse
from ehrextract.schema import FieldSpec, Schema, Task

VALID = json.dumps({"diagnosis": "d", "smoker": "Y"})
BAD_ENUM = json.dumps({"diagnosis": "d", "smoker": "MAYBE"})


def _task() -> Task:
    schema = Schema(fields=(
        FieldSpec(name="diagnosis", kind="string"),
        FieldSpec(name="smoker", kind="enum", enum_values=("Y", "N")),
    ))
    return Task(name="t", schema=schema, prompt="extract", generation={})


class _StatusError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


def test_repair_disabled_by_default(mock_provider_cls):
    """v0.2 behavior preserved exactly: one call, error row, no repair."""
    p = mock_provider_cls(["not json"])
    e = Extractor(p, _task(), on_egress="silent")
    r = e.run_one("note")
    assert len(p.calls) == 1
    assert r.parse_success is False
    assert r.repair_attempts == 0


def test_negative_max_repairs_rejected(mock_provider_cls):
    with pytest.raises(ValueError, match="max_repairs"):
        Extractor(mock_provider_cls(), _task(), on_egress="silent", max_repairs=-1)


def test_repair_fixes_parse_failure(mock_provider_cls):
    p = mock_provider_cls(["not json", VALID])
    e = Extractor(p, _task(), on_egress="silent", max_repairs=1)
    out = e.run(["note"], max_concurrency=1)
    assert len(p.calls) == 2
    assert bool(out["parse_success"].iloc[0])
    assert out["repair_attempts"].iloc[0] == 1
    assert out["raw_response"].iloc[0] == ""  # success keeps the output clean


def test_repair_fixes_validation_failure(mock_provider_cls):
    """Repair fires on field validation errors, not just JSON parse failures."""
    p = mock_provider_cls([BAD_ENUM, VALID])
    e = Extractor(p, _task(), on_egress="silent", max_repairs=1)
    r = e.run_one("note")
    assert len(p.calls) == 2
    assert r.parse_success is True
    assert r.fields["smoker"] == "Y"
    assert r.repair_attempts == 1


def test_repair_messages_contain_assistant_echo_and_errors(mock_provider_cls):
    p = mock_provider_cls(["not json", VALID])
    e = Extractor(p, _task(), on_egress="silent", max_repairs=1)
    e.run_one("note text")
    first_messages, _, _ = p.calls[0]
    repair_messages, _, _ = p.calls[1]
    assert repair_messages[: len(first_messages)] == first_messages  # original turns intact
    assert repair_messages[-2] == {"role": "assistant", "content": "not json"}
    assert repair_messages[-1]["role"] == "user"
    assert "_response:coercion_failed" in repair_messages[-1]["content"]
    assert "ONLY the corrected JSON" in repair_messages[-1]["content"]


def test_repair_bounded_by_max_repairs(mock_provider_cls):
    p = mock_provider_cls(["bad 0", "bad 1", "bad 2"])
    e = Extractor(p, _task(), on_egress="silent", max_repairs=2)
    r = e.run_one("note")
    assert len(p.calls) == 3  # initial + 2 repairs, then stop
    assert r.parse_success is False
    assert r.repair_attempts == 2
    assert r.raw_response == "bad 2"  # the final attempt's text


def test_repair_not_triggered_on_provider_error(mock_provider_cls):
    calls: list[int] = []

    class P(mock_provider_cls):
        def generate(self, messages, config, json_schema=None):
            calls.append(1)
            raise _StatusError("unauthorized", 401)

    e = Extractor(P(), _task(), on_egress="silent", max_repairs=3)
    r = e.run_one("note")
    assert len(calls) == 1
    assert r.repair_attempts == 0
    assert r.finish_reason == "error"


def test_repair_not_triggered_on_empty_note(mock_provider_cls):
    p = mock_provider_cls()
    e = Extractor(p, _task(), on_egress="silent", max_repairs=2)
    out = e.run([""], max_concurrency=1)
    assert p.calls == []
    assert out["finish_reason"].iloc[0] == "skipped"
    assert out["repair_attempts"].iloc[0] == 0


def test_repair_stops_on_mid_loop_provider_error(mock_provider_cls):
    """A failed repair keeps the previous parse result; the row must not
    become a provider_error row."""
    calls: list[int] = []

    class P(mock_provider_cls):
        def generate(self, messages, config, json_schema=None):
            calls.append(1)
            if len(calls) == 1:
                return ProviderResponse(text="not json", finish_reason="stop", usage=None, raw=None)
            raise _StatusError("unauthorized", 401)

    e = Extractor(P(), _task(), on_egress="silent", max_repairs=3)
    r = e.run_one("note")
    assert len(calls) == 2  # initial + the one failed repair; loop stopped
    assert r.parse_success is False
    assert r.finish_reason == "stop"
    assert r.repair_attempts == 1
    assert all(err.code != "provider_error" for err in r.validation_errors)
    assert r.raw_response == "not json"


def test_usage_summed_across_repair_attempts(mock_provider_cls):
    p = mock_provider_cls(["not json", VALID], usage={"input_tokens": 10, "output_tokens": 5})
    e = Extractor(p, _task(), on_egress="silent", max_repairs=1)
    out = e.run(["n"], max_concurrency=1)
    assert out["input_tokens"].iloc[0] == 20
    assert out["output_tokens"].iloc[0] == 10
    assert out["repair_attempts"].iloc[0] == 1


def test_extract_passes_max_repairs(mock_provider_cls, monkeypatch):
    p = mock_provider_cls(["not json", VALID])
    monkeypatch.setattr(pipeline, "load_provider", lambda name, **kw: p)
    out = pipeline.extract(
        ["note"], _task(), provider="openai", model="m",
        on_egress="silent", max_repairs=1, max_concurrency=1,
    )
    assert bool(out["parse_success"].iloc[0])
    assert out["repair_attempts"].iloc[0] == 1
