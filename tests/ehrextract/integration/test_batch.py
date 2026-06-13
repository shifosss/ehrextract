"""Batch-orchestration tests (v0.3 D5): Extractor.run(batch=True) routing."""

import json

import pytest

import ehrextract.pipeline as pipeline
from ehrextract.pipeline import Extractor
from ehrextract.providers import ProviderResponse
from ehrextract.schema import FieldSpec, Schema, Task, to_json_schema

VALID = json.dumps({"diagnosis": "d", "smoker": "Y"})
VALID2 = json.dumps({"diagnosis": "e", "smoker": "N"})


def _task() -> Task:
    schema = Schema(fields=(
        FieldSpec(name="diagnosis", kind="string"),
        FieldSpec(name="smoker", kind="enum", enum_values=("Y", "N")),
    ))
    return Task(name="t", schema=schema, prompt="extract", generation={})


def _pr(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, finish_reason="stop",
                            usage={"input_tokens": 10, "output_tokens": 5}, raw=None)


def _batch_provider(mock_provider_cls, batch_results, responses=None, egress=None):
    class P(mock_provider_cls):
        supports_batch = True

        def __init__(self):
            super().__init__(responses or [])
            self.batch_calls: list[tuple] = []

        def generate_batch(self, batch_messages, config, json_schema=None):
            self.batch_calls.append((batch_messages, config, json_schema))
            return list(batch_results)

        def egress_destination(self):
            return egress

    return P()


def test_run_batch_routes_through_generate_batch(mock_provider_cls):
    p = _batch_provider(mock_provider_cls, [_pr(VALID), _pr(VALID2)])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(["n0", "n1"], batch=True)
    assert p.calls == []  # sync generate never used
    assert len(p.batch_calls) == 1
    batch_messages, _, json_schema = p.batch_calls[0]
    assert len(batch_messages) == 2
    assert json_schema == to_json_schema(_task().schema)
    assert list(out["diagnosis"]) == ["d", "e"]
    assert list(out["parse_success"]) == [True, True]


def test_run_batch_skips_empty_notes_locally(mock_provider_cls):
    p = _batch_provider(mock_provider_cls, [_pr(VALID)])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(["", "n1"], batch=True)
    batch_messages, _, _ = p.batch_calls[0]
    assert len(batch_messages) == 1  # the empty note never left the machine
    assert out["finish_reason"].iloc[0] == "skipped"
    assert "empty_note" in out["validation_errors"].iloc[0]
    assert bool(out["parse_success"].iloc[1])  # order preserved


def test_run_batch_exception_slot_becomes_provider_error_row(mock_provider_cls):
    p = _batch_provider(mock_provider_cls, [_pr(VALID), RuntimeError("poisoned row")])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(["n0", "n1"], batch=True)
    assert bool(out["parse_success"].iloc[0])
    assert out["finish_reason"].iloc[1] == "error"
    assert "provider_error" in out["validation_errors"].iloc[1]
    assert "poisoned row" in out["validation_errors"].iloc[1]


def test_run_batch_failed_rows_repaired_synchronously(mock_provider_cls):
    p = _batch_provider(mock_provider_cls, [_pr("not json"), _pr(VALID2)], responses=[VALID])
    e = Extractor(p, _task(), on_egress="silent", max_repairs=1)
    out = e.run(["n0", "n1"], batch=True, max_concurrency=1)
    assert len(p.calls) == 1  # exactly one sync repair call, for the failed row
    assert list(out["parse_success"]) == [True, True]
    assert list(out["repair_attempts"]) == [1, 0]


def test_run_batch_output_columns_match_sync(mock_provider_cls):
    p = _batch_provider(mock_provider_cls, [_pr(VALID)])
    e = Extractor(p, _task(), on_egress="silent")
    out_batch = e.run(["n"], batch=True)
    p2 = mock_provider_cls([VALID])
    out_sync = Extractor(p2, _task(), on_egress="silent").run(["n"], max_concurrency=1)
    assert list(out_batch.columns) == list(out_sync.columns)
    assert "repair_attempts" in out_batch.columns


def test_run_batch_mismatched_result_count_raises(mock_provider_cls):
    p = _batch_provider(mock_provider_cls, [_pr(VALID)])
    e = Extractor(p, _task(), on_egress="silent")
    with pytest.raises(RuntimeError, match="batch results"):
        e.run(["n0", "n1"], batch=True)


def test_run_batch_unsupported_provider_raises_valueerror(mock_provider_cls):
    e = Extractor(mock_provider_cls(), _task(), on_egress="silent")
    with pytest.raises(ValueError, match="batch"):
        e.run(["n"], batch=True)


def test_extract_batch_with_huggingface_rejected_before_provider_build(monkeypatch):
    monkeypatch.setattr(
        pipeline, "load_provider", lambda *a, **k: pytest.fail("provider must not be built")
    )
    with pytest.raises(ValueError, match="batch"):
        pipeline.extract("note", _task(), provider="huggingface", model="m", batch=True)


def test_run_batch_egress_warned_once(mock_provider_cls, capsys):
    p = _batch_provider(mock_provider_cls, [_pr(VALID)],
                        egress="batch-test.example.invalid")
    e = Extractor(p, _task(), on_egress="warn")
    e.run(["n"], batch=True)
    err = capsys.readouterr().err
    assert err.count("batch-test.example.invalid") == 1
