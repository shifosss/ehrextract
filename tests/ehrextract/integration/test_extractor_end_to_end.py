"""End-to-end Extractor tests using a mock provider."""

import json
import logging
import threading
import time

import pandas as pd
import pytest

import ehrextract.pipeline as pipeline
import ehrextract.providers as providers
from ehrextract.pipeline import Extractor
from ehrextract.providers import GenerationConfig
from ehrextract.schema import FieldSpec, Schema, Task, to_json_schema

VALID = json.dumps({"diagnosis": "d", "smoker": "Y"})


def _task(generation: dict | None = None) -> Task:
    schema = Schema(fields=(
        FieldSpec(name="diagnosis", kind="string"),
        FieldSpec(name="smoker", kind="enum", enum_values=("Y", "N")),
    ))
    return Task(name="t", schema=schema, prompt="extract", generation=generation or {})


def test_run_one_returns_extraction_result(mock_provider_cls):
    p = mock_provider_cls([json.dumps({"diagnosis": "asthma", "smoker": "Y"})],
                          usage={"input_tokens": 10, "output_tokens": 5})
    e = Extractor(p, _task(), on_egress="silent")
    result = e.run_one("note text", note_id="n1")
    assert result.note_id == "n1"
    assert result.fields == {"diagnosis": "asthma", "smoker": "Y"}
    assert result.parse_success is True
    assert result.validation_errors == []
    assert result.usage == {"input_tokens": 10, "output_tokens": 5}


def test_json_schema_passed_to_generate(mock_provider_cls):
    p = mock_provider_cls([json.dumps({"diagnosis": "x", "smoker": "Y"})])
    task = _task()
    e = Extractor(p, task, on_egress="silent")
    e.run_one("note")
    _, _, json_schema = p.calls[0]
    assert json_schema == to_json_schema(task.schema)


def test_run_with_dataframe(mock_provider_cls):
    p = mock_provider_cls([
        json.dumps({"diagnosis": "asthma", "smoker": "Y"}),
        json.dumps({"diagnosis": "diabetes", "smoker": "N"}),
    ])
    e = Extractor(p, _task(), on_egress="silent")
    notes = pd.DataFrame({"note_id": ["a", "b"], "note_text": ["n1", "n2"]})
    out = e.run(notes, max_concurrency=1)
    assert list(out["note_id"]) == ["a", "b"]
    assert list(out["diagnosis"]) == ["asthma", "diabetes"]
    assert list(out["smoker"]) == ["Y", "N"]
    assert all(out["parse_success"])


def test_run_with_inline_string(mock_provider_cls):
    p = mock_provider_cls([json.dumps({"diagnosis": "x", "smoker": "Y"})])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run("a single note", max_concurrency=1)
    assert len(out) == 1
    assert out["diagnosis"].iloc[0] == "x"


def test_run_with_list_of_strings(mock_provider_cls):
    p = mock_provider_cls([
        json.dumps({"diagnosis": "x", "smoker": "Y"}),
        json.dumps({"diagnosis": "y", "smoker": "N"}),
    ])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(["note1", "note2"], max_concurrency=1)
    assert list(out["note_id"]) == [0, 1]
    assert list(out["diagnosis"]) == ["x", "y"]


def test_run_with_list_of_dicts(mock_provider_cls):
    p = mock_provider_cls([
        json.dumps({"diagnosis": "asthma", "smoker": "Y"}),
        json.dumps({"diagnosis": "diabetes", "smoker": "N"}),
    ])
    e = Extractor(p, _task(), on_egress="silent")
    notes = [
        {"note_id": "x", "note_text": "first"},
        {"note_id": "y", "note_text": "second"},
    ]
    out = e.run(notes, max_concurrency=1)
    assert list(out["note_id"]) == ["x", "y"]
    assert list(out["diagnosis"]) == ["asthma", "diabetes"]


def test_run_with_path_to_jsonl(mock_provider_cls, tmp_path):
    notes_file = tmp_path / "notes.jsonl"
    notes_file.write_text(
        '{"note_id": "a", "note_text": "first"}\n'
        '{"note_id": "b", "note_text": "second"}\n'
    )
    p = mock_provider_cls([
        json.dumps({"diagnosis": "x", "smoker": "Y"}),
        json.dumps({"diagnosis": "y", "smoker": "N"}),
    ])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(notes_file, max_concurrency=1)
    assert list(out["note_id"]) == ["a", "b"]
    assert list(out["diagnosis"]) == ["x", "y"]


def test_run_with_empty_list_returns_empty_frame(mock_provider_cls):
    e = Extractor(mock_provider_cls(), _task(), on_egress="silent")
    out = e.run([])
    assert len(out) == 0
    assert "diagnosis" in out.columns
    assert "parse_success" in out.columns


def test_raw_response_empty_on_success(mock_provider_cls):
    p = mock_provider_cls([json.dumps({"diagnosis": "x", "smoker": "Y"})])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(["n"], max_concurrency=1)
    assert out["raw_response"].iloc[0] == ""


def test_raw_response_full_text_on_failure(mock_provider_cls):
    """Design decision 9: full raw text on parse failure (no 500-char truncation)."""
    raw = "totally not json " + "x" * 600
    p = mock_provider_cls([raw])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(["n"], max_concurrency=1)
    assert not bool(out["parse_success"].iloc[0])
    assert out["raw_response"].iloc[0] == raw
    assert out["validation_errors"].iloc[0]


def test_provider_exception_does_not_crash_batch(mock_provider_cls):
    class BoomProvider(mock_provider_cls):
        def generate(self, messages, config, json_schema=None):
            raise RuntimeError("network fail")

    e = Extractor(BoomProvider(), _task(), on_egress="silent", max_retries=1)
    out = e.run(["n"], max_concurrency=1)
    assert len(out) == 1
    assert not bool(out["parse_success"].iloc[0])
    assert "provider_error" in out["validation_errors"].iloc[0]
    assert out["finish_reason"].iloc[0] == "error"


def test_concurrency_preserves_order(mock_provider_cls):
    """Output rows must follow input order even when futures complete out of order.

    The shared-list MockProvider is racy on which note gets which response
    under concurrency, so we use identical responses and verify the contract
    that actually matters: input order is preserved at the row level.
    """
    p = mock_provider_cls([
        json.dumps({"diagnosis": "d", "smoker": "Y"}) for _ in range(10)
    ])
    e = Extractor(p, _task(), on_egress="silent")
    notes = [f"note-{i}" for i in range(10)]
    out = e.run(notes, max_concurrency=4)
    assert list(out["note_id"]) == list(range(10))
    assert len(out) == 10


def test_output_dataframe_columns(mock_provider_cls):
    p = mock_provider_cls([json.dumps({"diagnosis": "x", "smoker": "Y"})])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(["n"], max_concurrency=1)
    expected = {
        "note_id", "diagnosis", "smoker",
        "parse_success", "validation_errors",
        "raw_response", "finish_reason",
        "input_tokens", "output_tokens",
    }
    assert expected.issubset(set(out.columns))


# --- generation precedence: GenerationConfig defaults < task.generation < arg ---


def test_generation_defaults(mock_provider_cls):
    e = Extractor(mock_provider_cls(), _task(), on_egress="silent")
    assert e.generation == GenerationConfig()


def test_task_generation_overrides_defaults(mock_provider_cls):
    e = Extractor(
        mock_provider_cls(),
        _task(generation={"max_new_tokens": 500, "temperature": 0.7}),
        on_egress="silent",
    )
    assert e.generation.max_new_tokens == 500
    assert e.generation.temperature == 0.7
    assert e.generation.repetition_penalty == 1.0


def test_generation_dict_arg_overrides_task(mock_provider_cls):
    e = Extractor(
        mock_provider_cls(),
        _task(generation={"max_new_tokens": 500, "temperature": 0.7}),
        generation={"max_new_tokens": 100},
        on_egress="silent",
    )
    assert e.generation.max_new_tokens == 100
    assert e.generation.temperature == 0.7  # untouched task value survives


def test_generation_config_arg_is_full_override(mock_provider_cls):
    """A GenerationConfig sets every field explicitly, so it replaces task values."""
    e = Extractor(
        mock_provider_cls(),
        _task(generation={"max_new_tokens": 500, "temperature": 0.7}),
        generation=GenerationConfig(max_new_tokens=64),
        on_egress="silent",
    )
    assert e.generation.max_new_tokens == 64
    assert e.generation.temperature == 0.0


def test_merged_generation_reaches_provider(mock_provider_cls):
    p = mock_provider_cls([json.dumps({"diagnosis": "x", "smoker": "Y"})])
    e = Extractor(p, _task(generation={"max_new_tokens": 500}), on_egress="silent")
    e.run_one("note")
    _, config, _ = p.calls[0]
    assert config.max_new_tokens == 500


# --- egress check inside the Extractor ---


def test_run_warns_once_for_remote_provider(mock_provider_cls, monkeypatch, capsys):
    monkeypatch.setattr(providers, "_WARNED_DESTINATIONS", set())
    monkeypatch.delenv("ACK_EGRESS", raising=False)

    class RemoteProvider(mock_provider_cls):
        def egress_destination(self):
            return "remote.example.com"

    p = RemoteProvider([json.dumps({"diagnosis": "x", "smoker": "Y"}) for _ in range(3)])
    e = Extractor(p, _task(), on_egress="warn")
    e.run(["a", "b", "c"], max_concurrency=2)
    err = capsys.readouterr().err
    assert err.count("remote.example.com") == 1


def test_run_silent_mode_suppresses_warning(mock_provider_cls, monkeypatch, capsys):
    monkeypatch.setattr(providers, "_WARNED_DESTINATIONS", set())
    monkeypatch.delenv("ACK_EGRESS", raising=False)

    class RemoteProvider(mock_provider_cls):
        def egress_destination(self):
            return "silent.example.com"

    p = RemoteProvider([json.dumps({"diagnosis": "x", "smoker": "Y"})])
    e = Extractor(p, _task(), on_egress="silent")
    e.run(["a"], max_concurrency=1)
    assert capsys.readouterr().err == ""


# --- A5: input normalization, empty-note error rows, duplicate-id warning ---


def test_dataframe_without_id_column_synthesizes_ids(mock_provider_cls):
    p = mock_provider_cls([VALID, VALID])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run(pd.DataFrame({"note_text": ["a", "b"]}), max_concurrency=1)
    assert list(out["note_id"]) == [0, 1]


def test_list_of_dicts_without_id_column_synthesizes_ids(mock_provider_cls):
    p = mock_provider_cls([VALID, VALID])
    e = Extractor(p, _task(), on_egress="silent")
    out = e.run([{"note_text": "a"}, {"note_text": "b"}], max_concurrency=1)
    assert list(out["note_id"]) == [0, 1]


def test_missing_text_column_raises_naming_it(mock_provider_cls):
    e = Extractor(mock_provider_cls(), _task(), on_egress="silent")
    with pytest.raises(ValueError, match="note_text"):
        e.run(pd.DataFrame({"body": ["a"]}), max_concurrency=1)


def test_empty_notes_become_error_rows_without_provider_call(mock_provider_cls):
    p = mock_provider_cls([VALID])
    e = Extractor(p, _task(), on_egress="silent")
    notes = pd.DataFrame({
        "note_id": [1, 2, 3, 4],
        "note_text": ["good note", "", "   ", None],
    })
    out = e.run(notes, max_concurrency=1)
    assert len(p.calls) == 1  # provider called for the good note only
    assert list(out["finish_reason"]) == ["stop", "skipped", "skipped", "skipped"]
    assert bool(out["parse_success"].iloc[0])
    assert not any(bool(v) for v in out["parse_success"].iloc[1:])
    assert "_input:empty_note:note text is empty" in out["validation_errors"].iloc[1]
    assert "_input:empty_note:note text is empty" in out["validation_errors"].iloc[2]
    assert "_input:empty_note:note text is null" in out["validation_errors"].iloc[3]
    assert out["raw_response"].iloc[1] == ""
    assert pd.isna(out["input_tokens"].iloc[1])
    assert pd.isna(out["output_tokens"].iloc[1])


def test_duplicate_ids_warn_once(mock_provider_cls, caplog):
    p = mock_provider_cls([VALID, VALID])
    e = Extractor(p, _task(), on_egress="silent")
    notes = pd.DataFrame({"note_id": ["a", "a"], "note_text": ["x", "y"]})
    with caplog.at_level(logging.WARNING, logger="ehrextract.pipeline"):
        out = e.run(notes, max_concurrency=1)
    dup_records = [r for r in caplog.records if "duplicate" in r.message]
    assert len(dup_records) == 1
    assert len(out) == 2  # duplicates still processed


# --- A6: concurrency validation, local clamp, worker-failure cancellation ---


def test_max_concurrency_below_one_rejected(mock_provider_cls):
    e = Extractor(mock_provider_cls([VALID]), _task(), on_egress="silent")
    with pytest.raises(ValueError, match="max_concurrency"):
        e.run(["n"], max_concurrency=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        e.run(["n"], max_concurrency=-2)


def test_local_provider_clamps_concurrency_with_warning(mock_provider_cls, caplog):
    p = mock_provider_cls([VALID, VALID])  # egress None, default_concurrency 4
    e = Extractor(p, _task(), on_egress="silent")
    with caplog.at_level(logging.WARNING, logger="ehrextract.pipeline"):
        out = e.run(["a", "b"], max_concurrency=64)
    assert len(out) == 2
    clamp = [r for r in caplog.records if "clamp" in r.getMessage()]
    assert len(clamp) == 1
    assert "mock" in clamp[0].getMessage()
    assert "4" in clamp[0].getMessage()


def test_remote_provider_concurrency_not_clamped(mock_provider_cls, caplog):
    class RemoteProvider(mock_provider_cls):
        def egress_destination(self):
            return "remote.example.com"

    p = RemoteProvider([VALID, VALID])
    e = Extractor(p, _task(), on_egress="silent")
    with caplog.at_level(logging.WARNING, logger="ehrextract.pipeline"):
        e.run(["a", "b"], max_concurrency=64)
    assert not any("clamp" in r.getMessage() for r in caplog.records)


def test_worker_exception_cancels_pending_futures(mock_provider_cls):
    """First worker failure stops the queue (no further PHI egress).

    KeyboardInterrupt is a BaseException, so it escapes the retry loop's
    `except Exception` and propagates out of the worker.
    """
    lock = threading.Lock()
    calls: list[int] = []

    class InterruptingProvider(mock_provider_cls):
        def generate(self, messages, config, json_schema=None):
            with lock:
                calls.append(1)
                first = len(calls) == 1
            if first:
                raise KeyboardInterrupt
            time.sleep(0.3)
            return super().generate(messages, config, json_schema)

    p = InterruptingProvider([VALID] * 10)
    e = Extractor(p, _task(), on_egress="silent")
    with pytest.raises(KeyboardInterrupt):
        e.run([f"note-{i}" for i in range(10)], max_concurrency=2)
    # Without cancel_futures the executor would drain all 10 queued notes
    # before run() re-raised. With it, only in-flight work completes.
    assert len(calls) < 10


# --- A9: retry policy (no retry on client errors; jittered backoff) ---


class _StatusError(RuntimeError):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


def test_client_error_401_not_retried(mock_provider_cls, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    calls: list[int] = []

    class P(mock_provider_cls):
        def generate(self, messages, config, json_schema=None):
            calls.append(1)
            raise _StatusError("unauthorized", 401)

    e = Extractor(P(), _task(), on_egress="silent", max_retries=3)
    out = e.run(["n"], max_concurrency=1)
    assert len(calls) == 1
    assert sleeps == []
    assert out["finish_reason"].iloc[0] == "error"
    assert "provider_error" in out["validation_errors"].iloc[0]


def test_server_error_500_retried_with_jittered_backoff(mock_provider_cls, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    calls: list[int] = []

    class P(mock_provider_cls):
        def generate(self, messages, config, json_schema=None):
            calls.append(1)
            raise _StatusError("server blew up", 500)

    e = Extractor(P(), _task(), on_egress="silent", max_retries=3)
    out = e.run(["n"], max_concurrency=1)
    assert len(calls) == 3
    assert len(sleeps) == 2  # no sleep after the final attempt
    assert 1.0 <= sleeps[0] < 1.5  # 2**0 + uniform(0, 0.5)
    assert 2.0 <= sleeps[1] < 2.5  # 2**1 + uniform(0, 0.5)
    assert out["finish_reason"].iloc[0] == "error"
