"""Tests for the extract() one-call convenience function (provider mocked)."""

import importlib
import inspect
import json

import pandas as pd
import pytest

import ehrextract.pipeline as pipeline_mod
from ehrextract import extract
from ehrextract.schema import FieldSpec, Schema

CLINICAL_VARS_RESPONSE = json.dumps({
    "tube_feeding": "Y",
    "oral_feeding": "N",
    "aspiration_risk": "Y",
    "ni_progressive_or_static": "N/A",
})


def _schema():
    return Schema(fields=(
        FieldSpec(name="diagnosis", kind="string"),
        FieldSpec(name="smoker", kind="enum", enum_values=("Y", "N")),
    ))


@pytest.fixture
def patched_provider(monkeypatch, mock_provider_cls):
    """Capture load_provider calls and hand back a MockProvider."""
    created: dict = {}

    def fake_load_provider(name, **kwargs):
        created["name"] = name
        created["kwargs"] = kwargs
        created["provider"] = mock_provider_cls(created.pop("responses", None))
        return created["provider"]

    monkeypatch.setattr(pipeline_mod, "load_provider", fake_load_provider)
    return created


def test_pipeline_is_module_and_extract_is_function():
    """A1 pin: after the extract.py -> pipeline.py rename there is no submodule
    shadowing. `ehrextract.pipeline` is a plain module; the package attribute
    `ehrextract.extract` is the function it exports."""
    import ehrextract

    assert inspect.ismodule(pipeline_mod)
    assert inspect.isfunction(ehrextract.extract)
    assert ehrextract.extract is pipeline_mod.extract
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ehrextract.extract")


def test_extract_with_schema_object(patched_provider):
    patched_provider["responses"] = [json.dumps({"diagnosis": "x", "smoker": "Y"})]
    df = extract(["note"], _schema(), provider="openai", model="gpt-4o-mini",
                 on_egress="silent")
    assert patched_provider["name"] == "openai"
    assert df["diagnosis"].iloc[0] == "x"
    assert bool(df["parse_success"].iloc[0])


def test_extract_builtin_task_name(patched_provider):
    patched_provider["responses"] = [CLINICAL_VARS_RESPONSE]
    df = extract(["note"], "clinical_vars", provider="openai", model="m",
                 on_egress="silent")
    assert bool(df["parse_success"].iloc[0])
    assert df["ni_progressive_or_static"].iloc[0] == "N/A"


def test_extract_requires_model(patched_provider):
    with pytest.raises(ValueError, match="model"):
        extract(["note"], _schema(), provider="openai")


def test_adapter_rejected_for_non_hf_provider(patched_provider):
    with pytest.raises(ValueError, match="adapter"):
        extract(["note"], _schema(), provider="openai", model="m", adapter="/x")


def test_output_written(patched_provider, tmp_path):
    patched_provider["responses"] = [json.dumps({"diagnosis": "x", "smoker": "Y"})]
    out = tmp_path / "results.csv"
    extract(["note"], _schema(), provider="openai", model="m", output=out,
            on_egress="silent")
    loaded = pd.read_csv(out)
    assert list(loaded["diagnosis"]) == ["x"]
    assert "parse_success" in loaded.columns


def test_prompt_override_reaches_system_message(patched_provider):
    extract(["note"], _schema(), provider="openai", model="m",
            prompt="CUSTOM SYS", on_egress="silent")
    messages, _, _ = patched_provider["provider"].calls[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("CUSTOM SYS")


def test_adapter_system_prompt_used_verbatim(patched_provider, tmp_path):
    (tmp_path / "system_prompt.txt").write_text("ADAPTER PROMPT\n", encoding="utf-8")
    extract(["note"], _schema(), provider="huggingface", model="m",
            adapter=str(tmp_path), on_egress="silent")
    assert patched_provider["kwargs"]["adapter_path"] == str(tmp_path)
    messages, _, _ = patched_provider["provider"].calls[0]
    assert messages[0]["content"] == "ADAPTER PROMPT\n"


def test_base_url_forwarded_only_to_openai(patched_provider):
    extract(["note"], _schema(), provider="anthropic", model="m",
            base_url="http://x", api_key="k", on_egress="silent")
    assert "base_url" not in patched_provider["kwargs"]
    assert patched_provider["kwargs"]["api_key"] == "k"

    extract(["note"], _schema(), provider="openai", model="m",
            base_url="http://x", api_key="k", on_egress="silent")
    assert patched_provider["kwargs"]["base_url"] == "http://x"


def test_trust_remote_code_and_dtype_forwarded_to_hf(patched_provider):
    """A2: the HF loader knobs flow from extract() into the provider kwargs."""
    extract(["note"], _schema(), provider="huggingface", model="m",
            trust_remote_code=True, dtype="float16", on_egress="silent")
    assert patched_provider["kwargs"]["trust_remote_code"] is True
    assert patched_provider["kwargs"]["dtype"] == "float16"

    extract(["note"], _schema(), provider="openai", model="m",
            trust_remote_code=True, dtype="float16", on_egress="silent")
    assert "trust_remote_code" not in patched_provider["kwargs"]
    assert "dtype" not in patched_provider["kwargs"]


# --- A7: fail-fast ordering -- bad inputs error before the provider is built ---


@pytest.fixture
def provider_must_not_build(monkeypatch):
    def boom(name, **kwargs):
        raise AssertionError("load_provider must not be reached")

    monkeypatch.setattr(pipeline_mod, "load_provider", boom)


def test_bad_output_extension_fails_before_provider(provider_must_not_build):
    with pytest.raises(ValueError, match="extension"):
        extract(["note"], _schema(), provider="openai", model="m",
                output="results.bogus", on_egress="silent")


def test_bad_notes_fail_before_provider(provider_must_not_build):
    with pytest.raises(ValueError, match="note_text"):
        extract(pd.DataFrame({"wrong_col": ["x"]}), _schema(),
                provider="openai", model="m", on_egress="silent")
