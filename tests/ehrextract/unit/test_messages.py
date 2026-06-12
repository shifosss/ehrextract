"""Tests for chat-message construction and adapter-prompt resolution."""

import logging

import pytest

from ehrextract.pipeline import build_default_prompt, build_messages, resolve_adapter_prompt
from ehrextract.schema import FieldSpec, Schema, Task


def _schema():
    return Schema(
        name="t",
        description="extract these",
        fields=(
            FieldSpec(name="diagnosis", kind="string"),
            FieldSpec(name="smoker", kind="enum", enum_values=("Y", "N")),
        ),
    )


def _task(**kwargs):
    defaults = {"name": "t", "schema": _schema(), "prompt": "SYSTEM PROMPT"}
    defaults.update(kwargs)
    return Task(**defaults)


def test_messages_have_two_roles():
    msgs = build_messages(_task(), "NOTE", schema_native=False)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "SYSTEM PROMPT" in msgs[0]["content"]
    assert "NOTE" in msgs[1]["content"]


def test_non_native_includes_json_shape_in_system():
    msgs = build_messages(_task(), "NOTE", schema_native=False)
    system = msgs[0]["content"]
    assert "Respond with JSON ONLY" in system
    assert "diagnosis" in system
    assert "smoker" in system
    assert '"enum"' in system  # json.dumps of the JSON Schema
    assert "- smoker (enum" in system  # per-field summary line


def test_native_omits_json_shape():
    msgs = build_messages(_task(), "NOTE", schema_native=True)
    assert msgs[0]["content"] == "SYSTEM PROMPT"
    assert "Respond with JSON ONLY" not in msgs[0]["content"]


def test_verbatim_prompt_is_byte_for_byte():
    raw = "RAW PROMPT  \n\n"
    msgs = build_messages(_task(prompt=raw, prompt_verbatim=True), "NOTE", schema_native=False)
    assert msgs[0]["content"] == raw
    assert "Respond with JSON ONLY" not in msgs[0]["content"]


def test_missing_prompt_falls_back_to_default():
    msgs = build_messages(_task(prompt=None), "NOTE", schema_native=True)
    assert "Extract the following fields" in msgs[0]["content"]
    assert "extract these" in msgs[0]["content"]


def test_user_template_formats_note():
    t = _task(user_template="Clinical note: {note}\n\nGo:")
    msgs = build_messages(t, "NOTE", schema_native=True)
    assert msgs[1]["content"] == "Clinical note: NOTE\n\nGo:"


def test_build_default_prompt_without_description():
    schema = Schema(fields=(FieldSpec(name="x", kind="string"),))
    assert build_default_prompt(schema) == "Extract the following fields from the clinical note."


# --- resolve_adapter_prompt ---


def test_resolve_no_adapter_no_prompt_is_unchanged():
    t = _task()
    assert resolve_adapter_prompt(t, None, None) == t


def test_resolve_adapter_system_prompt_is_verbatim(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("ADAPTER PROMPT\n", encoding="utf-8")
    out = resolve_adapter_prompt(_task(), tmp_path, None)
    assert out.prompt == "ADAPTER PROMPT\n"
    assert out.prompt_verbatim is True


def test_resolve_adapter_without_prompt_file_is_unchanged(tmp_path):
    t = _task()
    assert resolve_adapter_prompt(t, tmp_path, None) == t


def test_resolve_explicit_prompt_wins():
    out = resolve_adapter_prompt(_task(), None, "EXPLICIT")
    assert out.prompt == "EXPLICIT"
    assert out.prompt_verbatim is False


def test_resolve_explicit_prompt_overrides_adapter_with_warning(tmp_path, caplog):
    (tmp_path / "system_prompt.txt").write_text("ADAPTER PROMPT\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="ehrextract.pipeline"):
        out = resolve_adapter_prompt(_task(), tmp_path, "EXPLICIT")
    assert out.prompt == "EXPLICIT"
    assert out.prompt_verbatim is False
    assert any("overrides" in r.message for r in caplog.records)


def test_resolve_empty_adapter_prompt_raises(tmp_path):
    """A3: an empty/whitespace-only system_prompt.txt is a broken adapter dir."""
    (tmp_path / "system_prompt.txt").write_text("   \n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        resolve_adapter_prompt(_task(), tmp_path, None)
