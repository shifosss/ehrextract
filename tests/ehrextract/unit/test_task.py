"""Tests for Task and load_task: built-in tasks and task YAML files."""

from pathlib import Path

import pytest

from ehrextract.schema import SchemaError, load_task

COMORBIDITY_YN_FIELDS = (
    "fsd",
    "neurologic",
    "genetic_or_syndromic",
    "prematurity",
    "gastrointestinal_hepatopancreatic",
    "renal_or_urologic",
    "cardiovascular",
    "respiratory",
    "metabolic",
    "endocrine",
    "hematology_oncology",
    "immunologic",
    "mental_health",
    "developmental_or_behavioural",
    "musculoskeletal",
    "infectious_disease",
)
CLINICAL_VARS_FIELDS = (
    "tube_feeding",
    "oral_feeding",
    "aspiration_risk",
    "ni_progressive_or_static",
)
TRAINING_USER_TEMPLATE = "Clinical note: {note}\n\nProvide the classification in JSON format:"


# --- built-in tasks ---


def test_load_comorbidity_builtin():
    t = load_task("comorbidity")
    assert t.name == "comorbidity"
    assert len(t.schema.fields) == 17
    assert t.schema.field_names() == ("comorbidities_list", *COMORBIDITY_YN_FIELDS)
    assert t.schema.get_field("comorbidities_list").kind == "string"
    for name in COMORBIDITY_YN_FIELDS:
        f = t.schema.get_field(name)
        assert f.kind == "enum"
        assert f.enum_values == ("Y", "N")
    assert t.user_template == TRAINING_USER_TEMPLATE
    assert t.generation == {"max_new_tokens": 500}
    assert t.prompt and "comorbidities_list" in t.prompt
    assert t.prompt_verbatim is False


def test_load_clinical_vars_builtin():
    t = load_task("clinical_vars")
    assert t.name == "clinical_vars"
    assert len(t.schema.fields) == 4
    assert t.schema.field_names() == CLINICAL_VARS_FIELDS
    for name in ("tube_feeding", "oral_feeding", "aspiration_risk"):
        assert t.schema.get_field(name).enum_values == ("Y", "N")
    assert t.schema.get_field("ni_progressive_or_static").enum_values == (
        "Progressive", "Static", "N/A",
    )
    assert t.user_template == TRAINING_USER_TEMPLATE
    assert t.generation == {"max_new_tokens": 300}
    assert t.prompt and "ni_progressive_or_static" in t.prompt


def test_load_full_builtin():
    t = load_task("full")
    assert t.name == "full"
    assert len(t.schema.fields) == 20
    assert t.schema.field_names() == (*COMORBIDITY_YN_FIELDS, *CLINICAL_VARS_FIELDS)
    assert "comorbidities_list" not in t.schema.field_names()
    for name in COMORBIDITY_YN_FIELDS:
        assert t.schema.get_field(name).enum_values == ("Y", "N")
    assert t.schema.get_field("ni_progressive_or_static").enum_values == (
        "Progressive", "Static", "N/A",
    )
    assert t.user_template == TRAINING_USER_TEMPLATE
    assert t.generation == {"max_new_tokens": 512}


def test_unknown_builtin_lists_available():
    with pytest.raises(SchemaError, match="clinical_vars"):
        load_task("nonexistent_task")


# --- A11: built-in name vs file path detection ---


def test_unknown_builtin_error_mentions_file_path_interpretation():
    with pytest.raises(SchemaError, match="task file"):
        load_task("nonexistent_task")


def test_uppercase_yaml_suffix_treated_as_file(tmp_path: Path, monkeypatch):
    (tmp_path / "T.YAML").write_text("fields:\n  x:\n    type: string\n")
    monkeypatch.chdir(tmp_path)
    t = load_task("T.YAML")  # no separator; only the case-insensitive suffix marks it
    assert t.schema.field_names() == ("x",)


def test_separator_means_file_path_not_builtin(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_task(str(tmp_path / "missing" / "task_file"))


def test_altsep_counts_as_path_marker(monkeypatch):
    import os

    from ehrextract.schema import _looks_like_path

    monkeypatch.setattr(os, "altsep", "\\")
    assert _looks_like_path("adapters\\my_task") is True
    assert _looks_like_path("comorbidity") is False


# --- task YAML files ---


def test_load_task_from_path_with_all_keys(tmp_path: Path):
    p = tmp_path / "my_task.yaml"
    p.write_text(
        "name: my_task\n"
        "prompt: Extract things.\n"
        'user_template: "NOTE: {note}"\n'
        "generation:\n"
        "  max_new_tokens: 64\n"
        "  temperature: 0.5\n"
        "  stop: ['<end>']\n"
        "fields:\n"
        "  x:\n"
        "    type: string\n"
    )
    t = load_task(p)
    assert t.name == "my_task"
    assert t.prompt == "Extract things."
    assert t.user_template == "NOTE: {note}"
    assert t.generation == {"max_new_tokens": 64, "temperature": 0.5, "stop": ("<end>",)}
    assert t.prompt_verbatim is False


def test_str_path_with_yaml_suffix_is_treated_as_file(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("fields:\n  x:\n    type: string\n")
    t = load_task(str(p))
    assert t.schema.field_names() == ("x",)
    assert t.name is None


def test_plain_schema_yaml_is_a_valid_task(fixtures_dir: Path):
    t = load_task(fixtures_dir / "schemas" / "clinical_features.yaml")
    assert t.name == "clinical_features"
    assert t.prompt is None
    assert t.user_template == "{note}"
    assert t.generation == {}


def test_user_template_without_note_placeholder_raises(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("user_template: 'no placeholder'\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="note"):
        load_task(p)


def test_unknown_generation_key_raises(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation:\n  max_tokens: 5\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="generation key"):
        load_task(p)


def test_generation_must_be_mapping(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation: 5\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="generation"):
        load_task(p)


def test_prompt_must_be_string(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("prompt: [a, b]\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="prompt"):
        load_task(p)


def test_unknown_top_level_key_raises(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("bogus: 1\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="unknown.*key"):
        load_task(p)


# --- A12: generation value validation + duplicate-key rejection ---


def test_unknown_generation_key_lists_valid_keys(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation:\n  max_tokens: 5\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="max_new_tokens"):
        load_task(p)


def test_generation_stop_string_rejected(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation:\n  stop: '<end>'\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="list of strings"):
        load_task(p)


def test_generation_stop_list_of_non_strings_rejected(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation:\n  stop: [1, 2]\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="list of strings"):
        load_task(p)


def test_generation_temperature_must_be_number(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation:\n  temperature: hot\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="number"):
        load_task(p)


def test_generation_bool_rejected_for_max_new_tokens(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation:\n  max_new_tokens: true\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="integer"):
        load_task(p)


def test_generation_top_p_null_allowed(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("generation:\n  top_p: null\nfields:\n  x:\n    type: string\n")
    t = load_task(p)
    assert t.generation == {"top_p": None}


def test_duplicate_top_level_key_raises(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("name: a\nname: b\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="duplicate"):
        load_task(p)


def test_duplicate_nested_field_key_raises(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text("fields:\n  x:\n    type: string\n  x:\n    type: integer\n")
    with pytest.raises(SchemaError, match="duplicate"):
        load_task(p)


# --- A13: user_template format probe at load time ---


def test_user_template_stray_brace_raises_at_load(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text('user_template: "{note} {"\nfields:\n  x:\n    type: string\n')
    with pytest.raises(SchemaError, match="format"):
        load_task(p)


def test_user_template_unknown_placeholder_raises_at_load(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text('user_template: "{note} {bogus}"\nfields:\n  x:\n    type: string\n')
    with pytest.raises(SchemaError, match="format"):
        load_task(p)
