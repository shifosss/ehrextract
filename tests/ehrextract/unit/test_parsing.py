"""Tests for JSON extraction and schema-driven validation."""

import json

from ehrextract.pipeline import clean_json_response, parse_and_validate
from ehrextract.schema import FieldSpec, Schema


def _schema():
    return Schema(fields=(
        FieldSpec(name="diagnosis", kind="string"),
        FieldSpec(name="age", kind="integer", required=False),
        FieldSpec(name="smoker", kind="enum", enum_values=("Y", "N", "Unknown")),
        FieldSpec(name="comorbidities", kind="list", item_kind="string"),
    ))


def test_clean_strips_markdown_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert clean_json_response(raw) == '{"a": 1}'


def test_clean_strips_qwen_think_block():
    raw = "<think>analyzing...</think>\n{\"a\": 1}"
    assert clean_json_response(raw) == '{"a": 1}'


def test_clean_strips_leading_prose():
    raw = 'Sure, here you go: {"a": 1}'
    cleaned = clean_json_response(raw)
    assert json.loads(cleaned) == {"a": 1}


def test_parse_well_formed():
    raw = json.dumps({"diagnosis": "x", "age": 7, "smoker": "Y", "comorbidities": ["a", "b"]})
    fields, errors = parse_and_validate(raw, _schema())
    assert errors == []
    assert fields == {"diagnosis": "x", "age": 7, "smoker": "Y", "comorbidities": ["a", "b"]}


def test_parse_optional_absent():
    raw = json.dumps({"diagnosis": "x", "smoker": "Y", "comorbidities": []})
    fields, errors = parse_and_validate(raw, _schema())
    assert "age" not in fields
    assert all(e.field != "age" for e in errors)


def test_parse_missing_required_field():
    raw = json.dumps({"smoker": "Y", "comorbidities": []})
    _, errors = parse_and_validate(raw, _schema())
    assert any(e.field == "diagnosis" and e.code == "missing" for e in errors)


def test_parse_invalid_enum():
    raw = json.dumps({"diagnosis": "x", "smoker": "Maybe", "comorbidities": []})
    _, errors = parse_and_validate(raw, _schema())
    assert any(e.field == "smoker" and e.code == "invalid_enum" for e in errors)


def test_parse_int_string_coerces():
    raw = json.dumps({"diagnosis": "x", "age": "7", "smoker": "Y", "comorbidities": []})
    fields, errors = parse_and_validate(raw, _schema())
    assert fields["age"] == 7
    assert all(e.code != "wrong_type" for e in errors)


def test_parse_uncoercible_int():
    raw = json.dumps({"diagnosis": "x", "age": "old", "smoker": "Y", "comorbidities": []})
    _, errors = parse_and_validate(raw, _schema())
    assert any(e.field == "age" and e.code == "coercion_failed" for e in errors)


def test_parse_invalid_list_item():
    raw = json.dumps({"diagnosis": "x", "smoker": "Y", "comorbidities": ["a", 5]})
    _, errors = parse_and_validate(raw, _schema())
    assert any(e.field == "comorbidities" and e.code == "invalid_list_item" for e in errors)


def test_parse_unparseable_json():
    fields, errors = parse_and_validate("definitely not json", _schema())
    assert fields == {}
    assert any(e.code == "coercion_failed" for e in errors)


def test_parse_top_level_non_object():
    fields, errors = parse_and_validate("[1, 2]", _schema())
    assert fields == {}
    assert any(e.field == "_response" and e.code == "wrong_type" for e in errors)


def test_parse_extra_keys_ignored():
    raw = json.dumps({"diagnosis": "x", "smoker": "Y", "comorbidities": [], "BOGUS": 1})
    fields, _ = parse_and_validate(raw, _schema())
    assert "BOGUS" not in fields


# --- A4: JSON null and container values are wrong_type, never coerced ---


def test_null_is_wrong_type_for_every_kind():
    raw = json.dumps({"diagnosis": None, "age": None, "smoker": None, "comorbidities": None})
    fields, errors = parse_and_validate(raw, _schema())
    for name in ("diagnosis", "age", "smoker", "comorbidities"):
        assert any(
            e.field == name and e.code == "wrong_type" and "null" in e.detail
            for e in errors
        ), f"no wrong_type/null error for {name}"
    assert fields == {}


def test_null_string_field_never_becomes_literal_none():
    """`"None"` is a meaningful comorbidities_list value; JSON null must not produce it."""
    raw = json.dumps({"diagnosis": None, "smoker": "Y", "comorbidities": []})
    fields, errors = parse_and_validate(raw, _schema())
    assert "diagnosis" not in fields
    assert any(e.field == "diagnosis" and e.code == "wrong_type" for e in errors)


def test_literal_none_string_is_preserved():
    raw = json.dumps({"diagnosis": "None", "smoker": "Y", "comorbidities": []})
    fields, errors = parse_and_validate(raw, _schema())
    assert fields["diagnosis"] == "None"
    assert all(e.field != "diagnosis" for e in errors)


def test_dict_for_string_field_is_wrong_type():
    raw = json.dumps({"diagnosis": {"a": 1}, "smoker": "Y", "comorbidities": []})
    fields, errors = parse_and_validate(raw, _schema())
    assert "diagnosis" not in fields
    assert any(e.field == "diagnosis" and e.code == "wrong_type" for e in errors)


def test_list_for_string_field_is_wrong_type():
    raw = json.dumps({"diagnosis": ["asthma"], "smoker": "Y", "comorbidities": []})
    fields, errors = parse_and_validate(raw, _schema())
    assert "diagnosis" not in fields
    assert any(e.field == "diagnosis" and e.code == "wrong_type" for e in errors)
