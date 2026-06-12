"""Tests for the schema dataclasses, YAML loader, and JSON Schema conversion."""

from pathlib import Path

import pytest

from ehrextract.schema import FieldSpec, Schema, SchemaError, load_schema, to_json_schema


# --- FieldSpec / Schema invariants ---


def test_fieldspec_string():
    f = FieldSpec(name="diagnosis", kind="string")
    assert f.name == "diagnosis"
    assert f.kind == "string"
    assert f.required is True
    assert f.enum_values is None


def test_fieldspec_enum_requires_values():
    with pytest.raises(ValueError, match="enum_values"):
        FieldSpec(name="status", kind="enum")


def test_fieldspec_list_requires_item_kind():
    with pytest.raises(ValueError, match="item_kind"):
        FieldSpec(name="tags", kind="list")


def test_fieldspec_list_of_enum_requires_item_enum_values():
    with pytest.raises(ValueError, match="item_enum_values"):
        FieldSpec(name="tags", kind="list", item_kind="enum")


def test_fieldspec_immutable():
    f = FieldSpec(name="x", kind="string")
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        f.name = "y"  # type: ignore[misc]


def test_schema_immutable():
    s = Schema(fields=(FieldSpec(name="x", kind="string"),))
    with pytest.raises(Exception):
        s.name = "renamed"  # type: ignore[misc]


def test_schema_rejects_duplicate_field_names():
    a = FieldSpec(name="x", kind="string")
    b = FieldSpec(name="x", kind="integer")
    with pytest.raises(ValueError, match="duplicate"):
        Schema(fields=(a, b))


def test_schema_rejects_empty_fields():
    with pytest.raises(ValueError, match="at least one"):
        Schema(fields=())


def test_schema_field_names_and_get_field():
    s = Schema(fields=(FieldSpec(name="a", kind="string"), FieldSpec(name="b", kind="integer")))
    assert s.field_names() == ("a", "b")
    assert s.get_field("b").kind == "integer"
    with pytest.raises(KeyError):
        s.get_field("nope")


# --- YAML loading ---


def test_load_reference_schema(fixtures_dir: Path):
    schema = load_schema(fixtures_dir / "schemas" / "clinical_features.yaml")
    assert schema.name == "clinical_features"
    assert schema.field_names() == ("diagnosis", "age", "smoker", "comorbidities")
    assert schema.get_field("age").required is False
    assert schema.get_field("smoker").enum_values == ("Y", "N", "Unknown")
    assert schema.get_field("comorbidities").item_kind == "string"


def test_unknown_type_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("fields:\n  x:\n    type: not_a_type\n")
    with pytest.raises(SchemaError, match="unknown type"):
        load_schema(p)


def test_missing_fields_section_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\n")
    with pytest.raises(SchemaError, match="fields"):
        load_schema(p)


def test_enum_without_values_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("fields:\n  x:\n    type: enum\n")
    with pytest.raises(SchemaError, match="values"):
        load_schema(p)


def test_list_without_item_type_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("fields:\n  x:\n    type: list\n")
    with pytest.raises(SchemaError, match="item_type"):
        load_schema(p)


def test_unknown_top_level_key_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nfields:\n  x:\n    type: string\nbogus_key: 1\n")
    with pytest.raises(SchemaError, match="unknown.*key"):
        load_schema(p)


def test_schema_loader_rejects_task_keys(tmp_path: Path):
    """`prompt` is a task-file key; plain load_schema rejects it."""
    p = tmp_path / "bad.yaml"
    p.write_text("prompt: hi\nfields:\n  x:\n    type: string\n")
    with pytest.raises(SchemaError, match="unknown.*key"):
        load_schema(p)


def test_optional_and_required_both_raise(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("fields:\n  x:\n    type: string\n    optional: true\n    required: true\n")
    with pytest.raises(SchemaError, match="optional.*required|required.*optional"):
        load_schema(p)


def test_unknown_field_key_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("fields:\n  x:\n    type: string\n    bogus: 1\n")
    with pytest.raises(SchemaError, match="unknown keys"):
        load_schema(p)


# --- JSON Schema conversion ---


def test_string_field():
    s = Schema(fields=(FieldSpec(name="x", kind="string"),))
    js = to_json_schema(s)
    assert js["type"] == "object"
    assert js["properties"]["x"] == {"type": "string"}
    assert js["required"] == ["x"]
    assert js["additionalProperties"] is False


def test_optional_field_omitted_from_required():
    s = Schema(fields=(FieldSpec(name="x", kind="string", required=False),))
    js = to_json_schema(s)
    assert js["required"] == []


def test_enum_field():
    s = Schema(fields=(FieldSpec(name="x", kind="enum", enum_values=("A", "B")),))
    js = to_json_schema(s)
    assert js["properties"]["x"] == {"type": "string", "enum": ["A", "B"]}


def test_integer_and_float_and_boolean():
    s = Schema(fields=(
        FieldSpec(name="i", kind="integer"),
        FieldSpec(name="f", kind="float"),
        FieldSpec(name="b", kind="boolean"),
    ))
    js = to_json_schema(s)
    assert js["properties"]["i"] == {"type": "integer"}
    assert js["properties"]["f"] == {"type": "number"}
    assert js["properties"]["b"] == {"type": "boolean"}


def test_list_of_string():
    s = Schema(fields=(FieldSpec(name="t", kind="list", item_kind="string"),))
    js = to_json_schema(s)
    assert js["properties"]["t"] == {"type": "array", "items": {"type": "string"}}


def test_list_of_enum():
    s = Schema(fields=(FieldSpec(
        name="t", kind="list", item_kind="enum", item_enum_values=("a", "b"),
    ),))
    js = to_json_schema(s)
    assert js["properties"]["t"] == {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}


def test_description_carried_over():
    s = Schema(fields=(FieldSpec(name="x", kind="string", description="hi"),))
    js = to_json_schema(s)
    assert js["properties"]["x"]["description"] == "hi"
