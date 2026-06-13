"""Schema IR, YAML loading, JSON Schema conversion, and task definitions."""

import os
from dataclasses import dataclass, field, fields as dataclass_fields
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import yaml

from ehrextract.providers import GenerationConfig

FieldKind = Literal["string", "integer", "float", "boolean", "enum", "list"]

VALID_KINDS: set[FieldKind] = {"string", "integer", "float", "boolean", "enum", "list"}
SCHEMA_TOP_LEVEL_KEYS = {"name", "description", "fields"}
TASK_TOP_LEVEL_KEYS = SCHEMA_TOP_LEVEL_KEYS | {"prompt", "user_template", "generation"}
FIELD_KEYS = {"type", "description", "optional", "required", "values", "item_type", "item_values"}

GENERATION_KEYS = frozenset(f.name for f in dataclass_fields(GenerationConfig))

PRIMITIVE_JSON_TYPE: dict[FieldKind, str] = {
    "string": "string",
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
}


class SchemaError(ValueError):
    """Raised when a schema or task file cannot be turned into a valid Schema or Task."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (PyYAML silently keeps the last)."""

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:  # unhashable key; SafeLoader raises its own error
                continue
            if duplicate:
                raise SchemaError(f"duplicate mapping key {key!r}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    required: bool = True
    description: str | None = None
    enum_values: tuple[str, ...] | None = None
    item_kind: FieldKind | None = None
    item_enum_values: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind == "enum" and not self.enum_values:
            raise ValueError(f"FieldSpec {self.name!r}: kind='enum' requires enum_values")
        if self.kind == "list":
            if self.item_kind is None:
                raise ValueError(f"FieldSpec {self.name!r}: kind='list' requires item_kind")
            if self.item_kind == "enum" and not self.item_enum_values:
                raise ValueError(
                    f"FieldSpec {self.name!r}: list-of-enum requires item_enum_values"
                )


@dataclass(frozen=True)
class Schema:
    fields: tuple[FieldSpec, ...]
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("Schema must have at least one field")
        seen: set[str] = set()
        for f in self.fields:
            if f.name in seen:
                raise ValueError(f"duplicate field name: {f.name!r}")
            seen.add(f.name)

    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def get_field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)


@dataclass(frozen=True)
class Task:
    name: str | None
    schema: Schema
    prompt: str | None
    user_template: str = "{note}"
    generation: dict[str, Any] = field(default_factory=dict)
    prompt_verbatim: bool = False


def _parse_kind(raw: Any, field_name: str) -> FieldKind:
    if not isinstance(raw, str) or raw not in VALID_KINDS:
        raise SchemaError(
            f"field {field_name!r}: unknown type {raw!r} (valid: {sorted(VALID_KINDS)})"
        )
    return raw  # type: ignore[return-value]


def _build_field(name: str, body: Any) -> FieldSpec:
    if not isinstance(body, dict):
        raise SchemaError(f"field {name!r}: expected mapping, got {type(body).__name__}")
    extra = set(body) - FIELD_KEYS
    if extra:
        raise SchemaError(f"field {name!r}: unknown keys {sorted(extra)}")
    if "type" not in body:
        raise SchemaError(f"field {name!r}: missing 'type'")

    kind = _parse_kind(body["type"], name)
    description = body.get("description")

    optional = body.get("optional", False)
    if "required" in body:
        if "optional" in body:
            raise SchemaError(
                f"field {name!r}: specify either 'optional' or 'required', not both"
            )
        optional = not bool(body["required"])
    required = not bool(optional)

    enum_values: tuple[str, ...] | None = None
    if kind == "enum":
        values = body.get("values")
        if not isinstance(values, list) or not values:
            raise SchemaError(f"field {name!r}: enum requires non-empty 'values' list")
        enum_values = tuple(str(v) for v in values)

    item_kind: FieldKind | None = None
    item_enum_values: tuple[str, ...] | None = None
    if kind == "list":
        if "item_type" not in body:
            raise SchemaError(f"field {name!r}: list requires 'item_type'")
        item_kind = _parse_kind(body["item_type"], f"{name}.item_type")
        if item_kind == "list":
            raise SchemaError(f"field {name!r}: nested lists are not supported")
        if item_kind == "enum":
            item_values = body.get("item_values")
            if not isinstance(item_values, list) or not item_values:
                raise SchemaError(
                    f"field {name!r}: list-of-enum requires non-empty 'item_values'"
                )
            item_enum_values = tuple(str(v) for v in item_values)

    return FieldSpec(
        name=name,
        kind=kind,
        required=required,
        description=description,
        enum_values=enum_values,
        item_kind=item_kind,
        item_enum_values=item_enum_values,
    )


def _parse_doc(text: str, source: str, allowed_keys: set[str]) -> dict:
    try:
        doc = yaml.load(text, Loader=_UniqueKeyLoader)
    except SchemaError as e:
        raise SchemaError(f"{source}: {e}") from None
    if not isinstance(doc, dict):
        raise SchemaError(f"{source}: top-level must be a mapping")
    extra = set(doc) - allowed_keys
    if extra:
        raise SchemaError(f"{source}: unknown top-level key(s): {sorted(extra)}")
    if "fields" not in doc:
        raise SchemaError(f"{source}: missing 'fields' section")
    return doc


def _build_schema(doc: dict, source: str) -> Schema:
    fields_block = doc["fields"]
    if not isinstance(fields_block, dict) or not fields_block:
        raise SchemaError(f"{source}: 'fields' must be a non-empty mapping")
    specs = tuple(_build_field(name, body) for name, body in fields_block.items())
    return Schema(fields=specs, name=doc.get("name"), description=doc.get("description"))


def load_schema(path: str | Path) -> Schema:
    """Load and validate a YAML schema file into a Schema."""
    source = str(path)
    doc = _parse_doc(Path(path).read_text(encoding="utf-8"), source, SCHEMA_TOP_LEVEL_KEYS)
    return _build_schema(doc, source)


def _field_to_json_property(f: FieldSpec) -> dict[str, Any]:
    if f.kind in PRIMITIVE_JSON_TYPE:
        prop: dict[str, Any] = {"type": PRIMITIVE_JSON_TYPE[f.kind]}
    elif f.kind == "enum":
        assert f.enum_values is not None
        prop = {"type": "string", "enum": list(f.enum_values)}
    elif f.kind == "list":
        assert f.item_kind is not None
        if f.item_kind == "enum":
            assert f.item_enum_values is not None
            items: dict[str, Any] = {"type": "string", "enum": list(f.item_enum_values)}
        else:
            items = {"type": PRIMITIVE_JSON_TYPE[f.item_kind]}
        prop = {"type": "array", "items": items}
    else:
        raise ValueError(f"unsupported field kind: {f.kind}")
    if f.description:
        prop["description"] = f.description
    return prop


def to_json_schema(schema: Schema) -> dict[str, Any]:
    """Convert a Schema to a JSON Schema object document."""
    properties = {f.name: _field_to_json_property(f) for f in schema.fields}
    required = [f.name for f in schema.fields if f.required]
    js: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if schema.description:
        js["description"] = schema.description
    return js


def _builtin_task_names() -> list[str]:
    names = []
    try:
        for entry in (files("ehrextract") / "tasks").iterdir():
            if entry.name.endswith(".yaml"):
                names.append(entry.name[: -len(".yaml")])
    except FileNotFoundError:
        pass
    return sorted(names)


def _check_generation(generation: dict, source: str) -> dict[str, Any]:
    unknown = set(generation) - GENERATION_KEYS
    if unknown:
        raise SchemaError(
            f"{source}: unknown generation key(s): {sorted(unknown)} "
            f"(valid: {sorted(GENERATION_KEYS)})"
        )
    checked = dict(generation)
    if "max_new_tokens" in checked:
        v = checked["max_new_tokens"]
        if isinstance(v, bool) or not isinstance(v, int):
            raise SchemaError(f"{source}: generation.max_new_tokens must be an integer")
    for key in ("temperature", "top_p", "repetition_penalty"):
        if key in checked:
            v = checked[key]
            if key == "top_p" and v is None:
                continue  # explicit null = GenerationConfig default (disabled)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise SchemaError(f"{source}: generation.{key} must be a number")
    if "stop" in checked:
        v = checked["stop"]
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            raise SchemaError(f"{source}: generation.stop must be a list of strings")
        checked["stop"] = tuple(v)
    if "constrained" in checked and not isinstance(checked["constrained"], bool):
        raise SchemaError(f"{source}: generation.constrained must be a boolean")
    return checked


def _build_task(doc: dict, source: str, default_name: str | None) -> Task:
    schema = _build_schema(doc, source)

    prompt = doc.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise SchemaError(f"{source}: 'prompt' must be a string")

    user_template = doc.get("user_template", "{note}")
    if not isinstance(user_template, str) or "{note}" not in user_template:
        raise SchemaError(f"{source}: 'user_template' must be a string containing '{{note}}'")
    try:
        user_template.format(note="x")
    except (KeyError, IndexError, ValueError) as e:
        # Stray braces fail here at load time, not as a per-note runtime error.
        raise SchemaError(f"{source}: 'user_template' has invalid format syntax: {e}") from None

    generation = doc.get("generation")
    if generation is None:
        generation = {}
    if not isinstance(generation, dict):
        raise SchemaError(f"{source}: 'generation' must be a mapping")
    generation = _check_generation(generation, source)

    return Task(
        name=doc.get("name") or default_name,
        schema=schema,
        prompt=prompt,
        user_template=user_template,
        generation=generation,
        prompt_verbatim=False,
    )


def _looks_like_path(s: str) -> bool:
    if os.sep in s or (os.altsep is not None and os.altsep in s):
        return True
    return s.lower().endswith((".yaml", ".yml"))


def load_task(name_or_path: str | Path) -> Task:
    """Load a Task from a built-in task name or a task/schema YAML file."""
    if isinstance(name_or_path, str) and not _looks_like_path(name_or_path):
        resource = files("ehrextract") / "tasks" / f"{name_or_path}.yaml"
        if not resource.is_file():
            raise SchemaError(
                f"unknown built-in task {name_or_path!r} "
                f"(available built-ins: {_builtin_task_names()}); "
                f"to load a task file instead, pass a path containing a "
                f"separator or ending in .yaml/.yml"
            )
        doc = _parse_doc(resource.read_text(encoding="utf-8"), name_or_path, TASK_TOP_LEVEL_KEYS)
        return _build_task(doc, name_or_path, default_name=name_or_path)

    source = str(name_or_path)
    doc = _parse_doc(Path(name_or_path).read_text(encoding="utf-8"), source, TASK_TOP_LEVEL_KEYS)
    return _build_task(doc, source, default_name=None)
