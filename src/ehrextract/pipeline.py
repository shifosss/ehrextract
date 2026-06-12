"""Message building, response parsing, and the extraction orchestrator."""

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ehrextract.io import load_notes, validate_output_path, write_results
from ehrextract.providers import (
    GenerationConfig,
    ProviderResponse,
    load_provider,
    warn_egress,
)
from ehrextract.schema import FieldSpec, Schema, Task, load_task, to_json_schema

logger = logging.getLogger(__name__)

DEFAULT_USER_TEMPLATE = "{note}"

ErrorCode = Literal[
    "missing",
    "wrong_type",
    "invalid_enum",
    "invalid_list_item",
    "coercion_failed",
    "provider_error",
    "empty_note",
]

# Client errors that retrying cannot fix (bad request/auth/route/payload).
_NO_RETRY_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


@dataclass(frozen=True)
class FieldError:
    field: str
    code: ErrorCode
    detail: str


_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_json_response(raw: str) -> str:
    """Strip common LLM scaffolding and return the JSON substring."""
    text = raw.strip()
    text = _THINK_RE.sub("", text).strip()
    fence = _MARKDOWN_FENCE_RE.match(text)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _coerce_primitive(value: Any, kind: str):
    """Best-effort coercion. Returns (coerced_value, error_or_None)."""
    try:
        if kind == "string":
            if isinstance(value, str):
                return value, None
            if isinstance(value, (dict, list)):
                # Never str() containers: "None" is a meaningful string value,
                # a stringified dict never is.
                return None, FieldError(
                    "", "wrong_type", f"expected string, got {type(value).__name__}"
                )
            return str(value), None
        if kind == "integer":
            if isinstance(value, bool):
                return None, FieldError("", "wrong_type", "bool is not int")
            if isinstance(value, int):
                return value, None
            if isinstance(value, str):
                return int(value), None
            if isinstance(value, float) and value.is_integer():
                return int(value), None
        if kind == "float":
            if isinstance(value, bool):
                return None, FieldError("", "wrong_type", "bool is not float")
            if isinstance(value, (int, float)):
                return float(value), None
            if isinstance(value, str):
                return float(value), None
        if kind == "boolean":
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                s = value.strip().lower()
                if s in {"true", "yes", "y", "1"}:
                    return True, None
                if s in {"false", "no", "n", "0"}:
                    return False, None
    except (ValueError, TypeError) as e:
        return None, FieldError("", "coercion_failed", str(e))
    return None, FieldError("", "coercion_failed", f"cannot coerce {value!r} to {kind}")


def _validate_field(f: FieldSpec, value: Any):
    errs: list[FieldError] = []
    if value is None:
        # JSON null is a wrong type for every kind, never coerced.
        errs.append(FieldError(f.name, "wrong_type", "field is null"))
        return None, errs
    if f.kind == "enum":
        s = str(value)
        if s not in (f.enum_values or ()):
            errs.append(FieldError(f.name, "invalid_enum", f"{s!r} not in {list(f.enum_values or ())}"))
            return None, errs
        return s, errs
    if f.kind == "list":
        if not isinstance(value, list):
            errs.append(FieldError(f.name, "wrong_type", f"expected list, got {type(value).__name__}"))
            return None, errs
        out = []
        for i, item in enumerate(value):
            if f.item_kind == "enum":
                s = str(item)
                if s not in (f.item_enum_values or ()):
                    errs.append(FieldError(
                        f.name, "invalid_list_item",
                        f"item[{i}]={s!r} not in {list(f.item_enum_values or ())}",
                    ))
                else:
                    out.append(s)
            else:
                # Strict per-kind type check for list items: a number for a string list is a violation.
                if f.item_kind == "string" and not isinstance(item, str):
                    errs.append(FieldError(
                        f.name, "invalid_list_item", f"item[{i}]={item!r} not a string",
                    ))
                    continue
                coerced, err = _coerce_primitive(item, f.item_kind or "string")
                if err is not None:
                    errs.append(FieldError(
                        f.name, "invalid_list_item", f"item[{i}]: {err.detail}",
                    ))
                else:
                    out.append(coerced)
        return out, errs
    coerced, err = _coerce_primitive(value, f.kind)
    if err is not None:
        errs.append(FieldError(f.name, err.code, err.detail))
        return None, errs
    return coerced, errs


def parse_and_validate(raw: str, schema: Schema):
    """Parse a model's raw response and validate it against the schema."""
    cleaned = clean_json_response(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {}, [FieldError("_response", "coercion_failed", f"JSON parse failed: {e}")]
    if not isinstance(obj, dict):
        return {}, [FieldError("_response", "wrong_type", f"top-level must be object, got {type(obj).__name__}")]

    fields: dict[str, Any] = {}
    errors: list[FieldError] = []
    for f in schema.fields:
        if f.name not in obj:
            if f.required:
                errors.append(FieldError(f.name, "missing", "field absent from response"))
            continue
        value = obj[f.name]
        coerced, errs = _validate_field(f, value)
        errors.extend(errs)
        if coerced is not None:
            fields[f.name] = coerced
    return fields, errors


def build_default_prompt(schema: Schema) -> str:
    """Fallback system prompt for tasks that do not define one."""
    prompt = "Extract the following fields from the clinical note."
    if schema.description:
        prompt += "\n" + schema.description
    return prompt


def _field_summary(f: FieldSpec) -> str:
    parts = [f"- {f.name} ({f.kind}"]
    if f.kind == "enum":
        parts.append(f", values: {list(f.enum_values or ())}")
    elif f.kind == "list":
        parts.append(f", items: {f.item_kind}")
        if f.item_kind == "enum":
            parts.append(f" with values {list(f.item_enum_values or ())}")
    parts.append(", required" if f.required else ", optional")
    parts.append(")")
    if f.description:
        parts.append(f" - {f.description}")
    return "".join(parts)


def build_messages(task: Task, note_text: str, *, schema_native: bool) -> list[dict]:
    """Build chat messages.

    `schema_native=True` means the provider enforces the schema natively
    (Anthropic forced tool-use); we omit the JSON-shape boilerplate.
    `schema_native=False` (HF, OpenAI-compat) embeds the JSON shape spec.
    `task.prompt_verbatim=True` uses the prompt byte-for-byte (the
    training/inference prompt-match invariant for fine-tuned adapters).
    """
    prompt = task.prompt or build_default_prompt(task.schema)
    if task.prompt_verbatim:
        system_content = prompt
    else:
        parts = [prompt.strip()]
        if not schema_native:
            parts.append("")
            parts.append("Respond with JSON ONLY (no commentary, no markdown fences) matching this exact shape:")
            parts.append(json.dumps(to_json_schema(task.schema), indent=2))
            parts.append("")
            parts.append("Field details:")
            for f in task.schema.fields:
                parts.append(_field_summary(f))
        system_content = "\n".join(parts)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": task.user_template.format(note=note_text)},
    ]


def resolve_adapter_prompt(
    task: Task,
    adapter_path: str | Path | None,
    explicit_prompt: str | None,
) -> Task:
    """Apply prompt precedence: explicit prompt > adapter system_prompt.txt > task prompt."""
    prompt_file = Path(adapter_path) / "system_prompt.txt" if adapter_path is not None else None
    if explicit_prompt is not None:
        if prompt_file is not None and prompt_file.exists():
            logger.warning(
                "explicit prompt overrides adapter system prompt at %s", prompt_file
            )
        return replace(task, prompt=explicit_prompt, prompt_verbatim=False)
    if prompt_file is not None and prompt_file.exists():
        prompt_text = prompt_file.read_text(encoding="utf-8")
        if not prompt_text.strip():
            raise ValueError(f"adapter system prompt file is empty: {prompt_file}")
        return replace(task, prompt=prompt_text, prompt_verbatim=True)
    return task


@dataclass(frozen=True)
class ExtractionResult:
    note_id: str | int
    fields: dict[str, Any]
    parse_success: bool
    validation_errors: list[FieldError]
    raw_response: str
    finish_reason: str | None
    usage: dict[str, int] | None


def _normalize_notes(notes, *, id_column: str, text_column: str) -> pd.DataFrame:
    """Normalize any supported notes input into a validated DataFrame.

    Type drives the branch. No "guess whether this string is a path"
    heuristic: str = inline note content, Path = file to load.
    """
    if isinstance(notes, pd.DataFrame):
        df = notes.reset_index(drop=True)
    elif isinstance(notes, str):
        df = pd.DataFrame({id_column: [0], text_column: [notes]})
    elif isinstance(notes, Path):
        df = load_notes(notes, id_column=id_column, text_column=text_column)
    elif isinstance(notes, list):
        if not notes:
            df = pd.DataFrame({id_column: [], text_column: []})
        elif isinstance(notes[0], dict):
            df = pd.DataFrame(notes)
        elif isinstance(notes[0], str):
            df = pd.DataFrame({id_column: list(range(len(notes))), text_column: notes})
        else:
            raise TypeError(f"unsupported notes list element type: {type(notes[0]).__name__}")
    else:
        raise TypeError(f"unsupported notes type: {type(notes).__name__}")

    if text_column not in df.columns:
        raise ValueError(
            f"input is missing required text column {text_column!r} (have: {list(df.columns)})"
        )
    if id_column not in df.columns:
        df[id_column] = range(len(df))
    return df


def _empty_note_reason(text: Any) -> str | None:
    """Return why a note cell is unusable (null / empty after strip), or None if usable."""
    if text is None or (pd.api.types.is_scalar(text) and pd.isna(text)):
        return "note text is null"
    if not str(text).strip():
        return "note text is empty"
    return None


class Extractor:
    def __init__(
        self,
        provider: Any,
        task: Task,
        *,
        generation: GenerationConfig | dict | None = None,
        id_column: str = "note_id",
        text_column: str = "note_text",
        on_egress: Literal["warn", "silent"] = "warn",
        max_retries: int = 3,
    ):
        self.provider = provider
        self.task = task
        # Precedence: GenerationConfig defaults < task.generation < `generation` arg.
        merged: dict[str, Any] = dict(task.generation)
        if isinstance(generation, GenerationConfig):
            merged.update(asdict(generation))
        elif isinstance(generation, dict):
            merged.update(generation)
        self.generation = GenerationConfig(**merged)
        self.id_column = id_column
        self.text_column = text_column
        self.on_egress = on_egress
        self.max_retries = max_retries
        self._json_schema = to_json_schema(task.schema)
        self._egress_checked = False

    def _check_egress(self) -> None:
        if self._egress_checked:
            return
        self._egress_checked = True
        dest = self.provider.egress_destination()
        if dest is not None:
            warn_egress(dest, mode=self.on_egress)

    def _generate_with_retry(self, messages: list[dict]) -> ProviderResponse | Exception:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.provider.generate(messages, self.generation, json_schema=self._json_schema)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                status = getattr(e, "status_code", None)
                if status in _NO_RETRY_STATUS_CODES:
                    logger.warning("provider %s failed with non-retryable status %s: %s",
                                   self.provider.name, status, e)
                    return e
                logger.warning("provider %s failed (attempt %d/%d): %s",
                               self.provider.name, attempt + 1, self.max_retries, e)
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt + random.uniform(0, 0.5))
        assert last_exc is not None
        return last_exc

    def run_one(self, text: str, note_id: str | int = 0) -> ExtractionResult:
        self._check_egress()
        messages = build_messages(
            self.task, text, schema_native=self.provider.uses_schema_natively
        )
        outcome = self._generate_with_retry(messages)
        if isinstance(outcome, Exception):
            return ExtractionResult(
                note_id=note_id,
                fields={},
                parse_success=False,
                validation_errors=[FieldError("_provider", "provider_error", str(outcome))],
                raw_response="",
                finish_reason="error",
                usage=None,
            )
        fields, errors = parse_and_validate(outcome.text, self.task.schema)
        return ExtractionResult(
            note_id=note_id,
            fields=fields,
            parse_success=(len(errors) == 0),
            validation_errors=errors,
            raw_response=outcome.text,
            finish_reason=outcome.finish_reason,
            usage=outcome.usage,
        )

    def _normalize_input(self, notes) -> pd.DataFrame:
        return _normalize_notes(notes, id_column=self.id_column, text_column=self.text_column)

    def _empty_note_result(self, note_id: str | int, reason: str) -> ExtractionResult:
        return ExtractionResult(
            note_id=note_id,
            fields={},
            parse_success=False,
            validation_errors=[FieldError("_input", "empty_note", reason)],
            raw_response="",
            finish_reason="skipped",
            usage=None,
        )

    def run(self, notes, *, max_concurrency: int | None = None) -> pd.DataFrame:
        self._check_egress()
        df = self._normalize_input(notes)
        if df.empty:
            return self._empty_result_frame()
        if df[self.id_column].duplicated().any():
            logger.warning("input contains duplicate values in id column %r", self.id_column)

        if max_concurrency is None:
            max_concurrency = self.provider.default_concurrency
        elif max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if (
            self.provider.egress_destination() is None
            and max_concurrency > self.provider.default_concurrency
        ):
            # A shared local model (HF) is not thread-safe; clamp.
            logger.warning(
                "provider %r runs locally; clamping max_concurrency to %d",
                self.provider.name, self.provider.default_concurrency,
            )
            max_concurrency = self.provider.default_concurrency

        def _work(i: int):
            row = df.iloc[i]
            note_id = row[self.id_column]
            text = row[self.text_column]
            reason = _empty_note_reason(text)
            if reason is not None:
                # Error row; the provider is never called for empty notes.
                return i, self._empty_note_result(note_id, reason)
            return i, self.run_one(str(text), note_id=note_id)

        if max_concurrency <= 1:
            results = []
            for i in range(len(df)):
                _, r = _work(i)
                results.append(r)
        else:
            results = [None] * len(df)
            with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                futures = [ex.submit(_work, i) for i in range(len(df))]
                try:
                    for fut in as_completed(futures):
                        i, r = fut.result()
                        results[i] = r
                except BaseException:
                    # First worker failure: cancel the queued notes (stop
                    # further PHI egress, don't run a doomed queue), re-raise.
                    ex.shutdown(wait=True, cancel_futures=True)
                    raise

        return self._results_to_frame(results)

    def _empty_result_frame(self) -> pd.DataFrame:
        cols = [self.id_column, *self.task.schema.field_names(),
                "parse_success", "validation_errors", "raw_response",
                "finish_reason", "input_tokens", "output_tokens"]
        return pd.DataFrame({c: [] for c in cols})

    def _results_to_frame(self, results: list) -> pd.DataFrame:
        rows = []
        for r in results:
            assert r is not None
            row = {self.id_column: r.note_id}
            for name in self.task.schema.field_names():
                row[name] = r.fields.get(name, None)
            row["parse_success"] = r.parse_success
            row["validation_errors"] = "; ".join(
                f"{e.field}:{e.code}:{e.detail}" for e in r.validation_errors
            )
            # Full text on failure for debugging, empty on success to keep output clean.
            row["raw_response"] = "" if r.parse_success else r.raw_response
            row["finish_reason"] = r.finish_reason
            row["input_tokens"] = (r.usage or {}).get("input_tokens")
            row["output_tokens"] = (r.usage or {}).get("output_tokens")
            rows.append(row)
        return pd.DataFrame(rows)


def extract(
    notes,
    task: str | Path | Task | Schema,
    *,
    provider: str = "huggingface",
    model: str | None = None,
    adapter: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    prompt: str | None = None,
    generation: GenerationConfig | dict | None = None,
    output: str | Path | None = None,
    max_concurrency: int | None = None,
    id_column: str = "note_id",
    text_column: str = "note_text",
    on_egress: Literal["warn", "silent"] = "warn",
    trust_remote_code: bool = False,
    dtype: str | None = "bfloat16",
) -> pd.DataFrame:
    """One-call extraction: resolve task, load notes, build provider, run, optionally write."""
    # 1. Resolve the task.
    if isinstance(task, Schema):
        task = Task(name=task.name, schema=task, prompt=None, user_template=DEFAULT_USER_TEMPLATE)
    elif isinstance(task, (str, Path)):
        task = load_task(task)

    if model is None:
        raise ValueError(f"model is required for provider {provider!r}")
    if adapter is not None and provider != "huggingface":
        raise ValueError("adapter is only supported with provider='huggingface'")

    task = resolve_adapter_prompt(task, adapter, prompt)

    # 2./3. Load+validate notes and the output extension BEFORE building the
    # provider: model load takes minutes, and a bad --output extension must
    # not discard a finished run.
    notes_df = _normalize_notes(notes, id_column=id_column, text_column=text_column)
    if output is not None:
        validate_output_path(output)

    # 4. Build the provider.
    kwargs: dict[str, Any] = {"model": model}
    if provider == "huggingface":
        if adapter is not None:
            kwargs["adapter_path"] = adapter
        kwargs["trust_remote_code"] = trust_remote_code
        kwargs["dtype"] = dtype
    if provider == "openai" and base_url is not None:
        kwargs["base_url"] = base_url
    if provider in {"openai", "anthropic"} and api_key is not None:
        kwargs["api_key"] = api_key
    provider_obj = load_provider(provider, **kwargs)

    # 5./6. Run, then write.
    extractor = Extractor(
        provider_obj,
        task,
        generation=generation,
        id_column=id_column,
        text_column=text_column,
        on_egress=on_egress,
    )
    df = extractor.run(notes_df, max_concurrency=max_concurrency)
    if output is not None:
        write_results(df, output)
    return df
