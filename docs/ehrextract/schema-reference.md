# Task-file reference

A task file is a YAML document that declares the fields to extract and,
optionally, the prompt and generation settings. The same grammar covers
both built-in tasks (`comorbidity`, `clinical_vars`, `full` — packaged
inside ehrextract) and your own files.

```yaml
name: my_task                 # optional
description: One-line description.   # optional
prompt: |                     # optional -- the system prompt
  Read the clinical note and extract the requested fields.
user_template: "Note: {note}\n\nReturn the JSON:"   # optional
generation:                   # optional
  max_new_tokens: 300
fields:                       # required
  field_name:
    type: enum
    values: [Y, N]
```

Load with `load_task("comorbidity")` (built-in name) or
`load_task("path/to/my_task.yaml")` / `--task` on the CLI. A string with no
path separator and no `.yaml`/`.yml` suffix (matched case-insensitively) is
treated as a built-in task name; anything else is a filesystem path.
Unknown built-in names raise an error that lists the available built-ins
and notes the file-path interpretation.

## Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `fields` | yes | Mapping of field name → field spec (below) |
| `name` | no | Task name; defaults to the built-in name when loading a packaged task |
| `description` | no | Included in the default prompt and the generated JSON Schema |
| `prompt` | no | System prompt. Without it, a generic instruction sentence is used |
| `user_template` | no | User-message template; **must contain `{note}`** and no other `{...}` placeholders (validated at load time) |
| `generation` | no | Mapping of generation settings (below); unknown keys rejected |

Unknown top-level keys are rejected, and so are duplicate mapping keys
anywhere in the YAML. **A plain schema file** — only
`name`/`description`/`fields` — **is a valid task file**: the optional keys
just take their defaults. (`load_schema()` is the stricter entry point
that accepts only those three keys and returns a `Schema` instead of a
`Task`.)

## Field grammar

```yaml
fields:
  field_name:
    type: <string|integer|float|boolean|enum|list>
    description: Optional -- surfaced in the prompt to disambiguate
    optional: true | false  # default false
    values: [A, B, C]       # required for type: enum
    item_type: <kind>       # required for type: list
    item_values: [X, Y]     # required for type: list with item_type: enum
```

- `required: true|false` is accepted as the inverse of `optional`;
  specifying both is an error.
- `values` / `item_values` entries are coerced to strings.
- Nested lists (`item_type: list`) are not supported.
- Unknown field keys are rejected.

## Generation settings

`generation` keys mirror `GenerationConfig`:

| Key | Default | Notes |
|---|---|---|
| `max_new_tokens` | 1024 | |
| `temperature` | 0.0 | HuggingFace samples only when > 0 |
| `top_p` | unset | |
| `repetition_penalty` | 1.0 | HuggingFace only |
| `stop` | none | List of stop strings (OpenAI-compatible only) |
| `constrained` | false | HuggingFace only -- token-constrained JSON decoding (see [`quickstart.md`](quickstart.md)); other providers log one INFO and ignore it. Validated pairing: `repetition_penalty: 1.0` |

Values are type-checked at load time: numeric keys must be numbers (not
booleans), `constrained` must be a strict boolean, and `stop` must be a
list of strings; unknown keys raise an error listing the valid ones.

Precedence, lowest to highest: `GenerationConfig` defaults < the task
file's `generation` block < the `generation=` argument to
`Extractor`/`extract()` (a dict overrides per-key; a full
`GenerationConfig` replaces everything) and the CLI flags
`--max-new-tokens` / `--temperature` / `--top-p` /
`--repetition-penalty` / `--constrained` / `--no-constrained` (only
explicitly passed flags override).

## How the schema is used

- **Prompting.** For providers without native structured output
  (HuggingFace, OpenAI-compatible), the JSON Schema rendering of the
  fields plus per-field summary lines are appended to the system prompt,
  unless the prompt is marked verbatim (adapter `system_prompt.txt` flow —
  see [`quickstart.md`](quickstart.md)). For the Anthropic provider the
  JSON Schema becomes the forced tool's `input_schema` and the boilerplate
  is omitted.
- **Validation.** Each response is parsed (markdown fences, `<think>`
  blocks, and surrounding text are stripped) and every field is checked:
  enum membership, list item types, primitive coercion (`"3"` → 3,
  `"yes"` → true, ...). Failures are reported per field in the
  `validation_errors` column; `parse_success` is true only when there are
  no errors.
- **Enum matching is strict.** Values must match one of `values` exactly —
  case-sensitive, no normalization (`y` does not match `Y`; `YES` does not
  match `Y`). Fine-tuned adapters emit the exact values; baseline models
  that paraphrase get visible `invalid_enum` errors instead of silent
  remapping. JSON `null` is never accepted for any field kind.

## Out of scope

- Nested objects (a field cannot itself be an object)
- Format constraints (no regex / range / length validators)
- Cross-field rules
