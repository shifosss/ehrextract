# Quickstart

Three ways to run, in the order most users need them:

1. [Built-in task + fine-tuned LoRA adapter on a local GPU](#1-built-in-task--fine-tuned-adapter-local-gpu)
2. [Your own task YAML](#2-custom-task-yaml)
3. [API providers (OpenAI-compatible / Anthropic)](#3-api-providers)

Every flow has a CLI form and a Python (`extract()`) form. In Python,
`notes` is type-driven: a `str` is inline note text, a `pathlib.Path` is a
file to load, and lists of strings/dicts and DataFrames also work.

## 1. Built-in task + fine-tuned adapter (local GPU)

The flagship flow: a base model plus a LoRA adapter fine-tuned for one of
the built-in tasks. Requires `pip install 'ehrextract[hf]'` and a GPU
(the model loads in bfloat16 with `device_map="auto"` by default).

```bash
ehrextract \
  --task comorbidity \
  --model Qwen/Qwen3.5-27B \
  --adapter /path/to/adapter \
  --input notes.csv --output results.csv
```

```python
from pathlib import Path
from ehrextract import extract

df = extract(
    Path("notes.csv"),
    "comorbidity",
    model="Qwen/Qwen3.5-27B",
    adapter="/path/to/adapter",
    output="results.csv",
)
```

### The `system_prompt.txt` convention

Fine-tuned models only perform as evaluated when inference uses the exact
system prompt they were trained with. ehrextract automates this: if the
adapter directory contains a `system_prompt.txt` file, that file becomes
the system prompt **verbatim** — no JSON-shape boilerplate is appended, the
prompt is used byte-for-byte. The adapter's tokenizer and chat template are
also preferred over the base model's when the adapter directory ships
tokenizer files (otherwise the base model's tokenizer is used, logged at
WARNING).

Prompt precedence: explicit `--prompt` (or `prompt=`) > adapter
`system_prompt.txt` > task `prompt`. Overriding an adapter that ships a
`system_prompt.txt` logs a warning, because it breaks the
training/inference prompt-match invariant.

> **Generation defaults match the published adapters.** Each built-in
> task's `generation` block is tuned to the published Qwen3.5-27B
> adapters. An adapter that was validated with different settings — for
> example, an earlier-generation comorbidity adapter validated at
> `max_new_tokens=2048` and `repetition_penalty=1.15` — needs explicit
> `--max-new-tokens` / `--repetition-penalty` overrides; the built-in
> defaults will not reproduce its validated behavior.

### HuggingFace loading options

The model loads in bfloat16 with `device_map="auto"`. Override the load
dtype with `--dtype {bfloat16,float16,float32}` (`dtype=` in Python;
default `"bfloat16"`). Models whose repositories ship custom code need
`--trust-remote-code` (`trust_remote_code=True`):

```python
df = extract(
    Path("notes.csv"),
    "comorbidity",
    model="/path/to/your-own-weights",
    dtype="float16",
    trust_remote_code=True,
)
```

### Input and output

Input formats: `.csv`, `.jsonl`, `.json` (array of objects), `.xlsx`,
`.txt` (one note per line). The text column defaults to `note_text`
(`--text-column`); the id column defaults to `note_id` (`--id-column`) and
is auto-generated when missing — for every input form (files, lists of
strings or dicts, DataFrames). `--input -` reads a single note from stdin.

Inputs and the output path are validated **before** the model loads: a
missing text column or an unsupported `--output` extension fails in
seconds, not after a multi-minute model load. Rows whose note text is null
or empty are never sent to the model; they appear in the output as error
rows with `finish_reason` `"skipped"` and a `validation_errors` entry of
the form `_input:empty_note:<reason>`.

Output format follows the `--output` extension: `.csv`, `.jsonl`, `.json`,
`.xlsx`, `.parquet`. Columns: the id column, one column per task field,
then `parse_success`, `validation_errors`, `raw_response` (full model text
when parsing failed, empty when it succeeded), `finish_reason`,
`input_tokens`, `output_tokens`. `.jsonl`/`.json` outputs carry real JSON
types — booleans and numbers stay typed, missing values become `null`,
and non-ASCII text is written unescaped.

## 2. Custom task YAML

A task file declares the fields to extract plus (optionally) your prompt,
the user-message template, and generation settings. Full grammar:
[`schema-reference.md`](schema-reference.md).

```yaml
# my_task.yaml
name: smoking_status
prompt: |
  Read the clinical note and extract the requested fields.
user_template: "Note: {note}\n\nReturn the JSON:"
generation:
  max_new_tokens: 200
fields:
  diagnosis:
    type: string
  smoker:
    type: enum
    values: [Y, N, Unknown]
```

```bash
ehrextract \
  --task my_task.yaml \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --input notes.jsonl --output results.csv
```

```python
from pathlib import Path
from ehrextract import extract

df = extract(
    Path("notes.jsonl"),
    "my_task.yaml",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
)
```

A plain schema file (just `fields`, no `prompt`) is also a valid task file:
ehrextract supplies a default instruction sentence, and for providers
without native structured output it appends a JSON-shape specification to
the system prompt so the model knows the exact output format.

## 3. API providers

Requires `pip install 'ehrextract[openai]'` or `'ehrextract[anthropic]'`.

### OpenAI-compatible

Works with OpenAI proper and any OpenAI-compatible endpoint (vLLM,
Together, Ollama, LiteLLM, ...) via `--base-url`.

```bash
export OPENAI_API_KEY=sk-...
ehrextract \
  --task clinical_vars \
  --provider openai --model gpt-4o-mini --api-key-env OPENAI_API_KEY \
  --input notes.jsonl --output results.json
```

```python
from pathlib import Path
from ehrextract import extract

df = extract(
    Path("notes.jsonl"),
    "clinical_vars",
    provider="openai",
    model="gpt-4o-mini",          # api_key=... or OPENAI_API_KEY env
    # base_url="http://localhost:8000/v1",  # any OpenAI-compatible server
)
```

The expected JSON shape is embedded in the system prompt; the response is
parsed and validated against the task schema.

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
ehrextract \
  --task full \
  --provider anthropic --model claude-sonnet-4-5 --api-key-env ANTHROPIC_API_KEY \
  --input notes.csv --output results.csv
```

```python
from pathlib import Path
from ehrextract import extract

df = extract(
    Path("notes.csv"),
    "full",
    provider="anthropic",
    model="claude-sonnet-4-5",
)
```

The Anthropic provider uses forced tool-use: the task schema becomes the
tool's input schema, so the model cannot return free text and the
JSON-shape boilerplate is omitted from the prompt.

### The egress notice

The first time a run sends data to an off-machine destination, ehrextract
writes a notice to stderr naming the destination and summarizing the PHI
requirements (BAA + Zero-Data-Retention). It is shown **once per process
per destination**, is purely informational, and **never blocks** — no
interactive prompt, no acknowledgement file, safe for batch/cluster jobs.

Suppress it when you have already made the data-handling decision:

```bash
ehrextract --ack-egress ...     # CLI flag
export ACK_EGRESS=1             # environment variable
```

```python
extract(..., on_egress="silent")
```

The local HuggingFace provider never triggers the notice — data stays on
your machine. Read [`data-handling.md`](data-handling.md) before sending
notes that may contain PHI to any API.

## CLI exit codes

- **0** — the run completed. Per-note failures (parse errors, individual
  provider errors) are recorded in the output rows, not the exit code.
- **1** — the run completed but **every** row is a provider error
  (`finish_reason == "error"`).
- **2** — usage or configuration error: bad flags, unknown task, a
  `--prompt` that looks like a file path but does not exist,
  `--api-key-env` naming an unset environment variable, a missing input
  column, an unsupported `--output` extension, ...

API provider calls are retried up to 3 times with exponential backoff and
jitter, except client errors (HTTP 400/401/403/404/422), which fail
immediately. At the end of each run the CLI logs a one-line summary —
rows / parsed / provider errors — at INFO level.
