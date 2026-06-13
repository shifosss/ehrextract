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
`repair_attempts`, `input_tokens`, `output_tokens`. `.jsonl`/`.json` outputs carry real JSON
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

## Constrained decoding (HuggingFace provider)

With `generation.constrained: true` (task YAML) or `--constrained`,
generation is token-constrained to the task's JSON Schema via
lm-format-enforcer: the model *cannot* emit anything but a structurally
valid, schema-conformant JSON object — exact keys, enum fields limited to
their allowed values. The built-in `full` task enables it by default,
paired with `repetition_penalty: 1.0` (its validated configuration; the
enforcer makes repetition penalties unnecessary).

Details that matter in practice:

- HuggingFace provider only. Other providers log one INFO line and proceed
  unconstrained (Anthropic already forces the schema via tool-use).
- Requires `lm-format-enforcer`, included in `pip install 'ehrextract[hf]'`.
  A missing install fails fast — before the model loads — with the fix in
  the error message. Escape hatch: `--no-constrained`.
- The first constrained generation scans the tokenizer vocabulary once
  (seconds on large vocabularies); later notes reuse the scan.
- The only remaining failure mode is truncation: if `max_new_tokens` runs
  out, the JSON cannot be closed and the row reports
  `finish_reason == "length"`. Raise `--max-new-tokens` — a repair loop
  would just truncate again.
- Constrained output starts at `{` from the first token, so thinking-model
  preambles (`<think>`) are suppressed entirely.

## Batch mode (API providers)

`--batch` (or `extract(batch=True)`) submits the whole input as **one**
provider-side batch — an OpenAI Batch or an Anthropic Message Batch — at
**50% of the synchronous API price**:

```bash
ehrextract --task clinical_vars \
  --provider anthropic --model claude-sonnet-4-5 --api-key-env ANTHROPIC_API_KEY \
  --input notes.csv --output results.csv --batch
```

Semantics:

- **Blocking.** The process polls until the batch finishes: typically
  under an hour on Anthropic, up to 24 h on OpenAI. On a scheduler, set
  the wall time accordingly.
- The **batch id is logged at INFO immediately after creation** and on
  every poll. If the process dies mid-poll, nothing is lost: results stay
  retrievable by id (OpenAI keeps the output file; Anthropic retains
  results for 29 days). Manual retrieval:

  ```python
  # OpenAI
  batch = client.batches.retrieve("batch_...")
  text = client.files.content(batch.output_file_id).text
  # Anthropic
  for entry in client.messages.batches.results("msgbatch_..."):
      print(entry.custom_id, entry.result.type)
  ```

- Per-request failures become normal `provider_error` rows; only
  batch-level failures (a rejected submission, a `failed`/`expired`
  status) abort the run, always with the batch id in the message.
- Rows with empty note text never leave the machine, exactly as in
  synchronous mode, and the egress notice applies unchanged.
- HuggingFace + `--batch` is a usage error (exit 2). Generic
  OpenAI-compatible servers (vLLM, Ollama, Together) do not implement
  `/v1/batches` — the run fails loudly at submission.
- API caps apply (OpenAI: 50k requests / 200 MB; Anthropic: 100k / 256 MB);
  ehrextract does not chunk in v0.3.

## Repair on parse failure

`--max-repairs N` (default **0** — off) re-prompts the model when a
response fails to parse or validate: the failed output is echoed back as
an assistant turn followed by the exact `field:code:detail` errors and an
instruction to return only the corrected JSON. Up to N repair rounds per
note; the `repair_attempts` output column records how many ran, and the
token columns sum all attempts (the honest cost number).

Caveats: each repair re-sends the note text (cost — and egress, on API
providers), so it is opt-in; provider errors and empty notes are never
repaired; under constrained decoding, failures are truncation — raise
`--max-new-tokens` instead. Batch runs repair their failed rows with
synchronous calls after the batch returns.

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
