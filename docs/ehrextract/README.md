# ehrextract

[![PyPI](https://img.shields.io/pypi/v/ehrextract)](https://pypi.org/project/ehrextract/)
[![Python](https://img.shields.io/pypi/pyversions/ehrextract)](https://pypi.org/project/ehrextract/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](../../src/ehrextract/LICENSE)
[![tests](https://github.com/shifosss/ehrextract/actions/workflows/test.yml/badge.svg)](https://github.com/shifosss/ehrextract/actions/workflows/test.yml)

Structured feature extraction from clinical notes. Three steps:

1. **Bring your notes** — CSV, JSONL, JSON, XLSX, plain text, or a pandas
   DataFrame.
2. **Pick a task** — a built-in task (`comorbidity`, `clinical_vars`, `full`)
   or your own YAML file with your own fields and prompt.
3. **Pick a model** — a fine-tuned LoRA adapter on a local base model, your
   own local HuggingFace weights, or an API model (OpenAI-compatible or
   Anthropic).

One command (or one function call) later you have a results table —
CSV, JSONL, JSON, XLSX, or Parquet — with one column per extracted field.

> **Important — read before use.**
> ehrextract is **research-grade software**. It is **NOT a medical device**,
> is **NOT FDA-cleared / Health Canada-approved**, and **MUST NOT** be used
> for clinical decision-making, patient triage, eligibility determination,
> re-identification, surveillance, or any setting where its outputs affect
> a person's access to care, insurance, employment, or legal status.
> Outputs may hallucinate; any research use requires per-row human review.
> The egress-warning system is informational, not a privacy compliance
> control. **Users are solely responsible for HIPAA / PHIPA / PIPEDA / GDPR
> / REB compliance.** See [`NOTICE`](../../src/ehrextract/NOTICE) for the
> full acceptable-use scope.

## Install

```bash
pip install ehrextract                  # core (~50 MB)
pip install 'ehrextract[hf]'            # + torch + transformers + peft (~3 GB)
pip install 'ehrextract[openai]'        # + openai SDK
pip install 'ehrextract[anthropic]'     # + anthropic SDK
```

Python ≥ 3.11. For a development install from a clone, see
[CONTRIBUTING.md](https://github.com/shifosss/ehrextract/blob/main/CONTRIBUTING.md).

## 30-second example

```bash
ehrextract \
  --task comorbidity \
  --model Qwen/Qwen3.5-27B --adapter /path/to/adapter \
  --input notes.csv --output results.csv
```

or, as a library:

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

The input needs a `note_text` column (configurable via `--text-column`); a
`note_id` column is added automatically when absent. The output has one
column per task field plus `parse_success`, `validation_errors`,
`raw_response`, `finish_reason`, and token counts.

## Built-in tasks

| Task | Fields | What it extracts |
|---|---|---|
| `comorbidity` | 17 | Free-text diagnosis list + 16 Y/N comorbidity categories |
| `clinical_vars` | 4 | Feeding and neurologic variables (tube/oral feeding, aspiration risk, NI trajectory) |
| `full` | 20 | Joint task: the 16 comorbidity categories + the 4 clinical variables |

Built-in tasks ship inside the package; `--task <name>` works without any
extra files. Define your own task in YAML — see
[`schema-reference.md`](schema-reference.md).

> **Note on the `full` task.** The research pipeline that produced the
> published evaluation numbers for the joint 20-field task used constrained
> JSON decoding to force the output shape. ehrextract v0.2.0 does **not**
> constrain decoding (planned as a future feature), so `full`-task outputs
> can diverge from the published numbers on hard notes — watch the
> `parse_success` and `validation_errors` columns.

## Data handling

If your input may contain PHI, read [`data-handling.md`](data-handling.md)
BEFORE running with any API provider. The package writes a data-egress
notice to stderr (once per process per destination) on API use; it never
blocks, and it does not (and cannot) guarantee compliance for you. The
local HuggingFace provider keeps all data on your machine.

## Documentation

- [`quickstart.md`](quickstart.md) — fine-tuned adapters, custom tasks, API providers
- [`schema-reference.md`](schema-reference.md) — the task-file YAML reference
- [`data-handling.md`](data-handling.md) — PHI, egress notice, BAA-eligible providers
- [`extending-providers.md`](extending-providers.md) — plug in a custom provider

## Authors and institutions

ehrextract was developed by:

- **Chen Zhang** (lead author)
- **Yibing Xia** (co-author)
- **Sanjay Mahant, MD** -- supervisor, The Hospital for Sick Children (SickKids)
- **Nathan Taback, PhD** -- supervisor, University of Toronto

at **The Hospital for Sick Children** (Toronto, Canada) and the
**University of Toronto** (Toronto, Canada). Please cite the project if
you use it in published work.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](../../src/ehrextract/LICENSE)
for the full license text and [`NOTICE`](../../src/ehrextract/NOTICE) for
attribution, the no-endorsement clause, the clinical-use disclaimer, and the
acceptable-use restrictions that supplement (but do not override) the License.
