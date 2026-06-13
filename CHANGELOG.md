# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-13

### Added
- Constrained JSON decoding for the local HuggingFace provider (lm-format-enforcer): `generation.constrained` task key and `--constrained` / `--no-constrained` CLI flags. The built-in `full` task now enables it by default (paired with `repetition_penalty: 1.0`, matching its validated configuration).
- Provider-side batch mode (`--batch` / `extract(batch=True)`) for the OpenAI and Anthropic providers -- one OpenAI Batch / Anthropic Message Batch per run at 50% API cost, with a blocking poll, per-request failure isolation, and (OpenAI) server-side batch-file deletion after retrieval.
- Bounded repair loop (`--max-repairs` / `extract(max_repairs=N)`, default 0 = off): on a parse or validation failure the model is re-prompted with the exact field errors. New `repair_attempts` output column; token counts are summed across attempts.

### Changed
- The `hf` extra now includes `lm-format-enforcer`; running the `full` task on the HuggingFace provider with an older `[hf]` install raises an actionable ImportError (escape hatch: `--no-constrained`).
- `openai` extra floor raised to 1.35 and `anthropic` to 0.42 (batch-capable SDKs).
- Output frames gain the `repair_attempts` column after `finish_reason` (positional consumers take note; column-name access is unaffected).

## [0.2.0] - 2026-06-12

### Added
- Public release of `ehrextract` -- structured feature extraction from clinical notes via LLMs.
- Built-in task definitions shipped with the package: `comorbidity`, `clinical_vars`, `full`.
- Three providers out of the box: HuggingFace local (`hf` extra, incl. LoRA adapters), OpenAI-compatible (`openai` extra), and Anthropic (`anthropic` extra).
- YAML task files: typed schema fields plus optional `prompt`, `user_template`, and `generation` keys; schema-aware prompting and structured-output parsing (Anthropic uses forced tool-use).
- CLI entry point `ehrextract` with a PHI egress notice (stderr, once per process per destination) before any text leaves the machine; silence with `--ack-egress` or `ACK_EGRESS=1`.
- Apache 2.0 license + supplemental NOTICE covering attribution, no-endorsement, not-a-medical-device, PHI/regulatory scope, acceptable-use restrictions, and research-use guidance.
- CPU-only test suite; GPU-dependent tests carry the `gpu` marker.

[0.3.0]: https://github.com/shifosss/ehrextract/releases/tag/v0.3.0
[0.2.0]: https://github.com/shifosss/ehrextract/releases/tag/v0.2.0
