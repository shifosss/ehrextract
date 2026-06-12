# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-12

### Added
- Public release of `ehrextract` -- structured feature extraction from clinical notes via LLMs.
- Built-in task definitions shipped with the package: `comorbidity`, `clinical_vars`, `full`.
- Three providers out of the box: HuggingFace local (`hf` extra, incl. LoRA adapters), OpenAI-compatible (`openai` extra), and Anthropic (`anthropic` extra).
- YAML task files: typed schema fields plus optional `prompt`, `user_template`, and `generation` keys; schema-aware prompting and structured-output parsing (Anthropic uses forced tool-use).
- CLI entry point `ehrextract` with a PHI egress notice (stderr, once per process per destination) before any text leaves the machine; silence with `--ack-egress` or `ACK_EGRESS=1`.
- Apache 2.0 license + supplemental NOTICE covering attribution, no-endorsement, not-a-medical-device, PHI/regulatory scope, acceptable-use restrictions, and research-use guidance.
- CPU-only test suite; GPU-dependent tests carry the `gpu` marker.

[0.2.0]: https://github.com/shifosss/ehrextract/releases/tag/v0.2.0
