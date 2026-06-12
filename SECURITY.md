# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Email security reports to: **alexandar.zhang@mail.utoronto.ca**

Include: a clear description, reproduction steps, the affected version, and any proof-of-concept code. We aim to acknowledge within 5 business days and to release a fix within 90 days from confirmation.

## Scope

In scope:

- Code-execution, injection, or sandbox-escape vulnerabilities in the `ehrextract` library or CLI.
- PHI-leakage paths -- anywhere note text could be persisted, logged, or transmitted other than to the configured provider destination.
- Bugs in the egress-warning machinery (`ehrextract/providers.py`) that suppress the PHI egress notice without `--ack-egress` or `ACK_EGRESS=1`.

Out of scope:

- Vulnerabilities in third-party providers (OpenAI / Anthropic / HuggingFace) -- please report to those vendors.
- Issues that require a malicious schema YAML provided by the operator (the operator is trusted).
- Provider model behavior, hallucinations, or factual inaccuracies -- these are research-quality concerns, see NOTICE section 6.

## What is *not* a security control

The egress warning is **informational** -- it nudges operators toward BAA / Zero-Data-Retention awareness but does not technically block PHI from leaving the machine. The user must enforce data-handling policy themselves. See NOTICE section 4 and `docs/ehrextract/data-handling.md` for full caveats.
