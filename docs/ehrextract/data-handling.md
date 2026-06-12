# Data handling and PHI

> **The acceptable-use restrictions, no-endorsement clause, and clinical-use
> disclaimers in [`NOTICE`](../../src/ehrextract/NOTICE) apply to everything
> on this page. Read NOTICE before deploying ehrextract on any data that
> may contain Protected Health Information (PHI).**

`ehrextract` does not detect PHI. The user is responsible for classifying
their data and choosing an appropriate provider.

## Local-first

The default provider is local HuggingFace inference: model weights run on
your hardware and note text never leaves the machine. Local providers
report no egress destination and never trigger the egress notice. If your
data is PHI and you have no BAA in place, this is the path to use.

## Safe paths for PHI

| Provider | PHI-safe? | Requirements |
|---|---|---|
| `huggingface` (local) | Yes | Data never leaves your machine |
| `openai` (api.openai.com) | Only with BAA + ZDR enrollment | Contact OpenAI Enterprise |
| `openai` with `--base-url` | Depends on the endpoint | Self-hosted vLLM/Ollama is local; a HIPAA-eligible gateway (e.g. AWS Bedrock behind an OpenAI-compatible proxy) needs its own BAA |
| `anthropic` (api.anthropic.com) | Not directly | For PHI, use HIPAA-eligible Claude via AWS Bedrock instead of the direct API |

## The egress notice

When a run is about to send note text off-machine, ehrextract writes a
notice to stderr naming the destination (`api.openai.com`,
`api.anthropic.com`, or the `--base-url` hostname) and summarizing the
requirements for PHI. Semantics:

- **Shown once per process per destination.** A batch of 10,000 notes
  produces one notice, not 10,000.
- **Informational only — it never blocks.** There is no interactive
  prompt and no acknowledgement file, so it is safe in batch jobs,
  cluster schedulers, and CI. **It is NOT a privacy compliance control.**
- **No state is written to disk.** Every new process shows the notice
  again. There is no per-machine acknowledgement cache to go stale —
  each run re-surfaces the decision.

Suppress it once you have made the data-handling decision:

```bash
ehrextract --ack-egress ...     # CLI: sets "silent" mode for this run
export ACK_EGRESS=1             # environment variable, e.g. for batch jobs
```

```python
extract(..., on_egress="silent")            # library
Extractor(provider, task, on_egress="silent")
```

## Outputs can contain PHI too

Extracted field values derive from the notes, and the `raw_response`
column contains the model's full raw output whenever parsing fails.
Treat results files with the same confidentiality as the input notes.

## Logging

The CLI logs at INFO level by default (`-q` for warnings only, `-v` for
DEBUG). Note text is never logged; provider errors and retries are logged
with the exception message, and prompt-override warnings name the adapter
path. As a library, ehrextract is silent until you configure logging
(standard `logging` module, logger namespace `ehrextract`).
