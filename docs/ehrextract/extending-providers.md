# Extending: bring your own provider

There is no provider base class and no registry. A provider is **any
object** with this duck-typed interface:

| Member | Type | Meaning |
|---|---|---|
| `name` | `str` attribute | Used in log messages |
| `default_concurrency` | `int` attribute | Thread count when `max_concurrency` is not given (use 1 unless `generate` is thread-safe) |
| `uses_schema_natively` | `bool` attribute | `True` if the backend enforces the schema itself (e.g. forced tool-use); the JSON-shape boilerplate is then omitted from the prompt |
| `generate(messages, config, json_schema=None)` | method | Run one completion; returns a `ProviderResponse` |
| `egress_destination()` | method | Hostname the data is sent to, or `None` if it stays local |

`messages` is a list of `{"role": ..., "content": ...}` dicts; `config` is
a `GenerationConfig`; `json_schema` is the JSON Schema rendering of the
task's fields, passed on **every** call — ignore it unless your backend can
enforce it. Return the model text in `ProviderResponse.text` (for a
schema-native backend, serialize the structured result to a JSON string).
`ProviderResponse.usage`, when not `None`, should be
`{"input_tokens": ..., "output_tokens": ...}` — it feeds the token columns
of the results table.

Two **optional** capability members extend the interface; both are checked
with `getattr(provider, ..., False)`, so providers written against earlier
versions keep working unchanged:

| Member | Meaning |
|---|---|
| `supports_constrained` | `True` declares that `generate` honors `GenerationConfig.constrained` (token-constrained decoding). Without it, a constrained run logs one INFO and proceeds unconstrained. |
| `supports_batch` + `generate_batch(batch_messages, config, json_schema=None)` | Enables `run(batch=True)`. Receives a list of message lists; must return one entry per input, **in input order**: a `ProviderResponse` for success or an `Exception` instance for a per-request failure (mapped to a `provider_error` row). Batch-level failures should raise, ideally naming the backend's batch/job id. |

## Example

```python
from ehrextract import Extractor, GenerationConfig, load_task
from ehrextract.providers import ProviderResponse

class CohereProvider:
    name = "cohere"
    default_concurrency = 8
    uses_schema_natively = False

    def __init__(self, model: str, api_key: str):
        import cohere  # lazy -- keep heavy SDKs out of import time
        self._client = cohere.ClientV2(api_key)
        self.model = model

    def egress_destination(self) -> str | None:
        return "api.cohere.com"

    def generate(self, messages, config: GenerationConfig, json_schema=None) -> ProviderResponse:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        return ProviderResponse(
            text=resp.message.content[0].text,
            finish_reason="stop",
            usage=None,
            raw=None,
        )

task = load_task("comorbidity")
extractor = Extractor(CohereProvider("command-r", api_key="..."), task)
df = extractor.run(["45-year-old admitted for G-tube assessment..."])
```

Pass the provider instance directly to `Extractor`. (The string-named
shortcut — `extract(..., provider="huggingface" | "openai" | "anthropic")`
and the CLI `--provider` flag — covers only the three built-ins; custom
providers always go through `Extractor`.)

## What the Extractor does with your provider

- Calls `egress_destination()` once per `Extractor` (before the first
  generation); a non-`None` value triggers the stderr egress notice, which
  is itself deduplicated per process per destination (see
  [`data-handling.md`](data-handling.md)).
- Retries `generate()` up to 3 times (`max_retries`) with exponential
  backoff plus jitter, then emits a `provider_error` row and continues
  with the remaining notes. Exceptions carrying a `status_code` attribute
  in {400, 401, 403, 404, 422} are treated as non-retryable client errors
  and fail immediately — set `status_code` on your exceptions if your
  backend distinguishes them.
- Dispatches notes across a thread pool sized by `max_concurrency` (or
  your `default_concurrency`); output row order always matches input
  order. `max_concurrency` must be ≥ 1; for local providers
  (`egress_destination()` returns `None`) a value above
  `default_concurrency` is clamped to it with a warning, because a shared
  local model is typically not thread-safe. If a worker raises (rather
  than returning an error row), queued notes are cancelled and the
  exception propagates.
- Never calls your provider for rows whose note text is null or empty —
  those become error rows (`finish_reason` `"skipped"`) directly. In batch
  mode the same applies: empty rows are handled locally and only live rows
  reach `generate_batch`.
- Parses and validates `ProviderResponse.text` against the task schema —
  markdown fences and `<think>` blocks are stripped first, so providers
  do not need to clean those up themselves.
- With `max_repairs > 0`, re-calls `generate()` with the failed output and
  the field errors appended as extra chat turns (also for rows that failed
  inside a batch — repairs are always synchronous calls).
