"""LLM providers: GenerationConfig, ProviderResponse, concrete providers, egress notice.

There is no provider base class. A provider is any object with:

  - attributes ``name: str``, ``default_concurrency: int``,
    ``uses_schema_natively: bool``
  - ``generate(messages, config, json_schema=None) -> ProviderResponse``
  - ``egress_destination() -> str | None`` (None means the data stays local)

Optional capability members (checked with ``getattr(..., False)``):

  - ``supports_constrained: bool`` -- honors ``GenerationConfig.constrained``
  - ``supports_batch: bool`` plus
    ``generate_batch(batch_messages, config, json_schema=None) ->
    list[ProviderResponse | Exception]`` -- one slot per input, in input
    order; per-request failures are Exception slots, lifecycle failures raise.
"""

import json
import logging
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DOCS_URL = "https://github.com/shifosss/ehrextract/blob/main/docs/ehrextract"
PROVIDER_NAMES: tuple[str, ...] = ("anthropic", "huggingface", "openai")
TOOL_NAME = "extract"

EgressMode = Literal["warn", "silent"]

_OPENAI_BATCH_POLL_SECONDS = 30.0
_ANTHROPIC_BATCH_POLL_SECONDS = 10.0
_OPENAI_BATCH_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


def _custom_id_index(custom_id: str) -> int:
    """Positional index from an internal 'req-{i}' custom id."""
    return int(custom_id.rsplit("-", 1)[1])


def _resolve_custom_id(custom_id: Any, n: int, batch_id: str) -> int | None:
    """Map a batch-result custom_id back to its input slot, or None if anomalous.

    An unparseable or out-of-range custom_id from the API must not crash the
    retrieval of an already-paid batch: the entry is skipped with a WARNING
    and the unmatched input slot keeps its pre-filled no-result Exception.
    """
    try:
        idx = _custom_id_index(str(custom_id))
    except (ValueError, IndexError):
        idx = -1
    if not 0 <= idx < n:
        logger.warning(
            "batch %s returned unknown custom_id %r; entry ignored", batch_id, custom_id
        )
        return None
    return idx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_WARNED_DESTINATIONS: set[str] = set()
_WARNED_LOCK = threading.Lock()

_WARNING_TEMPLATE = """\
================================================================
  ehrextract -- data egress notice
================================================================
You are about to send note text to: {destination}

If your input may contain Protected Health Information (PHI),
this destination requires:
  * A signed Business Associate Agreement (BAA), AND
  * Zero-Data-Retention enrollment with the provider

Safer paths for PHI:
  * Use --provider huggingface with a local model
  * Use AWS Bedrock with HIPAA-eligible Claude (BAA available)
  * See: {docs_url}/data-handling.md

This warning is shown once per process. Set ACK_EGRESS=1 to suppress.
================================================================
"""


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float | None = None
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = ()
    # Constrained JSON decoding (lm-format-enforcer). Honored only by providers
    # with supports_constrained=True (HuggingFace); others log one INFO and
    # proceed unconstrained.
    constrained: bool = False


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    finish_reason: str | None
    usage: dict[str, int] | None
    raw: dict | None


def warn_egress(destination: str, mode: EgressMode = "warn") -> None:
    """Write the PHI egress notice to stderr once per process per destination."""
    if mode == "silent" or os.environ.get("ACK_EGRESS") == "1":
        return
    with _WARNED_LOCK:
        if destination in _WARNED_DESTINATIONS:
            return
        _WARNED_DESTINATIONS.add(destination)
    sys.stderr.write(_WARNING_TEMPLATE.format(destination=destination, docs_url=DOCS_URL))
    sys.stderr.flush()


class HuggingFaceProvider:
    """Local transformers provider with optional PEFT adapter loading."""

    name = "huggingface"
    default_concurrency = 1
    uses_schema_natively = False
    supports_constrained = True

    def __init__(
        self,
        model: str,
        *,
        adapter_path: str | None = None,
        device_map: Any = "auto",
        dtype: str | None = "bfloat16",
        trust_remote_code: bool = False,
    ):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HuggingFaceProvider requires the 'transformers' and 'torch' packages. "
                "Install with: pip install 'ehrextract[hf]'"
            ) from e

        self.model = model
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._model, self._tokenizer = self._load_model_and_tokenizer()
        # Per-tokenizer lm-format-enforcer data; built lazily on the first
        # constrained generate() (the vocab scan is expensive).
        self._lmfe_tokenizer_data: Any = None

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        # Fine-tuned adapters ship their own tokenizer + chat template; prefer it.
        # Fall back to the base model only when the adapter dir genuinely lacks
        # tokenizer files -- corruption errors must propagate, not be masked.
        if self.adapter_path is not None:
            adapter_dir = Path(self.adapter_path)
            if any(
                (adapter_dir / fname).is_file()
                for fname in ("tokenizer.json", "tokenizer_config.json")
            ):
                return AutoTokenizer.from_pretrained(
                    self.adapter_path, trust_remote_code=self.trust_remote_code
                )
            logger.warning(
                "no tokenizer files (tokenizer.json/tokenizer_config.json) in adapter %s; "
                "falling back to base model tokenizer",
                self.adapter_path,
            )
        return AutoTokenizer.from_pretrained(self.model, trust_remote_code=self.trust_remote_code)

    def _load_model_and_tokenizer(self):
        import torch
        from transformers import AutoModelForCausalLM

        if self.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.dtype == "float16":
            dtype = torch.float16
        elif self.dtype == "float32" or self.dtype is None:
            dtype = torch.float32  # explicit opt-in; ~4x the memory of bf16
        else:
            raise ValueError(
                f"unsupported dtype {self.dtype!r} "
                "(use 'bfloat16', 'float16', 'float32', or None)"
            )

        tokenizer = self._load_tokenizer()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )

        if self.adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)

        if self.device_map is None and torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model, tokenizer

    def egress_destination(self) -> str | None:
        return None

    def _constrained_prefix_fn(self, json_schema: dict | None):
        from ehrextract import constrained

        if json_schema is None:
            raise ValueError(
                "constrained decoding requires a json_schema; "
                "Extractor passes it automatically"
            )
        if self._lmfe_tokenizer_data is None:
            self._lmfe_tokenizer_data = constrained.get_tokenizer_data(self._tokenizer)
        return constrained.build_prefix_allowed_tokens_fn(
            self._lmfe_tokenizer_data, json_schema
        )

    def generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
        json_schema: dict | None = None,
    ) -> ProviderResponse:
        # Without constrained decoding, json_schema is unused here: the output
        # shape comes from the prompt-embedded hint.
        import torch

        prefix_fn = self._constrained_prefix_fn(json_schema) if config.constrained else None

        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # add_special_tokens=False: the chat template already emits BOS;
        # re-adding it double-prepends BOS on Gemma/Llama-family bases.
        inputs = self._tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(self._model.device)

        do_sample = config.temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        # Sampling params are only meaningful (and warning-free) when sampling.
        if do_sample:
            gen_kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                gen_kwargs["top_p"] = config.top_p
        if prefix_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_fn

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = _THINK_RE.sub("", text).strip()

        usage = {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(new_tokens.shape[0]),
        }
        return ProviderResponse(
            text=text,
            finish_reason="length" if new_tokens.shape[0] >= config.max_new_tokens else "stop",
            usage=usage,
            raw=None,
        )


class OpenAIProvider:
    """OpenAI-compatible chat provider (OpenAI proper, vLLM, Together, Ollama, LiteLLM, etc.)."""

    name = "openai"
    default_concurrency = 8
    uses_schema_natively = False
    supports_batch = True

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenAIProvider requires the 'openai' package. "
                "Install with: pip install 'ehrextract[openai]'"
            ) from e

        self.model = model
        self.base_url = base_url
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def egress_destination(self) -> str | None:
        url = self.base_url or "https://api.openai.com"
        return urlparse(url).hostname

    def generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
        json_schema: dict | None = None,
    ) -> ProviderResponse:
        # json_schema is ignored: output shape comes from the prompt-embedded hint.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        }
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.stop:
            kwargs["stop"] = list(config.stop)

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        usage = None
        if completion.usage is not None:
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
        return ProviderResponse(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
            raw=completion.model_dump() if hasattr(completion, "model_dump") else None,
        )

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        config: GenerationConfig,
        json_schema: dict | None = None,
    ) -> list:
        """Submit everything as one OpenAI Batch (50% cost), poll, retrieve.

        Blocks until the batch reaches a terminal state (the completion window
        is 24h). Returns one ProviderResponse or Exception per input, in input
        order. Per-request failures are Exception slots; lifecycle failures
        (rejected create, failed/expired status, servers without /v1/batches)
        raise with the batch id.
        """
        lines = []
        for i, messages in enumerate(batch_messages):
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": config.max_new_tokens,
                "temperature": config.temperature,
            }
            if config.top_p is not None:
                body["top_p"] = config.top_p
            if config.stop:
                body["stop"] = list(config.stop)
            lines.append(json.dumps({
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))
        payload = "\n".join(lines).encode("utf-8")
        input_file = self._client.files.create(
            file=("ehrextract_batch.jsonl", payload), purpose="batch"
        )
        batch = self._client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # Log the id BEFORE the first poll: an interrupted job keeps it in the log.
        logger.info("openai batch %s created (%d requests)", batch.id, len(batch_messages))
        while batch.status not in _OPENAI_BATCH_TERMINAL:
            time.sleep(_OPENAI_BATCH_POLL_SECONDS + random.uniform(0, 3))
            batch = self._client.batches.retrieve(batch.id)
            logger.info("openai batch %s: %s (%s)", batch.id, batch.status, batch.request_counts)
        if batch.status != "completed":
            raise RuntimeError(f"openai batch {batch.id} ended with status {batch.status!r}")

        results: list[Any] = [
            RuntimeError(f"openai batch {batch.id}: no result for custom_id 'req-{i}'")
            for i in range(len(batch_messages))
        ]
        if batch.output_file_id:
            for line in self._client.files.content(batch.output_file_id).text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                idx = _resolve_custom_id(obj.get("custom_id"), len(batch_messages), batch.id)
                if idx is None:
                    continue
                response = obj.get("response") or {}
                if obj.get("error") or response.get("status_code") != 200:
                    detail = obj.get("error") or f"status {response.get('status_code')}"
                    results[idx] = RuntimeError(
                        f"openai batch request {obj['custom_id']} failed: {detail}"
                    )
                    continue
                body = response["body"]
                choice = body["choices"][0]
                usage = body.get("usage")
                results[idx] = ProviderResponse(
                    text=choice["message"].get("content") or "",
                    finish_reason=choice.get("finish_reason"),
                    usage=(
                        {
                            "input_tokens": usage["prompt_tokens"],
                            "output_tokens": usage["completion_tokens"],
                        }
                        if usage else None
                    ),
                    raw=obj,
                )
        if batch.error_file_id:
            for line in self._client.files.content(batch.error_file_id).text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                idx = _resolve_custom_id(obj.get("custom_id"), len(batch_messages), batch.id)
                if idx is None:
                    continue
                detail = obj.get("error") or obj.get("response")
                results[idx] = RuntimeError(
                    f"openai batch request {obj['custom_id']} failed: {detail}"
                )
        # PHI hygiene: batch files persist server-side; delete best-effort.
        for fid in (batch.input_file_id, batch.output_file_id, batch.error_file_id):
            if fid:
                try:
                    self._client.files.delete(fid)
                except Exception as e:  # noqa: BLE001
                    logger.warning("could not delete openai batch file %s: %s", fid, e)
        return results


class AnthropicProvider:
    """Anthropic provider -- uses forced tool-use for structured output."""

    name = "anthropic"
    default_concurrency = 8
    uses_schema_natively = True
    supports_batch = True

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "AnthropicProvider requires the 'anthropic' package. "
                "Install with: pip install 'ehrextract[anthropic]'"
            ) from e

        self.model = model
        self._client = Anthropic(api_key=api_key, timeout=timeout)

    def egress_destination(self) -> str | None:
        return "api.anthropic.com"

    def _request_params(
        self,
        messages: list[dict],
        config: GenerationConfig,
        json_schema: dict | None,
    ) -> dict[str, Any]:
        """Messages-API params shared by generate() and generate_batch()."""
        if json_schema is None:
            raise RuntimeError(
                "AnthropicProvider requires json_schema for forced tool-use; "
                "Extractor passes it automatically."
            )
        system_text = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})
        tool = {
            "name": TOOL_NAME,
            "description": "Return the extracted fields.",
            "input_schema": json_schema,
        }
        return {
            "model": self.model,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "system": system_text or "You are a clinical-feature extraction assistant.",
            "messages": user_messages,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }

    @staticmethod
    def _extract_tool_input(content) -> dict | None:
        for block in content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
                return dict(block.input)
        return None

    def generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
        json_schema: dict | None = None,
    ) -> ProviderResponse:
        resp = self._client.messages.create(**self._request_params(messages, config, json_schema))

        tool_input = self._extract_tool_input(resp.content)
        if tool_input is None:
            # Forced tool_choice should make this unreachable. If it ever
            # happens, surface the API anomaly rather than fabricating output.
            raise RuntimeError(
                f"Anthropic returned no '{TOOL_NAME}' tool_use block "
                f"despite forced tool_choice; stop_reason={resp.stop_reason}"
            )

        usage = None
        if resp.usage is not None:
            usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}

        return ProviderResponse(
            text=json.dumps(tool_input),
            finish_reason=resp.stop_reason,
            usage=usage,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        config: GenerationConfig,
        json_schema: dict | None = None,
    ) -> list:
        """Submit everything as one Message Batch (50% cost), poll, retrieve.

        Blocks until processing ends (typically <1h). Returns one
        ProviderResponse or Exception per input, in input order. Results are
        retained server-side for 29 days. Per-request failures are Exception
        slots; lifecycle failures raise with the batch id.
        """
        requests = [
            {"custom_id": f"req-{i}", "params": self._request_params(m, config, json_schema)}
            for i, m in enumerate(batch_messages)
        ]
        batch = self._client.messages.batches.create(requests=requests)
        # Log the id BEFORE the first poll: an interrupted job keeps it in the log.
        logger.info("anthropic batch %s created (%d requests)", batch.id, len(requests))
        while batch.processing_status != "ended":
            time.sleep(_ANTHROPIC_BATCH_POLL_SECONDS + random.uniform(0, 2))
            batch = self._client.messages.batches.retrieve(batch.id)
            logger.info(
                "anthropic batch %s: %s (%s)",
                batch.id, batch.processing_status, batch.request_counts,
            )

        results: list[Any] = [
            RuntimeError(f"anthropic batch {batch.id}: no result for custom_id 'req-{i}'")
            for i in range(len(batch_messages))
        ]
        for entry in self._client.messages.batches.results(batch.id):
            idx = _resolve_custom_id(entry.custom_id, len(batch_messages), batch.id)
            if idx is None:
                continue
            if entry.result.type != "succeeded":
                results[idx] = RuntimeError(
                    f"anthropic batch request {entry.custom_id} {entry.result.type}: "
                    f"{getattr(entry.result, 'error', None)}"
                )
                continue
            msg = entry.result.message
            tool_input = self._extract_tool_input(msg.content)
            if tool_input is None:
                results[idx] = RuntimeError(
                    f"anthropic batch request {entry.custom_id} returned no "
                    f"'{TOOL_NAME}' tool_use block; stop_reason={msg.stop_reason}"
                )
                continue
            usage = None
            if msg.usage is not None:
                usage = {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                }
            results[idx] = ProviderResponse(
                text=json.dumps(tool_input),
                finish_reason=msg.stop_reason,
                usage=usage,
                raw=None,
            )
        return results


def load_provider(name: str, **kwargs: Any) -> Any:
    """Construct a built-in provider by name."""
    if name == "huggingface":
        return HuggingFaceProvider(**kwargs)
    elif name == "openai":
        return OpenAIProvider(**kwargs)
    elif name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise KeyError(f"unknown provider {name!r} (valid: {sorted(PROVIDER_NAMES)})")
