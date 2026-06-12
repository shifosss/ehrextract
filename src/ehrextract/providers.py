"""LLM providers: GenerationConfig, ProviderResponse, concrete providers, egress notice.

There is no provider base class. A provider is any object with:

  - attributes ``name: str``, ``default_concurrency: int``,
    ``uses_schema_natively: bool``
  - ``generate(messages, config, json_schema=None) -> ProviderResponse``
  - ``egress_destination() -> str | None`` (None means the data stays local)
"""

import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DOCS_URL = "https://github.com/shifosss/ehrextract/blob/main/docs/ehrextract"
PROVIDER_NAMES: tuple[str, ...] = ("anthropic", "huggingface", "openai")
TOOL_NAME = "extract"

EgressMode = Literal["warn", "silent"]

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

    def generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
        json_schema: dict | None = None,
    ) -> ProviderResponse:
        # json_schema is ignored: output shape comes from the prompt-embedded hint.
        import torch

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


class AnthropicProvider:
    """Anthropic provider -- uses forced tool-use for structured output."""

    name = "anthropic"
    default_concurrency = 8
    uses_schema_natively = True

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

    def generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
        json_schema: dict | None = None,
    ) -> ProviderResponse:
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

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            system=system_text or "You are a clinical-feature extraction assistant.",
            messages=user_messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )

        tool_input: dict | None = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
                tool_input = dict(block.input)
                break
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


def load_provider(name: str, **kwargs: Any) -> Any:
    """Construct a built-in provider by name."""
    if name == "huggingface":
        return HuggingFaceProvider(**kwargs)
    elif name == "openai":
        return OpenAIProvider(**kwargs)
    elif name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise KeyError(f"unknown provider {name!r} (valid: {sorted(PROVIDER_NAMES)})")
