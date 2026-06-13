"""HuggingFaceProvider tests. The loader is mocked except for the GPU smoke test."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from ehrextract.providers import GenerationConfig, HuggingFaceProvider

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _mocked_provider(**kwargs) -> HuggingFaceProvider:
    with patch.object(HuggingFaceProvider, "_load_model_and_tokenizer", return_value=(None, None)):
        return HuggingFaceProvider(model="dummy", **kwargs)


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 7
    eos_token_id = 7

    def __init__(self):
        self.encode_calls: list[dict] = []
        self.decoded_text = '{"x": "y"}'

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "RENDERED CHAT"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True, **kw):
        self.encode_calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return _FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens=True):
        return self.decoded_text


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.gen_kwargs: dict | None = None

    def generate(self, input_ids=None, **kwargs):
        self.gen_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])


def _provider_with_fakes(**kwargs):
    model, tok = _FakeModel(), _FakeTokenizer()
    with patch.object(
        HuggingFaceProvider, "_load_model_and_tokenizer", return_value=(model, tok)
    ):
        p = HuggingFaceProvider(model="dummy", **kwargs)
    return p, model, tok


_MSGS = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "hi"},
]


def test_duck_typed_attributes():
    p = _mocked_provider()
    assert p.name == "huggingface"
    assert p.uses_schema_natively is False
    assert p.default_concurrency == 1


def test_egress_destination_is_none():
    p = _mocked_provider()
    assert p.egress_destination() is None


def test_defaults_bfloat16_and_device_map_auto():
    """Adapter-ergonomics defaults (design decision 7)."""
    p = _mocked_provider()
    assert p.dtype == "bfloat16"
    assert p.device_map == "auto"
    assert p.adapter_path is None


def test_unknown_dtype_raises():
    with pytest.raises(ValueError, match="dtype"):
        HuggingFaceProvider(model="dummy", dtype="float64")


# --- A2: generate() kwargs, double-BOS fix, dtype= loader kwarg ---


def test_greedy_decoding_omits_sampling_params():
    """temperature=0 -> do_sample=False and NO temperature/top_p keys at all."""
    p, model, _ = _provider_with_fakes()
    p.generate(_MSGS, GenerationConfig(temperature=0.0, top_p=0.9))
    assert model.gen_kwargs is not None
    assert model.gen_kwargs["do_sample"] is False
    assert "temperature" not in model.gen_kwargs
    assert "top_p" not in model.gen_kwargs
    assert model.gen_kwargs["pad_token_id"] == 7


def test_sampling_params_forwarded_when_temperature_positive():
    p, model, _ = _provider_with_fakes()
    p.generate(_MSGS, GenerationConfig(temperature=0.7, top_p=0.9, repetition_penalty=1.15))
    assert model.gen_kwargs["do_sample"] is True
    assert model.gen_kwargs["temperature"] == 0.7
    assert model.gen_kwargs["top_p"] == 0.9
    assert model.gen_kwargs["repetition_penalty"] == 1.15


def test_chat_template_encoded_without_special_tokens():
    """Double-BOS fix: the rendered template already contains BOS."""
    p, _, tok = _provider_with_fakes()
    p.generate(_MSGS, GenerationConfig())
    assert len(tok.encode_calls) == 1
    assert tok.encode_calls[0]["text"] == "RENDERED CHAT"
    assert tok.encode_calls[0]["add_special_tokens"] is False


@pytest.mark.parametrize(
    "dtype_arg,expected",
    [("bfloat16", "bfloat16"), ("float16", "float16"), ("float32", "float32"), (None, "float32")],
)
def test_from_pretrained_uses_dtype_kwarg(dtype_arg, expected):
    """A2: `dtype=` (not the deprecated `torch_dtype=`) reaches from_pretrained."""
    with patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as model_fp, \
         patch.object(transformers.AutoTokenizer, "from_pretrained") as tok_fp:
        tok_fp.return_value = MagicMock(pad_token_id=1)
        HuggingFaceProvider(model="dummy-model", dtype=dtype_arg)
    kwargs = model_fp.call_args.kwargs
    assert kwargs["dtype"] is getattr(torch, expected)
    assert "torch_dtype" not in kwargs
    assert kwargs["device_map"] == "auto"


# --- A3: adapter tokenizer fallback semantics ---


def test_adapter_tokenizer_preferred_when_files_present(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    p = _mocked_provider(adapter_path=str(tmp_path))
    with patch.object(transformers.AutoTokenizer, "from_pretrained") as fp:
        p._load_tokenizer()
    fp.assert_called_once_with(str(tmp_path), trust_remote_code=False)


def test_adapter_without_tokenizer_files_falls_back_with_warning(tmp_path, caplog):
    p = _mocked_provider(adapter_path=str(tmp_path))
    with patch.object(transformers.AutoTokenizer, "from_pretrained") as fp, \
         caplog.at_level(logging.WARNING, logger="ehrextract.providers"):
        p._load_tokenizer()
    fp.assert_called_once_with("dummy", trust_remote_code=False)
    assert any("falling back" in r.message for r in caplog.records)


def test_adapter_tokenizer_load_error_propagates(tmp_path):
    """Corruption in a present adapter tokenizer must not be masked by the fallback."""
    (tmp_path / "tokenizer.json").write_text("corrupt", encoding="utf-8")
    p = _mocked_provider(adapter_path=str(tmp_path))
    with patch.object(
        transformers.AutoTokenizer, "from_pretrained", side_effect=OSError("corrupt tokenizer")
    ):
        with pytest.raises(OSError, match="corrupt tokenizer"):
            p._load_tokenizer()


def test_trust_remote_code_forwarded_to_tokenizer():
    p = _mocked_provider(trust_remote_code=True)
    with patch.object(transformers.AutoTokenizer, "from_pretrained") as fp:
        p._load_tokenizer()
    fp.assert_called_once_with("dummy", trust_remote_code=True)


# --- v0.3: constrained decoding wiring (enforcer itself tested in unit/test_constrained.py) ---

_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
    "additionalProperties": False,
}


def test_supports_constrained_attr():
    assert HuggingFaceProvider.supports_constrained is True


def test_constrained_injects_prefix_allowed_tokens_fn():
    p, model, _ = _provider_with_fakes()

    def sentinel(batch_id, sent):
        return [1]

    with patch("ehrextract.constrained.get_tokenizer_data", return_value=object()), \
         patch("ehrextract.constrained.build_prefix_allowed_tokens_fn", return_value=sentinel):
        p.generate(_MSGS, GenerationConfig(constrained=True), json_schema=_SCHEMA)
    assert model.gen_kwargs["prefix_allowed_tokens_fn"] is sentinel


def test_unconstrained_omits_prefix_allowed_tokens_fn():
    """v0.2 regression guard: default config leaves generate kwargs untouched."""
    p, model, _ = _provider_with_fakes()
    p.generate(_MSGS, GenerationConfig(), json_schema=_SCHEMA)
    assert "prefix_allowed_tokens_fn" not in model.gen_kwargs


def test_constrained_without_json_schema_raises():
    p, _, _ = _provider_with_fakes()
    with pytest.raises(ValueError, match="json_schema"):
        p.generate(_MSGS, GenerationConfig(constrained=True))


def test_tokenizer_data_cached_on_provider_instance():
    p, _, _ = _provider_with_fakes()
    with patch("ehrextract.constrained.get_tokenizer_data", return_value=object()) as gtd, \
         patch("ehrextract.constrained.build_prefix_allowed_tokens_fn",
               return_value=lambda b, s: [1]) as build:
        p.generate(_MSGS, GenerationConfig(constrained=True), json_schema=_SCHEMA)
        p.generate(_MSGS, GenerationConfig(constrained=True), json_schema=_SCHEMA)
    gtd.assert_called_once()
    assert build.call_count == 2  # fresh enforcer per call, shared tokenizer data


@pytest.mark.gpu
def test_constrained_smoke_tiny_model():
    """The killer demo: a model too small to reliably emit JSON does so under constraint."""
    if os.environ.get("RUN_HF_TESTS") != "1":
        pytest.skip("set RUN_HF_TESTS=1 to run HF integration tests")
    import json

    from ehrextract.schema import load_task, to_json_schema

    task = load_task("full")
    p = HuggingFaceProvider(model="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="bfloat16")
    msgs = [
        {"role": "system", "content": task.prompt},
        {"role": "user", "content": task.user_template.format(
            note="Synthetic note: healthy child, eats orally, no feeding tube."
        )},
    ]
    resp = p.generate(
        msgs,
        GenerationConfig(max_new_tokens=512, constrained=True),
        json_schema=to_json_schema(task.schema),
    )
    data = json.loads(resp.text)
    assert set(data) <= set(task.schema.field_names())


@pytest.mark.gpu
def test_smoke_generation_against_tiny_model():
    """End-to-end test on a small model. Gated by gpu marker AND RUN_HF_TESTS=1."""
    if os.environ.get("RUN_HF_TESTS") != "1":
        pytest.skip("set RUN_HF_TESTS=1 to run HF integration tests")

    p = HuggingFaceProvider(
        model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        dtype="bfloat16",
    )
    msgs = [
        {"role": "system", "content": "Reply with the single token: OK"},
        {"role": "user", "content": "ping"},
    ]
    resp = p.generate(msgs, GenerationConfig(max_new_tokens=4, temperature=0.0))
    assert resp.text
