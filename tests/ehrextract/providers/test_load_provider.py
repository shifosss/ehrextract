"""Tests for the load_provider factory and GenerationConfig defaults."""

import pytest

from ehrextract.providers import (
    AnthropicProvider,
    GenerationConfig,
    OpenAIProvider,
    load_provider,
)


def test_unknown_provider_raises_with_valid_names():
    with pytest.raises(KeyError, match="huggingface"):
        load_provider("nonsense")


def test_load_openai():
    p = load_provider("openai", model="gpt-4o-mini", api_key="sk-test")
    assert isinstance(p, OpenAIProvider)
    assert p.model == "gpt-4o-mini"


def test_load_anthropic():
    p = load_provider("anthropic", model="claude-haiku-4-5", api_key="k")
    assert isinstance(p, AnthropicProvider)
    assert p.model == "claude-haiku-4-5"


def test_generation_config_defaults():
    g = GenerationConfig()
    assert g.max_new_tokens == 1024
    assert g.temperature == 0.0
    assert g.top_p is None
    assert g.repetition_penalty == 1.0
    assert g.stop == ()
