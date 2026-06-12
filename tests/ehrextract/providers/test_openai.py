"""Tests for OpenAIProvider via mocked httpx transport.

The openai SDK uses httpx for HTTP. We mock at that layer using
httpx.MockTransport injected via OpenAI(http_client=...). This is the
canonical openai-python testing pattern and verifies request marshaling
at the wire level.
"""

import json
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from ehrextract.providers import GenerationConfig, OpenAIProvider


@pytest.fixture
def fixture_response(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "responses" / "openai_well_formed.json").read_text())


def _mock_client(handler) -> OpenAI:
    """Build an OpenAI client whose HTTP layer is replaced by `handler`."""
    transport = httpx.MockTransport(handler)
    return OpenAI(api_key="sk-test", http_client=httpx.Client(transport=transport))


def test_basic_call(fixture_response):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=fixture_response)

    p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    p._client = _mock_client(handler)

    resp = p.generate([{"role": "user", "content": "hi"}], GenerationConfig())
    assert "asthma" in resp.text
    assert resp.finish_reason == "stop"
    assert resp.usage == {"input_tokens": 100, "output_tokens": 30}
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-4o-mini"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]


def test_json_schema_param_is_ignored(fixture_response):
    """Decision 6: openai is NOT schema-native; the shape hint lives in the prompt."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=fixture_response)

    p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    p._client = _mock_client(handler)

    p.generate([{"role": "user", "content": "hi"}], GenerationConfig(),
               json_schema={"type": "object"})
    assert "response_format" not in captured["payload"]
    assert "tools" not in captured["payload"]


def test_stop_and_top_p_forwarded(fixture_response):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=fixture_response)

    p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    p._client = _mock_client(handler)

    p.generate(
        [{"role": "user", "content": "hi"}],
        GenerationConfig(max_new_tokens=64, top_p=0.9, stop=("<end>",)),
    )
    assert captured["payload"]["max_tokens"] == 64
    assert captured["payload"]["top_p"] == 0.9
    assert captured["payload"]["stop"] == ["<end>"]


def test_uses_schema_natively_is_false():
    """Regression for the v0.1.1 structured-output bug (design decision 6)."""
    p = OpenAIProvider(model="gpt-4o-mini", api_key="sk")
    assert p.uses_schema_natively is False
    assert p.name == "openai"
    assert p.default_concurrency == 8


def test_egress_destination_default():
    p = OpenAIProvider(model="m", api_key="sk-test")
    assert p.egress_destination() == "api.openai.com"


def test_egress_destination_custom_base_url():
    p = OpenAIProvider(model="m", api_key="sk", base_url="http://localhost:8000/v1")
    assert p.egress_destination() == "localhost"


def test_api_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    p = OpenAIProvider(model="m", api_key="sk")
    p._client = _mock_client(handler)

    with pytest.raises(Exception):
        p.generate([{"role": "user", "content": "hi"}], GenerationConfig())
