"""Tests for AnthropicProvider via mocked httpx transport.

The anthropic SDK uses httpx for HTTP. We mock at that layer using
httpx.MockTransport injected via Anthropic(http_client=...).
"""

import json
from pathlib import Path

import httpx
import pytest
from anthropic import Anthropic

from ehrextract.providers import AnthropicProvider, GenerationConfig
from ehrextract.schema import FieldSpec, Schema, to_json_schema


@pytest.fixture
def fixture_response(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "responses" / "anthropic_well_formed.json").read_text())


def _json_schema() -> dict:
    return to_json_schema(Schema(fields=(
        FieldSpec(name="diagnosis", kind="string"),
        FieldSpec(name="smoker", kind="enum", enum_values=("Y", "N")),
        FieldSpec(name="comorbidities", kind="list", item_kind="string"),
    )))


def _mock_client(handler) -> Anthropic:
    transport = httpx.MockTransport(handler)
    return Anthropic(api_key="k", http_client=httpx.Client(transport=transport))


def test_egress_destination():
    p = AnthropicProvider(model="m", api_key="k")
    assert p.egress_destination() == "api.anthropic.com"


def test_uses_schema_natively_is_true():
    p = AnthropicProvider(model="m", api_key="k")
    assert p.uses_schema_natively is True
    assert p.name == "anthropic"
    assert p.default_concurrency == 8


def test_basic_call_with_tool_use(fixture_response):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=fixture_response)

    p = AnthropicProvider(model="claude-haiku-4-5", api_key="k")
    p._client = _mock_client(handler)

    js = _json_schema()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    resp = p.generate(messages, GenerationConfig(), json_schema=js)
    parsed = json.loads(resp.text)
    assert parsed == {"diagnosis": "asthma", "smoker": "N", "comorbidities": []}
    assert resp.usage == {"input_tokens": 80, "output_tokens": 25}
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["payload"]["tool_choice"] == {"type": "tool", "name": "extract"}
    assert captured["payload"]["tools"][0]["input_schema"] == js
    assert captured["payload"]["system"] == "SYS"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]


def test_generate_without_json_schema_raises():
    """json_schema is required for forced tool-use (Extractor passes it automatically)."""
    p = AnthropicProvider(model="m", api_key="k")
    with pytest.raises(RuntimeError, match="json_schema"):
        p.generate([{"role": "user", "content": "hi"}], GenerationConfig())
