"""Pytest fixtures for the ehrextract test suite."""

from pathlib import Path

import pytest

from ehrextract.providers import ProviderResponse


class MockProvider:
    """Duck-typed provider returning canned responses. No network, no model."""

    name = "mock"
    default_concurrency = 4
    uses_schema_natively = False

    def __init__(self, responses: list[str] | None = None, usage: dict | None = None):
        self._responses = list(responses or [])
        self._usage = usage
        self.calls: list[tuple[list[dict], object, dict | None]] = []

    def generate(self, messages, config, json_schema=None):
        self.calls.append((messages, config, json_schema))
        text = self._responses.pop(0) if self._responses else "{}"
        return ProviderResponse(text=text, finish_reason="stop", usage=self._usage, raw=None)

    def egress_destination(self):
        return None


@pytest.fixture
def mock_provider_cls():
    """The MockProvider class (instantiate per test with canned responses)."""
    return MockProvider


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures."""
    return Path(__file__).parent / "fixtures"
