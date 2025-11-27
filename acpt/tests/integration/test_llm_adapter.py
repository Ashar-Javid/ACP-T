"""Integration tests for the Cerebras LLM adapter stub."""

from __future__ import annotations

from acpt.utils import CerebrasConfig, CerebrasLLMAdapter


def test_generate_returns_mock_response() -> None:
    adapter = CerebrasLLMAdapter(CerebrasConfig(mock_mode=True))
    response = adapter.generate("Hello world", context=["greeting"], max_tokens=64)

    assert response.startswith("[mock]model=")
    assert "reply=" in response


def test_stream_generate_iterates_tokens() -> None:
    adapter = CerebrasLLMAdapter(CerebrasConfig(mock_mode=True))
    stream = list(adapter.stream_generate("stream me", context=[]))

    assert stream
    joined = " ".join(stream)
    assert joined.startswith("[mock]model=")
