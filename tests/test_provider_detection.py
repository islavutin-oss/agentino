"""Which wire protocol a base URL implies.

The detector used to return `openai-codex` for everything that was not
Anthropic, so a plain OpenAI-compatible endpoint received Codex-shaped
`/codex/responses` SSE. That worked only because every call went through a
Router that spoke Codex; pointed at vLLM, Ollama or api.openai.com — the first
thing a new reader tries — it failed.
"""

from __future__ import annotations

import pytest

from agentino.core.llm import _detect_provider


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.1:11434/v1",
        "https://openrouter.ai/api/v1",
        "",
    ],
)
def test_an_ordinary_openai_compatible_endpoint_is_not_codex(url):
    assert _detect_provider(url) == "openai"


@pytest.mark.parametrize(
    "url",
    [
        "https://chatgpt.com/backend-api",
        "https://router.example.com/codex/responses",
        "https://ROUTER.example.com/CODEX",
    ],
)
def test_codex_is_detected_where_it_actually_lives(url):
    assert _detect_provider(url) == "openai-codex"


@pytest.mark.parametrize(
    "url",
    ["https://api.anthropic.com", "https://router.example.com/anthropic/v1"],
)
def test_anthropic_still_wins(url):
    assert _detect_provider(url) == "anthropic"


def test_detection_is_case_insensitive():
    assert _detect_provider("HTTPS://API.ANTHROPIC.COM") == "anthropic"


def test_a_plain_endpoint_gets_a_plain_default_model():
    """A Codex model name against a non-Codex endpoint is a 404 at request
    time, which reads as 'the library is broken' rather than 'wrong model'."""
    from agentino.core.llm import CODEX_DEFAULT_MODEL, OPENAI_DEFAULT_MODEL

    assert OPENAI_DEFAULT_MODEL != CODEX_DEFAULT_MODEL
    assert "codex" not in OPENAI_DEFAULT_MODEL
