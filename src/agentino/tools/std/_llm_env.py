"""Resolve the OpenAI-compatible endpoint the standard tools call.

Configure with ``AGENTINO_BASE_URL`` / ``AGENTINO_API_KEY``, or with the
conventional ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` pair. Any endpoint
speaking the OpenAI chat-completions API works.
"""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4-codex"


def _first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def llm_base_url() -> str:
    """Base URL of the chat-completions endpoint, without a trailing slash."""
    url = _first("AGENTINO_BASE_URL", "OPENAI_BASE_URL")
    return (url or DEFAULT_BASE_URL).rstrip("/")


def llm_api_key() -> str | None:
    """API key for that endpoint, or None when the caller should degrade."""
    return _first("AGENTINO_API_KEY", "OPENAI_API_KEY")
