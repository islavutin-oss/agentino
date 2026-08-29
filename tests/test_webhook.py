"""Tests for the webhook transport."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentino import Agent
from agentino.core.llm import LLMResponse
from agentino.core.message import Message, Usage
from agentino.transport.webhook import WebhookHandler


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=50, completion_tokens=10),
    )


class TestWebhookHandler:
    @pytest.mark.asyncio
    async def test_basic_response(self):
        agent = Agent(model="test", tools=[])
        handler = WebhookHandler(agent=agent)

        agent._llm.chat = AsyncMock(return_value=_make_text_response("Hello!"))
        result = await handler.handle(message="Hi")

        assert result["reply"] == "Hello!"
        assert result["usage"]["prompt_tokens"] == 50

    @pytest.mark.asyncio
    async def test_with_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(model="test", tools=[])
            handler = WebhookHandler(agent=agent, session_dir=tmp)

            agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi!"))
            result = await handler.handle(message="Hello", session_id="user-123")

            assert result["reply"] == "Hi!"
            assert result["session_id"] == "user-123"
            assert (Path(tmp) / "user-123.jsonl").exists()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        agent = Agent(model="test", tools=[])
        handler = WebhookHandler(agent=agent)

        agent._llm.chat = AsyncMock(side_effect=RuntimeError("API down"))
        result = await handler.handle(message="Hi")

        assert "error" in result
        assert "API down" in result["error"]

    @pytest.mark.asyncio
    async def test_no_session_dir(self):
        agent = Agent(model="test", tools=[])
        handler = WebhookHandler(agent=agent)

        agent._llm.chat = AsyncMock(return_value=_make_text_response("Ok"))
        result = await handler.handle(message="Hi", session_id="ignored")

        assert result["reply"] == "Ok"
