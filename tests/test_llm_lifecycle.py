"""Tests for LLMClient resource lifecycle (creation, close, context manager).

Validates that:
- LLMClient.close() actually closes the underlying httpx.AsyncClient
- Agent exposes close() that propagates to its LLMClient
- Using Agent as async context manager closes resources
- Double-close is safe (idempotent)
"""

from unittest.mock import AsyncMock

import pytest

from agentino.core.agent import Agent
from agentino.core.llm import LLMClient

# ---------------------------------------------------------------------------
# 1. LLMClient.close() closes underlying httpx client
# ---------------------------------------------------------------------------


class TestLLMClientClose:
    @pytest.mark.asyncio
    async def test_close_calls_aclose(self):
        """close() should call _client.aclose()."""
        client = LLMClient(api_key="sk-test", base_url="https://api.openai.com/v1")
        client._client.aclose = AsyncMock()

        await client.close()
        client._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_double_close_is_safe(self):
        """Calling close() twice should not raise."""
        client = LLMClient(api_key="sk-test", base_url="https://api.openai.com/v1")
        # First close — real
        await client.close()
        # Second close — should not raise (httpx handles this gracefully)
        await client.close()


# ---------------------------------------------------------------------------
# 2. Agent propagates close to LLMClient
# ---------------------------------------------------------------------------


class TestAgentClose:
    @pytest.mark.asyncio
    async def test_agent_close_propagates(self):
        """Agent.close() should close the underlying LLM client."""
        agent = Agent(model="gpt-4o", api_key="sk-test")
        agent._llm.close = AsyncMock()

        await agent.close()
        agent._llm.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_as_context_manager(self):
        """Agent should work as async context manager, closing on exit."""
        agent = Agent(model="gpt-4o", api_key="sk-test")
        agent._llm.close = AsyncMock()

        async with agent:
            pass  # use agent

        agent._llm.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_context_manager_closes_on_exception(self):
        """Agent context manager should close even if body raises."""
        agent = Agent(model="gpt-4o", api_key="sk-test")
        agent._llm.close = AsyncMock()

        with pytest.raises(ValueError):
            async with agent:
                raise ValueError("boom")

        agent._llm.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. LLMClient provider auto-detection sanity
# ---------------------------------------------------------------------------


class TestLLMClientProviderDetection:
    def test_anthropic_key_sets_provider(self):
        client = LLMClient(api_key="sk-ant-api03-test", base_url="https://api.openai.com/v1")
        assert client.provider == "anthropic"

    def test_an_openai_key_against_openai_is_not_codex(self):
        """Codex is a specific wire protocol, not the default for everything
        that is not Anthropic."""
        client = LLMClient(api_key="sk-proj-test", base_url="https://api.openai.com/v1")
        assert client.provider == "openai"

    def test_explicit_provider_overrides(self):
        client = LLMClient(
            api_key="sk-test", base_url="https://custom.proxy.com/v1", provider="anthropic"
        )
        assert client.provider == "anthropic"
