"""Tests for the runner — async framework execution engine."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentino import Agent
from agentino.config import Config
from agentino.core.llm import LLMResponse
from agentino.core.message import Message, Usage
from agentino.core.runner import Runner, create_runner
from agentino.core.tool import tool


@tool
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=50, completion_tokens=10),
    )


class TestRunner:
    @pytest.mark.asyncio
    async def test_send_to_default_agent(self):
        agent = Agent(model="test", tools=[])
        config = Config(agents={"bot": agent})
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl")
            agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi!"))
            reply = await runner.send("Hello")
        assert reply == "Hi!"

    @pytest.mark.asyncio
    async def test_send_to_named_agent(self):
        a1 = Agent(model="test", tools=[], name="bot1")
        a2 = Agent(model="test", tools=[], name="bot2")
        config = Config(agents={"bot1": a1, "bot2": a2})
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl")
            a2._llm.chat = AsyncMock(return_value=_make_text_response("From bot2"))
            reply = await runner.send("Hello", agent_name="bot2")
        assert reply == "From bot2"

    @pytest.mark.asyncio
    async def test_default_agent_from_config(self):
        a1 = Agent(model="test", tools=[], name="first")
        a2 = Agent(model="test", tools=[], name="preferred")
        config = Config(
            agents={"first": a1, "preferred": a2},
            raw={"default_agent": "preferred"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl")
            a2._llm.chat = AsyncMock(return_value=_make_text_response("I'm preferred"))
            reply = await runner.send("Hello")
        assert reply == "I'm preferred"

    @pytest.mark.asyncio
    async def test_session_auto_created(self):
        agent = Agent(model="test", tools=[])
        config = Config(agents={"bot": agent})
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl")
            agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi!"))
            await runner.send("Hello", session_id="user-123")
            assert (Path(tmp) / "bot--user-123.jsonl").exists()

    @pytest.mark.asyncio
    async def test_no_session_persists_nothing(self):
        """no_session=True → sessions are ephemeral; no history file is written."""
        agent = Agent(model="test", tools=[])
        config = Config(agents={"bot": agent})
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(
                config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl", no_session=True
            )
            agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi!"))
            await runner.send("Hello", session_id="user-123")
            assert not (Path(tmp) / "bot--user-123.jsonl").exists()
            assert runner.get_session("bot", "user-123").ephemeral is True

    @pytest.mark.asyncio
    async def test_usage_tracked_automatically(self):
        agent = Agent(model="test", tools=[])
        config = Config(agents={"bot": agent})
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl")
            agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi!"))
            await runner.send("Hello")
            assert runner.usage_tracker.total.total_tokens > 0

    def test_list_agents(self):
        a1 = Agent(model="gpt-4o", instructions="You are a bot.", tools=[greet])
        a2 = Agent(model="gpt-4o-mini", instructions="You are another bot.", tools=[])
        config = Config(agents={"bot1": a1, "bot2": a2})
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl")
            agents = runner.list_agents()
        assert len(agents) == 2
        assert agents[0]["name"] == "bot1"
        assert "greet" in agents[0]["tools"]

    @pytest.mark.asyncio
    async def test_one_shot_no_session(self):
        agent = Agent(model="test", tools=[])
        config = Config(agents={"bot": agent})
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner(config, session_dir=tmp, usage_file=Path(tmp) / "usage.jsonl")
            agent._llm.chat = AsyncMock(return_value=_make_text_response("Reply"))
            reply = await runner.one_shot("Hello")
        assert reply == "Reply"


class TestCreateRunner:
    def test_from_yaml(self):
        yaml_content = """\
agents:
  bot:
    model: gpt-4o
    instructions: "You are a bot."
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    runner = create_runner(
                        f.name, session_dir=tmp, usage_file=Path(tmp) / "u.jsonl"
                    )
                    assert "bot" in runner.config.agents
            finally:
                os.unlink(f.name)
