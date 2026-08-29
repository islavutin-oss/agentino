"""Tests for the Agent class — mocks the LLM client to test the async loop."""

from unittest.mock import AsyncMock

import pytest

from agentino import Agent, Event, Message, ToolCall, Usage, tool
from agentino.core.llm import LLMResponse


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=50, completion_tokens=10),
    )


def _make_tool_response(tool_name: str, tool_args: dict, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=tool_args)],
        ),
        usage=Usage(prompt_tokens=50, completion_tokens=10),
    )


class TestAgentBasic:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        agent = Agent(model="test", tools=[])
        agent._llm.chat = AsyncMock(return_value=_make_text_response("Hello!"))
        result = await agent.run("Hi")
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        agent = Agent(model="test", tools=[])
        agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi"))
        await agent.run("Hello")
        assert agent.last_usage.prompt_tokens == 50
        assert agent.last_usage.completion_tokens == 10
        assert agent.total_usage.total_tokens == 60

    @pytest.mark.asyncio
    async def test_cumulative_usage(self):
        agent = Agent(model="test", tools=[])
        agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi"))
        await agent.run("First")
        await agent.run("Second")
        assert agent.total_usage.prompt_tokens == 100
        assert agent.total_usage.completion_tokens == 20


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_tool_called_and_result_sent_back(self):
        @tool
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        agent = Agent(model="test", tools=[add])

        responses = [
            _make_tool_response("add", {"a": 2, "b": 3}),
            _make_text_response("The answer is 5"),
        ]
        agent._llm.chat = AsyncMock(side_effect=responses)
        result = await agent.run("What is 2+3?")

        assert result == "The answer is 5"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        agent = Agent(model="test", tools=[])

        responses = [
            _make_tool_response("nonexistent", {"q": "test"}),
            _make_text_response("Sorry, I couldn't do that."),
        ]
        agent._llm.chat = AsyncMock(side_effect=responses)
        result = await agent.run("Do something")

        assert result == "Sorry, I couldn't do that."

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        @tool
        def bad_tool(x: str) -> str:
            """Breaks."""
            raise RuntimeError("boom")

        agent = Agent(model="test", tools=[bad_tool])

        responses = [
            _make_tool_response("bad_tool", {"x": "test"}),
            _make_text_response("There was an error."),
        ]
        agent._llm.chat = AsyncMock(side_effect=responses)
        result = await agent.run("Do the thing")

        assert result == "There was an error."

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Async tools should work natively."""

        @tool
        async def async_add(a: int, b: int) -> str:
            """Add two numbers async."""
            return str(a + b)

        agent = Agent(model="test", tools=[async_add])

        responses = [
            _make_tool_response("async_add", {"a": 2, "b": 3}),
            _make_text_response("The answer is 5"),
        ]
        agent._llm.chat = AsyncMock(side_effect=responses)
        result = await agent.run("What is 2+3?")

        assert result == "The answer is 5"


class TestLoopGuard:
    @pytest.mark.asyncio
    async def test_duplicate_tool_calls_detected(self):
        @tool
        def search(q: str) -> str:
            """Search."""
            return "result"

        agent = Agent(model="test", tools=[search])

        tool_resp = _make_tool_response("search", {"q": "test"})
        responses = [tool_resp, tool_resp, _make_text_response("Done")]
        agent._llm.chat = AsyncMock(side_effect=responses)
        result = await agent.run("Search")

        assert result == "Done"

    @pytest.mark.asyncio
    async def test_max_turns_safety(self):
        @tool
        def loop_tool(x: str) -> str:
            """Loops."""
            return "again"

        agent = Agent(model="test", tools=[loop_tool], max_turns=3)

        responses = [
            _make_tool_response("loop_tool", {"x": f"iter-{i}"}, call_id=f"c{i}") for i in range(5)
        ]
        mock_chat = AsyncMock(side_effect=responses)
        agent._llm.chat = mock_chat
        await agent.run("Go")
        assert mock_chat.call_count == 3


class TestEventCallbacks:
    @pytest.mark.asyncio
    async def test_on_event_called(self):
        events: list[Event] = []
        agent = Agent(model="test", tools=[], on_event=events.append)

        agent._llm.chat = AsyncMock(return_value=_make_text_response("Hi"))
        await agent.run("Hello")

        types = [e.type for e in events]
        assert "llm_response" in types

    @pytest.mark.asyncio
    async def test_tool_events_emitted(self):
        @tool
        def my_tool(x: str) -> str:
            """Tool."""
            return "result"

        events: list[Event] = []
        agent = Agent(model="test", tools=[my_tool], on_event=events.append)

        responses = [
            _make_tool_response("my_tool", {"x": "test"}),
            _make_text_response("Done"),
        ]
        agent._llm.chat = AsyncMock(side_effect=responses)
        await agent.run("Do it")

        types = [e.type for e in events]
        assert "tool_start" in types
        assert "tool_result" in types


class TestSessionIntegration:
    @pytest.mark.asyncio
    async def test_session_persisted(self, tmp_path):
        from agentino import Session

        session = Session(tmp_path / "test.jsonl")
        agent = Agent(model="test", instructions="Be nice", tools=[])

        agent._llm.chat = AsyncMock(return_value=_make_text_response("Hello!"))
        await agent.run("Hi", session=session)

        loaded = session.load()
        assert len(loaded) == 2  # user + assistant
        assert loaded[0].role == "user"
        assert loaded[0].content == "Hi"
        assert loaded[1].role == "assistant"
        assert loaded[1].content == "Hello!"

    @pytest.mark.asyncio
    async def test_session_history_sent_to_llm(self, tmp_path):
        from agentino import Session

        session = Session(tmp_path / "test.jsonl")
        session.save(
            [
                Message(role="user", content="My name is Alex"),
                Message(role="assistant", content="Nice to meet you, Alex!"),
            ]
        )

        agent = Agent(model="test", tools=[])

        mock_chat = AsyncMock(return_value=_make_text_response("Alex!"))
        agent._llm.chat = mock_chat
        await agent.run("What's my name?", session=session)

        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args[0][0]
        contents = [m.content for m in messages]
        assert "My name is Alex" in contents
        assert "What's my name?" in contents

    @pytest.mark.asyncio
    async def test_tool_calls_in_session(self, tmp_path):
        from agentino import Session

        @tool
        def greet(name: str) -> str:
            """Greet."""
            return f"Hi {name}!"

        session = Session(tmp_path / "test.jsonl")
        agent = Agent(model="test", tools=[greet])

        responses = [
            _make_tool_response("greet", {"name": "Alex"}),
            _make_text_response("I greeted Alex!"),
        ]
        agent._llm.chat = AsyncMock(side_effect=responses)
        await agent.run("Greet Alex", session=session)

        loaded = session.load()
        # user + assistant(tool_call) + tool_result + assistant(text)
        assert len(loaded) == 4
        assert loaded[1].tool_calls is not None
        assert loaded[2].role == "tool"
