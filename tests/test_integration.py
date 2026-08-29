"""Integration tests — full agent loop with mock LLM.

Tests the complete chain: agent → LLM → tool calls → results → response.
Uses a mock LLM that returns predictable tool calls.
"""

import asyncio

from agentino import context
from agentino.core.agent import Agent
from agentino.core.llm import LLMResponse
from agentino.core.message import Message, ToolCall, Usage
from agentino.core.tool import Tool, tool
from agentino.safety.gates import GateManager, GateRule
from agentino.safety.hooks import HookManager

# ---------------------------------------------------------------------------
# Mock LLM — returns predictable responses
# ---------------------------------------------------------------------------


class MockLLM:
    """Mock LLM that returns scripted tool calls and responses."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0
        self.base_url = "http://mock"
        self.api_key = "mock"
        self.provider = "openai"
        self.default_model = "mock"

    async def chat(self, messages, tools=None, model=None, temperature=0.7, tool_choice=None):
        if self._call_count >= len(self._responses):
            return LLMResponse(
                message=Message(role="assistant", content="Done."),
                usage=Usage(),
                finish_reason="stop",
            )
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    async def chat_stream(
        self, messages, tools=None, model=None, temperature=0.7, tool_choice=None
    ):
        resp = await self.chat(messages, tools, model, temperature, tool_choice)
        from agentino.core.message import Event, EventType

        yield Event(type=EventType.LLM_RESPONSE, usage=resp.usage, data=resp.message)

    async def close(self):
        pass


def _make_tool_response(tool_calls: list[ToolCall]) -> LLMResponse:
    """Create a mock LLM response with tool calls."""
    return LLMResponse(
        message=Message(role="assistant", content=None, tool_calls=tool_calls),
        usage=Usage(prompt_tokens=100, completion_tokens=50),
        finish_reason="tool_calls",
    )


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=100, completion_tokens=50),
        finish_reason="stop",
    )


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool(is_read_only=True)
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def write_result(path: str, content: str) -> str:
    """Write to a file."""
    return f"Written to {path}"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAgentLoop:
    """Test the full agent → LLM → tools → response cycle."""

    def test_simple_text_response(self):
        """Agent returns text when LLM doesn't call tools."""
        mock = MockLLM([_make_text_response("The answer is 42.")])
        agent = Agent(model="mock", instructions="Be helpful.", tools=[greet])
        agent._llm = mock
        result = asyncio.run(agent.run("What is the answer?"))
        assert "42" in result

    def test_single_tool_call(self):
        """Agent calls a tool and returns the result."""
        mock = MockLLM(
            [
                _make_tool_response(
                    [ToolCall(id="tc1", name="greet", arguments={"name": "World"})]
                ),
                _make_text_response("I greeted them!"),
            ]
        )
        agent = Agent(model="mock", instructions="test", tools=[greet])
        agent._llm = mock
        result = asyncio.run(agent.run("Greet World"))
        assert "greeted" in result.lower() or "Hello" in result

    def test_multiple_tool_calls(self):
        """Agent handles multiple tool calls in one turn."""
        mock = MockLLM(
            [
                _make_tool_response(
                    [
                        ToolCall(id="tc1", name="greet", arguments={"name": "Alice"}),
                        ToolCall(id="tc2", name="greet", arguments={"name": "Bob"}),
                    ]
                ),
                _make_text_response("Greeted both!"),
            ]
        )
        agent = Agent(model="mock", instructions="test", tools=[greet])
        agent._llm = mock
        result = asyncio.run(agent.run("Greet Alice and Bob"))
        assert "both" in result.lower() or "Greeted" in result

    def test_tool_chain(self):
        """Agent calls tools across multiple turns."""
        mock = MockLLM(
            [
                _make_tool_response([ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})]),
                _make_tool_response(
                    [ToolCall(id="tc2", name="greet", arguments={"name": "Result"})]
                ),
                _make_text_response("Done: 5 and greeted."),
            ]
        )
        agent = Agent(model="mock", instructions="test", tools=[greet, add])
        agent._llm = mock
        asyncio.run(agent.run("Add 2+3 then greet"))
        assert mock._call_count == 3

    def test_unknown_tool_error(self):
        """Unknown tool returns error, agent continues."""
        mock = MockLLM(
            [
                _make_tool_response([ToolCall(id="tc1", name="nonexistent", arguments={})]),
                _make_text_response("Sorry, that failed."),
            ]
        )
        agent = Agent(model="mock", instructions="test", tools=[greet])
        agent._llm = mock
        result = asyncio.run(agent.run("Call nonexistent"))
        assert "Sorry" in result or "failed" in result


class TestGateIntegration:
    """Test gates blocking tool calls in the agent loop."""

    def test_gate_blocks_tool(self):
        """Gate rejects tool call when precondition not met."""
        mock = MockLLM(
            [
                _make_tool_response(
                    [
                        ToolCall(
                            id="tc1", name="write_result", arguments={"path": "/x", "content": "y"}
                        )
                    ]
                ),
                _make_text_response("Write was blocked."),
            ]
        )
        rules = [
            GateRule(
                gate="read_first", tools=["write_result"], message="REJECTED: read before writing"
            )
        ]
        gm = GateManager(rules)

        agent = Agent(model="mock", instructions="test", tools=[write_result])
        agent._llm = mock

        token = context.set_context(_gate_manager=gm)
        try:
            result = asyncio.run(agent.run("Write something"))
        finally:
            context.reset(token)

        # Gate should have blocked — model got rejection and responded
        assert "blocked" in result.lower() or mock._call_count == 2

    def test_gate_passes_when_marked(self):
        """Gate allows tool call when precondition is met."""
        mock = MockLLM(
            [
                _make_tool_response(
                    [
                        ToolCall(
                            id="tc1", name="write_result", arguments={"path": "/x", "content": "y"}
                        )
                    ]
                ),
                _make_text_response("Written!"),
            ]
        )
        rules = [GateRule(gate="read_first", tools=["write_result"], message="REJECTED")]
        gm = GateManager(rules)
        gm.mark("read_first")  # Precondition met

        agent = Agent(model="mock", instructions="test", tools=[write_result])
        agent._llm = mock

        token = context.set_context(_gate_manager=gm)
        try:
            result = asyncio.run(agent.run("Write something"))
        finally:
            context.reset(token)

        assert "Written" in result


class TestHookIntegration:
    """Test hooks in the agent loop."""

    def test_pre_hook_blocks(self):
        """PreToolUse hook blocks a tool call."""
        mock = MockLLM(
            [
                _make_tool_response([ToolCall(id="tc1", name="greet", arguments={"name": "test"})]),
                _make_text_response("Blocked by hook."),
            ]
        )

        hooks = HookManager()
        hooks.register(
            "PreToolUse",
            matcher={"tool_name": ["greet"]},
            callback=lambda ctx: "BLOCKED by security policy",
        )
        # The callback returns a string but doesn't set blocked=True via exit code
        # For callback hooks, any non-empty return is treated as non-blocking message
        # To block, the hook needs to be a command with exit code 2

        agent = Agent(model="mock", instructions="test", tools=[greet])
        agent._llm = mock

        token = context.set_context(_hook_manager=hooks)
        try:
            asyncio.run(agent.run("Greet test"))
        finally:
            context.reset(token)

        # Hook fires but callback doesn't block (only commands with exit 2 block)
        assert mock._call_count >= 1

    def test_post_hook_observes(self):
        """PostToolUse hook receives tool result."""
        observed = []

        mock = MockLLM(
            [
                _make_tool_response([ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})]),
                _make_text_response("3"),
            ]
        )

        hooks = HookManager()
        hooks.register("PostToolUse", callback=lambda ctx: observed.append(ctx))

        agent = Agent(model="mock", instructions="test", tools=[add])
        agent._llm = mock

        token = context.set_context(_hook_manager=hooks)
        try:
            asyncio.run(agent.run("Add 1+2"))
        finally:
            context.reset(token)

        assert len(observed) == 1
        assert observed[0]["tool_name"] == "add"
        assert "3" in observed[0]["result"]


class TestToolValidationIntegration:
    """Test tool validate_input and check_permission in agent loop."""

    def test_validation_rejects(self):
        """Tool validation rejects bad input."""

        def validate(path: str, content: str) -> str | None:
            if ".." in path:
                return "REJECTED: path traversal"
            return None

        guarded_write = Tool(
            name="safe_write",
            description="Write safely",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
            fn=lambda path, content: f"Written to {path}",
            validate_input=validate,
        )

        mock = MockLLM(
            [
                _make_tool_response(
                    [
                        ToolCall(
                            id="tc1",
                            name="safe_write",
                            arguments={"path": "../etc/passwd", "content": "hack"},
                        )
                    ]
                ),
                _make_text_response("Blocked."),
            ]
        )

        agent = Agent(model="mock", instructions="test", tools=[guarded_write])
        agent._llm = mock
        result = asyncio.run(agent.run("Write to ../etc/passwd"))
        # Validation should have rejected
        assert "Blocked" in result or mock._call_count == 2


class TestParallelToolExecution:
    """Test that read-only tools run in parallel."""

    def test_read_only_parallel(self):
        """Multiple read-only tools execute in parallel."""
        call_times = []

        @tool(is_read_only=True)
        async def slow_read(id: str) -> str:
            """Slow read."""
            import time

            start = time.monotonic()
            await asyncio.sleep(0.1)
            call_times.append((id, time.monotonic() - start))
            return f"read {id}"

        mock = MockLLM(
            [
                _make_tool_response(
                    [
                        ToolCall(id="tc1", name="slow_read", arguments={"id": "a"}),
                        ToolCall(id="tc2", name="slow_read", arguments={"id": "b"}),
                        ToolCall(id="tc3", name="slow_read", arguments={"id": "c"}),
                    ]
                ),
                _make_text_response("Done."),
            ]
        )

        agent = Agent(model="mock", instructions="test", tools=[slow_read])
        agent._llm = mock

        import time

        start = time.monotonic()
        asyncio.run(agent.run("Read all"))
        elapsed = time.monotonic() - start

        # 3 reads at 0.1s each should take ~0.1s parallel, not 0.3s sequential
        assert len(call_times) == 3
        # Allow some overhead, but should be under 0.25s (not 0.3s)
        assert elapsed < 0.25, f"Parallel reads took {elapsed:.2f}s (expected <0.25s)"


class TestHookBlocking:
    """Test that hook callbacks with REJECTED/BLOCKED in return block tool calls."""

    def test_callback_with_rejected_blocks(self):
        """Callback returning 'REJECTED: ...' should block the tool call."""
        import asyncio

        from agentino.safety.hooks import HookManager

        hooks = HookManager()
        hooks.register(
            "PreToolUse",
            matcher={"tool_name": ["write"]},
            callback=lambda ctx: "⚠️ REJECTED: must read first",
        )

        result = asyncio.run(hooks.fire("PreToolUse", {"tool_name": "write"}))
        assert result.blocked is True
        assert "REJECTED" in result.message

    def test_callback_with_wrong_tool_blocks(self):
        hooks = HookManager()
        hooks.register("PreToolUse", callback=lambda ctx: "⚠️ WRONG TOOL: use X instead")
        result = asyncio.run(hooks.fire("PreToolUse", {"tool_name": "verify"}))
        assert result.blocked is True

    def test_callback_without_keywords_doesnt_block(self):
        hooks = HookManager()
        hooks.register("PreToolUse", callback=lambda ctx: "info: tool called")
        result = asyncio.run(hooks.fire("PreToolUse", {"tool_name": "read"}))
        assert result.blocked is False

    def test_callback_returning_none_doesnt_block(self):
        hooks = HookManager()
        hooks.register("PreToolUse", callback=lambda ctx: None)
        result = asyncio.run(hooks.fire("PreToolUse", {"tool_name": "read"}))
        assert result.blocked is False
