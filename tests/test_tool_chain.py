"""Tests for Tool 3-stage execution chain and is_read_only flag."""

import asyncio

from agentino.core.tool import Tool, tool


class TestToolValidation:
    """Test validate_input stage."""

    def test_validation_passes(self):
        def validate(x: str) -> str | None:
            return None  # no error

        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda x: f"ok: {x}",
            validate_input=validate,
        )
        result = asyncio.run(t.execute({"x": "hello"}))
        assert result == "ok: hello"

    def test_validation_rejects(self):
        def validate(x: str) -> str | None:
            if not x.startswith("/"):
                return "Error: path must start with /"
            return None

        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda x: f"ok: {x}",
            validate_input=validate,
        )
        result = asyncio.run(t.execute({"x": "no-slash"}))
        assert "Error: path must start with /" in result

    def test_validation_error_returns_formatted(self):
        def validate(x: str) -> str | None:
            raise ValueError("bad input")

        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda x: "ok",
            validate_input=validate,
        )
        result = asyncio.run(t.execute({"x": "anything"}))
        assert "validation" in result.lower()


class TestToolPermission:
    """Test check_permission stage."""

    def test_permission_passes(self):
        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda: "ok",
            check_permission=lambda: None,
        )
        result = asyncio.run(t.execute({}))
        assert result == "ok"

    def test_permission_rejects(self):
        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda: "ok",
            check_permission=lambda: "REJECTED: not allowed",
        )
        result = asyncio.run(t.execute({}))
        assert "REJECTED" in result

    def test_validation_before_permission(self):
        """Validation runs before permission — fail fast."""
        call_order = []

        def validate(x: str) -> str | None:
            call_order.append("validate")
            return "validation failed"

        def permission(x: str) -> str | None:
            call_order.append("permission")
            return None

        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda x: "ok",
            validate_input=validate,
            check_permission=permission,
        )
        result = asyncio.run(t.execute({"x": "test"}))
        assert call_order == ["validate"]  # permission never called
        assert "validation failed" in result


class TestToolReadOnly:
    """Test is_read_only flag."""

    def test_default_not_read_only(self):
        @tool
        def my_tool(x: str) -> str:
            """test"""
            return x

        assert my_tool.is_read_only is False

    def test_marked_read_only(self):
        @tool(is_read_only=True)
        def my_reader(x: str) -> str:
            """test"""
            return x

        assert my_reader.is_read_only is True

    def test_decorator_with_all_kwargs(self):
        def val(x: str) -> str | None:
            return None

        def perm(x: str) -> str | None:
            return None

        @tool(is_read_only=True, validate_input=val, check_permission=perm, timeout=10)
        def full_tool(x: str) -> str:
            """test"""
            return x

        assert full_tool.is_read_only is True
        assert full_tool.validate_input is val
        assert full_tool.check_permission is perm
        assert full_tool.timeout == 10


class TestAsyncValidation:
    """Test async validate/permission functions."""

    def test_async_validation(self):
        async def async_validate(x: str) -> str | None:
            return "async rejection"

        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda x: "ok",
            validate_input=async_validate,
        )
        result = asyncio.run(t.execute({"x": "test"}))
        assert "async rejection" in result

    def test_async_permission(self):
        async def async_perm(x: str) -> str | None:
            return None

        t = Tool(
            name="test",
            description="test",
            parameters={},
            fn=lambda x: "ok",
            check_permission=async_perm,
        )
        result = asyncio.run(t.execute({"x": "test"}))
        assert result == "ok"


# ----------------------------------------------------------------------
# Borrow #4 — per-tool execution_mode override of batching policy
# ----------------------------------------------------------------------


class TestExecutionMode:
    def test_default_unset(self):
        @tool
        def t(x: str) -> str:
            """t"""
            return x

        assert t.execution_mode is None

    def test_marked_parallel(self):
        @tool(execution_mode="parallel")
        def slow_api(q: str) -> str:
            """rate-limited read"""
            return q

        assert slow_api.execution_mode == "parallel"

    def test_marked_sequential(self):
        @tool(execution_mode="sequential")
        def write_to_shared(q: str) -> str:
            """shared resource"""
            return q

        assert write_to_shared.execution_mode == "sequential"

    def test_agent_batches_parallel_overrides_into_parallel_batch(self):
        """Two tools both flagged execution_mode='parallel' — batched together
        even though neither is_read_only. Proves the override beats the default policy."""
        import asyncio

        from agentino.core.agent import Agent
        from agentino.core.message import ToolCall

        timeline: list[str] = []

        async def slow(q: str) -> str:
            timeline.append(f"start:{q}")
            await asyncio.sleep(0.05)
            timeline.append(f"end:{q}")
            return q

        from agentino.core.tool import Tool

        a_tool = Tool(
            name="api_a",
            description="rate-limited",
            parameters={"type": "object"},
            fn=slow,
            execution_mode="parallel",
        )
        b_tool = Tool(
            name="api_b",
            description="rate-limited",
            parameters={"type": "object"},
            fn=slow,
            execution_mode="parallel",
        )

        agent = Agent(model="x", api_key="x", tools=[a_tool, b_tool], base_url="http://localhost")
        calls = [
            ToolCall(id="1", name="api_a", arguments={"q": "A"}),
            ToolCall(id="2", name="api_b", arguments={"q": "B"}),
        ]
        msgs = []
        asyncio.run(agent._execute_tools(calls, msgs, prev_turn_calls=set()))

        # Parallel: both starts come before either end.
        assert timeline[:2] == ["start:A", "start:B"] or timeline[:2] == ["start:B", "start:A"]
        assert set(timeline[2:]) == {"end:A", "end:B"}

    def test_agent_batches_sequential_override_blocks_parallel(self):
        """A tool flagged execution_mode='sequential' is NOT batched in parallel
        even when is_read_only=True."""
        import asyncio

        from agentino.core.agent import Agent
        from agentino.core.message import ToolCall

        timeline: list[str] = []

        async def slow(q: str) -> str:
            timeline.append(f"start:{q}")
            await asyncio.sleep(0.05)
            timeline.append(f"end:{q}")
            return q

        from agentino.core.tool import Tool

        # is_read_only=True normally → parallel; but execution_mode="sequential" overrides.
        a_tool = Tool(
            name="seq_a",
            description="serial reader",
            parameters={"type": "object"},
            fn=slow,
            is_read_only=True,
            execution_mode="sequential",
        )
        b_tool = Tool(
            name="seq_b",
            description="serial reader",
            parameters={"type": "object"},
            fn=slow,
            is_read_only=True,
            execution_mode="sequential",
        )

        agent = Agent(model="x", api_key="x", tools=[a_tool, b_tool], base_url="http://localhost")
        calls = [
            ToolCall(id="1", name="seq_a", arguments={"q": "A"}),
            ToolCall(id="2", name="seq_b", arguments={"q": "B"}),
        ]
        msgs = []
        asyncio.run(agent._execute_tools(calls, msgs, prev_turn_calls=set()))

        # Sequential: each tool fully finishes before the next starts.
        assert timeline == ["start:A", "end:A", "start:B", "end:B"]
