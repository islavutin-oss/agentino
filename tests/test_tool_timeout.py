"""Tests for Tool.timeout enforcement via asyncio.wait_for.

Validates that:
- Tools with timeout set raise TimeoutError (returned as error string) when they exceed it
- Tools without timeout run indefinitely (no wrapping)
- Sync tools with timeout are also enforced
- FinalResult passthrough still works with timeout set
"""

import asyncio

import pytest

from agentino.core.tool import FinalResult, Tool, tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _slow_async_fn(delay: float = 5.0) -> str:
    """Async tool that sleeps longer than any reasonable timeout."""
    await asyncio.sleep(delay)
    return "finished"


async def _fast_async_fn(delay: float = 0.0) -> str:
    await asyncio.sleep(delay)
    return "fast_result"


def _slow_sync_fn() -> str:
    """Sync tool that blocks (time.sleep) — timeout should still apply."""
    import time

    time.sleep(5)
    return "finished"


def _fast_sync_fn() -> str:
    return "sync_ok"


async def _final_result_fn() -> FinalResult:
    return FinalResult("done!")


# ---------------------------------------------------------------------------
# 1. Async tool with timeout — should return error on timeout
# ---------------------------------------------------------------------------


class TestAsyncToolTimeout:
    @pytest.mark.asyncio
    async def test_timeout_triggers_on_slow_tool(self):
        t = Tool(
            name="slow",
            description="slow tool",
            parameters={"type": "object", "properties": {}},
            fn=_slow_async_fn,
            timeout=0.1,  # 100ms — tool sleeps 5s
        )
        result = await t.execute({"delay": 5.0})
        assert isinstance(result, str)
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_fast_tool_with_timeout_succeeds(self):
        t = Tool(
            name="fast",
            description="fast tool",
            parameters={"type": "object", "properties": {}},
            fn=_fast_async_fn,
            timeout=5.0,  # generous timeout
        )
        result = await t.execute({"delay": 0.0})
        assert result == "fast_result"

    @pytest.mark.asyncio
    async def test_no_timeout_lets_tool_run(self):
        """Tool without timeout should complete (testing with a fast tool)."""
        t = Tool(
            name="no_timeout",
            description="no timeout tool",
            parameters={"type": "object", "properties": {}},
            fn=_fast_async_fn,
            timeout=None,
        )
        result = await t.execute({"delay": 0.0})
        assert result == "fast_result"


# ---------------------------------------------------------------------------
# 2. Sync tool with timeout
# ---------------------------------------------------------------------------


class TestSyncToolTimeout:
    @pytest.mark.asyncio
    async def test_sync_timeout_triggers(self):
        t = Tool(
            name="slow_sync",
            description="slow sync tool",
            parameters={"type": "object", "properties": {}},
            fn=_slow_sync_fn,
            timeout=0.1,
        )
        result = await t.execute({})
        assert isinstance(result, str)
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_sync_fast_with_timeout_succeeds(self):
        t = Tool(
            name="fast_sync",
            description="fast sync tool",
            parameters={"type": "object", "properties": {}},
            fn=_fast_sync_fn,
            timeout=5.0,
        )
        result = await t.execute({})
        assert result == "sync_ok"


# ---------------------------------------------------------------------------
# 3. FinalResult passthrough with timeout
# ---------------------------------------------------------------------------


class TestFinalResultWithTimeout:
    @pytest.mark.asyncio
    async def test_final_result_passthrough_with_timeout(self):
        t = Tool(
            name="final",
            description="final result tool",
            parameters={"type": "object", "properties": {}},
            fn=_final_result_fn,
            timeout=5.0,
        )
        result = await t.execute({})
        assert isinstance(result, FinalResult)
        assert result.text == "done!"


# ---------------------------------------------------------------------------
# 4. @tool decorator preserves timeout
# ---------------------------------------------------------------------------


class TestToolDecoratorTimeout:
    def test_decorator_passes_timeout(self):
        @tool(timeout=10.0)
        async def my_tool(query: str) -> str:
            """A tool."""
            return query

        assert isinstance(my_tool, Tool)
        assert my_tool.timeout == 10.0

    def test_decorator_default_no_timeout(self):
        @tool
        def simple(x: str) -> str:
            """Simple."""
            return x

        assert isinstance(simple, Tool)
        assert simple.timeout is None
