"""Tests for the subagent spawn mechanism."""

import json
from unittest.mock import AsyncMock

import pytest

from agentino import Agent, Message, Usage, make_spawn_tool
from agentino.core.llm import LLMResponse


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=50, completion_tokens=10),
    )


def _make_agent(name: str = "test", response: str = "done") -> Agent:
    """Create a minimal agent with mocked LLM."""
    agent = Agent(model="test", tools=[], name=name)
    agent._llm.chat = AsyncMock(return_value=_make_text_response(response))
    return agent


class TestMakeSpawnTool:
    def test_creates_tool_with_correct_name(self):
        agents = {"coder": _make_agent("coder")}
        t = make_spawn_tool(agents)
        assert t.name == "spawn_agent"

    def test_description_lists_agents(self):
        agents = {"coder": _make_agent(), "reviewer": _make_agent()}
        t = make_spawn_tool(agents, allowed=["coder", "reviewer"])
        assert "coder" in t.description
        assert "reviewer" in t.description

    def test_allowed_filters_agents(self):
        agents = {"coder": _make_agent(), "reviewer": _make_agent()}
        t = make_spawn_tool(agents, allowed=["coder"])
        assert "coder" in t.description
        assert t.parameters["properties"]["agent_id"]["enum"] == ["coder"]

    def test_raises_on_invalid_allowed(self):
        agents = {"coder": _make_agent()}
        with pytest.raises(ValueError, match="not found"):
            make_spawn_tool(agents, allowed=["nonexistent"])

    def test_parameters_schema(self):
        agents = {"coder": _make_agent()}
        t = make_spawn_tool(agents)
        assert "agent_id" in t.parameters["properties"]
        assert "task" in t.parameters["properties"]
        assert t.parameters["required"] == ["agent_id", "task"]


class TestSpawnExecution:
    @pytest.mark.asyncio
    async def test_spawn_runs_subagent_and_returns_structured(self):
        sub = _make_agent("coder", response="Found the bug in line 42")
        agents = {"coder": sub}
        t = make_spawn_tool(agents)
        result = await t.execute({"agent_id": "coder", "task": "Find the bug"})
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["agent_id"] == "coder"
        assert "wf-" in parsed["workflow_id"]
        assert parsed["result_text"] == "Found the bug in line 42"

    @pytest.mark.asyncio
    async def test_spawn_wraps_json_output(self):
        """When subagent returns JSON, it's parsed and nested under 'result'."""
        findings = json.dumps({"root_cause": "tiling change", "confidence": "HIGH"})
        sub = _make_agent("coder", response=findings)
        agents = {"coder": sub}
        t = make_spawn_tool(agents)
        result = await t.execute({"agent_id": "coder", "task": "Investigate"})
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["result"]["root_cause"] == "tiling change"
        assert parsed["result"]["confidence"] == "HIGH"

    @pytest.mark.asyncio
    async def test_spawn_rejects_unlisted_agent(self):
        agents = {"coder": _make_agent(), "reviewer": _make_agent()}
        t = make_spawn_tool(agents, allowed=["coder"])
        result = await t.execute({"agent_id": "reviewer", "task": "Review code"})
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_spawn_handles_agent_error(self):
        sub = _make_agent("coder")
        sub._llm.chat = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        agents = {"coder": sub}
        t = make_spawn_tool(agents)
        result = await t.execute({"agent_id": "coder", "task": "Investigate"})
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "RuntimeError" in parsed["error"]

    @pytest.mark.asyncio
    async def test_spawn_handles_timeout(self):
        async def _slow_chat(**kwargs):
            import asyncio

            await asyncio.sleep(10)
            return _make_text_response("too late")

        sub = _make_agent("coder")
        sub._llm.chat = AsyncMock(side_effect=_slow_chat)
        agents = {"coder": sub}
        t = make_spawn_tool(agents, timeout=0.1)
        result = await t.execute({"agent_id": "coder", "task": "Slow task"})
        parsed = json.loads(result)
        assert parsed["status"] == "timeout"
        assert "timed out" in parsed["error"]

    @pytest.mark.asyncio
    async def test_spawn_empty_response(self):
        sub = _make_agent("coder", response="")
        agents = {"coder": sub}
        t = make_spawn_tool(agents)
        result = await t.execute({"agent_id": "coder", "task": "Do something"})
        parsed = json.loads(result)
        assert parsed["status"] == "empty"
        assert parsed["result"] is None

    @pytest.mark.asyncio
    async def test_spawn_includes_workflow_id_in_context(self):
        """Verify the subagent receives workflow_id and depth in its context."""
        sub = _make_agent("coder", response="done")
        agents = {"coder": sub}
        t = make_spawn_tool(agents)
        await t.execute({"agent_id": "coder", "task": "Find the bug"})
        # The copy's LLM won't be mocked — check the original was copied
        # Since _copy_agent copies the agent, the mock is shared via shallow copy
        sub._llm.chat.assert_called_once()
        call_args = sub._llm.chat.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = [m for m in messages if m.role == "user"]
        assert len(user_msg) == 1
        assert "workflow_id" in user_msg[0].content
        assert "wf-" in user_msg[0].content
        assert "spawn_depth" in user_msg[0].content
        assert "Find the bug" in user_msg[0].content

    @pytest.mark.asyncio
    async def test_spawn_unique_workflow_ids(self):
        """Each spawn call gets a unique workflow ID."""
        sub = _make_agent("coder", response="done")
        agents = {"coder": sub}
        t = make_spawn_tool(agents)
        r1 = await t.execute({"agent_id": "coder", "task": "Task 1"})
        # Reset mock for second call
        sub._llm.chat = AsyncMock(return_value=_make_text_response("done"))
        r2 = await t.execute({"agent_id": "coder", "task": "Task 2"})
        wf1 = json.loads(r1)["workflow_id"]
        wf2 = json.loads(r2)["workflow_id"]
        assert wf1 != wf2


class TestRecursionGuard:
    @pytest.mark.asyncio
    async def test_default_max_depth_blocks_nested_spawn(self):
        """Default max_depth=1 means subagents cannot spawn further."""
        sub = _make_agent("coder", response="done")
        agents = {"coder": sub}
        # Default max_depth=1: depth 0 → ok, depth 1 → blocked
        t = make_spawn_tool(agents)
        # _spawn is called with _depth=0 by default (via tool.execute)
        # which is fine. To test depth blocking, call _spawn directly.
        result = await t.fn("coder", "task", _depth=1)
        parsed = json.loads(result)
        assert parsed["status"] == "forbidden"
        assert "Max spawn depth" in parsed["error"]

    @pytest.mark.asyncio
    async def test_max_depth_zero_blocks_all_spawns(self):
        """max_depth=0 prevents any spawning."""
        sub = _make_agent("coder", response="done")
        agents = {"coder": sub}
        t = make_spawn_tool(agents, max_depth=0)
        result = await t.fn("coder", "task", _depth=0)
        parsed = json.loads(result)
        assert parsed["status"] == "forbidden"

    @pytest.mark.asyncio
    async def test_max_depth_2_allows_one_level(self):
        """max_depth=2 allows depth 0 and 1, blocks depth 2."""
        sub = _make_agent("coder", response="done")
        agents = {"coder": sub}
        t = make_spawn_tool(agents, max_depth=2)
        # depth=0 should work
        r0 = await t.execute({"agent_id": "coder", "task": "task"})
        assert json.loads(r0)["status"] == "ok"
        # depth=1 should work
        r1 = await t.fn("coder", "task", _depth=1)
        assert json.loads(r1)["status"] == "ok"
        # depth=2 should be blocked
        r2 = await t.fn("coder", "task", _depth=2)
        assert json.loads(r2)["status"] == "forbidden"


class TestAgentIsolation:
    @pytest.mark.asyncio
    async def test_spawn_does_not_mutate_original_usage(self):
        """Spawning should not affect the original agent's usage counters."""
        sub = _make_agent("coder", response="done")
        original_total = sub.total_usage
        agents = {"coder": sub}
        t = make_spawn_tool(agents)
        await t.execute({"agent_id": "coder", "task": "Do work"})
        # Original agent's total_usage should be unchanged
        assert sub.total_usage == original_total

    @pytest.mark.asyncio
    async def test_concurrent_spawns_dont_interfere(self):
        """Multiple concurrent spawns of the same agent should be independent."""
        import asyncio

        call_count = 0

        async def _counting_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return _make_text_response(f"result-{call_count}")

        sub = _make_agent("coder")
        sub._llm.chat = AsyncMock(side_effect=_counting_chat)
        agents = {"coder": sub}
        t = make_spawn_tool(agents)

        # Run two spawns concurrently
        r1, r2 = await asyncio.gather(
            t.execute({"agent_id": "coder", "task": "Task A"}),
            t.execute({"agent_id": "coder", "task": "Task B"}),
        )
        p1 = json.loads(r1)
        p2 = json.loads(r2)
        assert p1["status"] == "ok"
        assert p2["status"] == "ok"
        # Both completed independently
        assert p1["workflow_id"] != p2["workflow_id"]


class TestAddTool:
    def test_add_tool_updates_map(self):
        """Agent.add_tool() should update both tools list and _tool_map."""
        agent = _make_agent("test")
        agents = {"other": _make_agent("other")}
        spawn_tool = make_spawn_tool(agents)
        agent.add_tool(spawn_tool)
        assert spawn_tool in agent.tools
        assert agent._tool_map["spawn_agent"] is spawn_tool
