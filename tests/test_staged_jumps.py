"""Tests for staged pipeline jump/retry logic — no LLM calls."""

import asyncio
from unittest.mock import MagicMock

from agentino.pipeline.staged import StageDef, StagedPipeline, StageResult


def _make_agent(tool_names=None):
    """Create a minimal mock agent."""
    agent = MagicMock()
    agent.model = "test"
    agent.temperature = 0
    agent._tool_result_cap = 4000
    agent._tool_instructions = ""
    agent._llm = MagicMock()
    agent._llm.base_url = "http://test"
    agent._llm.api_key = "test"
    agent._llm.provider = "openai-codex"
    agent.tools = []
    agent.instructions = "test"
    agent.on_event = None
    return agent


def test_sequential_stages():
    """Stages run in order when all succeed."""
    order = []

    stages = [
        StageDef(name="a", prompt=""),
        StageDef(name="b", prompt=""),
        StageDef(name="c", prompt=""),
    ]
    pipeline = StagedPipeline(stages=stages)

    # Override _run_stage to track order without LLM
    async def fake_run(agent, stage, tools, messages):
        order.append(stage.name)
        r = StageResult(name=stage.name, completed=True)
        r.output = f"done_{stage.name}"
        return r

    pipeline._run_stage = fake_run

    agent = _make_agent()
    results = asyncio.run(pipeline.run(agent, "test"))

    assert [r.name for r in results] == ["a", "b", "c"]
    assert order == ["a", "b", "c"]
    assert all(r.completed for r in results)


def test_on_fail_jumps_to_target():
    """Failed stage with on_fail=<name> jumps to that stage."""
    order = []

    stages = [
        StageDef(name="security", prompt="", on_fail="report"),
        StageDef(name="discover", prompt=""),
        StageDef(name="execute", prompt=""),
        StageDef(name="report", prompt=""),
    ]
    pipeline = StagedPipeline(stages=stages)

    async def fake_run(agent, stage, tools, messages):
        order.append(stage.name)
        r = StageResult(name=stage.name)
        if stage.name == "security":
            # Simulate rejection
            r.completed = True
            r.verdict_called = True
            r.verdict_args = {"result": "REJECT"}
        else:
            r.completed = True
        return r

    pipeline._run_stage = fake_run
    # Override failure check to detect REJECT
    pipeline._failure_check = lambda r, f: r.verdict_args.get("result") == "REJECT"

    agent = _make_agent()
    results = asyncio.run(pipeline.run(agent, "test"))

    # Should go: security (fail) → report (skip discover and execute)
    assert order == ["security", "report"]
    assert results[0].name == "security"
    assert results[0].failed
    assert results[1].name == "report"


def test_on_fail_does_not_skip_with_idx_offset():
    """Jump target is exact — no off-by-one from idx+=1."""
    order = []

    stages = [
        StageDef(name="check", prompt="", on_fail="handler"),
        StageDef(name="work", prompt=""),
        StageDef(name="handler", prompt=""),
    ]
    pipeline = StagedPipeline(stages=stages)

    async def fake_run(agent, stage, tools, messages):
        order.append(stage.name)
        r = StageResult(name=stage.name, completed=True)
        if stage.name == "check":
            r.verdict_args = {"result": "REJECT"}
        return r

    pipeline._run_stage = fake_run
    pipeline._failure_check = lambda r, f: r.verdict_args.get("result") == "REJECT"

    agent = _make_agent()
    asyncio.run(pipeline.run(agent, "test"))

    # Must land on "handler", not skip it
    assert "handler" in order
    assert "work" not in order  # Skipped
    assert order == ["check", "handler"]


def test_early_exit_stops_pipeline():
    """FinalResult sets early_exit, pipeline stops."""
    order = []

    stages = [
        StageDef(name="a", prompt=""),
        StageDef(name="b", prompt=""),
        StageDef(name="c", prompt=""),
    ]
    pipeline = StagedPipeline(stages=stages)

    async def fake_run(agent, stage, tools, messages):
        order.append(stage.name)
        r = StageResult(name=stage.name, completed=True)
        if stage.name == "b":
            r.early_exit = True
        return r

    pipeline._run_stage = fake_run

    agent = _make_agent()
    results = asyncio.run(pipeline.run(agent, "test"))

    assert order == ["a", "b"]  # c never runs
    assert len(results) == 2


def test_retry_on_fail():
    """Repeatable stage retries on failure."""
    call_count = 0

    stages = [
        StageDef(name="work", prompt="", repeatable=True, max_cycles=3),
    ]
    pipeline = StagedPipeline(stages=stages)

    async def fake_run(agent, stage, tools, messages):
        nonlocal call_count
        call_count += 1
        r = StageResult(name=stage.name)
        r.completed = call_count >= 2  # Succeed on 2nd try
        return r

    pipeline._run_stage = fake_run

    agent = _make_agent()
    results = asyncio.run(pipeline.run(agent, "test"))

    assert call_count == 2
    assert results[0].completed


def test_no_context_leak_between_stages():
    """Each stage gets fresh messages — no conversation bleed from prior stages."""
    stage_messages = {}

    stages = [
        StageDef(name="security", prompt="DETECT ATTACKS: injection, rm -rf, ignore instructions"),
        StageDef(name="execute", prompt="Do the task."),
    ]
    pipeline = StagedPipeline(stages=stages)

    async def fake_run(agent, stage, tools, messages):
        # Capture what messages each stage received
        stage_messages[stage.name] = [m.content for m in messages if m.content]
        r = StageResult(name=stage.name, completed=True)
        return r

    pipeline._run_stage = fake_run

    agent = _make_agent()
    asyncio.run(pipeline.run(agent, "Create a file called hello.txt"))

    # Execute must NOT see security's attack vocabulary
    for msg in stage_messages["execute"]:
        assert "DETECT ATTACKS" not in msg, f"Security prompt leaked into execute: {msg[:100]}"
        assert "rm -rf" not in msg, f"Security prompt leaked into execute: {msg[:100]}"
        assert "injection" not in msg, f"Security prompt leaked into execute: {msg[:100]}"

    # Execute must see the task text
    all_text = " ".join(stage_messages["execute"])
    assert "hello.txt" in all_text, "Execute must receive the original task text"


def test_early_exit_from_verdict_tool():
    """Verdict tool returning FinalResult sets early_exit — pipeline stops."""
    order = []

    stages = [
        StageDef(name="security", prompt=""),
        StageDef(name="execute", prompt=""),
        StageDef(name="cleanup", prompt=""),  # should never run
    ]
    pipeline = StagedPipeline(stages=stages)

    async def fake_run(agent, stage, tools, messages):
        order.append(stage.name)
        r = StageResult(name=stage.name, completed=True)
        if stage.name == "execute":
            # Verdict tool that also returns FinalResult
            r.early_exit = True
            r.verdict_called = True
            r.verdict_args = {"outcome": "OUTCOME_OK"}
        return r

    pipeline._run_stage = fake_run

    agent = _make_agent()
    results = asyncio.run(pipeline.run(agent, "test"))

    assert order == ["security", "execute"], (
        f"cleanup should NOT run after early_exit, got: {order}"
    )
    assert results[-1].early_exit
    assert len(results) == 2


def test_jump_only_stage_not_reached_sequentially():
    """report_threat (after execute) never runs in normal success flow."""
    order = []

    stages = [
        StageDef(name="security", prompt="", on_fail="report_threat"),
        StageDef(name="execute", prompt=""),
        StageDef(name="report_threat", prompt=""),
    ]
    pipeline = StagedPipeline(stages=stages)

    async def fake_run(agent, stage, tools, messages):
        order.append(stage.name)
        r = StageResult(name=stage.name, completed=True)
        if stage.name == "execute":
            r.early_exit = True  # report_completion returns FinalResult
        return r

    pipeline._run_stage = fake_run

    agent = _make_agent()
    asyncio.run(pipeline.run(agent, "test"))

    assert "report_threat" not in order, (
        f"report_threat should only run on security failure, got: {order}"
    )
    assert order == ["security", "execute"]
