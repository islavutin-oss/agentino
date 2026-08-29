"""Tests for pipeline — async sequence and router patterns."""

from unittest.mock import AsyncMock, patch

import pytest

from agentino import Agent
from agentino.pipeline import ParallelPipeline, Pipeline, RouterPipeline, Step


def _mock_agent(reply: str = "done") -> Agent:
    """Create an agent that returns a fixed reply."""
    agent = Agent(model="test", tools=[])
    return agent


def _patch_run(agent: Agent, reply: str):
    """Patch agent.run to return a fixed reply (async)."""
    return patch.object(agent, "run", new_callable=AsyncMock, return_value=reply)


# ---------------------------------------------------------------------------
# Pipeline (sequence)
# ---------------------------------------------------------------------------


class TestSequencePipeline:
    @pytest.mark.asyncio
    async def test_single_step(self):
        agent = _mock_agent()
        pipeline = Pipeline([("step1", agent, "Do something")])
        with _patch_run(agent, "result1"):
            result = await pipeline.run()
        assert result == {"step1": "result1"}

    @pytest.mark.asyncio
    async def test_multi_step(self):
        a1 = _mock_agent()
        a2 = _mock_agent()
        pipeline = Pipeline(
            [
                ("first", a1, "Start"),
                ("second", a2, lambda ctx: f"Continue from: {ctx['first']}"),
            ]
        )
        with _patch_run(a1, "step1_done"), _patch_run(a2, "step2_done"):
            result = await pipeline.run()
        assert result == {"first": "step1_done", "second": "step2_done"}

    @pytest.mark.asyncio
    async def test_previous_substitution(self):
        a1 = _mock_agent()
        a2 = _mock_agent()
        pipeline = Pipeline(
            [
                ("first", a1, "Start"),
                ("second", a2, "Process {previous}"),
            ]
        )
        with _patch_run(a1, "data123"), _patch_run(a2, "ok") as m2:
            await pipeline.run()
        m2.assert_called_once_with("Process data123")

    @pytest.mark.asyncio
    async def test_named_substitution(self):
        a1 = _mock_agent()
        a2 = _mock_agent()
        pipeline = Pipeline(
            [
                ("checker", a1, "Check CI"),
                ("reporter", a2, "Report on {checker}"),
            ]
        )
        with _patch_run(a1, "3 failures"), _patch_run(a2, "report") as m2:
            await pipeline.run()
        m2.assert_called_once_with("Report on 3 failures")

    @pytest.mark.asyncio
    async def test_initial_message(self):
        agent = _mock_agent()
        pipeline = Pipeline([("step1", agent, lambda ctx: f"Input: {ctx['input']}")])
        with _patch_run(agent, "done") as m:
            await pipeline.run("hello world")
        m.assert_called_once_with("Input: hello world")

    @pytest.mark.asyncio
    async def test_condition_skip(self):
        a1 = _mock_agent()
        a2 = _mock_agent()
        pipeline = Pipeline(
            [
                ("first", a1, "Start"),
                Step(
                    name="second",
                    agent=a2,
                    message="Only if failed",
                    condition=lambda ctx: "fail" in ctx.get("first", ""),
                ),
            ]
        )
        with _patch_run(a1, "all good"), _patch_run(a2, "never") as m2:
            result = await pipeline.run()
        assert "second" not in result
        m2.assert_not_called()

    @pytest.mark.asyncio
    async def test_condition_run(self):
        a1 = _mock_agent()
        a2 = _mock_agent()
        pipeline = Pipeline(
            [
                ("first", a1, "Start"),
                Step(
                    name="second",
                    agent=a2,
                    message="Handle failure",
                    condition=lambda ctx: "failure" in ctx.get("first", ""),
                ),
            ]
        )
        with _patch_run(a1, "detected failure"), _patch_run(a2, "fixed"):
            result = await pipeline.run()
        assert result["second"] == "fixed"

    @pytest.mark.asyncio
    async def test_step_from_tuple_with_condition(self):
        a1 = _mock_agent()
        pipeline = Pipeline(
            [
                ("step1", a1, "Go", lambda ctx: True),
            ]
        )
        with _patch_run(a1, "done"):
            result = await pipeline.run()
        assert result == {"step1": "done"}

    def test_step_tuple_validation(self):
        try:
            Pipeline([(1, 2)])  # too few elements
            assert False, "Should have raised"
        except (ValueError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Router pipeline
# ---------------------------------------------------------------------------


class TestRouterPipeline:
    @pytest.mark.asyncio
    async def test_exact_route(self):
        router = _mock_agent()
        booking = _mock_agent()
        wine = _mock_agent()

        pipeline = Pipeline.router(
            router=router,
            routes={"booking": booking, "wine": wine},
        )

        with _patch_run(router, "wine"), _patch_run(wine, "Try a Pinot Noir"):
            result = await pipeline.run("Wine for salmon?")
        assert result == "Try a Pinot Noir"

    @pytest.mark.asyncio
    async def test_fuzzy_route(self):
        router = _mock_agent()
        booking = _mock_agent()

        pipeline = Pipeline.router(
            router=router,
            routes={"booking": booking},
        )

        with _patch_run(router, "I think this is about booking"), _patch_run(booking, "Booked!"):
            result = await pipeline.run("Reserve a table")
        assert result == "Booked!"

    @pytest.mark.asyncio
    async def test_default_fallback(self):
        router = _mock_agent()
        general = _mock_agent()

        pipeline = Pipeline.router(
            router=router,
            routes={"booking": _mock_agent(), "general": general},
            default="general",
        )

        with _patch_run(router, "unknown_intent"), _patch_run(general, "How can I help?"):
            result = await pipeline.run("What's the weather?")
        assert result == "How can I help?"

    @pytest.mark.asyncio
    async def test_classify_only(self):
        router = _mock_agent()
        pipeline = Pipeline.router(
            router=router,
            routes={"booking": _mock_agent()},
        )
        with _patch_run(router, "BOOKING"):
            intent = await pipeline.classify("Reserve a table")
        assert intent == "booking"

    @pytest.mark.asyncio
    async def test_router_message_override(self):
        router = _mock_agent()
        booking = _mock_agent()
        pipeline = RouterPipeline(
            router=router,
            routes={"booking": booking},
            router_message="Classify the intent",
        )
        with _patch_run(router, "booking") as m_router, _patch_run(booking, "ok"):
            await pipeline.run("Table for 2")
        assert "Classify the intent" in m_router.call_args[0][0]


# ---------------------------------------------------------------------------
# Parallel pipeline
# ---------------------------------------------------------------------------


class TestParallelPipeline:
    @pytest.mark.asyncio
    async def test_runs_all_agents(self):
        a1 = _mock_agent()
        a2 = _mock_agent()
        pipeline = ParallelPipeline({"analysis": a1, "summary": a2})
        with _patch_run(a1, "analyzed"), _patch_run(a2, "summarized"):
            result = await pipeline.run("Review this PR")
        assert result == {"analysis": "analyzed", "summary": "summarized"}
