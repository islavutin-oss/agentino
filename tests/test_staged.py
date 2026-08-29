"""Tests for staged pipeline."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from agentino.core.message import EventType
from agentino.pipeline.staged import FactStore, StageDef, StagedPipeline, StageResult


@dataclass
class TestFacts:
    """Test fact store."""

    task_type: str = ""
    answer: str = ""

    def to_context(self) -> str:
        lines = []
        if self.task_type:
            lines.append(f"task_type={self.task_type}")
        if self.answer:
            lines.append(f"answer={self.answer}")
        return "\n".join(lines)


def test_fact_store_protocol():
    """FactStore protocol works with concrete implementations."""
    facts = TestFacts(task_type="lookup", answer="42")
    assert isinstance(facts, FactStore)
    assert "task_type=lookup" in facts.to_context()
    assert "answer=42" in facts.to_context()


def test_stage_def_skip_condition():
    """skip_condition controls stage skipping."""
    facts = TestFacts(task_type="lookup")

    stage_skip = StageDef(
        name="validate",
        skip_condition=lambda f: f.task_type == "lookup",
    )
    stage_run = StageDef(
        name="execute",
        skip_condition=lambda f: f.task_type == "inbox",
    )

    assert stage_skip.should_skip(facts) is True
    assert stage_run.should_skip(facts) is False


def test_stage_def_no_skip_without_condition():
    """Stages without skip_condition always run."""
    facts = TestFacts()
    stage = StageDef(name="discover")
    assert stage.should_skip(facts) is False


def test_stage_def_no_skip_without_facts():
    """Stages don't skip when no facts provided."""
    stage = StageDef(
        name="validate",
        skip_condition=lambda f: True,
    )
    assert stage.should_skip(None) is False


def test_stage_result_defaults():
    """StageResult has sensible defaults."""
    r = StageResult(name="test")
    assert r.completed is False
    assert r.verdict_called is False
    assert r.verdict_args == {}
    assert r.early_exit is False


def test_staged_pipeline_init():
    """StagedPipeline initializes correctly."""
    stages = [
        StageDef(name="a", prompt="do A"),
        StageDef(name="b", prompt="do B"),
    ]
    facts = TestFacts()
    pipeline = StagedPipeline(stages=stages, facts=facts)
    assert len(pipeline.stages) == 2
    assert pipeline.facts is facts
    assert pipeline.max_reprompts == 3


def test_staged_pipeline_skip_events():
    """Skipped stages emit STAGE_SKIP events."""
    facts = TestFacts(task_type="lookup")
    stages = [
        StageDef(name="discover"),
        StageDef(name="validate", skip_condition=lambda f: f.task_type == "lookup"),
        StageDef(name="complete"),
    ]
    StagedPipeline(stages=stages, facts=facts)

    events = []
    # We can't run the full pipeline without a real agent,
    # but we can verify skip logic directly
    for stage in stages:
        if stage.should_skip(facts):
            events.append(EventType.STAGE_SKIP)
        else:
            events.append(EventType.STAGE_START)

    assert events == [EventType.STAGE_START, EventType.STAGE_SKIP, EventType.STAGE_START]


def test_stage_def_retry_defaults():
    """StageDef has retry defaults."""
    stage = StageDef(name="test")
    assert stage.repeatable is False
    assert stage.max_cycles == 3
    assert stage.on_fail == "retry"


def test_stage_def_repeatable():
    """Repeatable stage with jump target."""
    stage = StageDef(name="test", repeatable=True, max_cycles=5, on_fail="implement")
    assert stage.repeatable is True
    assert stage.max_cycles == 5
    assert stage.on_fail == "implement"


def test_stage_result_failed():
    """StageResult tracks failure and cycles."""
    r = StageResult(name="test", failed=True, cycles_used=3)
    assert r.failed is True
    assert r.cycles_used == 3


def test_pipeline_global_max_cycles():
    """Pipeline respects global_max_cycles."""
    pipeline = StagedPipeline(
        stages=[StageDef(name="a"), StageDef(name="b")],
        global_max_cycles=10,
    )
    assert pipeline.global_max_cycles == 10


def test_pipeline_custom_failure_check():
    """Pipeline accepts custom failure check."""

    def check(r, f):
        return r.verdict_args.get("status") == "FAIL"

    pipeline = StagedPipeline(
        stages=[],
        failure_check=check,
    )
    # Test the check
    r = StageResult(name="test", verdict_args={"status": "FAIL"})
    assert pipeline._failure_check(r, None) is True

    r2 = StageResult(name="test", verdict_args={"status": "PASS"})
    assert pipeline._failure_check(r2, None) is False


# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------


def test_parse_verdict_accept():
    from agentino.pipeline.staged import parse_verdict

    assert parse_verdict("VERDICT:ACCEPT") == "ACCEPT"


def test_parse_verdict_fail():
    from agentino.pipeline.staged import parse_verdict

    assert parse_verdict("some output\nVERDICT:FAIL") == "FAIL"


def test_parse_verdict_reject():
    from agentino.pipeline.staged import parse_verdict

    assert parse_verdict("review done\nVERDICT:REJECT") == "REJECT"


def test_parse_verdict_none():
    from agentino.pipeline.staged import parse_verdict

    assert parse_verdict("no verdict here") is None


def test_parse_verdict_embedded():
    from agentino.pipeline.staged import parse_verdict

    assert parse_verdict("tool returned VERDICT:ACCEPT and then continued") == "ACCEPT"


# ---------------------------------------------------------------------------
# Helpers for async pipeline tests (mock _run_stage, no LLM)
# ---------------------------------------------------------------------------

from unittest.mock import patch  # noqa: E402

from agentino.core.agent import Agent  # noqa: E402
from agentino.core.message import Message  # noqa: E402


def _accept(**extra) -> StageResult:
    return StageResult(
        name="",
        completed=True,
        verdict_called=True,
        verdict_args={"result": "ACCEPT", **extra},
    )


def _reject(summary: str = "", reason: str = "", **extra) -> StageResult:
    args = {"result": "REJECT", **extra}
    if summary:
        args["summary"] = summary
    if reason:
        args["reason"] = reason
    return StageResult(
        name="",
        completed=True,
        verdict_called=True,
        verdict_args=args,
    )


def _make_template():
    """Minimal mock template agent — no LLM calls."""
    agent = Agent(model="test", tools=[])
    agent._llm = MagicMock()
    agent._llm.base_url = "http://test"
    agent._llm.api_key = "test"
    agent._llm.provider = "openai"
    agent._tool_result_cap = 4000
    agent._tool_instructions = ""
    return agent


def _msgs_text(messages: list[Message]) -> list[str]:
    """Extract non-empty text content from messages."""
    return [m.content for m in messages if m.content]


# ---------------------------------------------------------------------------
# Jump: verify → respond with rejection feedback
# ---------------------------------------------------------------------------


class TestJumpRejectionFeedback:
    """When verify rejects and on_fail jumps to respond,
    the rejection reason must appear in respond's messages."""

    @pytest.mark.asyncio
    async def test_rejection_summary_passed_on_jump(self):
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if stage.name == "verify" and len(call_log) == 2:
                return _reject(summary="Links inside code blocks")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        # respond(1), verify(reject), respond(2), verify(accept)
        assert len(call_log) >= 3
        second_respond_texts = call_log[2][1]
        assert any("Links inside code blocks" in t for t in second_respond_texts), (
            f"Rejection feedback missing: {second_respond_texts}"
        )

    @pytest.mark.asyncio
    async def test_rejection_reason_field_fallback(self):
        """Uses 'reason' field when 'summary' is absent."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if stage.name == "verify" and len(call_log) == 2:
                return _reject(reason="No diagram found")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        second_respond_texts = call_log[2][1]
        assert any("No diagram found" in t for t in second_respond_texts)

    @pytest.mark.asyncio
    async def test_rejected_by_prefix_in_feedback(self):
        """Feedback message starts with 'REJECTED by STAGENAME stage.'"""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if stage.name == "verify" and len(call_log) == 2:
                return _reject(summary="Bad format")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        second_respond_texts = call_log[2][1]
        assert any("REJECTED by VERIFY stage" in t for t in second_respond_texts)

    @pytest.mark.asyncio
    async def test_no_feedback_when_no_reason(self):
        """No feedback injected when reject has no summary or reason."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if stage.name == "verify" and len(call_log) == 2:
                return _reject()  # no summary, no reason
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        second_respond_texts = call_log[2][1]
        assert not any("REJECTED" in t for t in second_respond_texts)

    @pytest.mark.asyncio
    async def test_no_feedback_on_first_run(self):
        """First respond run has no rejection feedback."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v"),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        first_respond_texts = call_log[0][1]
        assert not any("REJECTED" in t for t in first_respond_texts)


# ---------------------------------------------------------------------------
# Retry: same-stage retry with rejection feedback
# ---------------------------------------------------------------------------


class TestRetryRejectionFeedback:
    """When a repeatable stage fails, the rejection reason
    must appear in the retry's messages."""

    @pytest.mark.asyncio
    async def test_retry_includes_rejection_reason(self):
        stages = [
            StageDef(
                name="verify", prompt="Check.", verdict_tool="v", repeatable=True, max_cycles=3
            ),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if len(call_log) == 1:
                return _reject(summary="Missing diagram section")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        assert len(call_log) == 2
        retry_texts = call_log[1][1]
        assert any("Missing diagram section" in t for t in retry_texts), (
            f"Retry reason missing: {retry_texts}"
        )

    @pytest.mark.asyncio
    async def test_retry_includes_rejection_reason_field(self):
        """Retry uses 'reason' field when 'summary' absent."""
        stages = [
            StageDef(
                name="check", prompt="Check.", verdict_tool="v", repeatable=True, max_cycles=2
            ),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if len(call_log) == 1:
                return _reject(reason="Plain text paths detected")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        retry_texts = call_log[1][1]
        assert any("Plain text paths detected" in t for t in retry_texts)

    @pytest.mark.asyncio
    async def test_retry_message_without_reason(self):
        """Retry still gets a generic failure message even without reason."""
        stages = [
            StageDef(
                name="check", prompt="Check.", verdict_tool="v", repeatable=True, max_cycles=2
            ),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if len(call_log) == 1:
                return _reject()  # no summary or reason
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        retry_texts = call_log[1][1]
        assert any("failed" in t.lower() for t in retry_texts)

    @pytest.mark.asyncio
    async def test_multiple_retries_each_get_reason(self):
        """Each retry gets its own rejection reason."""
        stages = [
            StageDef(
                name="check", prompt="Check.", verdict_tool="v", repeatable=True, max_cycles=4
            ),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            n = len(call_log)
            if n == 1:
                return _reject(summary="Issue A")
            if n == 2:
                return _reject(summary="Issue B")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        assert len(call_log) == 3
        assert any("Issue A" in t for t in call_log[1][1])
        assert any("Issue B" in t for t in call_log[2][1])


# ---------------------------------------------------------------------------
# Feedback consumed: not leaked to subsequent stages
# ---------------------------------------------------------------------------


class TestFeedbackConsumed:
    @pytest.mark.asyncio
    async def test_feedback_not_leaked_to_next_stage(self):
        """After jump, feedback consumed — next stages don't see it."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
            StageDef(name="publish", prompt="Publish.", verdict_tool="v"),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if stage.name == "verify" and len(call_log) == 2:
                return _reject(summary="Bad formatting")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        # Find publish stage
        publish_entries = [(n, t) for n, t in call_log if n == "publish"]
        assert len(publish_entries) == 1
        publish_texts = publish_entries[0][1]
        assert not any("REJECTED" in t for t in publish_texts), (
            f"Feedback leaked to publish: {publish_texts}"
        )
        assert not any("Bad formatting" in t for t in publish_texts), (
            f"Rejection reason leaked to publish: {publish_texts}"
        )

    @pytest.mark.asyncio
    async def test_feedback_consumed_after_jump(self):
        """After jump to respond, feedback consumed — second verify doesn't see it."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
        ]
        pipeline = StagedPipeline(stages=stages)
        call_log = []

        async def mock_run(agent, stage, tools, messages):
            call_log.append((stage.name, _msgs_text(messages)))
            if stage.name == "verify" and len(call_log) == 2:
                return _reject(summary="Fix links")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        # respond(1), verify(reject), respond(2, gets feedback), verify(accept)
        second_verify = [t for n, t in call_log if n == "verify"]
        assert len(second_verify) == 2
        # Second verify should NOT have the rejection feedback
        assert not any("Fix links" in t for t in second_verify[1])


# ---------------------------------------------------------------------------
# Multiple jumps: feedback refreshed each time
# ---------------------------------------------------------------------------


class TestMultipleJumps:
    @pytest.mark.asyncio
    async def test_different_feedback_per_jump(self):
        """Each jump carries fresh feedback from the latest rejection."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(
                name="verify",
                prompt="Check.",
                verdict_tool="v",
                repeatable=True,
                max_cycles=3,
                on_fail="respond",
            ),
        ]
        pipeline = StagedPipeline(stages=stages, global_max_cycles=20)
        call_log = []
        verify_count = 0

        async def mock_run(agent, stage, tools, messages):
            nonlocal verify_count
            call_log.append((stage.name, _msgs_text(messages)))
            if stage.name == "verify":
                verify_count += 1
                if verify_count == 1:
                    return _reject(summary="Problem alpha")
                if verify_count == 2:
                    return _reject(summary="Problem beta")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        # respond(1), verify(reject alpha), respond(2, gets alpha),
        # verify(reject beta), respond(3, gets beta), verify(accept)
        respond_entries = [(n, t) for n, t in call_log if n == "respond"]
        assert len(respond_entries) >= 3
        # 2nd respond sees "Problem alpha"
        assert any("Problem alpha" in t for t in respond_entries[1][1])
        # 3rd respond sees "Problem beta", NOT "Problem alpha"
        third_texts = respond_entries[2][1]
        assert any("Problem beta" in t for t in third_texts)
        assert not any("Problem alpha" in t for t in third_texts)


# ---------------------------------------------------------------------------
# Global max cycles budget
# ---------------------------------------------------------------------------


class TestGlobalBudget:
    @pytest.mark.asyncio
    async def test_stops_at_global_max(self):
        """Pipeline stops after global_max_cycles even in a reject loop."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
        ]
        pipeline = StagedPipeline(stages=stages, global_max_cycles=5)
        call_count = 0

        async def mock_run(agent, stage, tools, messages):
            nonlocal call_count
            call_count += 1
            if stage.name == "verify":
                return _reject(summary="Always bad")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task")

        assert call_count <= 5, f"Exceeded budget: {call_count} calls"


# ---------------------------------------------------------------------------
# Stage events emitted correctly
# ---------------------------------------------------------------------------


class TestStageEvents:
    @pytest.mark.asyncio
    async def test_fail_event_on_jump(self):
        """STAGE_FAIL event emitted with jump_to when verify rejects."""
        stages = [
            StageDef(name="respond", prompt="Write.", verdict_tool="v"),
            StageDef(name="verify", prompt="Check.", verdict_tool="v", on_fail="respond"),
        ]
        pipeline = StagedPipeline(stages=stages)
        events = []
        verify_count = 0

        async def mock_run(agent, stage, tools, messages):
            nonlocal verify_count
            if stage.name == "verify":
                verify_count += 1
                if verify_count == 1:
                    return _reject(summary="Bad")
            return _accept()

        with patch.object(pipeline, "_run_stage", side_effect=mock_run):
            await pipeline.run(_make_template(), "task", on_event=lambda e: events.append(e))

        fail_events = [e for e in events if e.type == EventType.STAGE_FAIL]
        assert len(fail_events) >= 1
        assert fail_events[0].data.get("jump_to") == "respond"
