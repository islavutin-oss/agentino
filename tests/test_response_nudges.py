"""The built-in retry nudges: the framework's own, minimal, universal rules.

Anything beyond these — refusal detection, length floors, output-shape rules —
belongs to the application and goes through `response_filter`.
"""

from __future__ import annotations

import pytest

from agentino import Agent

pytestmark = pytest.mark.asyncio


@pytest.fixture
def agent():
    return Agent(instructions="test", max_turns=5)


async def test_empty_response_before_the_limit_is_retried(agent):
    nudge = await agent._check_text_response("", [], turn=0, had_tool_calls=False)
    assert nudge and "empty response" in nudge.lower()


async def test_empty_response_at_the_limit_after_tool_calls_asks_for_a_summary(agent):
    """The turn budget is gone and the model returned nothing, but it did do
    the work. Asking for more tool calls would waste the last turn; asking it
    to summarise is the only useful move left."""
    nudge = await agent._check_text_response("", [], turn=4, had_tool_calls=True)
    assert nudge and "summarize" in nudge.lower()
    assert "do not call any more tools" in nudge.lower()


async def test_at_the_limit_with_no_tool_calls_is_not_nudged(agent):
    """Nothing was collected, so there is nothing to summarise."""
    assert await agent._check_text_response("", [], turn=4, had_tool_calls=False) is None


async def test_a_real_answer_at_the_limit_passes(agent):
    assert (
        await agent._check_text_response("Here is the answer.", [], turn=4, had_tool_calls=True)
        is None
    )


async def test_whitespace_only_counts_as_empty(agent):
    assert await agent._check_text_response("   \n\t ", [], turn=4, had_tool_calls=True) is not None


async def test_the_application_filter_is_not_consulted_at_the_limit():
    """At the limit there is no turn left to spend on the app's own retry."""
    calls = []

    def never_satisfied(text, turn, max_turns, had_tool_calls):
        calls.append(text)
        return "try harder"

    a = Agent(instructions="test", max_turns=5, response_filter=never_satisfied)
    assert await a._check_text_response("fine", [], turn=4, had_tool_calls=True) is None
    assert calls == []
