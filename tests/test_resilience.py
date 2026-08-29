"""Tests for resilience utilities."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agentino.core.message import Message, ToolCall, Usage
from agentino.reliability.resilience import (
    ThinkFilterStream,
    compact_history,
    estimate_tokens,
    repair_messages,
    retry_with_backoff,
    strip_think_tags,
    truncate_result,
)

# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        fn = AsyncMock(return_value="ok")
        assert await retry_with_backoff(fn) == "ok"
        fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}

        fn = AsyncMock(
            side_effect=[
                httpx.HTTPStatusError("rate limited", request=MagicMock(), response=resp_429),
                "ok",
            ]
        )
        result = await retry_with_backoff(fn, max_attempts=3, initial_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_non_retryable(self):
        resp_400 = MagicMock()
        resp_400.status_code = 400
        resp_400.headers = {}

        fn = AsyncMock(
            side_effect=httpx.HTTPStatusError("bad", request=MagicMock(), response=resp_400)
        )
        with pytest.raises(httpx.HTTPStatusError):
            await retry_with_backoff(fn, max_attempts=3)
        fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.headers = {}

        fn = AsyncMock(
            side_effect=httpx.HTTPStatusError("down", request=MagicMock(), response=resp_500)
        )
        with pytest.raises(httpx.HTTPStatusError):
            await retry_with_backoff(fn, max_attempts=2, initial_delay=0.01)
        assert fn.call_count == 2

    @pytest.mark.asyncio
    async def test_honors_retry_after(self):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"retry-after": "0.01"}

        fn = AsyncMock(
            side_effect=[
                httpx.HTTPStatusError("rate limited", request=MagicMock(), response=resp_429),
                "ok",
            ]
        )
        result = await retry_with_backoff(fn, max_attempts=2, initial_delay=0.001)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self):
        fn = AsyncMock(
            side_effect=[
                httpx.ConnectError("refused"),
                "ok",
            ]
        )
        result = await retry_with_backoff(fn, max_attempts=2, initial_delay=0.01)
        assert result == "ok"


# ---------------------------------------------------------------------------
# repair_messages
# ---------------------------------------------------------------------------


class TestRepairMessages:
    def test_empty(self):
        assert repair_messages([]) == []

    def test_drops_orphaned_tool_results(self):
        messages = [
            Message(role="user", content="hi"),
            Message(role="tool", content="result", tool_call_id="orphan_123"),
        ]
        repaired = repair_messages(messages)
        assert len(repaired) == 1
        assert repaired[0].role == "user"

    def test_inserts_synthetic_for_unmatched_calls(self):
        messages = [
            Message(role="user", content="do it"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "test"})],
            ),
            # Missing tool result for call_1
        ]
        repaired = repair_messages(messages)
        tool_msgs = [m for m in repaired if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call_1"
        assert "interrupted" in tool_msgs[0].content

    def test_removes_empty_assistant(self):
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content=None, tool_calls=None),
            Message(role="assistant", content="hello"),
        ]
        repaired = repair_messages(messages)
        assert len(repaired) == 2
        assert repaired[1].content == "hello"

    def test_merges_consecutive_user_messages(self):
        messages = [
            Message(role="user", content="first"),
            Message(role="user", content="second"),
        ]
        repaired = repair_messages(messages)
        assert len(repaired) == 1
        assert "first" in repaired[0].content
        assert "second" in repaired[0].content

    def test_valid_messages_unchanged(self):
        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        repaired = repair_messages(messages)
        assert len(repaired) == 2

    def test_matched_tool_calls_preserved(self):
        messages = [
            Message(role="user", content="search"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"q": "test"})],
            ),
            Message(role="tool", content="found it", tool_call_id="call_1", name="search"),
            Message(role="assistant", content="Here's what I found"),
        ]
        repaired = repair_messages(messages)
        assert len(repaired) == 4


# ---------------------------------------------------------------------------
# truncate_result
# ---------------------------------------------------------------------------


class TestTruncateResult:
    def test_short_text_unchanged(self):
        assert truncate_result("hello", 100) == "hello"

    def test_truncates_at_newline(self):
        text = "line1\nline2\nline3\n" + "x" * 5000
        result = truncate_result(text, max_chars=30)
        assert "truncated" in result
        assert len(result) < len(text)

    def test_truncates_without_newline(self):
        text = "a" * 10000
        result = truncate_result(text, max_chars=100)
        assert "truncated" in result

    def test_exact_limit(self):
        text = "a" * 100
        assert truncate_result(text, max_chars=100) == text


# ---------------------------------------------------------------------------
# strip_think_tags
# ---------------------------------------------------------------------------


class TestStripThinkTags:
    def test_removes_think_block(self):
        text = "Before <think>internal reasoning</think> After"
        assert strip_think_tags(text) == "Before  After"

    def test_removes_multiline_think(self):
        text = "Start <think>\nthinking\nmore thinking\n</think> End"
        assert strip_think_tags(text) == "Start  End"

    def test_no_think_tags(self):
        text = "Normal response"
        assert strip_think_tags(text) == "Normal response"

    def test_empty_think(self):
        text = "A <think></think> B"
        assert strip_think_tags(text) == "A  B"

    def test_multiple_think_blocks(self):
        text = "<think>a</think> middle <think>b</think> end"
        assert strip_think_tags(text) == "middle  end"


# ---------------------------------------------------------------------------
# ThinkFilterStream
# ---------------------------------------------------------------------------


class TestThinkFilterStream:
    def test_complete_block_in_one_chunk(self):
        f = ThinkFilterStream()
        assert f.feed("Before <think>thought</think> After") == "Before  After"

    def test_split_across_chunks(self):
        f = ThinkFilterStream()
        out1 = f.feed("Hello <thi")
        out2 = f.feed("nk>secret</think> world")
        assert "Hello" in out1 + out2
        assert "world" in out1 + out2
        assert "secret" not in out1 + out2

    def test_no_think_tags(self):
        f = ThinkFilterStream()
        assert f.feed("plain text") == "plain text"


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens([]) == 0

    def test_text_messages(self):
        messages = [
            Message(role="user", content="Hello world"),  # 11 chars ≈ 2 tokens + 4 overhead
            Message(role="assistant", content="Hi there"),  # 8 chars ≈ 2 tokens + 4 overhead
        ]
        tokens = estimate_tokens(messages)
        assert tokens > 0
        assert tokens < 100  # sanity check

    def test_tool_calls_add_tokens(self):
        messages = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="1", name="search", arguments={"query": "test"}),
                ],
            ),
        ]
        tokens = estimate_tokens(messages)
        assert tokens > 20  # tool call overhead


# ---------------------------------------------------------------------------
# compact_history
# ---------------------------------------------------------------------------


class TestCompactHistory:
    @pytest.mark.asyncio
    async def test_no_compaction_when_under_threshold(self):
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ]
        result = await compact_history(messages, llm=MagicMock(), max_tokens=128_000)
        assert result == messages

    @pytest.mark.asyncio
    async def test_compaction_when_over_threshold(self):
        from agentino.core.llm import LLMResponse

        messages = [Message(role="system", content="Be helpful")]
        for i in range(100):
            messages.append(Message(role="user", content=f"Message {i} " + "x" * 500))
            messages.append(Message(role="assistant", content=f"Reply {i} " + "y" * 500))

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=LLMResponse(
                message=Message(role="assistant", content="Summary of conversation"),
                usage=Usage(prompt_tokens=100, completion_tokens=50),
            )
        )

        result = await compact_history(messages, llm=mock_llm, max_tokens=1000, threshold=0.1)
        assert len(result) < len(messages)
        summary_msgs = [m for m in result if m.content and "summary" in m.content.lower()]
        assert len(summary_msgs) > 0
        mock_llm.chat.assert_called_once()
