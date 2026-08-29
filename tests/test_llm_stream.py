"""Tests for LLM client streaming — SSE parsing, tool call assembly, usage extraction."""

import json
from unittest.mock import patch

import pytest

from agentino import Message
from agentino.core.llm import LLMClient


def _make_sse_lines(chunks: list[dict | str]) -> list[str]:
    """Build SSE lines from a list of chunk dicts (or raw strings like '[DONE]')."""
    lines = []
    for chunk in chunks:
        if isinstance(chunk, str):
            lines.append(f"data: {chunk}")
        else:
            lines.append(f"data: {json.dumps(chunk)}")
    return lines


def _text_delta(index: int, content: str) -> dict:
    """Build a streaming text delta chunk."""
    return {
        "choices": [{"index": 0, "delta": {"content": content}}],
    }


def _tool_call_start(idx: int, call_id: str, name: str) -> dict:
    """Build a streaming tool_call start chunk."""
    return {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": idx,
                            "id": call_id,
                            "function": {"name": name, "arguments": ""},
                        }
                    ],
                },
            }
        ],
    }


def _tool_call_args(idx: int, args_fragment: str) -> dict:
    """Build a streaming tool_call arguments fragment."""
    return {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": idx,
                            "function": {"arguments": args_fragment},
                        }
                    ],
                },
            }
        ],
    }


def _usage_chunk(prompt: int, completion: int) -> dict:
    """Build a usage chunk (typically the last real chunk before [DONE])."""
    return {
        "choices": [{}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class _FakeStreamResponse:
    """Fake httpx streaming response that yields SSE lines."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# -------------------------------------------------------------------
# Text streaming
# -------------------------------------------------------------------


class TestTextStreaming:
    @pytest.mark.asyncio
    async def test_simple_text_stream(self):
        chunks = [
            _text_delta(0, "Hello"),
            _text_delta(0, " world"),
            _usage_chunk(10, 5),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)

        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="Hi")]):
                events.append(event)

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 2
        assert text_events[0].data == "Hello"
        assert text_events[1].data == " world"

        llm_event = next(e for e in events if e.type == "llm_response")
        assert llm_event.data.content == "Hello world"
        assert llm_event.usage.prompt_tokens == 10
        assert llm_event.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Stream with only [DONE] — no content, no tool calls."""
        lines = _make_sse_lines(["[DONE]"])
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="Hi")]):
                events.append(event)

        llm_event = next(e for e in events if e.type == "llm_response")
        assert llm_event.data.content is None
        assert llm_event.data.tool_calls is None


# -------------------------------------------------------------------
# Tool call streaming
# -------------------------------------------------------------------


class TestToolCallStreaming:
    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        chunks = [
            _tool_call_start(0, "call_abc", "search"),
            _tool_call_args(0, '{"q":'),
            _tool_call_args(0, ' "cats"}'),
            _usage_chunk(20, 15),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="search cats")]):
                events.append(event)

        llm_event = next(e for e in events if e.type == "llm_response")
        msg = llm_event.data
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_abc"
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].arguments == {"q": "cats"}

    @pytest.mark.asyncio
    async def test_multiple_parallel_tool_calls(self):
        chunks = [
            _tool_call_start(0, "c1", "tool_a"),
            _tool_call_start(1, "c2", "tool_b"),
            _tool_call_args(0, '{"x": 1}'),
            _tool_call_args(1, '{"y": 2}'),
            _usage_chunk(30, 20),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="do both")]):
                events.append(event)

        llm_event = next(e for e in events if e.type == "llm_response")
        msg = llm_event.data
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].name == "tool_a"
        assert msg.tool_calls[0].arguments == {"x": 1}
        assert msg.tool_calls[1].name == "tool_b"
        assert msg.tool_calls[1].arguments == {"y": 2}

    @pytest.mark.asyncio
    async def test_malformed_tool_args_default_to_empty(self):
        chunks = [
            _tool_call_start(0, "c1", "bad_tool"),
            _tool_call_args(0, "not valid json{{{"),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="bad")]):
                events.append(event)

        llm_event = next(e for e in events if e.type == "llm_response")
        assert llm_event.data.tool_calls[0].arguments == {}


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------


class TestStreamEdgeCases:
    @pytest.mark.asyncio
    async def test_non_sse_lines_ignored(self):
        """Lines not starting with 'data: ' should be silently ignored."""
        lines = [
            ": keep-alive",
            "",
            "event: ping",
            f"data: {json.dumps(_text_delta(0, 'ok'))}",
            "data: [DONE]",
        ]
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="Hi")]):
                events.append(event)

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 1
        assert text_events[0].data == "ok"

    @pytest.mark.asyncio
    async def test_malformed_json_chunk_skipped(self):
        """Invalid JSON in SSE data should be skipped without error."""
        lines = [
            f"data: {json.dumps(_text_delta(0, 'before'))}",
            "data: {not json at all",
            f"data: {json.dumps(_text_delta(0, ' after'))}",
            "data: [DONE]",
        ]
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="Hi")]):
                events.append(event)

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 2
        llm_event = next(e for e in events if e.type == "llm_response")
        assert llm_event.data.content == "before after"

    @pytest.mark.asyncio
    async def test_text_and_tool_calls_mixed(self):
        """Stream that has both text content and tool calls."""
        chunks = [
            _text_delta(0, "Let me search"),
            _tool_call_start(0, "c1", "search"),
            _tool_call_args(0, '{"q": "test"}'),
            _usage_chunk(15, 10),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="search")]):
                events.append(event)

        llm_event = next(e for e in events if e.type == "llm_response")
        assert llm_event.data.content == "Let me search"
        assert llm_event.data.tool_calls is not None
        assert llm_event.data.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_usage_from_final_chunk(self):
        """Usage should come from the usage chunk, not accumulated."""
        chunks = [
            _text_delta(0, "hi"),
            {"choices": [{}], "usage": {"prompt_tokens": 42, "completion_tokens": 17}},
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="Hi")]):
                events.append(event)

        llm_event = next(e for e in events if e.type == "llm_response")
        assert llm_event.usage.prompt_tokens == 42
        assert llm_event.usage.completion_tokens == 17


# -------------------------------------------------------------------
# Borrow #1 — Fine-grained streaming events (toolcall_start/delta/end)
# -------------------------------------------------------------------


class TestGranularToolCallEvents:
    @pytest.mark.asyncio
    async def test_emits_start_delta_end(self):
        """A single tool call should emit start once, delta per fragment, end once."""
        chunks = [
            _tool_call_start(0, "call_x", "search"),
            _tool_call_args(0, '{"q":'),
            _tool_call_args(0, ' "cats"}'),
            _usage_chunk(20, 15),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = []
            async for event in client.chat_stream([Message(role="user", content="x")]):
                events.append(event)

        starts = [e for e in events if e.type == "toolcall_start"]
        deltas = [e for e in events if e.type == "toolcall_delta"]
        ends = [e for e in events if e.type == "toolcall_end"]

        assert len(starts) == 1
        assert starts[0].name == "search"
        assert starts[0].data == {"id": "call_x", "index": 0}

        assert len(deltas) == 2
        assert "".join(d.data["delta"] for d in deltas) == '{"q": "cats"}'

        assert len(ends) == 1
        assert ends[0].args == {"q": "cats"}
        assert ends[0].data == {"id": "call_x", "index": 0}

    @pytest.mark.asyncio
    async def test_start_emitted_only_once_per_call(self):
        """Even when id and name arrive in separate chunks, start should fire exactly once."""
        # First chunk has id but no name; second has name. Start fires when both known.
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_y", "function": {"arguments": ""}}
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"name": "do_it", "arguments": ""}}
                            ]
                        },
                    }
                ]
            },
            _tool_call_args(0, '{"k": 1}'),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = [e async for e in client.chat_stream([Message(role="user", content="x")])]

        starts = [e for e in events if e.type == "toolcall_start"]
        assert len(starts) == 1
        assert starts[0].name == "do_it"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_get_separate_events(self):
        """Two parallel tool calls each get their own start/end with the right index."""
        chunks = [
            _tool_call_start(0, "c1", "tool_a"),
            _tool_call_start(1, "c2", "tool_b"),
            _tool_call_args(0, '{"x": 1}'),
            _tool_call_args(1, '{"y": 2}'),
            _usage_chunk(30, 20),
            "[DONE]",
        ]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = [e async for e in client.chat_stream([Message(role="user", content="x")])]

        starts = [e for e in events if e.type == "toolcall_start"]
        ends = [e for e in events if e.type == "toolcall_end"]
        assert {s.data["index"] for s in starts} == {0, 1}
        assert {e.data["index"] for e in ends} == {0, 1}
        end_by_idx = {e.data["index"]: e.args for e in ends}
        assert end_by_idx[0] == {"x": 1}
        assert end_by_idx[1] == {"y": 2}

    @pytest.mark.asyncio
    async def test_no_toolcall_events_for_text_only_response(self):
        """Pure text response emits no toolcall_* events."""
        chunks = [_text_delta(0, "hello"), _usage_chunk(5, 5), "[DONE]"]
        lines = _make_sse_lines(chunks)
        client = LLMClient(api_key="test", default_model="gpt-4o", provider="openai")

        with patch.object(client._client, "stream", return_value=_FakeStreamResponse(lines)):
            events = [e async for e in client.chat_stream([Message(role="user", content="hi")])]

        assert not any(e.type.startswith("toolcall_") for e in events)
