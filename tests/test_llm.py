"""Tests for the LLM client — request building, response parsing, resilience."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentino import Message, Usage, tool
from agentino.core.llm import (
    _MAX_RETRIES,
    LLMClient,
    _build_timeout,
    _retry_delay,
)


class TestBuildBody:
    def test_basic_body(self):
        client = LLMClient(api_key="test", default_model="gpt-4o")
        messages = [Message(role="user", content="Hello")]
        body = client._build_body(messages, None, None, 0.7, stream=False)

        assert body["model"] == "gpt-4o"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert body["temperature"] == 0.7
        assert body["stream"] is False
        assert "tools" not in body

    def test_body_with_tools(self):
        @tool
        def search(q: str) -> str:
            """Search."""
            return q

        client = LLMClient(api_key="test")
        messages = [Message(role="user", content="Hi")]
        body = client._build_body(messages, [search], "custom-model", 0.5, stream=False)

        assert body["model"] == "custom-model"
        assert len(body["tools"]) == 1
        assert body["tools"][0]["function"]["name"] == "search"

    def test_stream_body_includes_usage_option(self):
        client = LLMClient(api_key="test")
        body = client._build_body(
            [Message(role="user", content="Hi")], None, None, 0.7, stream=True
        )
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_model_override(self):
        client = LLMClient(api_key="test", default_model="default-model")
        body = client._build_body(
            [Message(role="user", content="Hi")], None, "override-model", 0.7, stream=False
        )
        assert body["model"] == "override-model"


class TestParseResponse:
    def test_text_response(self):
        client = LLMClient(api_key="test")
        data = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = client._parse_response(data)
        assert resp.message.content == "Hello!"
        assert resp.usage.prompt_tokens == 10
        assert resp.finish_reason == "stop"

    def test_tool_call_response(self):
        client = LLMClient(api_key="test")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "cats"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        resp = client._parse_response(data)
        assert resp.message.tool_calls is not None
        assert len(resp.message.tool_calls) == 1
        assert resp.message.tool_calls[0].name == "search"
        assert resp.message.tool_calls[0].arguments == {"q": "cats"}
        assert resp.message.tool_calls[0].id == "call_123"

    def test_multiple_tool_calls(self):
        client = LLMClient(api_key="test")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "tool_a", "arguments": '{"x": 1}'},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "tool_b", "arguments": '{"y": 2}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 20},
        }
        resp = client._parse_response(data)
        assert len(resp.message.tool_calls) == 2
        assert resp.message.tool_calls[0].name == "tool_a"
        assert resp.message.tool_calls[1].name == "tool_b"

    def test_malformed_arguments_handled(self):
        client = LLMClient(api_key="test")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "bad", "arguments": "not json{{{"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        resp = client._parse_response(data)
        assert resp.message.tool_calls[0].arguments == {}


class TestHeaders:
    def test_auth_header(self):
        client = LLMClient(api_key="sk-test123")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer sk-test123"

    def test_no_auth_header_when_empty(self):
        with patch("agentino.core.llm.LLMClient._resolve_api_key", return_value=""):
            client = LLMClient(api_key="")
            headers = client._headers()
            assert "Authorization" not in headers

    def test_base_url_trailing_slash_stripped(self):
        client = LLMClient(base_url="http://localhost:8100/v1/", api_key="sk-test")
        assert client.base_url == "http://localhost:8100/v1"


# ---------------------------------------------------------------------------
# Resilience helpers — granular timeout + jittered backoff
# ---------------------------------------------------------------------------


class TestRetryHelpers:
    def test_retry_delay_grows_exponentially_with_jitter(self):
        for attempt in range(_MAX_RETRIES + 1):
            base = 2.0 * (2**attempt)
            delay = _retry_delay(attempt)
            # exponential floor, jitter adds at most +25%
            assert base <= delay <= base * 1.25

    def test_build_timeout_fast_connect_generous_read(self):
        t = _build_timeout(120.0)
        assert t.connect == 15.0  # stalled handshake fails fast → retried
        assert t.read == 120.0  # slow generation still gets the full window
        assert t.write == 30.0
        assert t.pool == 15.0

    def test_build_timeout_clamps_to_small_budget(self):
        # a budget below the connect cap clamps every phase to the budget
        t = _build_timeout(5.0)
        assert t.connect == 5.0
        assert t.read == 5.0
        assert t.write == 5.0


# ---------------------------------------------------------------------------
# _post_with_retry — retries transient HTTP status AND connection faults
#
# REGRESSION: the old retry loop only handled HTTP status codes. A stalled
# or dropped connection raised httpx.TransportError straight out of chat(),
# aborting the whole agent turn (~observed as the intermittent ~175s harness
# timeout). Connection faults are transient — they must be retried.
# ---------------------------------------------------------------------------


def _resp(status=200, headers=None, json_body=None):
    """A stand-in for an httpx.Response good enough for the retry loop."""
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json.return_value = json_body if json_body is not None else {}
    r.raise_for_status = MagicMock()
    return r


class TestPostWithRetry:
    async def test_no_retry_on_first_success(self):
        client = LLMClient(api_key="test", provider="openai")
        ok = _resp(200)
        client._client.post = AsyncMock(return_value=ok)
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()) as slept:
            resp = await client._post_with_retry("/x", {}, label="LLM")
        assert resp is ok
        client._client.post.assert_awaited_once()
        slept.assert_not_awaited()
        await client.close()

    async def test_recovers_from_read_timeout(self):
        client = LLMClient(api_key="test", provider="openai")
        ok = _resp(200)
        client._client.post = AsyncMock(side_effect=[httpx.ReadTimeout("stalled"), ok])
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()) as slept:
            resp = await client._post_with_retry("/x", {}, label="LLM")
        assert resp is ok
        assert client._client.post.await_count == 2
        slept.assert_awaited_once()
        await client.close()

    async def test_recovers_from_connect_error(self):
        client = LLMClient(api_key="test", provider="openai")
        ok = _resp(200)
        client._client.post = AsyncMock(side_effect=[httpx.ConnectError("refused"), ok])
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()):
            resp = await client._post_with_retry("/x", {}, label="LLM")
        assert resp is ok
        await client.close()

    async def test_recovers_from_dropped_connection(self):
        client = LLMClient(api_key="test", provider="openai")
        ok = _resp(200)
        client._client.post = AsyncMock(side_effect=[httpx.RemoteProtocolError("peer closed"), ok])
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()):
            resp = await client._post_with_retry("/x", {}, label="LLM")
        assert resp is ok
        await client.close()

    async def test_exhausts_retries_on_persistent_connection_fault(self):
        client = LLMClient(api_key="test", provider="openai")
        client._client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(httpx.ConnectError):
                await client._post_with_retry("/x", {}, label="LLM")
        assert client._client.post.await_count == _MAX_RETRIES + 1
        await client.close()

    async def test_recovers_from_transient_503(self):
        client = LLMClient(api_key="test", provider="openai")
        ok = _resp(200)
        client._client.post = AsyncMock(side_effect=[_resp(503), ok])
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()):
            resp = await client._post_with_retry("/x", {}, label="LLM")
        assert resp is ok
        await client.close()

    async def test_honors_retry_after_header(self):
        client = LLMClient(api_key="test", provider="openai")
        client._client.post = AsyncMock(
            side_effect=[_resp(429, headers={"retry-after": "7"}), _resp(200)]
        )
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()) as slept:
            await client._post_with_retry("/x", {}, label="LLM")
        assert slept.await_args.args[0] >= 7.0
        await client.close()

    async def test_raises_on_non_retryable_status(self):
        client = LLMClient(api_key="test", provider="openai")
        bad = _resp(400)
        bad.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad request", request=MagicMock(), response=MagicMock(status_code=400)
        )
        client._client.post = AsyncMock(return_value=bad)
        with pytest.raises(httpx.HTTPStatusError):
            await client._post_with_retry("/x", {}, label="LLM")
        client._client.post.assert_awaited_once()  # no retry on a hard 4xx
        await client.close()


class TestChatResilience:
    async def test_chat_openai_recovers_from_transient_timeout(self):
        client = LLMClient(api_key="test", provider="openai", default_model="gpt-4o")
        ok = _resp(
            200,
            json_body={
                "choices": [
                    {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
        client._client.post = AsyncMock(side_effect=[httpx.ReadTimeout("stall"), ok])
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()):
            resp = await client.chat([Message(role="user", content="hello")])
        assert resp.message.content == "hi"
        assert client._client.post.await_count == 2
        await client.close()

    async def test_chat_codex_retries_on_dropped_sse_connection(self):
        client = LLMClient(api_key="test", provider="openai-codex", default_model="gpt-5.3-codex")

        async def _fail(_body):
            raise httpx.RemoteProtocolError("peer closed connection")
            yield  # pragma: no cover — makes this an async generator

        async def _ok(_body):
            yield (
                Message(role="assistant", content="done"),
                Usage(prompt_tokens=2, completion_tokens=1),
            )

        client._consume_codex_sse = MagicMock(side_effect=[_fail({}), _ok({})])
        with patch("agentino.core.llm.asyncio.sleep", new=AsyncMock()):
            resp = await client.chat([Message(role="user", content="hi")])
        assert resp.message.content == "done"
        assert client._consume_codex_sse.call_count == 2
        await client.close()
