"""LLM client — async multi-provider support (OpenAI, Anthropic, any OpenAI-compatible)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from agentino.core.message import Event, EventType, Message, ToolCall, Usage
from agentino.core.tool import Tool

logger = logging.getLogger(__name__)

# Retry config for transient faults. Two fault classes are retried:
#   - HTTP status codes below (server told us it's transient)
#   - connection-level failures: httpx.TransportError covers timeouts
#     (connect/read/write/pool), network errors (connection reset/dropped)
#     and protocol errors (RemoteProtocolError). A stalled or dropped
#     connection to the provider is transient, not fatal — retry it instead
#     of letting the exception kill the agent turn.
_RETRY_STATUSES = {404, 429, 500, 502, 503}  # 404 for OpenRouter free model unavailability
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds, exponential base


def _retry_delay(attempt: int) -> float:
    """Exponential backoff with jitter.

    Jitter spreads out retries so many parallel agent trials that hit the
    same provider hiccup don't all wake and re-stampede in lockstep.
    """
    base = _RETRY_BASE_DELAY * (2**attempt)
    return base + random.uniform(0.0, base * 0.25)


def _build_timeout(read_timeout: float) -> httpx.Timeout:
    """Granular per-phase timeout.

    A blanket timeout applies the (necessarily generous) generation budget
    to the TCP/TLS connect phase too — so a stalled handshake blocks for the
    full read budget before failing. Splitting it out lets a stuck connect
    fail fast and be retried, while a legitimately slow generation still
    gets the full read window.
    """
    return httpx.Timeout(
        connect=min(15.0, read_timeout),
        read=read_timeout,
        write=min(30.0, read_timeout),
        pool=min(15.0, read_timeout),
    )


@dataclass
class LLMResponse:
    """Response from a single LLM call."""

    message: Message
    usage: Usage
    finish_reason: str = "stop"


CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_DEFAULT_MODEL = "gpt-5.4-codex"


def _detect_provider(base_url: str) -> str:
    """Detect provider from base URL."""
    if "anthropic" in base_url:
        return "anthropic"
    # Default to openai-codex (all calls go through Router → Codex)
    return "openai-codex"


def _is_setup_token(api_key: str) -> bool:
    """Check if the key is an Anthropic OAuth setup-token (from `claude setup-token`)."""
    return api_key.startswith("sk-ant-oat")


class LLMClient:
    """Async multi-provider LLM client.

    Supported providers:
    - openai-codex: Codex via Router or direct (/codex/responses SSE)
    - anthropic: Claude via Router or direct (Messages API)

    Configuration (in priority order):
    1. Constructor args: base_url, api_key, provider
    2. Environment: AGENTINO_BASE_URL, AGENTINO_API_KEY, AGENTINO_PROVIDER
    3. Key only: ANTHROPIC_API_KEY, ANTHROPIC_SETUP_TOKEN, then OPENAI_API_KEY
    4. Auth files: ~/.agentino/auth.json, ~/.codex/auth.json

    OPENAI_BASE_URL is not read — only AGENTINO_BASE_URL selects the endpoint,
    and it defaults to https://api.openai.com/v1. This docstring used to claim
    otherwise, which meant setting OPENAI_BASE_URL silently did nothing.

    All methods are async-native.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
        provider: str | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("AGENTINO_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")

        self.api_key = api_key or self._resolve_api_key()
        self.default_model = default_model

        # Resolve provider: arg → AGENTINO_PROVIDER → auto-detect
        env_provider = os.getenv("AGENTINO_PROVIDER")
        self.provider = provider or env_provider or _detect_provider(self.base_url)
        self._is_oauth = _is_setup_token(self.api_key)

        # Auto-detect Anthropic from key prefix
        if self.api_key.startswith("sk-ant-") and self.provider != "anthropic":
            self.provider = "anthropic"
            if "anthropic" not in self.base_url:
                self.base_url = "https://api.anthropic.com"

        # Auto-detect Codex from JWT token (only if no explicit provider set)
        if (
            not env_provider
            and not provider
            and self.provider == "openai-codex"
            and self.api_key
            and not self.api_key.startswith(("sk-", "pk_"))
            and self._is_codex_token(self.api_key)
        ):
            self.base_url = CODEX_BASE_URL

        # Default models per provider
        if not self.default_model:
            if self.provider == "openai-codex":
                self.default_model = CODEX_DEFAULT_MODEL
            elif self.provider == "anthropic":
                self.default_model = "claude-sonnet-4-20250514"
            else:
                self.default_model = CODEX_DEFAULT_MODEL

        if not self.api_key:
            import logging

            logging.getLogger(__name__).warning(
                "No API key found. Set AGENTINO_API_KEY, OPENAI_API_KEY, or run `agentino login`."
            )

        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=_build_timeout(timeout),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    @staticmethod
    def _is_codex_token(api_key: str) -> bool:
        """Check if the key is a Codex JWT (from ChatGPT subscription)."""
        from agentino.safety.auth import decode_jwt_claims

        claims = decode_jwt_claims(api_key)
        if not claims:
            return False
        aud = claims.get("aud", [])
        return any("openai.com" in a for a in aud) if isinstance(aud, list) else False

    @staticmethod
    def _resolve_api_key() -> str:
        """Resolve API key from env vars or stored OAuth credentials."""
        for var in ("AGENTINO_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_SETUP_TOKEN"):
            val = os.getenv(var)
            if val:
                return val
        try:
            from agentino.safety.auth import get_api_key

            token = get_api_key("openai")
            if token:
                return token
            token = get_api_key("anthropic")
            if token:
                return token
        except Exception:
            pass
        return ""

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            if self._is_oauth:
                h["Authorization"] = f"Bearer {self.api_key}"
                h["anthropic-version"] = "2023-06-01"
                h["anthropic-beta"] = "oauth-2025-04-20,claude-code-20250219"
            else:
                if self.api_key:
                    h["x-api-key"] = self.api_key
                h["anthropic-version"] = "2023-06-01"
        else:
            if self.api_key:
                h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _post_with_retry(self, path: str, body: dict, *, label: str) -> httpx.Response:
        """POST a JSON body, retrying transient HTTP statuses AND connection
        faults (timeouts, dropped/reset connections, protocol errors).

        The connection-fault arm is the important one: without it a single
        stalled or dropped request to the provider raises straight out of
        the chat call and aborts the whole agent turn. Such faults are
        transient — retry them with jittered backoff like a 503.
        """
        last_exc: httpx.TransportError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self._client.post(path, json=body)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    logger.error(
                        "%s connection error %s — retries exhausted (%d attempts)",
                        label,
                        type(exc).__name__,
                        _MAX_RETRIES + 1,
                    )
                    raise
                delay = _retry_delay(attempt)
                logger.warning(
                    "%s connection error %s (attempt %d/%d), retrying in %.1fs",
                    label,
                    type(exc).__name__,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                delay = _retry_delay(attempt)
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                logger.warning(
                    "%s HTTP %d (attempt %d/%d), retrying in %.1fs",
                    label,
                    resp.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        # Unreachable: the final attempt always returns or raises above.
        raise last_exc if last_exc else RuntimeError(f"{label}: retry loop exited")

    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        """Non-streaming chat completion."""
        if self.provider == "anthropic":
            return await self._chat_anthropic(
                messages, tools, model, temperature, tool_choice=tool_choice
            )
        if self.provider == "openai-codex":
            return await self._chat_codex(
                messages, tools, model, temperature, tool_choice=tool_choice
            )
        body = self._build_body(messages, tools, model, temperature, stream=False)
        if tool_choice and tools:
            body["tool_choice"] = tool_choice
        resp = await self._post_with_retry("/chat/completions", body, label="LLM")
        return self._parse_response(resp.json())

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        tool_choice: str | None = None,
    ) -> AsyncIterator[Event]:
        """Streaming chat completion. Yields Events."""
        # Route to provider-specific streaming
        if self.provider == "openai-codex":
            async for event in self._chat_stream_codex(messages, tools, model, temperature):
                yield event
            return

        body = self._build_body(messages, tools, model, temperature, stream=True)
        if tool_choice and tools:
            body["tool_choice"] = tool_choice

        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            message = Message(role="assistant")
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            usage = Usage()

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if "usage" in chunk and chunk["usage"]:
                    u = chunk["usage"]
                    usage = Usage(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                    )

                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                if "content" in delta and delta["content"]:
                    text = delta["content"]
                    if message.content is None:
                        message.content = ""
                    message.content += text
                    yield Event(type=EventType.TEXT, data=text)

                if "tool_calls" in delta:
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        is_new = idx not in tool_calls_acc
                        if is_new:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.get("id", ""),
                                "name": "",
                                "arguments": "",
                                "started_emitted": False,
                            }
                        acc = tool_calls_acc[idx]
                        if tc_delta.get("id"):
                            acc["id"] = tc_delta["id"]
                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        # Emit TOOLCALL_START once both id and name known
                        if not acc["started_emitted"] and acc["id"] and acc["name"]:
                            acc["started_emitted"] = True
                            yield Event(
                                type=EventType.TOOLCALL_START,
                                name=acc["name"],
                                data={"id": acc["id"], "index": idx},
                            )
                        if fn.get("arguments"):
                            acc["arguments"] += fn["arguments"]
                            yield Event(
                                type=EventType.TOOLCALL_DELTA,
                                name=acc["name"],
                                data={"id": acc["id"], "index": idx, "delta": fn["arguments"]},
                            )

            if tool_calls_acc:
                message.tool_calls = []
                for idx in sorted(tool_calls_acc):
                    acc = tool_calls_acc[idx]
                    try:
                        args = json.loads(acc["arguments"]) if acc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    message.tool_calls.append(
                        ToolCall(id=acc["id"], name=acc["name"], arguments=args)
                    )
                    yield Event(
                        type=EventType.TOOLCALL_END,
                        name=acc["name"],
                        data={"id": acc["id"], "index": idx},
                        args=args,
                    )

            yield Event(type=EventType.LLM_RESPONSE, usage=usage, data=message)

    # ------------------------------------------------------------------
    # OpenAI Codex (ChatGPT subscription) — Responses API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Codex Responses API (delegated to providers/codex.py)
    # ------------------------------------------------------------------

    async def _consume_codex_sse(
        self, body: dict[str, Any]
    ) -> AsyncIterator[str | tuple[Message, Usage]]:
        from agentino.providers.codex import consume_codex_sse

        async for item in consume_codex_sse(self._client, body):
            yield item

    async def _chat_stream_codex(self, messages, tools, model, temperature) -> AsyncIterator[Event]:
        from agentino.providers.codex import build_codex_body

        body = build_codex_body(messages, tools, model, temperature, self.default_model)
        async for item in self._consume_codex_sse(body):
            if isinstance(item, str):
                yield Event(type="text", data=item)
            else:
                message, usage = item
                yield Event(type=EventType.LLM_RESPONSE, usage=usage, data=message)

    async def _chat_codex(
        self, messages, tools, model, temperature, tool_choice=None
    ) -> LLMResponse:
        from agentino.providers.codex import build_codex_body

        body = build_codex_body(messages, tools, model, temperature, self.default_model)
        if tool_choice and tools:
            body["tool_choice"] = tool_choice
        for attempt in range(_MAX_RETRIES + 1):
            try:
                message = Usage()
                usage = Usage()
                async for item in self._consume_codex_sse(body):
                    if isinstance(item, tuple):
                        message, usage = item
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                    delay = _retry_delay(attempt)
                    logger.warning(
                        "Codex HTTP %d (attempt %d/%d), retrying in %.1fs",
                        e.response.status_code,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.TransportError as e:
                # Stalled / dropped SSE connection — transient, retry it.
                if attempt < _MAX_RETRIES:
                    delay = _retry_delay(attempt)
                    logger.warning(
                        "Codex connection error %s (attempt %d/%d), retrying in %.1fs",
                        type(e).__name__,
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Codex connection error %s — retries exhausted (%d attempts)",
                    type(e).__name__,
                    _MAX_RETRIES + 1,
                )
                raise
        finish_reason = (
            "tool_calls" if (isinstance(message, Message) and message.tool_calls) else "stop"
        )
        if not isinstance(message, Message):
            message = Message(role="assistant")
        return LLMResponse(message=message, usage=usage, finish_reason=finish_reason)

    def _build_codex_body(self, messages, tools, model, temperature):
        from agentino.providers.codex import build_codex_body

        return build_codex_body(messages, tools, model, temperature, self.default_model)

    @staticmethod
    def _tool_to_codex(t):
        from agentino.providers.codex import tool_to_codex

        return tool_to_codex(t)

    def _parse_codex_response(self, data):
        from agentino.providers.codex import parse_codex_response

        msg, usage, fr = parse_codex_response(data)
        return LLMResponse(message=msg, usage=usage, finish_reason=fr)

    # ------------------------------------------------------------------
    # Anthropic Messages API (delegated to providers/anthropic.py)
    # ------------------------------------------------------------------

    async def _chat_anthropic(
        self, messages, tools, model, temperature, tool_choice=None
    ) -> LLMResponse:
        from agentino.providers.anthropic import build_anthropic_body

        body = build_anthropic_body(messages, tools, model, temperature, self.default_model)
        if tool_choice and tools:
            body["tool_choice"] = {"type": "any"} if tool_choice == "required" else {"type": "auto"}
        resp = await self._post_with_retry("/v1/messages", body, label="Anthropic")
        from agentino.providers.anthropic import parse_anthropic_response

        msg, usage, fr = parse_anthropic_response(resp.json())
        return LLMResponse(message=msg, usage=usage, finish_reason=fr)

    def _build_anthropic_body(self, messages, tools, model, temperature):
        from agentino.providers.anthropic import build_anthropic_body

        return build_anthropic_body(messages, tools, model, temperature, self.default_model)

    @staticmethod
    def _tool_to_anthropic(t):
        from agentino.providers.anthropic import tool_to_anthropic

        return tool_to_anthropic(t)

    def _parse_anthropic_response(self, data):
        from agentino.providers.anthropic import parse_anthropic_response

        msg, usage, fr = parse_anthropic_response(data)
        return LLMResponse(message=msg, usage=usage, finish_reason=fr)

    # ------------------------------------------------------------------
    # OpenAI Chat Completions API
    # ------------------------------------------------------------------

    def _build_body(
        self,
        messages: list[Message],
        tools: list[Tool] | None,
        model: str | None,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.default_model or "gpt-4o",
            "messages": [m.to_api() for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            body["tools"] = [t.schema for t in tools]
        if stream:
            body["stream_options"] = {"include_usage": True}
        # Disable reasoning/thinking for models that support it (e.g. Qwen on OpenRouter)
        # Reasoning adds overhead and breaks tool calling flow
        if "openrouter" in self.base_url:
            body["reasoning"] = {"effort": "none"}
        return body

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        choice = data["choices"][0]
        msg_data = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        tool_calls = None
        if "tool_calls" in msg_data and msg_data["tool_calls"]:
            tool_calls = []
            for tc in msg_data["tool_calls"]:
                fn = tc["function"]
                try:
                    args = json.loads(fn["arguments"]) if fn["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=args))

        message = Message(
            role="assistant",
            content=msg_data.get("content"),
            tool_calls=tool_calls,
        )

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return LLMResponse(message=message, usage=usage, finish_reason=finish_reason)
