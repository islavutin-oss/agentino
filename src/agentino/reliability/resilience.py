"""Resilience utilities — async retry, session repair, truncation, think filter, token estimation, compaction."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from agentino.core.message import Message

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# 1. Retry with exponential backoff (async-native)
# ---------------------------------------------------------------------------


async def retry_with_backoff(
    fn: Callable[[], Coroutine],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.2,
    retry_on: tuple[int, ...] = (429, 500, 502, 503),
) -> Any:
    """Call async fn(), retrying on transient HTTP errors with exponential backoff.

    Honors the Retry-After header when present.
    """
    import httpx

    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in retry_on:
                raise
            last_exc = e
            delay = min(initial_delay * (2**attempt), max_delay)
            delay *= 1 + random.uniform(-jitter, jitter)
            retry_after = e.response.headers.get("retry-after")
            if retry_after:
                try:
                    delay = max(delay, min(float(retry_after), max_delay))
                except ValueError:
                    pass
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            last_exc = e
            delay = min(initial_delay * (2**attempt), max_delay)
            delay *= 1 + random.uniform(-jitter, jitter)
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Session repair — fix malformed message histories before sending to LLM
# ---------------------------------------------------------------------------


def repair_messages(messages: list[Message]) -> list[Message]:
    """Fix malformed message histories that would crash LLM APIs.

    4 phases:
    1. Drop orphaned tool results (no matching tool_call)
    2. Insert synthetic error for unmatched tool calls
    3. Remove empty assistant messages
    4. Merge consecutive same-role user messages
    """
    if not messages:
        return messages

    # Phase 1: Collect all tool_call IDs from assistant messages
    tool_call_ids: set[str] = set()
    for m in messages:
        if m.tool_calls:
            for tc in m.tool_calls:
                tool_call_ids.add(tc.id)

    # Drop orphaned tool results
    messages = [
        m
        for m in messages
        if m.role != "tool" or (m.tool_call_id and m.tool_call_id in tool_call_ids)
    ]

    # Phase 2: Find unmatched tool calls and insert synthetic errors
    result_ids: set[str] = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}
    synthetic: list[Message] = []
    for m in messages:
        if m.tool_calls:
            for tc in m.tool_calls:
                if tc.id not in result_ids:
                    synthetic.append(
                        Message(
                            role="tool",
                            content="[Error: tool execution was interrupted]",
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

    if synthetic:
        repaired: list[Message] = []
        for m in messages:
            repaired.append(m)
            if m.tool_calls:
                for s in synthetic:
                    if s.tool_call_id and any(tc.id == s.tool_call_id for tc in m.tool_calls):
                        repaired.append(s)
        messages = repaired

    # Phase 3: Remove empty assistant messages (no content, no tool calls)
    messages = [
        m for m in messages if not (m.role == "assistant" and not m.content and not m.tool_calls)
    ]

    # Phase 4: Merge consecutive same-role user messages
    merged: list[Message] = []
    for m in messages:
        if merged and m.role == "user" and merged[-1].role == "user":
            prev = merged[-1]
            prev_content = prev.content or ""
            new_content = m.content or ""
            merged[-1] = Message(
                role="user",
                content=f"{prev_content}\n{new_content}".strip(),
                timestamp=m.timestamp or prev.timestamp,
            )
        else:
            merged.append(m)

    return merged


# ---------------------------------------------------------------------------
# 3. Tool result truncation
# ---------------------------------------------------------------------------


def truncate_result(text: str, max_chars: int = 4000) -> str:
    """Truncate tool output to max_chars, breaking at line boundaries."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind("\n")
    if cut < max_chars // 2:
        cut = max_chars
    omitted = len(text) - cut
    return text[:cut] + f"\n\n[...truncated, {omitted} chars omitted]"


# ---------------------------------------------------------------------------
# 4. Think filter — strip <think>...</think> blocks
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output (DeepSeek-R1, Qwen3, etc)."""
    return _THINK_RE.sub("", text).strip()


class ThinkFilterStream:
    """Stateful filter for stripping <think> blocks from streaming chunks.

    Buffers partial tags across chunks so `<thi` + `nk>content</think>` is handled.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False

    def feed(self, chunk: str) -> str:
        """Feed a chunk, return the filtered output."""
        self._buffer += chunk
        output = ""

        while self._buffer:
            if self._inside_think:
                end = self._buffer.find("</think>")
                if end == -1:
                    if self._buffer.endswith(("</", "</t", "</th", "</thi", "</thin", "</think")):
                        break
                    self._buffer = ""
                    break
                self._buffer = self._buffer[end + 8 :]
                self._inside_think = False
            else:
                start = self._buffer.find("<think>")
                if start == -1:
                    for partial in ("<", "<t", "<th", "<thi", "<thin", "<think"):
                        if self._buffer.endswith(partial):
                            output += self._buffer[: -len(partial)]
                            self._buffer = partial
                            return output
                    output += self._buffer
                    self._buffer = ""
                else:
                    output += self._buffer[:start]
                    self._buffer = self._buffer[start + 7 :]
                    self._inside_think = True

        return output


# ---------------------------------------------------------------------------
# 5. Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(messages: list[Message]) -> int:
    """Approximate token count without API calls.

    Uses ~3 chars per token (conservative — code/JSON is denser than prose)
    plus a 20% safety margin to avoid context overflows.
    """
    total = 0
    for m in messages:
        if m.content:
            total += len(m.content) // 3
        if m.tool_calls:
            for tc in m.tool_calls:
                total += len(json.dumps(tc.arguments)) // 3
                total += 20  # overhead per tool call
    total += len(messages) * 4  # message framing overhead
    return int(total * 1.2)  # 20% safety margin


# Compaction moved to compaction.py — re-export for backward compat
from agentino.reliability.compaction import (  # noqa: E402,F401
    _extract_recent_files,
    compact_history,
)
