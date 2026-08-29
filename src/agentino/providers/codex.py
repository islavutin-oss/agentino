"""Codex Responses API provider — SSE streaming, function calls, body building."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agentino.core.message import Message, ToolCall, Usage
from agentino.core.tool import Tool

CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_DEFAULT_MODEL = "gpt-5.4-codex"


async def consume_codex_sse(
    client,
    body: dict[str, Any],
) -> AsyncIterator[str | tuple[Message, Usage]]:
    """Parse Codex SSE stream. Yields text deltas, then final (Message, Usage)."""
    body["stream"] = True
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage = Usage()
    current_fc: dict[str, Any] = {}

    async with client.stream("POST", "/codex/responses", json=body) as resp:
        resp.raise_for_status()
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

            event_type = chunk.get("type", "")

            if event_type == "response.output_text.delta":
                text = chunk.get("delta", "")
                text_parts.append(text)
                yield text
            elif event_type == "response.function_call_arguments.delta":
                if "arguments" not in current_fc:
                    current_fc["arguments"] = ""
                current_fc["arguments"] += chunk.get("delta", "")
            elif event_type == "response.output_item.added":
                item = chunk.get("item", {})
                if item.get("type") == "function_call":
                    current_fc = {
                        "call_id": item.get("call_id", item.get("id", "")),
                        "name": item.get("name", ""),
                        "arguments": "",
                    }
            elif event_type == "response.output_item.done":
                item = chunk.get("item", {})
                if item.get("type") == "function_call":
                    args_str = current_fc.get("arguments", item.get("arguments", "{}"))
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            id=current_fc.get("call_id", item.get("call_id", "")),
                            name=current_fc.get("name", item.get("name", "")),
                            arguments=args,
                        )
                    )
                    current_fc = {}
            elif event_type == "response.completed":
                resp_data = chunk.get("response", {})
                usage_data = resp_data.get("usage", {})
                usage = Usage(
                    prompt_tokens=usage_data.get("input_tokens", 0),
                    completion_tokens=usage_data.get("output_tokens", 0),
                )

    message = Message(
        role="assistant",
        content="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )
    yield (message, usage)


def build_codex_body(
    messages: list[Message],
    tools: list[Tool] | None,
    model: str | None,
    temperature: float,
    default_model: str = "",
) -> dict[str, Any]:
    """Build Codex Responses API request body."""
    instructions = ""
    input_messages: list[dict[str, Any]] = []

    for m in messages:
        if m.role == "system":
            instructions = m.content or ""
        elif m.role == "tool":
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id or "",
                    "output": m.content or "",
                }
            )
        elif m.role == "assistant":
            if m.tool_calls:
                if m.content:
                    input_messages.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": m.content}],
                        }
                    )
                for tc in m.tool_calls:
                    fc: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    }
                    if tc.id.startswith("fc_"):
                        fc["id"] = tc.id
                    input_messages.append(fc)
            else:
                input_messages.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": m.content or ""}],
                    }
                )
        else:
            if m.images:
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"type": "input_text", "text": m.content})
                for img in m.images:
                    parts.append({"type": "input_image", "image_url": img})
                input_messages.append({"role": "user", "content": parts})
            else:
                input_messages.append({"role": "user", "content": m.content or ""})

    body: dict[str, Any] = {
        "model": model or default_model or CODEX_DEFAULT_MODEL,
        "input": input_messages,
        "store": False,
    }
    body["instructions"] = instructions or "You are a helpful assistant."
    # Cap reasoning effort. Codex 5.3 defaults to medium-or-high internal
    # thinking; with heavy prompts (tool-using agents on big workspaces)
    # this burns the entire output budget on reasoning tokens, leaving
    # 0–4 visible tokens — agentino strips the think tags and the loop
    # nudges "you returned empty content" up to max_turns.
    # `low` keeps reasoning available for genuinely hard tasks but stops
    # the runaway-thinking failure mode. Override per-call by setting
    # `AGENTINO_CODEX_REASONING_EFFORT` (low | medium | high | minimal).
    import os as _os

    body["reasoning"] = {"effort": _os.environ.get("AGENTINO_CODEX_REASONING_EFFORT", "low")}
    if tools:
        body["tools"] = [tool_to_codex(t) for t in tools]
    return body


def tool_to_codex(t: Tool) -> dict[str, Any]:
    """Convert Tool to Codex function format."""
    return {
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
    }


def parse_codex_response(data: dict[str, Any]) -> tuple[Message, Usage, str]:
    """Parse Codex response. Returns (message, usage, finish_reason)."""
    output = data.get("output", [])
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in output:
        bt = block.get("type", "")
        if bt == "message":
            for item in block.get("content", []):
                if item.get("type") == "output_text":
                    text_parts.append(item.get("text", ""))
        elif bt == "output_text":
            text_parts.append(block.get("text", ""))
        elif bt == "function_call":
            args_str = block.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=block.get("call_id", block.get("id", "")),
                    name=block.get("name", ""),
                    arguments=args,
                )
            )

    if not text_parts and data.get("output_text"):
        text_parts.append(data["output_text"])

    message = Message(
        role="assistant",
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )
    usage_data = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_data.get("input_tokens", usage_data.get("prompt_tokens", 0)),
        completion_tokens=usage_data.get("output_tokens", usage_data.get("completion_tokens", 0)),
    )
    status = data.get("status", "completed")
    finish_reason = "tool_calls" if tool_calls else ("stop" if status == "completed" else status)
    return message, usage, finish_reason
