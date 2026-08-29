"""Anthropic Messages API provider — body building, response parsing."""

from __future__ import annotations

from typing import Any

from agentino.core.message import Message, ToolCall, Usage
from agentino.core.tool import Tool


def build_anthropic_body(
    messages: list[Message],
    tools: list[Tool] | None,
    model: str | None,
    temperature: float,
    default_model: str = "",
) -> dict[str, Any]:
    """Build Anthropic Messages API request body."""
    system_text = ""
    api_messages: list[dict[str, Any]] = []

    for m in messages:
        if m.role == "system":
            system_text = m.content or ""
        elif m.role == "assistant":
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            if m.tool_calls:
                for tc in m.tool_calls:
                    content.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
            api_messages.append({"role": "assistant", "content": content or m.content or ""})
        elif m.role == "tool":
            api_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id or "",
                            "content": m.content or "",
                        }
                    ],
                }
            )
        else:
            if m.images:
                user_content: list[dict[str, Any]] = []
                if m.content:
                    user_content.append({"type": "text", "text": m.content})
                for img in m.images:
                    if img.startswith("data:"):
                        header, b64 = img.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                    else:
                        media_type, b64 = "image/jpeg", img
                    user_content.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        }
                    )
                api_messages.append({"role": "user", "content": user_content})
            else:
                api_messages.append({"role": "user", "content": m.content or ""})

    body: dict[str, Any] = {
        "model": model or default_model or "claude-sonnet-4-20250514",
        "messages": api_messages,
        "max_tokens": 4096,
        "temperature": temperature,
    }
    if system_text:
        body["system"] = system_text
    if tools:
        body["tools"] = [tool_to_anthropic(t) for t in tools]
    return body


def tool_to_anthropic(t: Tool) -> dict[str, Any]:
    """Convert Tool to Anthropic format."""
    return {"name": t.name, "description": t.description, "input_schema": t.parameters}


def parse_anthropic_response(data: dict[str, Any]) -> tuple[Message, Usage, str]:
    """Parse Anthropic response. Returns (message, usage, finish_reason)."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in data.get("content", []):
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append(
                ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {}))
            )

    message = Message(
        role="assistant",
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )
    usage_data = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_data.get("input_tokens", 0),
        completion_tokens=usage_data.get("output_tokens", 0),
    )
    stop_reason = data.get("stop_reason", "end_turn")
    finish_reason = "stop" if stop_reason == "end_turn" else stop_reason
    return message, usage, finish_reason
