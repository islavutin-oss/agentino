"""Message types for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Attachment:
    """Dynamic content attached to a message (file deltas, agent listings, etc.).

    Ported from Claude Code's attachment system.
    """

    type: str  # "file_delta", "agent_listing", "skill_content", "tool_discovery"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    timestamp: float | None = None
    # Base64-encoded images (data:image/jpeg;base64,...) for multimodal input
    images: list[str] | None = None
    # Attachments — dynamic content injected into messages (#14)
    attachments: list[Attachment] | None = None

    def to_api(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        msg: dict[str, Any] = {"role": self.role}
        if self.images and self.role == "user":
            # Multimodal: list of text + image_url parts
            parts: list[dict[str, Any]] = []
            if self.content:
                parts.append({"type": "text", "text": self.content})
            for img in self.images:
                parts.append({"type": "image_url", "image_url": {"url": img}})
            msg["content"] = parts
        elif self.content is not None:
            # Append attachment content to message
            if self.attachments:
                attachment_text = "\n\n".join(
                    f"[{a.type}]\n{a.content}" for a in self.attachments if a.content
                )
                msg["content"] = (
                    f"{self.content}\n\n{attachment_text}" if self.content else attachment_text
                )
            else:
                msg["content"] = self.content
        # Codex API requires content field even on tool-call messages
        if "content" not in msg and self.tool_calls:
            msg["content"] = ""
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _json_dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    def to_jsonl(self) -> dict[str, Any]:
        """Convert to JSONL storage format."""
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "args": tc.arguments} for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.timestamp:
            d["ts"] = self.timestamp
        return d

    @classmethod
    def from_jsonl(cls, data: dict[str, Any]) -> Message:
        """Restore from JSONL storage format."""
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("args", {}))
                for tc in data["tool_calls"]
            ]
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            timestamp=data.get("ts"),
        )


@dataclass
class Usage:
    """Token usage from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __repr__(self) -> str:
        return f"Usage(prompt={self.prompt_tokens}, completion={self.completion_tokens})"


class EventType(str, Enum):
    """Event types emitted during agent streaming."""

    TEXT = "text"  # alias of TEXT_DELTA, kept for backward compat
    TEXT_DELTA = "text"  # one chunk of streamed assistant text
    TOOLCALL_START = "toolcall_start"  # LLM starts emitting a tool call (id + name known)
    TOOLCALL_DELTA = "toolcall_delta"  # streamed fragment of tool-call arguments
    TOOLCALL_END = "toolcall_end"  # tool call fully assembled, args parsed
    TOOL_START = "tool_start"  # tool execution begins
    TOOL_RESULT = "tool_result"
    LLM_RESPONSE = "llm_response"
    DONE = "done"
    ERROR = "error"
    FALLBACK = "fallback"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGE_FAIL = "stage_fail"
    STAGE_SKIP = "stage_skip"


@dataclass
class Event:
    """Streaming event from the agent loop."""

    type: EventType
    data: Any = None
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    usage: Usage | None = None


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj)
