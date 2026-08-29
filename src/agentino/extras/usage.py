"""Usage tracking — per-session, per-model token stats persisted to JSONL."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentino.core.message import Event, Usage


@dataclass
class UsageEntry:
    """A single usage record."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    timestamp: float
    session_id: str = ""
    agent_name: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_jsonl(self) -> dict[str, Any]:
        """Serialize usage entry to JSONL-compatible dict."""
        d: dict[str, Any] = {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ts": self.timestamp,
        }
        if self.session_id:
            d["session_id"] = self.session_id
        if self.agent_name:
            d["agent_name"] = self.agent_name
        return d

    @classmethod
    def from_jsonl(cls, data: dict[str, Any]) -> UsageEntry:
        """Deserialize usage entry from JSONL dict."""
        return cls(
            model=data.get("model", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            timestamp=data.get("ts", 0.0),
            session_id=data.get("session_id", ""),
            agent_name=data.get("agent_name", ""),
        )


# ---------------------------------------------------------------------------
# Known pricing ($ per 1M tokens) — updated as of 2025
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-20250506": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate cost in USD. Returns None if model pricing unknown."""
    pricing = _PRICING.get(model)
    if not pricing:
        # Try prefix match (e.g. "gpt-4o-2024-08-06" matches "gpt-4o")
        for key, val in _PRICING.items():
            if model.startswith(key):
                pricing = val
                break
    if not pricing:
        return None
    input_cost = (prompt_tokens / 1_000_000) * pricing[0]
    output_cost = (completion_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


class UsageTracker:
    """Track and persist token usage to JSONL.

    Usage:
        tracker = UsageTracker("./usage.jsonl")

        # Option 1: Pass as on_event callback
        agent = Agent(model="gpt-4o", on_event=tracker.on_event)
        agent.run("Hello")

        # Option 2: Record manually
        tracker.record(model="gpt-4o", usage=Usage(prompt_tokens=100, completion_tokens=50))

        # Query
        print(tracker.total)              # Usage(prompt=100, completion=50)
        print(tracker.by_model)           # {"gpt-4o": Usage(...)}
        print(tracker.cost_estimate)      # 0.00075
        print(tracker.summary())          # formatted string
    """

    def __init__(
        self,
        path: str | Path | None = None,
        session_id: str = "",
        agent_name: str = "",
    ):
        self.path = Path(path) if path else None
        self.session_id = session_id
        self.agent_name = agent_name
        self._entries: list[UsageEntry] = []
        self._current_model: str = ""

        # Load existing entries from disk
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        """Load entries from JSONL file."""
        if not self.path:
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._entries.append(UsageEntry.from_jsonl(data))
                except (json.JSONDecodeError, KeyError):
                    continue

    def _append_to_disk(self, entry: UsageEntry) -> None:
        """Append a single entry to the JSONL file."""
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_jsonl()) + "\n")

    def record(self, model: str, usage: Usage, session_id: str = "", agent_name: str = "") -> None:
        """Record a usage entry."""
        entry = UsageEntry(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            timestamp=time.time(),
            session_id=session_id or self.session_id,
            agent_name=agent_name or self.agent_name,
        )
        self._entries.append(entry)
        self._append_to_disk(entry)

    def on_event(self, event: Event) -> None:
        """Event callback for use with Agent(on_event=tracker.on_event).

        Captures usage from llm_response events.
        """
        if event.type == "llm_response" and event.usage:
            # Try to get model from event data (the assistant message)
            model = self._current_model or "unknown"
            self.record(model=model, usage=event.usage)

    def bind(self, model: str) -> UsageTracker:
        """Set the current model name for on_event tracking. Returns self for chaining."""
        self._current_model = model
        return self

    @property
    def total(self) -> Usage:
        """Total usage across all entries."""
        total = Usage()
        for e in self._entries:
            total = total + Usage(
                prompt_tokens=e.prompt_tokens, completion_tokens=e.completion_tokens
            )
        return total

    @property
    def by_model(self) -> dict[str, Usage]:
        """Usage broken down by model."""
        result: dict[str, Usage] = {}
        for e in self._entries:
            if e.model not in result:
                result[e.model] = Usage()
            result[e.model] = result[e.model] + Usage(
                prompt_tokens=e.prompt_tokens,
                completion_tokens=e.completion_tokens,
            )
        return result

    @property
    def cost_estimate(self) -> float:
        """Total estimated cost in USD across all entries."""
        total = 0.0
        for e in self._entries:
            cost = _estimate_cost(e.model, e.prompt_tokens, e.completion_tokens)
            if cost is not None:
                total += cost
        return total

    @property
    def entries(self) -> list[UsageEntry]:
        """All recorded entries."""
        return list(self._entries)

    def summary(self) -> str:
        """Human-readable usage summary."""
        lines = ["Usage Summary", "=" * 40]
        total = self.total
        lines.append(
            f"Total: {total.prompt_tokens:,} prompt + {total.completion_tokens:,} completion = {total.total_tokens:,} tokens"
        )

        cost = self.cost_estimate
        if cost > 0:
            lines.append(f"Estimated cost: ${cost:.4f}")

        by_model = self.by_model
        if by_model:
            lines.append("")
            lines.append("By model:")
            for model, usage in sorted(by_model.items()):
                model_cost = _estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
                cost_str = f" (${model_cost:.4f})" if model_cost is not None else ""
                lines.append(f"  {model}: {usage.total_tokens:,} tokens{cost_str}")

        lines.append(f"\nEntries: {len(self._entries)}")
        return "\n".join(lines)
