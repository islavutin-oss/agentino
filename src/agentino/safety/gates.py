"""Workflow gates — code-enforced preconditions for tool calls.

LLMs skip prompt-based instructions ~15% of the time. Gates make skipping
impossible by rejecting tool calls whose preconditions haven't been met.

Usage:
    from agentino.safety.gates import GateManager, GateRule

    rules = [GateRule(gate="security_checked", tools=["send_email"],
                      message="Run security_scan first.")]
    gm = GateManager(rules)
    gm.mark("security_checked")           # tool marks after completing check
    assert gm.check("send_email") is None # passes

Stored in agentino context so it's per-request and async-safe:
    context.set_context(_gate_manager=GateManager(rules=...))
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateRule:
    """A gate precondition for one or more tools.

    gate:      name of the gate that must be marked before calling the tools
    tools:     tool names that require this gate
    message:   rejection message returned when gate is not satisfied
    condition: only enforce this rule if this other gate is also marked
               (e.g. only require security_checked if external_content_read)
    """

    gate: str
    tools: list[str]
    message: str
    condition: str = ""


class GateManager:
    """Per-request gate state. Create one per agent invocation, store in context."""

    def __init__(self, rules: list[GateRule] | None = None):
        self._rules = list(rules or [])
        self._marked: set[str] = set()
        self._tracked: dict[str, str] = {}

    def mark(self, gate: str) -> None:
        """Mark a gate as satisfied (called by tools after completing checks)."""
        self._marked.add(gate)

    def is_marked(self, gate: str) -> bool:
        """Check if a gate has been marked."""
        return gate in self._marked

    def track(self, key: str, value: str) -> None:
        """Track arbitrary key-value state (e.g. current_email_id)."""
        self._tracked[key] = value

    def get_tracked(self, key: str, default: str = "") -> str:
        """Retrieve tracked state."""
        return self._tracked.get(key, default)

    def check(self, tool_name: str) -> str | None:
        """Check if a tool is allowed. Returns rejection message or None."""
        for rule in self._rules:
            if tool_name not in rule.tools:
                continue
            # If rule has a condition, only enforce when that condition gate is marked
            if rule.condition and rule.condition not in self._marked:
                continue
            # Gate must be marked
            if rule.gate not in self._marked:
                return rule.message
        return None

    def reset(self) -> None:
        """Reset all gate state (call at pipeline/request start)."""
        self._marked.clear()
        self._tracked.clear()

    @property
    def marked_gates(self) -> set[str]:
        """Return copy of currently marked gates (for debugging/logging)."""
        return set(self._marked)
