"""Global state management — session tracking, skill invocations, telemetry.

Singleton with lazy initialization. Leaf module — nothing imports it back.
Ported from Claude Code's bootstrap/state.ts pattern.

Usage:
    from agentino.core.state import get_session_id, record_skill, get_state

    sid = get_session_id()
    record_skill("commit", "/path/to/skill.md")
    state = get_state()
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """Global session state — created once per process."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_session_id: str = ""  # For session lineage tracking
    start_time: float = field(default_factory=time.time)
    original_cwd: str = field(default_factory=os.getcwd)
    cwd: str = field(default_factory=os.getcwd)

    # Usage tracking per model
    model_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    # Skill invocations (for post-compact re-injection)
    invoked_skills: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Slow operations log (for diagnostics)
    slow_operations: list[dict[str, Any]] = field(default_factory=list)

    # Feature flags (cached, may be stale)
    _feature_cache: dict[str, Any] = field(default_factory=dict)


# Singleton
_state: SessionState | None = None


def get_state() -> SessionState:
    """Get or create the global session state."""
    global _state
    if _state is None:
        _state = SessionState()
    return _state


def reset_state() -> None:
    """Reset state (for testing)."""
    global _state
    _state = None


def get_session_id() -> str:
    """Get current session ID."""
    return get_state().session_id


def record_skill(name: str, path: str, content: str = "") -> None:
    """Record that a skill was invoked (for post-compact re-injection)."""
    get_state().invoked_skills[name] = {
        "path": path,
        "content": content[:5000],  # Cap stored content
        "invoked_at": time.time(),
    }


def get_invoked_skills() -> dict[str, dict[str, Any]]:
    """Get skills invoked this session."""
    return get_state().invoked_skills


def record_model_usage(model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """Accumulate token usage per model."""
    state = get_state()
    if model not in state.model_usage:
        state.model_usage[model] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    state.model_usage[model]["prompt_tokens"] += prompt_tokens
    state.model_usage[model]["completion_tokens"] += completion_tokens
    state.model_usage[model]["calls"] += 1


def record_slow_operation(operation: str, duration_ms: int) -> None:
    """Record a slow operation for diagnostics."""
    state = get_state()
    state.slow_operations.append(
        {
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }
    )
    # Keep only last 50
    if len(state.slow_operations) > 50:
        state.slow_operations = state.slow_operations[-50:]


def get_feature(name: str, default: Any = None) -> Any:
    """Get a cached feature flag value (may be stale)."""
    return get_state()._feature_cache.get(name, default)


def set_feature(name: str, value: Any) -> None:
    """Set a feature flag value."""
    get_state()._feature_cache[name] = value


def elapsed_seconds() -> float:
    """Seconds since session start."""
    return time.time() - get_state().start_time
