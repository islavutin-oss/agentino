"""Async-native context — inject runtime data into tools without globals.

Uses contextvars.ContextVar which is natively async-safe: each asyncio.Task
inherits parent context, mutations are isolated per-task. No thread-local
hacks, no snapshots, no run_coroutine_threadsafe bridging needed.

Usage:
    # In your app (before running the agent):
    from agentino import context
    context.set_context(tenant_id="abc", sender_id="+123")

    # In your async tool:
    from agentino.core.context import get_context
    tenant_id = get_context("tenant_id")

    # For per-request isolation in async frameworks:
    token = context.set_context(tenant_id="abc")
    try:
        reply = await agent.run("hello")
    finally:
        context.reset(token)
"""

from __future__ import annotations

import contextvars
from typing import Any

_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "agentino_ctx",
    default={},
)


def set_context(**kwargs: Any) -> contextvars.Token:
    """Set context values. Returns a token for reset().

    Async-safe: each asyncio.Task inherits parent context,
    but set_context() creates an isolated copy for the current task.
    """
    current = _ctx.get().copy()
    current.update(kwargs)
    return _ctx.set(current)


def get_context(key: str, default: Any = None) -> Any:
    """Get a context value."""
    return _ctx.get().get(key, default)


def clear_context() -> contextvars.Token:
    """Clear all context values."""
    return _ctx.set({})


def reset(token: contextvars.Token) -> None:
    """Reset context to the state before a set_context() call."""
    _ctx.reset(token)
