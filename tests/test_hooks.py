"""Tests for HookManager — Python callback flavour.

Shell-hook coverage is harder to assert hermetically; the callback path
is the one app-level consumers (acme, lemana, etc.) actually use,
and that's what we pin here.
"""

from __future__ import annotations

import asyncio

import pytest

from agentino.safety.hooks import HOOK_EVENTS, HookManager


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_register_unknown_event_raises():
    hooks = HookManager()
    with pytest.raises(ValueError, match="Unknown hook event"):
        hooks.register("FakeEvent", callback=lambda c: None)


def test_callback_fires_on_matching_event():
    hooks = HookManager()
    fired_with: list[dict] = []

    def collect(ctx: dict) -> None:
        fired_with.append(ctx)

    hooks.register("PostToolUse", matcher={"tool_name": "chat"}, callback=collect)
    _run(hooks.fire("PostToolUse", {"tool_name": "chat", "content": "hi"}))

    assert len(fired_with) == 1
    assert fired_with[0]["content"] == "hi"


def test_callback_skipped_on_non_matching_event():
    hooks = HookManager()
    fired = []
    hooks.register("PostToolUse", matcher={"tool_name": "chat"}, callback=lambda c: fired.append(c))

    _run(hooks.fire("PostToolUse", {"tool_name": "shell"}))

    assert fired == []


def test_async_callback_awaited():
    hooks = HookManager()
    fired = []

    async def acollect(ctx: dict) -> None:
        await asyncio.sleep(0)
        fired.append(ctx)

    hooks.register("PreToolUse", callback=acollect)
    _run(hooks.fire("PreToolUse", {"tool_name": "x"}))

    assert len(fired) == 1


def test_callback_returning_REJECTED_blocks():
    hooks = HookManager()

    def reject(ctx: dict) -> str:
        return "REJECTED: writes to /etc are forbidden"

    hooks.register("PreToolUse", matcher={"tool_name": "write"}, callback=reject)
    result = _run(hooks.fire("PreToolUse", {"tool_name": "write", "path": "/etc/passwd"}))

    assert result.blocked is True
    assert "REJECTED" in result.message


def test_callback_returning_plain_string_does_not_block():
    """Strings that don't contain blocking keywords are warnings, not blocks."""
    hooks = HookManager()

    def warn(ctx: dict) -> str:
        return "FYI: tool was called"

    hooks.register("PostToolUse", callback=warn)
    result = _run(hooks.fire("PostToolUse", {"tool_name": "x"}))

    assert result.blocked is False


def test_first_blocking_hook_wins():
    """If two hooks match and one blocks, fire returns at the first block."""
    hooks = HookManager()
    second_called = []

    hooks.register("PreToolUse", callback=lambda c: "REJECTED: nope")
    hooks.register("PreToolUse", callback=lambda c: second_called.append(1))

    result = _run(hooks.fire("PreToolUse", {"tool_name": "x"}))

    assert result.blocked is True
    assert second_called == []  # second hook never ran


def test_non_blocking_callbacks_all_fire():
    """For PostToolUse-style fan-out, every matching callback should run."""
    hooks = HookManager()
    a_called, b_called = [], []

    hooks.register("PostToolUse", callback=lambda c: a_called.append(c["x"]))
    hooks.register("PostToolUse", callback=lambda c: b_called.append(c["x"]))

    _run(hooks.fire("PostToolUse", {"tool_name": "t", "x": 42}))

    assert a_called == [42]
    assert b_called == [42]


def test_callback_exception_is_swallowed_not_raised():
    """A buggy hook shouldn't crash the agent loop."""
    hooks = HookManager()

    def boom(ctx: dict) -> None:
        raise RuntimeError("oops")

    hooks.register("PostToolUse", callback=boom)
    # Should not raise:
    result = _run(hooks.fire("PostToolUse", {"tool_name": "x"}))
    # And should not be marked as blocking from the buggy hook:
    assert result.blocked is False


def test_has_hooks_and_list_hooks():
    hooks = HookManager()
    assert not hooks.has_hooks("PostToolUse")
    hooks.register("PostToolUse", callback=lambda c: None)
    hooks.register("PostToolUse", callback=lambda c: None)
    hooks.register("PreToolUse", callback=lambda c: None)
    assert hooks.has_hooks("PostToolUse")
    assert hooks.list_hooks() == {"PostToolUse": 2, "PreToolUse": 1}


def test_matcher_dict_with_list_value():
    """matcher={'tool_name': ['a', 'b']} should match either."""
    hooks = HookManager()
    fired = []
    hooks.register(
        "PreToolUse",
        matcher={"tool_name": ["read", "write"]},
        callback=lambda c: fired.append(c["tool_name"]),
    )

    _run(hooks.fire("PreToolUse", {"tool_name": "read"}))
    _run(hooks.fire("PreToolUse", {"tool_name": "write"}))
    _run(hooks.fire("PreToolUse", {"tool_name": "delete"}))

    assert fired == ["read", "write"]


def test_no_matcher_matches_all():
    """A hook with no matcher fires on every event."""
    hooks = HookManager()
    fired = []
    hooks.register("PostToolUse", callback=lambda c: fired.append(c["tool_name"]))

    _run(hooks.fire("PostToolUse", {"tool_name": "a"}))
    _run(hooks.fire("PostToolUse", {"tool_name": "b"}))

    assert fired == ["a", "b"]


def test_HOOK_EVENTS_exported():
    """The HOOK_EVENTS set is part of the public API; consumers may iterate it."""
    assert "PreToolUse" in HOOK_EVENTS
    assert "PostToolUse" in HOOK_EVENTS
    assert len(HOOK_EVENTS) >= 13


# ----------------------------------------------------------------------
# Borrow #3 — before_tool_call / after_tool_call mutation semantics
# ----------------------------------------------------------------------


def test_pre_hook_can_rewrite_arguments():
    """Returning {'arguments': {...}} from a PreToolUse callback rewrites args."""
    hooks = HookManager()

    def inject_tenant(ctx):
        new_args = {**ctx["arguments"], "tenant_id": "acme"}
        return {"arguments": new_args}

    hooks.register("PreToolUse", callback=inject_tenant)
    result = _run(hooks.fire("PreToolUse", {"tool_name": "x", "arguments": {"q": "hi"}}))

    assert result.arguments_override == {"q": "hi", "tenant_id": "acme"}
    assert not result.blocked


def test_pre_hooks_compose_in_order():
    """A second pre-hook sees the first hook's rewrites."""
    hooks = HookManager()

    def first(ctx):
        return {"arguments": {**ctx["arguments"], "lang": "en"}}

    def second(ctx):
        # Should observe lang already injected by first
        assert ctx["arguments"].get("lang") == "en"
        return {"arguments": {**ctx["arguments"], "tenant_id": "p1"}}

    hooks.register("PreToolUse", callback=first)
    hooks.register("PreToolUse", callback=second)
    result = _run(hooks.fire("PreToolUse", {"tool_name": "x", "arguments": {"q": "go"}}))

    assert result.arguments_override == {"q": "go", "lang": "en", "tenant_id": "p1"}


def test_pre_hook_dict_block_short_circuits():
    """Returning {'block': True, 'reason': '...'} blocks like the string form."""
    hooks = HookManager()
    hooks.register("PreToolUse", callback=lambda ctx: {"block": True, "reason": "not allowed"})
    result = _run(hooks.fire("PreToolUse", {"tool_name": "x", "arguments": {}}))

    assert result.blocked
    assert "not allowed" in result.message


def test_post_hook_can_override_result():
    """Returning {'result': '...'} from a PostToolUse callback rewrites the result."""
    hooks = HookManager()

    def redact(ctx):
        if "iban" in ctx["result"].lower():
            return {"result": "[REDACTED]"}

    hooks.register("PostToolUse", callback=redact)
    result = _run(
        hooks.fire(
            "PostToolUse",
            {
                "tool_name": "fetch_invoice",
                "arguments": {},
                "result": "Customer IBAN: CY07009005",
            },
        )
    )

    assert result.result_override == "[REDACTED]"


def test_post_hook_no_override_when_callback_returns_none():
    """A normal observe-only callback (returns None) does not set result_override."""
    hooks = HookManager()
    seen = []
    hooks.register("PostToolUse", callback=lambda ctx: seen.append(ctx["result"]))
    result = _run(
        hooks.fire(
            "PostToolUse",
            {
                "tool_name": "x",
                "arguments": {},
                "result": "ok",
            },
        )
    )

    assert result.result_override is None
    assert seen == ["ok"]


def test_pre_hook_rewrite_applied_in_agent_loop():
    """End-to-end: pre-hook rewrites tenant_id; tool sees it; post-hook redacts result."""
    import asyncio

    from agentino.core import context as ctx_module
    from agentino.core.agent import Agent
    from agentino.core.message import ToolCall
    from agentino.core.tool import tool as tool_decorator

    captured = {}

    @tool_decorator
    def whoami(tenant_id: str = "") -> str:
        captured["tenant_id"] = tenant_id
        return f"hello tenant {tenant_id} secret=42"

    hooks = HookManager()
    hooks.register(
        "PreToolUse", callback=lambda ctx: {"arguments": {**ctx["arguments"], "tenant_id": "acme"}}
    )
    hooks.register(
        "PostToolUse",
        callback=lambda ctx: {"result": ctx["result"].replace("secret=42", "[redacted]")},
    )

    a = Agent(model="x", api_key="x", tools=[whoami], base_url="http://localhost")
    call = ToolCall(id="c1", name="whoami", arguments={})

    # Run inside a context that has the hook manager bound
    async def go():
        token = ctx_module.set_context(_hook_manager=hooks)
        try:
            return await a._execute_one_tool(call)
        finally:
            ctx_module.reset(token)

    out = asyncio.new_event_loop().run_until_complete(go())
    assert captured["tenant_id"] == "acme"
    assert out == "hello tenant acme [redacted]"
