"""Hook system — event-driven pre/post hooks for tool calls and lifecycle events.

13 event types with exit code semantics:
- 0: Success (output shown or ignored per event)
- 2: Blocking error (tool call blocked, output shown to model)
- Other: Non-blocking warning (shown to user only)

Two hook flavours, register either:

    # Python callback (in-process, fast — preferred for app-level hooks
    # like audit log writes, history mirroring, metric emission):
    async def mirror_to_db(context):
        await db.insert("chat_messages", context)
    hooks.register("PostToolUse", matcher={"tool_name": "chat"}, callback=mirror_to_db)

    # Shell command (subprocess — useful for ops/CI integrations,
    # external linters, security validators):
    hooks.register(
        "PreToolUse", matcher={"tool_name": "shell"},
        command="bash /path/to/validator.sh",
    )

Both shapes deliver context as a dict. Shell hooks receive it as JSON on
stdin; callbacks receive it as a kwarg. Callbacks may be sync or async.
A callback that returns a string containing REJECTED / BLOCKED / WRONG
TOOL is treated as a blocking outcome (same semantics as exit code 2).

    # In agent loop:
    result = await hooks.fire("PreToolUse", {"tool_name": "shell", "input": {"command": "rm -rf /"}})
    if result.blocked:
        return result.message  # Block the tool call
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# All supported hook events (from Claude Code)
HOOK_EVENTS = {
    "PreToolUse",  # Before tool execution — can block
    "PostToolUse",  # After tool execution
    "PostToolUseFailure",  # After tool error
    "SessionStart",  # Session begins
    "Stop",  # Agent stops
    "StopFailure",  # Agent failed to stop cleanly
    "PreCompact",  # Before context compaction — exit 2 blocks
    "PostCompact",  # After compaction
    "SubagentStart",  # Worker/fork spawned
    "SubagentStop",  # Worker/fork finished
    "UserPromptSubmit",  # User sends message
    "PermissionDenied",  # Permission check blocked a tool
    "Notification",  # Generic notification
}


@dataclass
class HookMatcher:
    """Matches events by field values."""

    match_field: str = ""  # e.g. "tool_name", "agent_type"
    match_values: list[str] | None = None  # Match any of these

    def matches(self, context: dict[str, Any]) -> bool:
        """Check if this filter matches the given context."""
        if not self.match_field:
            return True  # No matcher = match all
        actual = str(context.get(self.match_field, ""))
        if not self.match_values:
            return True
        return actual in self.match_values


@dataclass
class HookDef:
    """A registered hook."""

    event: str
    matcher: HookMatcher = field(default_factory=HookMatcher)
    command: str = ""  # Shell command to run
    callback: Callable | None = None  # Python callback (alternative to command)
    timeout: float = 10.0


@dataclass
class HookResult:
    """Result from firing a hook."""

    blocked: bool = False  # Exit code 2 → blocked
    message: str = ""  # Output to show (stderr on block, stdout on success)
    exit_code: int = 0
    # Borrow #3 (pi): a PreToolUse callback may rewrite arguments before execution
    # by returning a dict like {"arguments": {...}}. None means "do not rewrite".
    arguments_override: dict[str, Any] | None = None
    # Borrow #3 (pi): a PostToolUse callback may override the tool result
    # by returning a dict like {"result": "..."}. None means "do not override".
    result_override: str | None = None


class HookManager:
    """Manages event hooks with exit code semantics."""

    def __init__(self):
        self._hooks: dict[str, list[HookDef]] = {}

    def register(
        self,
        event: str,
        command: str = "",
        callback: Callable | None = None,
        matcher: dict[str, Any] | HookMatcher | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Register a hook for an event.

        Args:
            event: Event name (PreToolUse, PostToolUse, etc.)
            command: Shell command to run (receives context as JSON on stdin)
            callback: Python callback (alternative to command)
            matcher: Filter which events to match (e.g. {"tool_name": ["shell"]})
            timeout: Max seconds for hook execution
        """
        if event not in HOOK_EVENTS:
            raise ValueError(
                f"Unknown hook event: {event}. Valid: {', '.join(sorted(HOOK_EVENTS))}"
            )

        if isinstance(matcher, dict):
            # Handle shorthand: {"tool_name": "shell"} or {"tool_name": ["shell", "write_file"]}
            first_key = list(matcher.keys())[0] if matcher else ""
            first_val = list(matcher.values())[0] if matcher else []
            if isinstance(first_val, str):
                first_val = [first_val]
            m = HookMatcher(match_field=first_key, match_values=first_val)
        else:
            m = matcher or HookMatcher()

        hook = HookDef(event=event, matcher=m, command=command, callback=callback, timeout=timeout)
        self._hooks.setdefault(event, []).append(hook)

    async def fire(self, event: str, context: dict[str, Any] | None = None) -> HookResult:
        """Fire all hooks for an event. Returns combined result.

        Exit code semantics:
        - 0: Success
        - 2: Blocking error (tool blocked, compaction blocked)
        - Other: Warning (non-blocking)
        """
        context = context or {}
        hooks = self._hooks.get(event, [])
        if not hooks:
            return HookResult()

        # Borrow #3: accumulate non-blocking overrides across hooks. Each hook sees the
        # latest mutated arguments dict (later hooks compose on earlier ones).
        accum = HookResult()
        working_args = (
            context.get("arguments") if isinstance(context.get("arguments"), dict) else None
        )
        for hook in hooks:
            if not hook.matcher.matches(context):
                continue

            result = await self._execute_hook(hook, context)
            if result.blocked:
                return result  # First blocking hook wins
            if result.arguments_override is not None:
                accum.arguments_override = result.arguments_override
                # Propagate the rewrite into context so later hooks see it
                if working_args is not None:
                    context["arguments"] = result.arguments_override
            if result.result_override is not None:
                accum.result_override = result.result_override
                if "result" in context:
                    context["result"] = result.result_override

        return accum

    async def _execute_hook(self, hook: HookDef, context: dict[str, Any]) -> HookResult:
        """Execute a single hook."""
        import json

        if hook.callback:
            try:
                result = hook.callback(context)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, str) and result:
                    # Callbacks that return a string with REJECTED/BLOCKED → block the tool
                    is_blocking = any(
                        kw in result.upper() for kw in ("REJECTED", "BLOCKED", "WRONG TOOL")
                    )
                    return HookResult(blocked=is_blocking, message=result)
                # Borrow #3: dict-shaped return → arg/result override or structured block
                if isinstance(result, dict):
                    return HookResult(
                        blocked=bool(result.get("block")),
                        message=str(result.get("reason", "")) if result.get("block") else "",
                        arguments_override=result.get("arguments")
                        if isinstance(result.get("arguments"), dict)
                        else None,
                        result_override=result.get("result")
                        if isinstance(result.get("result"), str)
                        else None,
                    )
                return HookResult()
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("Hook error: %s", e)
                return HookResult(blocked=False, message=f"Hook error: {e}")

        if hook.command:
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        hook.command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=hook.timeout,
                )
                stdout, stderr = await proc.communicate(json.dumps(context).encode())
                exit_code = proc.returncode or 0

                if exit_code == 2:
                    return HookResult(
                        blocked=True,
                        message=stderr.decode().strip() or "Hook blocked this action",
                        exit_code=2,
                    )
                return HookResult(
                    blocked=False,
                    message=stdout.decode().strip(),
                    exit_code=exit_code,
                )
            except asyncio.TimeoutError:
                return HookResult(blocked=False, message=f"Hook timed out after {hook.timeout}s")
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("Hook error: %s", e)
                return HookResult(blocked=False, message=f"Hook error: {e}")

        return HookResult()

    def has_hooks(self, event: str) -> bool:
        """Check if any hooks are registered for an event."""
        return bool(self._hooks.get(event))

    def list_hooks(self) -> dict[str, int]:
        """List registered hook counts per event."""
        return {event: len(hooks) for event, hooks in self._hooks.items()}
