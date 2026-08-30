# Hooks — Python callbacks for cross-cutting concerns

`HookManager` lets you observe or block tool calls and lifecycle events
without modifying the tool itself. Use it for audit logging, history
mirroring, metric emission, security scanning — anything that should
happen *around* a tool call but shouldn't be part of the tool's logic.

## When to reach for hooks

You have several tools that share a side effect ("after every chat,
mirror to Supabase", "before every shell command, run this validator")
and you don't want each tool reimplementing it. A hook is one
registration; the framework fans it out across the matching tools.

If only one tool needs the side effect, just put it in the tool. Hooks
shine when the side effect crosses tool boundaries.

## Two flavours

| Flavour | Use when | Cost |
|---|---|---|
| **Python callback** | The hook is in-process logic — DB write, log emit, in-memory metric | One function call. Fast. Default for app-level hooks. |
| **Shell command** | The hook is an external process — bash validator, security scanner, CI integration | Subprocess fork per event. Slow. Use sparingly. |

## Recipe: mirror chat history to Supabase (PostToolUse callback)

The classic problem: every agent chat completion should land in a
`chat_messages` table for auditability. Without hooks, every chat handler
has to remember to insert the row. With a hook:

```python
from agentino.safety.hooks import HookManager
from supabase import Client

async def mirror_to_supabase(context: dict) -> None:
    """PostToolUse hook — append every chat turn to chat_messages."""
    if context.get("tool_name") != "chat":
        return
    sb: Client = context["supabase"]  # passed in via fire(...) context
    sb.table("chat_messages").insert({
        "session_id": context["session_id"],
        "role": context["role"],
        "content": context["content"],
        "tool": context["tool_name"],
        "ts": context["ts"],
    }).execute()

hooks = HookManager()
hooks.register(
    "PostToolUse",
    matcher={"tool_name": "chat"},
    callback=mirror_to_supabase,
)
```

When the agent finishes a `chat` tool call, the framework fires
`PostToolUse` with the tool's result context. The hook gets called
exactly once per chat turn, no matter which agent or codepath
triggered it.

## Recipe: block tool calls based on inspection (PreToolUse callback)

A callback that returns a string containing `REJECTED`, `BLOCKED`, or
`WRONG TOOL` is treated as a blocking outcome — same semantics as a
shell hook returning exit code 2. The model sees the message and re-plans.

```python
async def block_writes_to_root(context: dict) -> str | None:
    if context.get("tool_name") != "write":
        return None
    path = context.get("input", {}).get("path", "")
    if path.startswith("/etc/") or path.startswith("/root/.ssh"):
        return f"REJECTED: writes under {path} are not allowed."
    return None

hooks.register("PreToolUse", matcher={"tool_name": "write"}, callback=block_writes_to_root)
```

## Recipe: shell hook for an external validator (PreToolUse command)

Useful when the validator is already a script, or when it lives in a
different language (e.g. a TypeScript schema validator):

```python
hooks.register(
    "PreToolUse",
    matcher={"tool_name": "shell"},
    command="bash /opt/validators/check_shell_safety.sh",
    timeout=5.0,
)
```

The script receives the context as JSON on stdin. Exit 2 → block; any
other non-zero → warning. stdout/stderr surface to the model.

## Wiring it into your agent

```python
from agentino import Agent
from agentino.safety.hooks import HookManager

hooks = HookManager()
hooks.register("PostToolUse", matcher={"tool_name": "chat"}, callback=mirror_to_supabase)

agent = Agent(
    name="ada",
    soul=...,
    tools=[...],
    hooks=hooks,   # passed at construction; agent fires events at the right moments
)
```

If you're using Runspace's `WorkspaceGateway`, pass the
`HookManager` into the gateway and it propagates to every registered
agent.

## When hooks are not the right tool

- **One-off auth check inside a single tool**: just put the check in
  the tool. A hook adds indirection without saving lines.
- **Mutating the tool's input/output**: hooks are observers/blockers,
  not transformers. If you need to rewrite arguments, wrap the tool
  itself with a decorator.
- **Heavy side effects on every tool call**: each hook is awaited
  inline. Don't run a 30-second job in a hook — schedule it via
  `agentino.scheduler` instead and have the hook just enqueue.

## Events at a glance

`PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SessionStart`,
`Stop`, `StopFailure`, `PreCompact`, `PostCompact`, `SubagentStart`,
`SubagentStop`, `UserPromptSubmit`, `PermissionDenied`, `Notification`.

Most apps use `PreToolUse` (block) and `PostToolUse` (mirror, audit).
The compaction and subagent events are useful when you're building
a long-running multi-turn agent that may compact or fork.
