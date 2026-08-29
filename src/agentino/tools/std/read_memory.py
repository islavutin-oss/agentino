"""read_memory — fetch the full body of a stored memory."""

from __future__ import annotations

from agentino.core.tool import tool

from ._agent_memory import read_memory as _read_memory


@tool(is_read_only=True)
async def read_memory(slug: str) -> str:
    """Fetch the full prose body of a stored memory by slug. Use when the description in your prompt's memory list isn't enough and you need the original quotes/context.

    Args:
        slug: The memory id (kebab-case, e.g. "prefers-window-seat").
    """
    try:
        m = _read_memory(slug)
    except ValueError as e:
        return f"read_memory: {e}"
    except Exception as e:
        return f"read_memory failed: {e}"
    if not m:
        return f"No memory with slug '{slug}'."
    parts = [
        f"# {m['slug']}",
        f"_{m.get('kind') or 'fact'} · created {m.get('created') or 'unknown'} · updated {m.get('updated') or 'unknown'}_",
        "",
        f"**{m['description']}**",
    ]
    if m.get("body"):
        parts.append("")
        parts.append(m["body"])
    return "\n".join(parts)
