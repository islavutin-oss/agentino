"""forget — archive a memory that's no longer accurate."""

from __future__ import annotations

from agentino.core.tool import tool

from ._agent_memory import forget_memory


@tool
async def forget(slug: str) -> str:
    """Archive a memory that's no longer accurate or useful. Soft-delete: the file is moved to an archive subdir, not destroyed. Use when a fact has been replaced (prefer update_memory for that), is wrong, or no longer applies.

    Args:
        slug: The memory id to archive.
    """
    try:
        ok = forget_memory(slug)
    except ValueError as e:
        return f"forget: {e}"
    except Exception as e:
        return f"forget failed: {e}"
    if not ok:
        return f"No memory with slug '{slug}' (already archived?)."
    return f"Forgot **{slug}** — moved to archive."
