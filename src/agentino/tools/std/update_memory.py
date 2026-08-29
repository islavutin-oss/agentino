"""update_memory — patch an existing memory in place."""

from __future__ import annotations

from agentino.core.tool import tool

from ._agent_memory import update_memory as _update_memory


@tool
async def update_memory(
    slug: str, description: str | None = None, body: str | None = None, kind: str | None = None
) -> str:
    """Patch an existing memory in place. Use when a fact CHANGES (preference shifts, budget raised, allergy resolved). Pass only the fields you want to change — others are preserved.

    Args:
        slug: The memory to update (must already exist).
        description: New one-line summary (optional).
        body: New full prose (optional).
        kind: New category (optional).
    """
    try:
        result = _update_memory(slug=slug, description=description, body=body, kind=kind)
    except FileNotFoundError as e:
        return f"update_memory: {e}"
    except ValueError as e:
        return f"update_memory: {e}"
    except Exception as e:
        return f"update_memory failed: {e}"
    return f"Updated **{result['slug']}** at {result['updated']}"
