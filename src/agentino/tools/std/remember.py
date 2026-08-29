"""remember — store a durable fact about the current user."""

from __future__ import annotations

from agentino.core.tool import tool

from ._agent_memory import write_memory


@tool
async def remember(slug: str, description: str, body: str = "", kind: str | None = None) -> str:
    """Store a durable fact about the current user. Loaded back into your prompt at the start of every future session, so use it for things you'd want to know next time.

    GOOD candidates: user preferences (wines, dishes, report format, language),
    decisions they approved (budgets, promos, vendors), constraints (allergies,
    supplier exclusivity), recurring requests, personal context that affects
    tone.

    BAD candidates: single-question lookups, data you can re-derive from
    tools, greetings/small-talk, sensitive data the user didn't ask you to
    record.

    Args:
        slug: Short kebab-case identifier (e.g. "prefers-window-seat"). Used
              as filename + reference handle. Lowercase letters/digits/hyphens
              only, max 64 chars.
        description: One-line summary (≤200 chars). This is what you see on
              every future session — keep it self-contained.
        body: Optional longer prose (quotes, context, exact user wording).
              Loaded only via read_memory(slug).
        kind: Optional category — one of: preference, decision, constraint,
              routine, personal, recommendation.

    Returns confirmation with the created slug + timestamp.
    """
    try:
        result = write_memory(slug=slug, description=description, body=body, kind=kind)
    except (ValueError, FileNotFoundError) as e:
        return f"remember failed: {e}"
    except Exception as e:
        return f"remember failed: {e}"
    return f"Remembered: **{result['slug']}** — {description}"
