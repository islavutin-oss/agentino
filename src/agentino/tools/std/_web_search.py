"""Web search tool — wraps agentino's built-in Brave/DuckDuckGo search."""

import asyncio

from agentino.core.tool import tool


@tool(is_read_only=True)
async def web_search(query: str) -> str:
    """Lightweight web search — returns titles, URLs, and short snippets. Use for: headlines, finding URLs, quick reference lookups (e.g. 'restaurant tech news 2026', 'AccuWeather Paphos URL'). DO NOT use this when the user asks for SPECIFIC DATA like temperature, price, score, schedule — snippets rarely contain real numbers. For specific data, call `fetch_web_data` instead, which auto-chains search + page extraction until it finds the actual values.

    Args:
        query: Search query string.
    """
    from agentino.builtin_tools import web_search as _builtin

    return await asyncio.to_thread(_builtin.fn, query)
