"""Read RSS / Atom feeds — industry news from any public feed URL."""

from __future__ import annotations

import json
import logging
import re

from agentino.core.tool import tool

log = logging.getLogger(__name__)


@tool(is_read_only=True)
async def read_rss(url: str, limit: int = 10) -> str:
    """Read recent articles from an RSS/Atom feed. Returns JSON list of {title, link, description, date}. Use for industry news (restaurants, hospitality, wine, tech) or any public feed — tenant-agnostic.

    Args:
        url: RSS/Atom feed URL (e.g. 'https://example.com/feed/')
        limit: Max articles to return (default 10)
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; agentino/1.0)"}
            )
            resp.raise_for_status()
            xml = resp.text
    except Exception as e:
        return f"Error fetching RSS: {e}"

    items = re.findall(r"<item[^>]*>(.*?)</item>", xml, re.DOTALL)
    if not items:
        items = re.findall(r"<entry[^>]*>(.*?)</entry>", xml, re.DOTALL)

    results = []
    for item in items[:limit]:
        title = re.search(r"<title[^>]*>(.*?)</title>", item, re.DOTALL)
        link = re.search(r'<link[^>]*href="([^"]+)"', item) or re.search(
            r"<link[^>]*>(.*?)</link>", item, re.DOTALL
        )
        desc = re.search(r"<description[^>]*>(.*?)</description>", item, re.DOTALL) or re.search(
            r"<summary[^>]*>(.*?)</summary>", item, re.DOTALL
        )
        pub_date = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", item) or re.search(
            r"<published[^>]*>(.*?)</published>", item
        )

        title_text = re.sub(r"<[^>]+>", "", title.group(1)).strip() if title else ""
        link_text = (link.group(1) if link else "").strip()
        desc_text = re.sub(r"<[^>]+>", "", desc.group(1)).strip()[:200] if desc else ""
        date_text = pub_date.group(1).strip() if pub_date else ""

        if title_text:
            results.append(
                {
                    "title": title_text,
                    "link": link_text,
                    "description": desc_text,
                    "date": date_text,
                }
            )

    if not results:
        return f"No articles found at {url}"
    return json.dumps(results, indent=2, ensure_ascii=False)
