"""fetch_web_data — agentic web fetcher with budget + LLM quality judge.

Runs web_search, then crawl_page on top URLs. After each step asks an LLM
judge "does what we have so far answer the user's question?". Loops until
the judge says yes OR the step budget is spent. Agents call ONE tool and
get real data.
"""

from __future__ import annotations

import json
import logging
import os
import re

from agentino.core.tool import tool
from agentino.tools.std._llm_env import llm_api_key, llm_base_url

log = logging.getLogger(__name__)


def _extract_urls(search_output: str, limit: int = 5) -> list[str]:
    """Pull the first `limit` distinct URLs from web_search output.

    Handles DuckDuckGo redirect URLs (`//duckduckgo.com/l/?uddg=<encoded>`)
    by unwrapping them to the real target. Also decodes percent-encoded URLs.
    """
    from urllib.parse import parse_qs, unquote, urlparse

    seen: set[str] = set()
    out: list[str] = []

    # 1. DDG redirect pattern: //duckduckgo.com/l/?uddg=<encoded>&...
    for match in re.finditer(r"duckduckgo\.com/l/\?([^\s\"'<>]*)", search_output):
        qs = parse_qs(match.group(1).replace("&amp;", "&"))
        if "uddg" in qs and qs["uddg"]:
            real = unquote(qs["uddg"][0]).rstrip(".,);:]")
            if real.startswith(("http://", "https://")) and real not in seen:
                seen.add(real)
                out.append(real)
                if len(out) >= limit:
                    return out

    # 2. Plain https:// URLs — skip DDG redirects (already handled) + tracker domains
    SKIP_HOSTS = {"duckduckgo.com", "bing.com", "google.com"}
    for match in re.finditer(r"https?://[^\s\"'<>]+", search_output):
        u = match.group(0).rstrip(".,);:]")
        try:
            host = urlparse(u).netloc.lower()
        except Exception:
            continue
        if host in SKIP_HOSTS or any(host.endswith("." + h) for h in SKIP_HOSTS):
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= limit:
            break
    return out


async def _llm_judge(user_question: str, evidence: str) -> dict:
    """Ask an LLM: does `evidence` answer `user_question`?

    Returns {answered: bool, missing: str}. Falls back to a heuristic
    (contains numeric data) if LLM is unavailable.
    """
    # Heuristic fallback — detect numeric data (numbers + common units).
    has_numbers = bool(
        re.search(
            r"\d+\.?\d*\s*(?:°[cf]|°|%|€|\$|£|km/?h|mph|mm|cm|m/s|kg|g|ml|bar)",
            evidence,
            re.IGNORECASE,
        )
    )

    api_key = llm_api_key()
    if not api_key:
        return {"answered": has_numbers, "missing": "(no LLM — heuristic fallback)"}

    try:
        import httpx

        base_url = llm_base_url()
        model = os.environ.get("CRAWLER_MODEL", "gpt-5.4-codex")
        prompt = (
            f"User's question: {user_question}\n\n"
            f"Evidence collected so far:\n{evidence[:3000]}\n\n"
            "Does the evidence contain CONCRETE data that fully answers the question? "
            "Respond with ONLY valid JSON: "
            '{"answered": true|false, "missing": "<short description of what\'s still needed, or empty>"}'
        )
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        # Extract JSON (LLM may wrap in prose)
        m = re.search(r"\{.*?\}", content, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            return {
                "answered": bool(parsed.get("answered")),
                "missing": str(parsed.get("missing", "")),
            }
    except Exception as e:
        log.warning("_llm_judge error: %s — falling back to heuristic", e)
    return {"answered": has_numbers, "missing": "(LLM judge failed — heuristic fallback)"}


@tool(is_read_only=True)
async def fetch_web_data(query: str, user_question: str, step_budget: int = 10) -> str:
    """Fetch SPECIFIC DATA from the web — temperature, prices, exchange rates, scores, schedules, live stats, etc. Auto-chains: search → static-page extraction → JS-rendered browser, looping until an LLM judge confirms the user's question is answered or the step budget is spent. Use this whenever the user asks for concrete real-world numbers/facts that aren't in internal business tools.

    PICK THIS over `web_search` when the user wants a VALUE (e.g. "what's the weather", "what's the EUR/USD rate", "what time is sunset", "what's competitor X's price"). Use `web_search` only when the user wants headlines, URLs, or general reference (e.g. "find articles about restaurant tech").

    Args:
        query: Search query, e.g. "paphos weather hourly today" or "vivino pricing 2026".
        user_question: The original user question in natural language. Used by an internal LLM judge to verify evidence is complete and to drive page extraction. Example: "current temperature (°C), wind (km/h), precipitation for 18-23h tonight in Paphos".
        step_budget: Max tool steps (1 search + crawls + 1 reserved for browser). Default 10 — enough for tough JS-heavy queries (multi-day forecasts, hidden prices).
    """
    # Private modules — fetch_web_data is the single public entrypoint
    from ._crawl_page import crawl_page
    from ._web_search import web_search

    try:
        from ._browse_web import browse_web

        _HAS_BROWSE = True
    except ImportError:
        _HAS_BROWSE = False

    steps = 0
    evidence_parts: list[str] = []

    # Step 1: web_search
    try:
        search_raw = await web_search.fn(query)
        steps += 1
    except Exception as e:
        return f"web_search failed: {e}"

    evidence_parts.append(f"--- web_search ({query}) ---\n{search_raw}")
    evidence = "\n\n".join(evidence_parts)

    verdict = await _llm_judge(user_question, evidence)
    if verdict["answered"]:
        return f"[answered in {steps} step(s)]\n\n{evidence[:3000]}"

    # Step 2+: crawl top URLs until answered or budget spent.
    # Reserve 1 slot at the end for tier-3 browse_web escalation when available,
    # so JS-heavy pages (forecasts, prices) get their last-resort shot.
    reserve = 1 if _HAS_BROWSE else 0
    urls = _extract_urls(search_raw, limit=max(1, step_budget - steps - reserve))
    for url in urls:
        if steps >= step_budget - reserve:
            break
        try:
            page = await crawl_page.fn(url, extract=user_question)
        except Exception as e:
            evidence_parts.append(f"--- crawl {url} FAILED: {e} ---")
            continue
        steps += 1
        evidence_parts.append(f"--- crawl {url} ---\n{page[:2000]}")
        evidence = "\n\n".join(evidence_parts)

        verdict = await _llm_judge(user_question, evidence)
        if verdict["answered"]:
            return f"[answered in {steps} step(s), source: {url}]\n\n{page[:3000]}"

    # Tier 3: escalate to JS-rendering browser if still not answered and budget left
    if _HAS_BROWSE and steps < step_budget and urls:
        try:
            browse_result = await browse_web.fn(
                task=user_question,
                url=urls[0],
                max_steps=8,
            )
            steps += 1
            evidence_parts.append(f"--- browse_web {urls[0]} ---\n{browse_result[:2000]}")
            evidence = "\n\n".join(evidence_parts)
            verdict = await _llm_judge(user_question, evidence)
            if verdict["answered"]:
                return f"[answered in {steps} step(s), browser source: {urls[0]}]\n\n{browse_result[:3000]}"
        except Exception as e:
            log.warning("browse_web escalation failed: %s", e)

    # Budget exhausted
    return (
        f"[budget of {step_budget} steps exhausted; missing: {verdict.get('missing', 'unknown')}]\n\n"
        f"Collected evidence:\n{evidence[:3000]}"
    )
