"""translate_text — clean translation between supported languages.

Used by:
  - The Knowledge Base UI (auto-fill RU/EL when the owner edits the
    EN source).
  - Any agent that needs deterministic, prompt-controlled
    translation rather than relying on the LLM's casual rendering
    of a non-source-language phrase mid-conversation.

Calls the same Router the agents use (gpt-5.3-codex), with a
tight system prompt that preserves proper nouns, URLs, prices,
markdown — the things naive translation usually breaks.

Two surfaces:

  1. `translate_text(...)` — plain async helper. Imported directly
     by the KB module.
  2. `@tool` wrapper at the bottom — agentino auto-discovers this
     and any agent that needs explicit translation gets it for free.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import httpx

from agentino.core.tool import tool

log = logging.getLogger(__name__)


# Supported languages this tool will translate between. ISO 639-1.
# Adding a new one is a single entry — no other code change.
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "ru": "Russian",
    "el": "Greek",
}

from agentino.tools.std._llm_env import (  # noqa: E402
    DEFAULT_MODEL,
    llm_api_key,
    llm_base_url,
)


async def translate_text(
    text: str,
    *,
    source_lang: str,
    target_lang: str,
    keywords: Iterable[str] = (),
    model: str = DEFAULT_MODEL,
) -> str:
    """One-shot translation, returning the translated string only.

    Args:
        text: Source-language text. Markdown, URLs, prices preserved.
        source_lang / target_lang: ISO 639-1 codes (`en`, `ru`, `el`).
        keywords: Optional source-language keyword hints — used by
            the Knowledge Base path to keep retrieval-relevant
            terminology consistent across languages. Empty for
            plain translation.
        model: Override default model. Useful if a caller wants to
            A/B-test or use a cheaper model for low-stakes content.

    Returns the translated text, or empty string if translation
    failed (network, missing API key, …). Caller decides how to
    surface the failure — KB shows "translation failed" badge;
    agents would just retry or skip.
    """
    if not text:
        return ""
    if source_lang == target_lang:
        return text

    src = LANG_NAMES.get(source_lang, source_lang)
    tgt = LANG_NAMES.get(target_lang, target_lang)
    key = llm_api_key()
    if not key:
        log.warning("[translate_text] no Router key in env — returning empty")
        return ""

    system = (
        f"You are a precise translator from {src} to {tgt} for a "
        f"hospitality-business knowledge base. Translate the text "
        f"below preserving:\n"
        f"  - tone (warm, hospitality, factual)\n"
        f"  - proper nouns (brand names, place names, IBANs, etc.) "
        f"untranslated\n"
        f"  - URLs, phone numbers, currency amounts verbatim\n"
        f"  - markdown formatting (links, bullets, line breaks)\n"
        f"Output ONLY the translation. No preamble, no quotes, no "
        f"explanation. Do not wrap in code fences."
    )
    user = text
    if keywords:
        user = (
            f"{text}\n\n"
            f"(For reference, the source-language keywords are: "
            f"{', '.join(keywords)} — keep equivalent meaning in {tgt}.)"
        )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(
                f"{llm_base_url().rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return (out or "").strip()
    except Exception as e:
        log.warning("[translate_text] %s→%s failed: %s", source_lang, target_lang, e)
        return ""


# ── Agent-facing tool wrapper ──────────────────────────────────────────


@tool(is_read_only=True)
async def translate(
    text: str,
    target_lang: str,
    source_lang: str = "en",
) -> dict:
    """Translate text between supported languages (en, ru, el).

    Use this when you need a clean, terminology-preserving
    translation — proper nouns, URLs, prices, and markdown are
    preserved verbatim. Do NOT use for casual mid-conversation
    paraphrasing; that's what your own multilingual capability is
    for. Use this when the OUTPUT must be a faithful rendering
    suitable for storing in a knowledge base or sending to a
    customer in a different language.

    Args:
        text: Source-language text to translate.
        target_lang: ISO 639-1 code of the target language
            (`en`, `ru`, `el`).
        source_lang: ISO 639-1 code of the source language. Defaults
            to `en` (English).

    Returns:
        Dict with `text` (the translation) and `ok` flag. Empty
        translation + ok=False indicates a transient failure —
        retry or skip.
    """
    if target_lang not in LANG_NAMES:
        return {
            "ok": False,
            "text": "",
            "error": f"unsupported target_lang: {target_lang!r}; allowed: {sorted(LANG_NAMES)}",
        }
    if source_lang not in LANG_NAMES:
        return {
            "ok": False,
            "text": "",
            "error": f"unsupported source_lang: {source_lang!r}; allowed: {sorted(LANG_NAMES)}",
        }
    out = await translate_text(
        text,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    return {"ok": bool(out), "text": out}
