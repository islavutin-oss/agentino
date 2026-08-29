"""Input sanitization — clean LLM-corrupted tool arguments.

LLMs occasionally corrupt file paths and text with JSON brackets, unicode
garbage, zero-width characters, and other artifacts. This module provides
deterministic cleanup that runs before tool execution.

Usage:
    from agentino.safety.sanitize import clean_path, normalize_text, sanitize_tool_args

    path, was_corrupted = clean_path("inbox/msg.txt|{extra}")
    # path = "inbox/msg.txt", was_corrupted = True

    clean_args = sanitize_tool_args({"path": "foo|bar"}, path_params=["path"])
"""

from __future__ import annotations

import re
import unicodedata

# Characters that indicate LLM corruption in file paths
_PATH_GARBAGE = re.compile(r"[{}()\[\]|<>\\;`~@#$%^&*+=]")


def clean_path(path: str) -> tuple[str, bool]:
    """Auto-fix LLM-corrupted paths.

    Strips unicode garbage, JSON brackets, pipe chars, etc.
    Returns (clean_path, was_corrupted).
    """
    clean = re.split(r"[{}()\[\]|<>\\;`~@#$%^&*+=]", path)[0].strip()
    clean = clean.encode("ascii", errors="ignore").decode("ascii").strip()
    corrupted = clean != path
    return (clean or path, corrupted)


def normalize_text(text: str) -> str:
    """Normalize text to expose hidden content.

    Phase 1: Unicode NFKC normalization (decomposes homoglyphs)
    Phase 2: Strip zero-width characters
    Phase 3: Detect and inline suspicious base64 blobs
    """
    import base64

    # Phase 1: Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Phase 2: Strip zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)

    # Phase 3: Detect base64 blobs (24+ chars of base64 alphabet)
    def _try_decode(m: re.Match) -> str:
        try:
            decoded = base64.b64decode(m.group(0)).decode("utf-8", errors="ignore")
            # Only inline if it contains suspicious patterns
            lowered = decoded.lower()
            if any(
                p in lowered
                for p in (
                    "ignore previous",
                    "ignore all",
                    "new rules",
                    "you are now",
                    "act as",
                    "pretend to be",
                    "bypass",
                    "rm -rf",
                )
            ):
                return f"[DECODED_BASE64: {decoded}]"
        except Exception:
            pass
        return m.group(0)

    text = re.sub(r"[A-Za-z0-9+/=]{24,}", _try_decode, text)
    return text


def sanitize_tool_args(
    args: dict,
    path_params: list[str] | None = None,
) -> dict:
    """Sanitize tool call arguments.

    Cleans path parameters using clean_path(). Returns a new dict.
    """
    if not path_params:
        return args

    result = dict(args)
    for param in path_params:
        if param in result and isinstance(result[param], str):
            result[param], _ = clean_path(result[param])
    return result
