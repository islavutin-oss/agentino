"""Regression tests for the typed @tool defensive-coercion pattern.

The footgun: LLMs occasionally pass empty strings (or "null"-strings) for
typed `int | None` / `bool | None` / `list | None` arguments instead of
omitting them. A naive `is not None` guard accepts the empty string and
forwards it to downstream code that isn't expecting it.

These tests pin the *recipe* in `docs/cookbook/typed-tool-guards.md` —
they don't test agentino itself; they test that the guard pattern works
when applied. New `@tool` authors should be able to copy-paste either
the pattern or the test as a starting point.

Real incident: acme Ada's `list_invoices(status, due_within_days)`
saw the LLM pass `due_within_days=""` and returned a bare argparse
error from a subprocess call, which agentino surfaced as a generic
"tool error" the model couldn't recover from. Diverse-prompt benchmark
dropped from 100% correctness to 50% until the guard landed.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# The recipe — copy this into your tool when you have an `int | None` arg.
# ---------------------------------------------------------------------------


def coerce_optional_int(value: Any) -> int | None:
    """Coerce LLM-supplied junk to a real int-or-None.

    Treats None, '', 'null' (case-insensitive) and any non-int-parseable
    value as "argument absent". Returns the parsed int otherwise.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip() == "" or value.strip().lower() == "null":
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_bool(value: Any) -> bool | None:
    """Same shape as coerce_optional_int but for bool | None.

    Recognised true-strings: 'true', 'yes', '1' (case-insensitive).
    Recognised false-strings: 'false', 'no', '0'.
    Anything else → None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def coerce_optional_list(value: Any, *, sep: str = ",") -> list[str] | None:
    """Tolerate LLM-passed `[]`, `""`, comma-string, or list. None otherwise."""
    if value is None:
        return None
    if isinstance(value, list):
        # Empty list = "argument intentionally absent" per most agents
        return value or None
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        parts = [p.strip() for p in s.split(sep) if p.strip()]
        return parts or None
    return None


# ---------------------------------------------------------------------------
# Tests — coerce_optional_int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),  # The actual LLM regression case
        ("   ", None),  # Whitespace
        ("null", None),  # Literal "null" string
        ("NULL", None),  # Case-insensitive
        (" Null ", None),  # Whitespace + case
        ("not a number", None),
        ("3.14", None),  # Float string is not int-parseable
        (3.14, 3),  # Float is int-coercible (truncates)
        (0, 0),  # Zero is meaningful — keep it
        (7, 7),
        ("7", 7),
        ("-3", -3),
    ],
)
def test_coerce_optional_int(value: Any, expected: int | None) -> None:
    assert coerce_optional_int(value) == expected


# ---------------------------------------------------------------------------
# Tests — coerce_optional_bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("1", True),
        (1, True),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("0", False),
        (0, False),
        ("maybe", None),
        ("null", None),
    ],
)
def test_coerce_optional_bool(value: Any, expected: bool | None) -> None:
    assert coerce_optional_bool(value) is expected


# ---------------------------------------------------------------------------
# Tests — coerce_optional_list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ([], None),
        ("", None),
        ("a,b,c", ["a", "b", "c"]),
        ("a, b, c", ["a", "b", "c"]),
        (["a", "b"], ["a", "b"]),
        ("single", ["single"]),
        (",,", None),  # all-separator string with no real values
    ],
)
def test_coerce_optional_list(value: Any, expected: list[str] | None) -> None:
    assert coerce_optional_list(value) == expected


# ---------------------------------------------------------------------------
# Integration test — applying the guard in a realistic @tool body
# ---------------------------------------------------------------------------


def _build_args(status: str = "open", due_within_days: Any = None) -> list[str]:
    """Mimics acme Ada's list_invoices wrapper, post-fix."""
    args = ["python", "skills/list_invoices.py", "--status", status or ""]
    n = coerce_optional_int(due_within_days)
    if n is not None:
        args += ["--due-within-days", str(n)]
    return args


def test_list_invoices_wrapper_omits_due_within_days_for_empty_string() -> None:
    """The actual acme regression — pin it with a test."""
    args = _build_args(status="open", due_within_days="")
    assert "--due-within-days" not in args, (
        "Empty string for `due_within_days` must NOT be forwarded to argparse "
        "— it would trip 'invalid int value: ''' and surface as a tool error."
    )


def test_list_invoices_wrapper_keeps_zero() -> None:
    """due_within_days=0 (today only) is a meaningful value, not 'absent'."""
    args = _build_args(status="open", due_within_days=0)
    assert "--due-within-days" in args
    assert args[args.index("--due-within-days") + 1] == "0"


def test_list_invoices_wrapper_handles_int_string() -> None:
    args = _build_args(status="open", due_within_days="7")
    assert args[args.index("--due-within-days") + 1] == "7"


def test_list_invoices_wrapper_omits_due_within_days_for_null_string() -> None:
    args = _build_args(status="open", due_within_days="null")
    assert "--due-within-days" not in args


def test_list_invoices_wrapper_omits_due_within_days_for_garbage() -> None:
    args = _build_args(status="open", due_within_days="not a number")
    assert "--due-within-days" not in args
