# Typed @tool guards — defensive coercion for LLM-supplied args

This recipe pins a footgun that bit a real production agent (an
Ada, 2026-05) and that is easy to re-introduce.

## The footgun

When you write a typed `@tool` function with `int | None` (or any optional
typed argument), the LLM will *occasionally* pass an empty string instead
of omitting the argument:

```python
from agentino import tool
import subprocess

@tool
def list_invoices(status: str = "open", due_within_days: int | None = None) -> str:
    """List invoices, optionally filtered by status and due-by-date."""
    args = ["python", "skills/list_invoices.py", "--status", status]
    if due_within_days is not None:                # ← seems right
        args += ["--due-within-days", str(due_within_days)]
    return subprocess.check_output(args, text=True)
```

The `is not None` guard looks correct. It isn't. Real LLM call patterns
include things like:

```json
{"name": "list_invoices", "arguments": {"status": "open", "due_within_days": ""}}
```

Empty string passes `is not None`. `str("")` is `""`. The subprocess sees
`["--due-within-days", ""]` and argparse rejects it:

```
list_invoices.py: error: argument --due-within-days: invalid int value: ''
```

The `@tool` framework surfaces this as a generic tool error the LLM
can't recover from. Ada's agentino-typed `list_invoices` was 50%
correct on diverse prompts because of this single bug; pi's bash path
was 90% correct on the same prompts because bash subshell error
handling lets the LLM see the failure and retry.

## The fix

Coerce LLM-supplied optional arguments explicitly. Treat empty string,
the literal `"null"`, and anything that doesn't `int()`-parse as
"argument absent":

```python
@tool
def list_invoices(status: str = "open", due_within_days: int | None = None) -> str:
    """List invoices, optionally filtered by status and due-by-date."""
    args = ["python", "skills/list_invoices.py", "--status", status or ""]

    # Coerce any LLM-supplied junk (empty string, "null", etc.) to None.
    n: int | None = None
    if due_within_days not in (None, "", "null"):
        try:
            n = int(due_within_days)
        except (TypeError, ValueError):
            n = None
    if n is not None:
        args += ["--due-within-days", str(n)]

    return subprocess.check_output(args, text=True)
```

## Rules of thumb

1. **Trust the LLM's *type intent*, not the type of the wire value.** If
   the parameter is annotated `int | None`, write the body as if the
   model meant either an int or no argument — and translate any other
   value to `None`. Models are inconsistent about omitting fields.
2. **Coerce at the tool boundary, not deeper.** The single-place coercion
   keeps the rest of your tool's logic strict and Pythonic.
3. **Don't `raise` on bad values from the LLM.** A `ValueError` becomes
   a tool error; the model can't recover. Coerce silently and move on.
4. **Pi/codex/claude's bash-tool path doesn't have this problem** —
   bash sees the literal command string the LLM produced and its
   subshell handles errors as stderr that the LLM can read. Typed tools
   own this defensiveness themselves.

## Test the guard

```python
def test_list_invoices_handles_empty_string_due_within_days(monkeypatch):
    """LLM sometimes passes due_within_days='' for unset numeric args."""
    captured = {}

    def fake_check_output(args, text=False, **kw):
        captured["args"] = args
        return ""

    monkeypatch.setattr("subprocess.check_output", fake_check_output)
    list_invoices(status="open", due_within_days="")  # type: ignore[arg-type]

    assert "--due-within-days" not in captured["args"], (
        "empty string for due_within_days must be treated as None — "
        "do not pass it to argparse"
    )
```

If you only test `due_within_days=None`, you're not testing the failure
mode the LLM actually hits.

## Where to apply this

Anywhere a `@tool` function takes an optional non-string typed argument:

- `int | None`, `float | None` — coerce or skip
- `bool | None` — model may pass `"true"`/`"false"`/`"yes"`/`""` strings
- `list[str] | None` — model may pass `[]`, `""`, `None`, sometimes a comma-string
- `dict | None` — model may pass `{}`

For each, add the explicit conversion at the tool boundary.

## Related

- [`hooks.md`](./hooks.md) — `PreToolUse` hooks for cross-cutting validation
- agentino `@tool` decorator implementation: `src/agentino/core/tool.py`
