"""Machine-readable output for `agentino run --mode json|jsonl`.

The default `agentino run --message "..."` writes ANSI-prettified markdown
to stdout via `_print_final()`. Foreign harnesses (openclaw plugins,
runspace runtime adapters, IDE extensions, polyglot stacks) can't parse
that — they need a structured contract.

Two modes:

  --mode json    → silent during the run, emit one envelope at the end:
                   {"type":"final","text":"...","tools_used":[...],
                    "tool_outputs":[...],"usage":{...},"model":"...",
                    "elapsed_ms":...}

  --mode jsonl   → emit one JSON event per line as the run progresses
                   (mirrors the pi/codex/claude-code CLI shapes), plus
                   a trailing `final` envelope. Useful for streaming UIs
                   and real-time tool tracking.

The emitter wraps the agent's `on_event` chain — it doesn't replace the
existing CLIRenderer, it bypasses it for these modes (verbose=False is
forced upstream so the human-readable renderer doesn't pollute stdout).
"""

from __future__ import annotations

import json
import sys
import time
from typing import IO, Any


class JsonEmitter:
    """Capture agent loop events; emit JSON envelope or JSONL stream."""

    def __init__(self, mode: str = "json", out: IO[str] | None = None) -> None:
        if mode not in ("json", "jsonl"):
            raise ValueError(f"mode must be 'json' or 'jsonl', got {mode!r}")
        self.mode = mode
        self.out = out if out is not None else sys.stdout
        self.events: list[dict[str, Any]] = []
        self.tools_used: list[str] = []
        self.tool_outputs: list[str] = []
        self._text_pieces: list[str] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._t0 = time.time()
        self._final_text: str | None = None

    def handle(self, event: Any) -> None:
        """Hook this in via `agent.on_event = emitter.handle` (chained)."""
        rec = self._record(event)
        self.events.append(rec)
        if self.mode == "jsonl":
            self._write_line(rec)

    def _record(self, event: Any) -> dict[str, Any]:
        et = getattr(event, "type", None)
        et_str = et.value if hasattr(et, "value") else str(et or "")
        rec: dict[str, Any] = {"type": et_str}

        if et_str == "tool_start":
            name = getattr(event, "name", "") or ""
            if name:
                rec["name"] = name
                self.tools_used.append(name)
            args = getattr(event, "args", None)
            if args:
                rec["args"] = args

        elif et_str == "tool_result":
            data = getattr(event, "data", None)
            if data is not None:
                txt = str(data)[:2000]
                rec["data"] = txt
                self.tool_outputs.append(txt)

        elif et_str == "text":
            chunk = getattr(event, "data", None)
            if chunk:
                rec["delta"] = str(chunk)
                self._text_pieces.append(str(chunk))

        elif et_str == "llm_response":
            usage = getattr(event, "usage", None)
            if usage:
                pt = getattr(usage, "prompt_tokens", 0) or 0
                ct = getattr(usage, "completion_tokens", 0) or 0
                rec["usage"] = {"prompt_tokens": pt, "completion_tokens": ct}
                self.usage["prompt_tokens"] += pt
                self.usage["completion_tokens"] += ct
            # When AGENTINO_LLM_TRACE=1 is set, agent.py attaches the full
            # prompt + completion to event.data — pass it through so the
            # consumer (harness dashboard / trial-detail view) can render it.
            data = getattr(event, "data", None)
            if isinstance(data, dict) and ("prompt" in data or "completion" in data):
                rec["trace"] = data

        elif et_str == "done":
            data = getattr(event, "data", None)
            if isinstance(data, str):
                self._final_text = data
                rec["text"] = data

        elif et_str.startswith("stage_"):
            name = getattr(event, "name", "") or ""
            if name:
                rec["stage"] = name

        return rec

    def emit_envelope(self, reply: str, model: str = "") -> None:
        """Write the final envelope. Always called at end of run.

        For --mode json this is the *only* output. For --mode jsonl it's
        the trailing summary line after the streamed events.
        """
        env = {
            "type": "final",
            "text": reply or self._final_text or "".join(self._text_pieces),
            "tools_used": list(self.tools_used),
            "tool_outputs": list(self.tool_outputs),
            "usage": dict(self.usage),
            "model": model,
            "elapsed_ms": int((time.time() - self._t0) * 1000),
        }
        self._write_line(env)

    def _write_line(self, obj: dict[str, Any]) -> None:
        self.out.write(json.dumps(obj, default=str) + "\n")
        self.out.flush()
