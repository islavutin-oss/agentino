"""Reusable CLI renderer for staged pipelines.

Supports both plain ANSI output and rich library rendering.

Usage:
    from agentino.cli.renderer import CLIRenderer

    # Plain ANSI (no dependencies)
    renderer = CLIRenderer()

    # Rich (if installed)
    renderer = CLIRenderer(use_rich=True)

    results = await pipeline.run(template, task, on_event=renderer.handle)
    renderer.print_summary(results)
"""

from __future__ import annotations

import sys
import time
from typing import Any

# ── Rich detection ──

_HAS_RICH = False
try:
    from rich.console import Console
    from rich.text import Text

    _HAS_RICH = True
except ImportError:
    pass


# ── ANSI helpers ──

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _IS_TTY else ""


_G = _c("\033[32m")
_R = _c("\033[31m")
_Y = _c("\033[33m")
_B = _c("\033[34m")
_M = _c("\033[35m")
_DIM = _c("\033[2m")
_BOLD = _c("\033[1m")
_N = _c("\033[0m")


class CLIRenderer:
    """Renders pipeline events to terminal.

    Supports plain ANSI or rich library output. Handles:
    STAGE_START, STAGE_COMPLETE, STAGE_FAIL, STAGE_SKIP,
    TOOL_START, TOOL_RESULT, TEXT, ERROR, DONE,
    LLM_RESPONSE (token/cost tracking), context (budget), worker events.
    """

    # Rough cost per 1K tokens (input/output averages)
    _COST_PER_1K: dict[str, float] = {
        "gpt-5.4-codex": 0.003,
        "gpt-5.3-codex": 0.003,
        "gpt-4o": 0.005,
        "claude-sonnet": 0.003,
        "claude-opus": 0.015,
        "default": 0.002,
    }

    def __init__(
        self,
        verbose: bool = False,
        show_timing: bool = True,
        show_tokens: bool = True,
        show_cost: bool = False,
        use_rich: bool = False,
        model: str = "",
    ):
        self.verbose = verbose
        self.show_timing = show_timing
        self.show_tokens = show_tokens
        self.show_cost = show_cost
        self._use_rich = use_rich and _HAS_RICH
        self._console = Console() if self._use_rich else None
        self._model = model

        self._stage_start: float = 0
        self._tool_start: float = 0
        self._task_start: float = time.time()
        self._current_stage: str = ""
        self._stage_count: int = 0
        self._total_stages: int = 0

        # Token/cost tracking
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._turn_tokens: int = 0
        self._llm_calls: int = 0
        self._tool_calls: int = 0

        # Context budget tracking
        self._context_max: int = 0
        self._context_used: int = 0
        self._was_compacted: bool = False

        # Worker tracking
        self._active_workers: dict[str, str] = {}  # id → description

    def set_total_stages(self, n: int) -> None:
        """Set the total number of stages for progress tracking."""
        self._total_stages = n

    def handle(self, event) -> None:
        """Main event handler — pass as on_event to StagedPipeline.run()."""
        from agentino.core.message import EventType

        t = event.type
        d = event.data

        if t == EventType.STAGE_START:
            self._on_stage_start(d)
        elif t == EventType.STAGE_COMPLETE:
            self._on_stage_complete(d)
        elif t == EventType.STAGE_FAIL:
            self._on_stage_fail(d)
        elif t == EventType.STAGE_SKIP:
            self._on_stage_skip(d)
        elif t == EventType.TOOL_START:
            name = getattr(event, "name", None) or (d.get("name") if isinstance(d, dict) else d)
            args_str = str(getattr(event, "args", "") or "")[:80]
            self._on_tool_start({"name": name or "?", "arguments": args_str})
        elif t == EventType.TOOL_RESULT:
            result = d or getattr(event, "data", "")
            self._on_tool_result(result)
        elif t == EventType.LLM_RESPONSE:
            self._on_llm_response(event)
        elif t == EventType.TEXT:
            self._on_text(d)
        elif t == EventType.ERROR:
            self._on_error(d)
        elif t == EventType.DONE:
            self._on_done(d)
        elif str(t) == "context":
            self._on_context(d)
        elif str(t) == "worker_start":
            self._on_worker_start(d, getattr(event, "name", ""))
        elif str(t) == "worker_done":
            self._on_worker_done(d, getattr(event, "name", ""))
        elif str(t) == "task_notification":
            self._on_task_notification(d, getattr(event, "name", ""))

    # ── Output helpers ──

    def _print(self, msg: str, style: str = "") -> None:
        if self._use_rich and self._console:
            self._console.print(msg, style=style, highlight=False)
        else:
            print(msg)

    def _separator(self) -> None:
        self._print(f"\n{'─' * 60}")

    # ── Stage events ──

    def _on_stage_start(self, data: Any) -> None:
        name = data if isinstance(data, str) else data.get("name", "?")
        self._current_stage = name
        self._stage_count += 1
        self._stage_start = time.time()

        bar = self._progress_bar()

        if self._use_rich and self._console:
            self._separator()
            line = Text()
            line.append(f" {bar}  ", style="")
            line.append(name.upper(), style="bold")
            self._console.print(line)
        else:
            self._separator()
            print(f" {bar}  {_BOLD}{name.upper()}{_N}")

    def _on_stage_complete(self, data: Any) -> None:
        name = data.get("name", self._current_stage) if isinstance(data, dict) else data
        elapsed = self._format_elapsed(self._stage_start)
        timing = f"  {elapsed}" if self.show_timing else ""

        if self._use_rich and self._console:
            line = Text()
            line.append(" ✓ ", style="green")
            line.append(f"{name.upper()} done", style="green")
            line.append(timing, style="dim")
            self._console.print(line)
        else:
            print(f" {_G}✓{_N} {_G}{name.upper()} done{_N}  {_DIM}{elapsed}{_N}")

    def _on_stage_fail(self, data: Any) -> None:
        if isinstance(data, dict):
            name = data.get("name", self._current_stage)
            if "jump_to" in data:
                self._print(f" ⚠ {name} failed → jumping to {data['jump_to']}", style="yellow")
            elif "retry" in data:
                self._print(f" ⚠ {name} — retry {data['retry']}", style="yellow")
            elif "reason" in data:
                self._print(f" ✗ {name}: {data['reason']}", style="red")
            else:
                self._print(f" ✗ {name} failed", style="red")
        else:
            self._print(f" ✗ {data} failed", style="red")

    def _on_stage_skip(self, data: Any) -> None:
        name = data if isinstance(data, str) else data.get("name", "?")
        self._stage_count += 1

        if self._use_rich and self._console:
            self._console.print(f" ── {name.upper()} (skipped) ──", style="dim")
        else:
            print(f" {_DIM}── {name.upper()} (skipped) ──{_N}")

    # ── Tool events ──

    def _on_tool_start(self, data: Any) -> None:
        self._tool_start = time.time()
        name = "?"
        args = ""
        if isinstance(data, dict):
            name = data.get("name", "?")
            raw_args = str(data.get("arguments", ""))
            # Always show args summary (truncated)
            if raw_args and raw_args != "{}":
                # Clean up JSON-like args for display
                args = raw_args.replace("{", "").replace("}", "").replace('"', "").replace("'", "")
                args = f" {args[:80]}"
        elif data is not None:
            name = str(data)

        if self._use_rich and self._console:
            line = Text()
            line.append(" ▸ ", style="cyan")
            line.append(name, style="bold")
            if args:
                line.append(args, style="dim")
            self._console.print(line)
        else:
            print(f" {_Y}▸{_N} {_BOLD}{name}{_N}{_DIM}{args}{_N}")

    def _on_tool_result(self, data: Any) -> None:
        result_full = str(data or "")
        if self.verbose and result_full.strip():
            # Debug mode: show full result
            if self._use_rich and self._console:
                self._console.print(f"   → {result_full}", style="dim")
            else:
                for line in result_full.split("\n"):
                    print(f"   {_DIM}│ {line}{_N}")
        else:
            # Normal: truncated preview
            result_str = result_full[:120].split("\n")[0].strip()
            if result_str:
                if self._use_rich and self._console:
                    self._console.print(f"   → {result_str}", style="dim")
                else:
                    print(f"   {_DIM}→ {result_str}{_N}")

    # ── Text / error ──

    def _on_text(self, data: Any) -> None:
        text = str(data or "")[:200]
        if text.strip():
            if self._use_rich and self._console:
                line = Text()
                line.append(" ● ", style="magenta")
                line.append(text, style="")
                self._console.print(line)
            else:
                print(f" {_M}●{_N} {text}")

    def _on_error(self, data: Any) -> None:
        self._print(f" ✗ {data}", style="red")

    def _on_done(self, data: Any) -> None:
        pass

    # ── LLM response (token/cost tracking) ──

    def _on_llm_response(self, event: Any) -> None:
        usage = getattr(event, "usage", None)
        if not usage:
            return
        self._llm_calls += 1
        tokens = getattr(usage, "total_tokens", 0) or (
            getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0)
        )
        self._turn_tokens = tokens
        self._total_tokens += tokens

        # Estimate cost
        cost_rate = self._COST_PER_1K.get(self._model, self._COST_PER_1K["default"])
        turn_cost = tokens / 1000 * cost_rate
        self._total_cost += turn_cost

        if self.show_tokens and self.verbose:
            token_str = f"{tokens:,}" if tokens < 10000 else f"{tokens / 1000:.1f}K"
            cost_str = f"${turn_cost:.3f}" if self.show_cost else ""
            self._print(f"   {_DIM}[{token_str} tokens{', ' + cost_str if cost_str else ''}]{_N}")

    # ── Context budget ──

    def _on_context(self, data: Any) -> None:
        if isinstance(data, dict):
            self._context_used = data.get("tokens", 0)
            self._context_max = data.get("max", 0)
            self._was_compacted = data.get("compacted", False)
            if data.get("reactive_compact"):
                self._print(f" {_Y}⟳{_N} {_DIM}Context compacted (reactive){_N}")
            elif self._was_compacted and self.verbose:
                pct = (self._context_used / self._context_max * 100) if self._context_max else 0
                self._print(
                    f" {_DIM}⟳ Context: {self._context_used:,}/{self._context_max:,} ({pct:.0f}%){_N}"
                )

    # ── Worker events ──

    def _on_worker_start(self, worker_id: Any, description: str) -> None:
        self._active_workers[str(worker_id)] = description
        self._print(f" {_B}⊕{_N} Worker spawned: {_BOLD}{description}{_N} {_DIM}({worker_id}){_N}")

    def _on_worker_done(self, worker_id: Any, description: str) -> None:
        self._active_workers.pop(str(worker_id), None)
        self._print(f" {_G}⊖{_N} Worker done: {description} {_DIM}({worker_id}){_N}")

    def _on_task_notification(self, data: Any, description: str) -> None:
        text = str(data or "")[:200]
        suffix = f" — {text}" if text else ""
        self._print(f" {_M}◆{_N} {_DIM}Task notification: {description}{suffix}{_N}")

    # ── Helpers ──

    def _progress_bar(self) -> str:
        total = self._total_stages or 5
        done = min(self._stage_count - 1, total)
        return "▰" * done + "▱" * (total - done)

    def _format_elapsed(self, start: float) -> str:
        elapsed = time.time() - start
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m {secs:02d}s"

    def total_elapsed(self) -> str:
        """Return formatted total elapsed time since task start."""
        return self._format_elapsed(self._task_start)

    def print_summary(self, results: list) -> None:
        """Print final summary after pipeline completes."""
        passed = sum(1 for r in results if r.completed and not r.failed)
        total = len(results)
        elapsed = self.total_elapsed()

        # Token/cost summary
        token_str = (
            f"{self._total_tokens:,}"
            if self._total_tokens < 100000
            else f"{self._total_tokens / 1000:.1f}K"
        )
        stage_str = f"{total} stages" if passed == total else f"{passed}/{total} stages passed"
        stats = f"{elapsed} · {stage_str} · {self._llm_calls} LLM calls · {token_str} tokens"
        ok = passed == total
        if self.show_cost and self._total_cost > 0:
            stats += f" · ${self._total_cost:.3f}"

        if self._use_rich and self._console:
            self._separator()
            line = Text()
            line.append(" ✓ " if ok else " ✗ ", style="green bold" if ok else "red bold")
            line.append("COMPLETE" if ok else "FAILED", style="green bold" if ok else "red bold")
            line.append(f"  {stats}", style="dim")
            self._console.print(line)
            self._console.print("─" * 60)
        else:
            self._separator()
            mark = f"{_G}{_BOLD}✓ COMPLETE{_N}" if ok else f"{_R}{_BOLD}✗ FAILED{_N}"
            print(f" {mark}  {_DIM}{stats}{_N}")
            print(f"{'─' * 60}")
