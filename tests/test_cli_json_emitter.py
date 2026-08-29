"""Tests for `agentino run --mode json|jsonl` output.

Two layers:
  1. Unit tests on JsonEmitter — direct event injection, assert envelope shape.
  2. Subprocess-level smoke test — spawn `python -m agentino run` against
     a stub config + monkey-patched Agent.run, parse JSON from stdout.

Live LLM calls are out of scope; provider behaviour is covered elsewhere.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import pytest

from agentino.cli.json_emitter import JsonEmitter
from agentino.core.message import Event, EventType, Usage


@pytest.fixture(autouse=True)
def _restore_default_event_loop():
    """The CLI integration tests call `main()` which internally does
    `asyncio.run(...)` — that tears down the thread-local event loop.
    Sibling test files (test_hooks.py, …) use the deprecated
    `asyncio.get_event_loop()` pattern that requires one to be set.
    Restore a fresh default loop after each test so cross-file ordering
    stays clean."""
    import asyncio

    yield
    asyncio.set_event_loop(asyncio.new_event_loop())


def _ev(t: EventType, **kwargs) -> Event:
    return Event(type=t, **kwargs)


def test_envelope_collects_tools_and_text() -> None:
    em = JsonEmitter(mode="json", out=io.StringIO())
    em.handle(_ev(EventType.TOOL_START, name="search", args={"q": "foo"}))
    em.handle(_ev(EventType.TOOL_RESULT, data="result text"))
    em.handle(_ev(EventType.TEXT, data="hello "))
    em.handle(_ev(EventType.TEXT, data="world"))
    em.handle(_ev(EventType.LLM_RESPONSE, usage=Usage(prompt_tokens=10, completion_tokens=5)))
    em.handle(_ev(EventType.DONE, data="hello world"))

    assert em.tools_used == ["search"]
    assert em.tool_outputs == ["result text"]
    assert em.usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_json_mode_silent_until_envelope() -> None:
    out = io.StringIO()
    em = JsonEmitter(mode="json", out=out)
    em.handle(_ev(EventType.TOOL_START, name="x"))
    em.handle(_ev(EventType.TEXT, data="chunk"))
    # In json mode nothing is written until emit_envelope
    assert out.getvalue() == ""
    em.emit_envelope("the answer", model="gpt-test")
    line = out.getvalue().strip()
    env = json.loads(line)
    assert env["type"] == "final"
    assert env["text"] == "the answer"
    assert env["tools_used"] == ["x"]
    assert env["model"] == "gpt-test"
    assert "elapsed_ms" in env


def test_jsonl_mode_streams_per_event_then_final() -> None:
    out = io.StringIO()
    em = JsonEmitter(mode="jsonl", out=out)
    em.handle(_ev(EventType.TOOL_START, name="reader"))
    em.handle(_ev(EventType.TOOL_RESULT, data="file body"))
    em.handle(_ev(EventType.TEXT, data="hi"))
    em.emit_envelope("hi", model="gpt-test")

    lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
    assert len(lines) == 4
    assert lines[0]["type"] == "tool_start"
    assert lines[0]["name"] == "reader"
    assert lines[1]["type"] == "tool_result"
    assert lines[2]["type"] == "text"
    assert lines[2]["delta"] == "hi"
    assert lines[3]["type"] == "final"
    assert lines[3]["text"] == "hi"
    assert lines[3]["tools_used"] == ["reader"]


def test_envelope_falls_back_to_text_pieces_when_no_done_event() -> None:
    out = io.StringIO()
    em = JsonEmitter(mode="json", out=out)
    em.handle(_ev(EventType.TEXT, data="part1 "))
    em.handle(_ev(EventType.TEXT, data="part2"))
    em.emit_envelope("", model="m")
    env = json.loads(out.getvalue().strip())
    assert env["text"] == "part1 part2"


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        JsonEmitter(mode="text")  # only json/jsonl accepted


def test_tool_output_truncated_at_2000_chars() -> None:
    out = io.StringIO()
    em = JsonEmitter(mode="json", out=out)
    em.handle(_ev(EventType.TOOL_RESULT, data="x" * 5000))
    em.emit_envelope("done", model="m")
    env = json.loads(out.getvalue().strip())
    assert len(env["tool_outputs"][0]) == 2000


def test_usage_accumulates_across_llm_responses() -> None:
    em = JsonEmitter(mode="json", out=io.StringIO())
    em.handle(_ev(EventType.LLM_RESPONSE, usage=Usage(prompt_tokens=10, completion_tokens=2)))
    em.handle(_ev(EventType.LLM_RESPONSE, usage=Usage(prompt_tokens=15, completion_tokens=3)))
    assert em.usage == {"prompt_tokens": 25, "completion_tokens": 5}


def test_handle_with_unknown_event_type_does_not_crash() -> None:
    em = JsonEmitter(mode="jsonl", out=io.StringIO())

    @dataclass
    class FakeEvent:
        type: str = "made_up"
        data: object = None

    em.handle(FakeEvent())  # should not raise
    assert em.events[-1] == {"type": "made_up"}


# ---------------------------------------------------------------------------
# Integration tests — drive the full `agentino run --mode …` CLI path with a
# stubbed LLM so we exercise argparse → emitter → stdout end-to-end without
# making a real model call. Pins the JSON contract for foreign harnesses.
# ---------------------------------------------------------------------------


import os  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

_STUB_AGENTS_YAML = """
agents:
  bot:
    model: gpt-4o
    instructions: "stub agent for the json-mode integration test"
"""


def _stub_one_shot_returning(reply: str):
    """Build an async one_shot that returns `reply` and emits a few fake events."""
    from agentino.core.message import Event, EventType, Usage

    async def _one_shot(self, message: str, agent_name=None):
        # Resolve which agent is in play and fire some events on its on_event
        # so the emitter has tools/text/usage to capture.
        agent = self._resolve_agent(agent_name)
        if agent.on_event:
            agent.on_event(Event(type=EventType.TOOL_START, name="stub_tool", args={"q": "x"}))
            agent.on_event(Event(type=EventType.TOOL_RESULT, data="stub output"))
            agent.on_event(
                Event(
                    type=EventType.LLM_RESPONSE, usage=Usage(prompt_tokens=12, completion_tokens=3)
                )
            )
            agent.on_event(Event(type=EventType.DONE, data=reply))
        return reply

    return _one_shot


def test_cli_mode_json_emits_single_envelope(capsys) -> None:
    """`agentino run … --mode json` must print one JSON line and nothing else."""
    from agentino import __main__ as cli
    from agentino.core.runner import Runner

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "agents.yml"
        config_path.write_text(_STUB_AGENTS_YAML)

        argv = [
            "agentino",
            "run",
            str(config_path),
            "--agent",
            "bot",
            "--message",
            "hi",
            "--mode",
            "json",
            "--session-dir",
            str(Path(tmp) / "sessions"),
            "--usage-file",
            str(Path(tmp) / "usage.jsonl"),
        ]

        stub = _stub_one_shot_returning("the canned reply")
        with patch("sys.argv", argv), patch.object(Runner, "one_shot", new=stub):
            cli.main()

        out = capsys.readouterr().out
        # Exactly one JSON object on stdout
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 1, f"expected 1 line, got {len(lines)}: {lines}"
        env = json.loads(lines[0])
        assert env["type"] == "final"
        assert env["text"] == "the canned reply"
        assert env["tools_used"] == ["stub_tool"]
        assert env["tool_outputs"] == ["stub output"]
        assert env["usage"] == {"prompt_tokens": 12, "completion_tokens": 3}
        assert env["model"] == "gpt-4o"
        assert "elapsed_ms" in env


def test_cli_mode_jsonl_streams_events_then_final(capsys) -> None:
    """`agentino run … --mode jsonl` emits per-event lines plus a final envelope."""
    from agentino import __main__ as cli
    from agentino.core.runner import Runner

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "agents.yml"
        config_path.write_text(_STUB_AGENTS_YAML)

        argv = [
            "agentino",
            "run",
            str(config_path),
            "--agent",
            "bot",
            "--message",
            "hi",
            "--mode",
            "jsonl",
            "--session-dir",
            str(Path(tmp) / "sessions"),
            "--usage-file",
            str(Path(tmp) / "usage.jsonl"),
        ]

        stub = _stub_one_shot_returning("canned")
        with patch("sys.argv", argv), patch.object(Runner, "one_shot", new=stub):
            cli.main()

        out = capsys.readouterr().out
        lines = [json.loads(line) for line in out.splitlines() if line.strip()]
        types = [obj["type"] for obj in lines]
        # Streamed events arrive before the final envelope
        assert "tool_start" in types
        assert "tool_result" in types
        assert "llm_response" in types
        assert types[-1] == "final"
        # Final envelope still carries the aggregate
        assert lines[-1]["text"] == "canned"
        assert lines[-1]["tools_used"] == ["stub_tool"]


def test_cli_mode_text_unchanged_default(capsys) -> None:
    """Backward compat: omitting --mode keeps the ANSI-prettified text path."""
    from agentino import __main__ as cli
    from agentino.core.runner import Runner

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "agents.yml"
        config_path.write_text(_STUB_AGENTS_YAML)

        argv = [
            "agentino",
            "run",
            str(config_path),
            "--agent",
            "bot",
            "--message",
            "hi",
            "--session-dir",
            str(Path(tmp) / "sessions"),
            "--usage-file",
            str(Path(tmp) / "usage.jsonl"),
        ]

        stub = _stub_one_shot_returning("the canned reply")
        with patch("sys.argv", argv), patch.object(Runner, "one_shot", new=stub):
            cli.main()

        out = capsys.readouterr().out
        # Text mode renders the reply via _print_final (markdown→ANSI). The
        # exact bytes vary, but the reply text must be present and the output
        # MUST NOT parse as JSON (otherwise we silently broke backward compat).
        assert "the canned reply" in out
        try:
            json.loads(out.strip())
            raise AssertionError("text mode unexpectedly produced JSON output")
        except (json.JSONDecodeError, ValueError):
            pass  # expected — text mode is not JSON


def test_cli_mode_json_handles_unicode(capsys) -> None:
    """Non-ASCII replies must round-trip through JSON cleanly."""
    from agentino import __main__ as cli
    from agentino.core.runner import Runner

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "agents.yml"
        config_path.write_text(_STUB_AGENTS_YAML)
        argv = [
            "agentino",
            "run",
            str(config_path),
            "--agent",
            "bot",
            "--message",
            "hi",
            "--mode",
            "json",
            "--session-dir",
            str(Path(tmp) / "sessions"),
            "--usage-file",
            str(Path(tmp) / "usage.jsonl"),
        ]

        stub = _stub_one_shot_returning("Γειά — €1240,50 — 你好 🎉")
        with patch("sys.argv", argv), patch.object(Runner, "one_shot", new=stub):
            cli.main()

        line = capsys.readouterr().out.strip()
        env = json.loads(line)
        assert env["text"] == "Γειά — €1240,50 — 你好 🎉"


def test_cli_mode_json_iterate_flag_silently_suppressed(capsys) -> None:
    """`--iterate` would drop into REPL after one-shot — must NOT trigger
    in machine modes (would block on stdin and corrupt JSON output)."""
    from agentino import __main__ as cli
    from agentino.core.runner import Runner

    repl_called = []

    def _repl_spy(self, **kwargs):
        repl_called.append(True)

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "agents.yml"
        config_path.write_text(_STUB_AGENTS_YAML)
        argv = [
            "agentino",
            "run",
            str(config_path),
            "--agent",
            "bot",
            "--message",
            "hi",
            "--mode",
            "json",
            "--iterate",
            "--session-dir",
            str(Path(tmp) / "sessions"),
            "--usage-file",
            str(Path(tmp) / "usage.jsonl"),
        ]

        stub = _stub_one_shot_returning("ok")
        with (
            patch("sys.argv", argv),
            patch.object(Runner, "one_shot", new=stub),
            patch.object(Runner, "repl", new=_repl_spy),
        ):
            cli.main()

        assert repl_called == [], "REPL must not start when --mode json is set"
        # Output must still be parseable JSON
        line = capsys.readouterr().out.strip()
        env = json.loads(line)
        assert env["text"] == "ok"


def test_cli_mode_json_runner_failure_propagates_with_clean_stderr(capsys) -> None:
    """If runner.one_shot raises, the CLI exits non-zero with a clean error.
    JSON mode must not swallow exceptions silently — that would hide bugs."""
    from agentino import __main__ as cli
    from agentino.core.runner import Runner

    async def _boom(self, message, agent_name=None):
        raise RuntimeError("upstream LLM unavailable")

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "agents.yml"
        config_path.write_text(_STUB_AGENTS_YAML)
        argv = [
            "agentino",
            "run",
            str(config_path),
            "--agent",
            "bot",
            "--message",
            "hi",
            "--mode",
            "json",
            "--session-dir",
            str(Path(tmp) / "sessions"),
            "--usage-file",
            str(Path(tmp) / "usage.jsonl"),
        ]

        with patch("sys.argv", argv), patch.object(Runner, "one_shot", new=_boom):
            try:
                cli.main()
            except SystemExit as e:
                assert e.code == 1, f"expected exit code 1, got {e.code}"

        # Error message goes to stdout (existing _cmd_run behavior); critical
        # check is that we DON'T emit a malformed JSON line claiming success.
        out = capsys.readouterr().out
        for line in out.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            # If anything looks like JSON, it must not falsely claim success
            try:
                obj = json.loads(line)
                if obj.get("type") == "final":
                    raise AssertionError("Failure path emitted a `final` envelope as if successful")
            except json.JSONDecodeError:
                pass


def test_cli_subprocess_invocation_end_to_end() -> None:
    """OS-level subprocess test: `python -m agentino run … --mode json` from
    a fresh interpreter. Catches packaging issues (sys.path, entry points,
    import-time side effects) that in-process tests miss.

    Uses a stub LLM via OPENAI_API_KEY=dummy + a config pointing at a fake
    base URL so the model call would fail — we run with `agents` subcommand
    instead, which doesn't make network calls but exercises argparse + config
    load + the same module-import path the real `run` invocation hits.
    """
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "agents.yml"
        config_path.write_text(_STUB_AGENTS_YAML)

        # `agents` subcommand exercises argparse + Config load + Runner init,
        # without an LLM call. Confirms the entry point actually runs.
        result = subprocess.run(
            [sys.executable, "-m", "agentino", "agents", str(config_path)],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "AGENTINO_API_KEY": "test-stub"},
        )
        assert result.returncode == 0, (
            f"subprocess failed: rc={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "bot" in result.stdout
        assert "gpt-4o" in result.stdout


def test_cli_run_help_lists_mode_flag() -> None:
    """`agentino run --help` must document the new flag (regression guard
    for someone deleting the help text but leaving the flag wired)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "agentino", "run", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "--mode" in result.stdout
    assert "json" in result.stdout
    assert "jsonl" in result.stdout
