"""Forked agents — lightweight background workers that share the parent's prompt cache.

Unlike spawn.py (full agent copy), forked agents:
1. Share the parent's system prompt prefix → API prompt cache hits
2. Clone file state for isolation (edits don't leak back)
3. Can run fire-and-forget (don't block parent)
4. Track usage separately for cost attribution

Ported from Claude Code's forkedAgent.ts pattern.

Usage:
    from agentino.fork import fork_agent, ForkConfig

    # Fork a background verification worker
    result = await fork_agent(
        parent=agent,
        task="Run tests and report results",
        config=ForkConfig(label="test-verify", fire_and_forget=False),
    )

    # Fork fire-and-forget memory extraction
    await fork_agent(
        parent=agent,
        task="Extract key facts from this conversation",
        config=ForkConfig(label="memory", fire_and_forget=True),
    )
"""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agentino.builtin_tools import _file_read_state
from agentino.core.agent import Agent
from agentino.core.message import Event, EventType, Message, Usage


@dataclass
class ForkConfig:
    """Configuration for a forked agent."""

    label: str = "fork"  # For logging/attribution
    fire_and_forget: bool = False  # Don't await result
    max_turns: int = 15  # Lower than parent (focused task)
    timeout: float = 120.0  # Seconds
    tools: list | None = None  # Override tools (None = inherit parent's)
    instructions_override: str = ""  # Extra instructions prepended
    on_event: Callable[[Event], None] | None = None


@dataclass
class ForkResult:
    """Result from a forked agent."""

    label: str
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    elapsed_ms: int = 0
    status: str = "completed"  # completed, failed, timeout, cancelled
    error: str = ""


def _clone_file_state() -> dict:
    """Clone the file read state cache for isolation."""
    return dict(_file_read_state)


def _restore_file_state(snapshot: dict) -> None:
    """Restore file state (used if fork's edits should propagate back)."""
    _file_read_state.update(snapshot)


async def fork_agent(
    parent: Agent,
    task: str,
    config: ForkConfig | None = None,
    parent_messages: list[Message] | None = None,
) -> ForkResult:
    """Fork a lightweight worker from a parent agent.

    The fork shares the parent's LLM client (prompt cache) but gets:
    - Isolated usage counters
    - Cloned file state cache
    - Its own message history (optionally prefixed with parent's for cache hits)

    Args:
        parent: The parent agent to fork from
        task: Task description for the fork
        config: Fork configuration (defaults if None)
        parent_messages: Parent's message prefix for prompt cache sharing.
            If provided, the fork's messages start with these (cache hit on API).
    """
    cfg = config or ForkConfig()

    # Create isolated agent copy
    fork = copy.copy(parent)
    fork.last_usage = Usage()
    fork.total_usage = Usage()
    fork.max_turns = cfg.max_turns

    if cfg.tools is not None:
        fork.tools = cfg.tools

    # Clone file state for isolation
    file_state_snapshot = _clone_file_state()

    # Build fork's messages — share parent's prefix for cache hits
    instructions = parent.instructions
    if cfg.instructions_override:
        instructions = cfg.instructions_override + "\n\n" + instructions

    # Prepare messages with shared prefix
    if parent_messages:
        # Reuse parent's system + early messages (cache hit on API)
        system_msgs = [m for m in parent_messages if m.role == "system"]
        messages = list(system_msgs) + [Message(role="user", content=task)]
    else:
        messages = fork._prepare_messages(task, session=None)

    start = time.monotonic()
    result = ForkResult(label=cfg.label)

    async def _run_fork() -> str:
        nonlocal result
        try:
            # Run the fork's agent loop inline
            prev_turn_calls: set[str] = set()
            response = None

            for turn in range(cfg.max_turns):
                from agentino.reliability.resilience import repair_messages

                messages_repaired = repair_messages(messages)

                response = await fork._llm_call_with_fallback(messages_repaired)
                fork.last_usage = response.usage
                fork.total_usage = fork.total_usage + response.usage
                messages.append(response.message)

                if cfg.on_event:
                    cfg.on_event(Event(type=EventType.LLM_RESPONSE, usage=response.usage))

                if not response.message.tool_calls:
                    return response.message.content or ""

                # Execute tools
                msgs_with_results, final = await fork._execute_tools(
                    response.message.tool_calls, messages, prev_turn_calls
                )
                messages.extend(m for m in msgs_with_results if m not in messages)
                prev_turn_calls = {fork._call_hash(c) for c in response.message.tool_calls}

                if final is not None:
                    return final.text

            return (response.message.content if response else "") or ""

        except Exception as e:
            result.status = "failed"
            result.error = f"{type(e).__name__}: {e}"
            return f"Fork error: {result.error}"

    if cfg.fire_and_forget:
        # Launch and don't wait
        async def _background():
            try:
                text = await asyncio.wait_for(_run_fork(), timeout=cfg.timeout)
                result.text = text
                result.status = "completed"
            except asyncio.TimeoutError:
                result.status = "timeout"
            except Exception as e:
                result.status = "failed"
                result.error = str(e)
            finally:
                result.elapsed_ms = int((time.monotonic() - start) * 1000)
                result.usage = fork.total_usage

        asyncio.create_task(_background())
        result.status = "launched"
        return result
    else:
        # Await result
        try:
            text = await asyncio.wait_for(_run_fork(), timeout=cfg.timeout)
            result.text = text
            if result.status != "failed":
                result.status = "completed"
        except asyncio.TimeoutError:
            result.status = "timeout"
            result.text = "Fork timed out"
        finally:
            result.elapsed_ms = int((time.monotonic() - start) * 1000)
            result.usage = fork.total_usage

        # Restore parent's file state (fork's edits are isolated)
        _file_read_state.clear()
        _file_read_state.update(file_state_snapshot)

        return result
