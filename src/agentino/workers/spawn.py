"""Subagent spawning — let an agent delegate tasks to other agents via tool calls.

Design follows OpenClaw patterns:
- Fresh agent copy per spawn (no shared mutable state)
- Depth tracking with configurable max (default 1, like OpenClaw)
- Cancellation propagation via asyncio.Task

Usage:
    from agentino.spawn import make_spawn_tool

    agents = {"coder": coder_agent, "reviewer": reviewer_agent}
    spawn_tool = make_spawn_tool(agents, allowed=["coder"])
    parent_agent.add_tool(spawn_tool)
"""

from __future__ import annotations

import asyncio
import copy
import json
import uuid

from agentino.core.agent import Agent
from agentino.core.tool import Tool

# Default limits (aligned with OpenClaw: DEFAULT_SUBAGENT_MAX_SPAWN_DEPTH = 1)
DEFAULT_MAX_DEPTH = 1


_SUBAGENT_PREFIX = (
    "You are a subagent spawned for a specific task. "
    "Complete the task thoroughly and return your findings. "
    "Do not ask follow-up questions — deliver results.\n\n"
)


def _copy_agent(agent: Agent) -> Agent:
    """Create a shallow copy of an agent with isolated mutable state.

    Copies usage counters so concurrent spawns don't race on the same instance.
    The LLM client is shared (stateless HTTP client — safe to share).
    """
    clone = copy.copy(agent)
    # Isolate mutable state
    from agentino.core.message import Usage

    clone.last_usage = Usage()
    clone.total_usage = Usage()
    # Tools list is shared (read-only during run) — no need to deep copy
    return clone


def make_spawn_tool(
    agents: dict[str, Agent],
    allowed: list[str] | None = None,
    timeout: float = 300.0,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Tool:
    """Create a tool that spawns a subagent to handle a delegated task.

    Args:
        agents: Dict of available agents keyed by ID.
        allowed: Optional allowlist of agent IDs. If None, all agents are available.
        timeout: Max seconds for the subagent to run (default 300s).
        max_depth: Max spawn nesting depth (default 1 — subagents are leaves).

    Returns:
        A Tool instance that the LLM can call with {"agent_id": "...", "task": "..."}.
    """
    available = allowed if allowed is not None else list(agents.keys())

    # Validate allowlist
    for name in available:
        if name not in agents:
            raise ValueError(f"spawn_agent: allowed agent '{name}' not found in agents dict")

    agent_list = ", ".join(available)
    description = (
        f"Delegate a task to a specialized subagent. "
        f"Available agents: [{agent_list}]. "
        f"The subagent runs independently with its own tools and returns a structured result. "
        f"Use this when the task requires specialized investigation or analysis."
    )

    # Track active tasks for cancellation
    _active_tasks: dict[str, asyncio.Task] = {}

    async def _spawn(agent_id: str, task: str, _depth: int = 0) -> str:
        if agent_id not in available:
            return f"Error: agent '{agent_id}' is not available. Choose from: {agent_list}"

        agent = agents.get(agent_id)
        if agent is None:
            return f"Error: agent '{agent_id}' not found"

        # Recursion guard
        if _depth >= max_depth:
            return json.dumps(
                {
                    "agent_id": agent_id,
                    "status": "forbidden",
                    "error": f"Max spawn depth reached ({_depth}/{max_depth}). "
                    f"Subagents at depth {max_depth} cannot spawn further.",
                }
            )

        # Generate workflow ID for correlation
        workflow_id = f"wf-{uuid.uuid4().hex[:12]}"

        # Create isolated copy to avoid shared state races
        agent_copy = _copy_agent(agent)

        try:
            context = (
                f"{_SUBAGENT_PREFIX}"
                f"[workflow_id: {workflow_id}]\n"
                f"[parent_agent: calling agent]\n"
                f"[subagent: {agent_id}]\n"
                f"[spawn_depth: {_depth + 1}/{max_depth}]\n\n"
                f"{task}"
            )

            # Wrap in a Task for cancellation propagation
            run_task = asyncio.create_task(agent_copy.run(context))
            _active_tasks[workflow_id] = run_task

            try:
                result = await asyncio.wait_for(
                    asyncio.shield(run_task),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                run_task.cancel()
                # Wait briefly for clean cancellation
                try:
                    await asyncio.wait_for(run_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                return json.dumps(
                    {
                        "workflow_id": workflow_id,
                        "agent_id": agent_id,
                        "status": "timeout",
                        "error": f"Subagent timed out after {timeout}s",
                    }
                )
            finally:
                _active_tasks.pop(workflow_id, None)

            if not result:
                return json.dumps(
                    {
                        "workflow_id": workflow_id,
                        "agent_id": agent_id,
                        "status": "empty",
                        "result": None,
                    }
                )

            # Try to parse as JSON (structured output from format_findings etc.)
            # If it's already JSON, wrap with metadata. If not, return as-is with metadata.
            try:
                parsed = json.loads(result)
                return json.dumps(
                    {
                        "workflow_id": workflow_id,
                        "agent_id": agent_id,
                        "status": "ok",
                        "result": parsed,
                    }
                )
            except (json.JSONDecodeError, TypeError):
                # Subagent returned free text — wrap it
                return json.dumps(
                    {
                        "workflow_id": workflow_id,
                        "agent_id": agent_id,
                        "status": "ok",
                        "result_text": result,
                    }
                )

        except asyncio.CancelledError:
            return json.dumps(
                {
                    "workflow_id": workflow_id,
                    "agent_id": agent_id,
                    "status": "cancelled",
                    "error": "Subagent was cancelled",
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    "workflow_id": workflow_id,
                    "agent_id": agent_id,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    return Tool(
        name="spawn_agent",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": f"ID of the agent to spawn. Available: {agent_list}",
                    "enum": available,
                },
                "task": {
                    "type": "string",
                    "description": "The task description for the subagent to execute.",
                },
            },
            "required": ["agent_id", "task"],
        },
        fn=_spawn,
        timeout=None,  # We handle timeout internally
    )
