"""Coordinator mode — orchestrate multiple worker agents in parallel.

Ported from Claude Code's coordinator pattern. The coordinator:
1. Plans the work
2. Spawns workers for research/implementation/verification
3. Receives results via task notifications
4. Synthesizes findings before delegating follow-up
5. Never predicts or fabricates worker results

Usage:
    coordinator = Coordinator(
        llm=llm_client,
        worker_tools=worker_tools,
        max_workers=3,
    )
    result = await coordinator.run("Fix the auth bug in src/auth/validate.ts")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from agentino.core.agent import Agent
from agentino.core.llm import LLMClient
from agentino.core.message import Event, Message
from agentino.core.tool import Tool, tool

# ---------------------------------------------------------------------------
# Worker — an autonomous agent running a subtask
# ---------------------------------------------------------------------------


@dataclass
class WorkerTask:
    """A worker executing a subtask."""

    id: str
    description: str
    prompt: str
    status: str = "running"  # running, completed, failed, stopped
    result: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    llm_calls: int = 0
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# Task notification format (XML, matches Claude Code)
# ---------------------------------------------------------------------------


def format_task_notification(task: WorkerTask) -> str:
    """Format worker result as a task notification (injected as user message)."""
    parts = [
        "<task-notification>",
        f"<task-id>{task.id}</task-id>",
        f"<status>{task.status}</status>",
        f'<summary>Worker "{task.description}" {task.status}</summary>',
    ]
    if task.result:
        parts.append(f"<result>{task.result}</result>")
    elapsed = int((task.end_time or time.time()) - task.start_time)
    parts.append(
        f"<usage><duration_ms>{elapsed * 1000}</duration_ms>"
        f"<llm_calls>{task.llm_calls}</llm_calls>"
        f"<tool_calls>{task.tool_calls}</tool_calls></usage>"
    )
    parts.append("</task-notification>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Coordinator system prompt
# ---------------------------------------------------------------------------

COORDINATOR_SYSTEM_PROMPT = """You are a coordinator that orchestrates work across multiple workers.

## Your Role
- Help achieve the goal by directing workers
- Synthesize results — NEVER predict or fabricate worker results
- Answer directly when possible — don't delegate trivial work

## Your Tools
- spawn_worker(description, prompt) — Launch a worker with a specific task
- send_message(worker_id, message) — Continue an existing worker
- stop_worker(worker_id) — Stop a running worker

## Workflow
1. **Research** — Spawn workers in parallel to investigate
2. **Synthesis** — Read findings, understand the problem, craft precise specs
3. **Implementation** — Direct workers with specific file paths and line numbers
4. **Verification** — Spawn fresh worker to independently verify

## Critical Rules
- After launching workers, briefly tell the user what you launched and END your response
- When worker results arrive, SYNTHESIZE them — include file paths, line numbers, what to change
- NEVER write "based on your findings" — that's lazy delegation
- Workers can't see your conversation — every prompt must be self-contained
- Continue workers with high context overlap, spawn fresh for low overlap
- Parallelism is your superpower — launch independent workers concurrently

## Worker Prompt Quality
Good: "Fix null pointer in src/auth/validate.ts:42. Add null check before user.id access — return 401 if null."
Bad: "Fix the auth bug we discussed." (workers can't see your conversation)
"""


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class Coordinator:
    """Orchestrates multiple worker agents following the coordinator pattern.

    The coordinator agent plans and synthesizes while workers execute.
    Workers run as independent Agent instances with their own tools.
    """

    def __init__(
        self,
        llm: LLMClient,
        worker_tools: list[Tool],
        worker_instructions: str = "",
        model: str | None = None,
        max_workers: int = 5,
        max_turns: int = 30,
        on_event: Callable[[Event], None] | None = None,
    ):
        self._llm = llm
        self._worker_tools = worker_tools
        self._worker_instructions = worker_instructions
        self._model = model
        self._max_workers = max_workers
        self._max_turns = max_turns
        self._on_event = on_event
        self._workers: dict[str, WorkerTask] = {}
        self._running: dict[str, asyncio.Task] = {}

    def _emit(self, event: Event) -> None:
        if self._on_event:
            self._on_event(event)

    def _create_coordinator_tools(self) -> list[Tool]:
        """Create the coordinator's tools: spawn, send, stop workers."""

        @tool
        async def spawn_worker(description: str, prompt: str) -> str:
            """Spawn a new worker to execute a task autonomously.
            description: Short label (3-5 words).
            prompt: Complete, self-contained instructions for the worker."""
            if len(self._running) >= self._max_workers:
                return f"Error: max {self._max_workers} concurrent workers. Wait for one to finish or stop one."

            worker_id = f"worker-{uuid.uuid4().hex[:8]}"
            task = WorkerTask(id=worker_id, description=description, prompt=prompt)
            self._workers[worker_id] = task

            # Create worker agent
            agent = Agent(
                model=self._model,
                instructions=self._worker_instructions
                or "You are a worker agent. Execute the task precisely. Report findings with file paths and line numbers.",
                tools=self._worker_tools,
                base_url=self._llm.base_url,
                api_key=self._llm.api_key,
                provider=self._llm.provider,
                max_turns=20,
            )

            # Run worker async
            async def _run_worker():
                try:
                    self._emit(Event(type="worker_start", data=worker_id, name=description))
                    result = await agent.run(prompt)
                    task.result = result
                    task.status = "completed"
                    task.llm_calls = agent.total_usage.prompt_tokens // 1000  # rough
                    task.tool_calls = sum(1 for _ in [])  # TODO: track properly
                except Exception as e:
                    task.result = f"Error: {type(e).__name__}: {e}"
                    task.status = "failed"
                finally:
                    task.end_time = time.time()
                    self._emit(Event(type="worker_done", data=worker_id, name=description))

            self._running[worker_id] = asyncio.create_task(_run_worker())
            return f"Worker '{description}' spawned with ID {worker_id}. You'll receive a task notification when it completes."

        @tool
        async def send_message(worker_id: str, message: str) -> str:
            """Send a follow-up message to a completed worker to continue its work.
            The worker retains its full context from the previous run."""
            task = self._workers.get(worker_id)
            if not task:
                return f"Error: worker {worker_id} not found"
            if task.status == "running":
                return f"Error: worker {worker_id} is still running. Wait for it to complete."

            # Create fresh agent with worker's context
            task.status = "running"
            task.prompt = message

            agent = Agent(
                model=self._model,
                instructions=self._worker_instructions
                or "You are a worker agent. Execute the task precisely.",
                tools=self._worker_tools,
                base_url=self._llm.base_url,
                api_key=self._llm.api_key,
                provider=self._llm.provider,
                max_turns=20,
            )

            async def _run_continued():
                try:
                    self._emit(Event(type="worker_start", data=worker_id, name=task.description))
                    result = await agent.run(message)
                    task.result = result
                    task.status = "completed"
                except Exception as e:
                    task.result = f"Error: {type(e).__name__}: {e}"
                    task.status = "failed"
                finally:
                    task.end_time = time.time()
                    self._emit(Event(type="worker_done", data=worker_id, name=task.description))

            self._running[worker_id] = asyncio.create_task(_run_continued())
            return f"Continued worker '{task.description}' ({worker_id}). You'll receive a task notification when it completes."

        @tool
        async def stop_worker(worker_id: str) -> str:
            """Stop a running worker."""
            if worker_id in self._running:
                self._running[worker_id].cancel()
                task = self._workers.get(worker_id)
                if task:
                    task.status = "stopped"
                    task.end_time = time.time()
                return f"Worker {worker_id} stopped."
            return f"Worker {worker_id} not found or already finished."

        return [spawn_worker, send_message, stop_worker]

    async def run(self, task: str) -> str:
        """Run the coordinator loop.

        The coordinator plans, spawns workers, receives notifications, and synthesizes.
        """
        coordinator_tools = self._create_coordinator_tools()

        coordinator = Agent(
            model=self._model,
            instructions=COORDINATOR_SYSTEM_PROMPT,
            tools=coordinator_tools,
            base_url=self._llm.base_url,
            api_key=self._llm.api_key,
            provider=self._llm.provider,
            max_turns=self._max_turns,
        )
        if self._on_event:
            coordinator.on_event = self._on_event

        messages = coordinator._prepare_messages(task, session=None)

        for turn in range(self._max_turns):
            # Check for completed workers → inject task notifications
            for wid, wtask in list(self._workers.items()):
                if wtask.status in ("completed", "failed", "stopped") and wid in self._running:
                    notification = format_task_notification(wtask)
                    messages.append(Message(role="user", content=notification))
                    del self._running[wid]
                    self._emit(
                        Event(
                            type="task_notification",
                            data=notification,
                            name=wtask.description,
                        )
                    )

            # If workers are still running, wait briefly for them
            if self._running and not any(
                m.role == "user" and "<task-notification>" in (m.content or "")
                for m in messages[-3:]
            ):
                # Wait for at least one worker to complete
                done, _ = await asyncio.wait(
                    self._running.values(),
                    timeout=60,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    continue  # Loop back to inject notifications

            # Coordinator turn
            response = await coordinator._llm_call_with_fallback(messages)
            messages.append(response.message)

            if not response.message.tool_calls:
                # Coordinator responded with text — check if work is done
                if not self._running:
                    return response.message.content or ""
                # Workers still running, wait and loop
                continue

            # Execute coordinator tools (spawn/send/stop)
            messages, final = await coordinator._execute_tools(
                response.message.tool_calls, messages, set()
            )

            # After spawning, give workers time to start
            if self._running:
                await asyncio.sleep(0.5)

        return messages[-1].content if messages else "Coordinator reached max turns."
