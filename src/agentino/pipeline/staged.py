"""Staged pipeline — runs an agent through a sequence of stages with per-stage tools.

Each stage has its own prompt, tool set, verdict tool, and optional skip condition.
A shared FactStore carries structured data between stages (no text summarization).

Usage:
    from agentino.pipeline.staged import StagedPipeline, StageDef, FactStore

    class MyFacts(FactStore):
        task_type: str = ""
        def to_context(self) -> str:
            return f"task_type={self.task_type}"

    stages = [
        StageDef(name="classify", prompt="...", tools=["read"], verdict_tool="classify_verdict"),
        StageDef(name="execute", prompt="...", tools=["read", "write"]),
        StageDef(name="complete", prompt="...", tools=["report"], verdict_tool="report"),
    ]

    pipeline = StagedPipeline(stages=stages, facts=MyFacts())
    result = await pipeline.run(agent_template, task_text)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agentino.core.agent import Agent
from agentino.core.message import Event, EventType, Message
from agentino.core.tool import Tool
from agentino.reliability.resilience import repair_messages


@runtime_checkable
class FactStore(Protocol):
    """Protocol for shared fact stores between stages.

    Applications define concrete classes with domain-specific fields.
    The pipeline uses to_context() to inject facts into stage prompts.
    """

    def to_context(self) -> str:
        """Serialize facts as structured text for LLM prompt injection."""
        ...


@dataclass
class StageDef:
    """Definition of a single pipeline stage."""

    name: str
    prompt: str = ""
    number: int = 0  # display order (optional, for rendering)
    gate: str = ""  # completion criteria (for LLM judge or manual check)
    tools: list[str] | None = None  # None = all tools
    verdict_tool: str = ""  # tool that signals stage completion
    max_turns: int = 10
    skip_condition: Callable[[FactStore], bool] | None = None

    # Retry / jump support
    repeatable: bool = False  # can this stage retry on failure?
    max_cycles: int = 3  # max retries for this stage
    on_fail: str = "retry"  # "retry" = same stage, or stage name to jump to

    def should_skip(self, facts: FactStore | None) -> bool:
        """Check if this stage should be skipped based on facts."""
        if self.skip_condition and facts:
            return self.skip_condition(facts)
        return False


@dataclass
class StageResult:
    """Result from running a single stage."""

    name: str
    completed: bool = False
    failed: bool = False  # stage failed gate check
    verdict_called: bool = False
    verdict_args: dict[str, Any] = field(default_factory=dict)
    turns_used: int = 0
    cycles_used: int = 1
    early_exit: bool = False  # pipeline should stop (e.g., security threat)
    output: str = ""  # last assistant text output from this stage


class StagedPipeline:
    """Runs an agent through a sequence of stages with per-stage tool filtering.

    Principles applied:
    - #1 Structured output: verdict tools, not text signals
    - #2 Deterministic gates: code reads verdict args
    - #3 Durable fact store: FactStore injected, not summarized
    - #5 Minimum tools per stage: filtered per StageDef
    - #8 Structured handover: facts.to_context() in each stage prompt
    """

    def __init__(
        self,
        stages: list[StageDef],
        facts: FactStore | None = None,
        max_reprompts: int = 3,
        global_max_cycles: int = 20,
        failure_check: Callable[[StageResult, FactStore | None], bool] | None = None,
    ):
        self.stages = stages
        self.facts = facts
        self.max_reprompts = max_reprompts
        self.global_max_cycles = global_max_cycles

        # Optional app-provided failure check. Returns True if stage failed.
        # Default: stage failed if not completed.
        def _default_failure_check(r, f):
            if not r.completed:
                return True
            # Check verdict args for REJECT/FAIL
            verdict = str(r.verdict_args.get("result", r.verdict_args.get("verdict", ""))).upper()
            return verdict in ("REJECT", "FAIL")

        self._failure_check = failure_check or _default_failure_check

    async def run(
        self,
        template: Agent,
        task_text: str,
        base_instructions: str = "",
        on_event: Callable[[Event], None] | None = None,
        session: Any = None,
    ) -> list[StageResult]:
        """Run all stages with retry/jump support. Returns list of StageResults."""
        import uuid

        all_tools = {t.name: t for t in template.tools}
        messages: list[Message] = []
        results: list[StageResult] = []
        stage_index = {s.name: i for i, s in enumerate(self.stages)}
        # Generate unique working file path for document-based stages
        working_file = f"/tmp/agentino_answer_{uuid.uuid4().hex[:8]}.md"

        # Load chat history for context (recent messages from session)
        chat_history: list[Message] = []
        if session:
            chat_history = session.load()

        # Reset gate state at pipeline start (clean slate per invocation)
        from agentino.core.context import get_context

        gm = get_context("_gate_manager")
        if gm:
            gm.reset()

        global_cycles = 0
        idx = 0
        next_idx = None  # Explicit jump target, None = advance sequentially
        rejection_feedback = ""  # Feedback from a failed stage to pass to the jump target

        while idx < len(self.stages):
            stage = self.stages[idx]

            # Global budget check
            global_cycles += 1
            if global_cycles > self.global_max_cycles:
                if on_event:
                    on_event(
                        Event(
                            type=EventType.STAGE_FAIL,
                            data={"name": "BUDGET", "reason": "global_max_cycles exceeded"},
                        )
                    )
                break

            # Check skip condition
            if stage.should_skip(self.facts):
                results.append(StageResult(name=stage.name, completed=True))
                if on_event:
                    on_event(Event(type=EventType.STAGE_SKIP, data=stage.name))
                idx += 1
                continue

            next_idx = None  # Reset jump target

            # Filter tools
            if stage.tools is not None:
                stage_tools = [all_tools[n] for n in stage.tools if n in all_tools]
            else:
                stage_tools = list(template.tools)

            # Auto-add verdict tool if stage requires it and it's not already present
            if stage.verdict_tool:
                has_verdict = any(t.name == stage.verdict_tool for t in stage_tools)
                if not has_verdict:
                    # Check template agent's tools first (custom verdicts)
                    for t in template.tools:
                        if t.name == stage.verdict_tool:
                            stage_tools.append(t)
                            has_verdict = True
                            break
                    # Then check builtins
                    if not has_verdict:
                        from agentino.builtin_tools import BUILTIN_TOOLS

                        for bt in BUILTIN_TOOLS:
                            if bt.name == stage.verdict_tool:
                                stage_tools.append(bt)
                                break

            # Build stage prompt — inject working_file path
            facts_context = self.facts.to_context() if self.facts else ""
            prompt_text = stage.prompt.replace("{{working_file}}", working_file)
            stage_prompt = (
                f"{base_instructions}\n\n"
                f"## Stage: {stage.name.upper()}\n\n"
                f"{prompt_text}\n\n"
                f"{facts_context}\n"
            )

            # Create per-stage agent — inherit tool_instructions from template
            agent = Agent(
                model=template.model,
                instructions=stage_prompt,
                tools=stage_tools,
                temperature=template.temperature,
                max_turns=stage.max_turns,
                tool_result_cap=template._tool_result_cap
                if hasattr(template, "_tool_result_cap")
                else 4000,
                tool_instructions=template._tool_instructions
                if hasattr(template, "_tool_instructions")
                else "",
                base_url=template._llm.base_url,
                api_key=template._llm.api_key,
                provider=template._llm.provider,
            )
            # Forward agent events (tool calls, text) to the pipeline's on_event
            if on_event:
                agent.on_event = on_event

            # Each stage starts fresh — only its own prompt + task text
            messages = agent._prepare_messages(task_text, session=None)
            # Inject previous stage output so data flows between stages
            if results and results[-1].output:
                prev = results[-1]
                messages.append(
                    Message(
                        role="user",
                        content=f"OUTPUT FROM PREVIOUS STAGE ({prev.name}):\n{prev.output[-8000:]}",
                    )
                )
            # Inject chat history as context for the first stage (understand)
            if chat_history and idx == 0:
                history_lines = []
                for m in chat_history[-20:]:  # last 20 messages max
                    role = "User" if m.role == "user" else "Assistant"
                    if m.content:
                        history_lines.append(f"{role}: {m.content[:500]}")
                if history_lines:
                    history_text = "\n".join(history_lines)
                    messages.append(
                        Message(
                            role="user",
                            content=f"CONVERSATION HISTORY (prior messages in this thread):\n{history_text}\n\nNow answer the latest message above. Use the history for context.",
                        )
                    )
            # If prior stages wrote to working file, tell this stage about it
            if Path(working_file).exists():
                hint = f'Prior stage findings are in: {working_file}\nCall read_file("{working_file}") to see them before proceeding.'
                if rejection_feedback:
                    hint = f"{rejection_feedback}\n\n{hint}"
                    rejection_feedback = ""  # consumed
                messages.append(Message(role="user", content=hint))
            elif rejection_feedback:
                messages.append(Message(role="user", content=rejection_feedback))
                rejection_feedback = ""  # consumed

            if on_event:
                on_event(Event(type=EventType.STAGE_START, data=stage.name))

            # Run stage (with per-stage retry)
            cycle = 0
            result = StageResult(name=stage.name)

            while True:
                cycle += 1
                result = await self._run_stage(agent, stage, stage_tools, messages)
                result.cycles_used = cycle

                # Check failure
                failed = self._failure_check(result, self.facts)
                result.failed = failed

                if not failed:
                    break  # Success — move to next stage

                # Failed — decide: jump, retry, or give up
                if stage.on_fail != "retry" and stage.on_fail in stage_index:
                    # Jump to named stage — carry rejection feedback
                    next_idx = stage_index[stage.on_fail]
                    reason = result.verdict_args.get("summary", "") or result.verdict_args.get(
                        "reason", ""
                    )
                    rejection_feedback = (
                        (
                            f"REJECTED by {stage.name.upper()} stage. "
                            f"You MUST fix these issues:\n{reason}"
                        )
                        if reason
                        else ""
                    )
                    if on_event:
                        on_event(
                            Event(
                                type=EventType.STAGE_FAIL,
                                data={"name": stage.name, "jump_to": stage.on_fail},
                            )
                        )
                    break

                if stage.repeatable and cycle < stage.max_cycles:
                    # Retry same stage — include rejection reason
                    reason = result.verdict_args.get("summary", "") or result.verdict_args.get(
                        "reason", ""
                    )
                    retry_msg = f"Stage {stage.name} failed. Retry ({cycle}/{stage.max_cycles})."
                    if reason:
                        retry_msg += (
                            f"\nREJECTION REASON: {reason}\nYou MUST fix these specific issues."
                        )
                    if on_event:
                        on_event(
                            Event(
                                type=EventType.STAGE_FAIL, data={"name": stage.name, "retry": cycle}
                            )
                        )
                    messages.append(Message(role="user", content=retry_msg))
                    continue

                break  # Give up

            results.append(result)

            if on_event:
                etype = (
                    EventType.STAGE_COMPLETE
                    if result.completed and not result.failed
                    else EventType.STAGE_FAIL
                )
                on_event(Event(type=etype, data={"name": stage.name, **result.verdict_args}))

            if result.early_exit:
                break

            # Advance: jump target or next sequential stage
            idx = next_idx if next_idx is not None else idx + 1

        self.last_working_file = working_file

        # Save exchange to session for conversation continuity
        if session:
            import time

            reply = ""
            if working_file and Path(working_file).exists():
                reply = Path(working_file).read_text(encoding="utf-8").strip()
            if not reply:
                for r in reversed(results):
                    if r.output:
                        reply = r.output
                        break
            session.append(
                [
                    Message(role="user", content=task_text, timestamp=time.time()),
                    Message(
                        role="assistant", content=reply or "(no response)", timestamp=time.time()
                    ),
                ]
            )

        return results

    async def _run_stage(
        self,
        agent: Agent,
        stage: StageDef,
        tools: list[Tool],
        messages: list[Message],
    ) -> StageResult:
        """Execute a single stage with verdict enforcement."""
        seen_calls: set[str] = set()
        reprompt_count = 0
        result = StageResult(name=stage.name)

        for turn in range(stage.max_turns):
            result.turns_used = turn + 1
            messages = repair_messages(messages)

            # LLM call — force tool call on first turn when verdict_tool is set
            tc = "required" if stage.verdict_tool and turn == 0 else None
            response = await agent._llm_call_with_fallback(messages, tool_choice=tc)
            messages.append(response.message)
            agent._emit(Event(type=EventType.LLM_RESPONSE, usage=response.usage))

            # No tool calls — text response from LLM
            if not response.message.tool_calls:
                if not stage.verdict_tool:
                    # No verdict required — text response = stage complete
                    result.completed = True
                    result.output = response.message.content or ""
                    break
                reprompt_count += 1
                if reprompt_count >= self.max_reprompts:
                    break
                messages.append(
                    Message(
                        role="user",
                        content=f"You must call {stage.verdict_tool} to complete this stage.",
                    )
                )
                continue

            reprompt_count = 0

            # Check for verdict tool call
            verdict_called = (
                any(tc.name == stage.verdict_tool for tc in response.message.tool_calls)
                if stage.verdict_tool
                else False
            )

            # Execute tools
            messages, final = await agent._execute_tools(
                response.message.tool_calls,
                messages,
                seen_calls,
            )

            # FinalResult from any tool → early exit
            if final is not None:
                result.completed = True
                result.early_exit = True

            if verdict_called:
                # Extract verdict args
                verdict_tc_id = None
                for tc in response.message.tool_calls:
                    if tc.name == stage.verdict_tool:
                        result.verdict_called = True
                        result.verdict_args = dict(tc.arguments)
                        verdict_tc_id = tc.id
                        break
                # Check if the verdict tool returned a rejection (gate/hook blocked it)
                # Search for the specific verdict tool's result, not just last message
                verdict_result = ""
                for m in reversed(messages):
                    if m.role == "tool" and m.tool_call_id == verdict_tc_id:
                        verdict_result = m.content or ""
                        break
                    elif m.role == "tool" and m.name == stage.verdict_tool:
                        verdict_result = m.content or ""
                        break
                if (
                    "REJECTED" in str(verdict_result)
                    or "WRONG TOOL" in str(verdict_result)
                    or "BLOCKED" in str(verdict_result)
                ):
                    result.completed = False
                    result.verdict_called = False
                else:
                    result.completed = True

            if result.completed:
                # Capture output: prefer last assistant text, fallback to last tool result
                if not result.output:
                    for m in reversed(messages):
                        if m.role == "assistant" and m.content:
                            result.output = m.content
                            break
                        if m.role == "tool" and m.content:
                            result.output = m.content
                            break
                break

        return result


# ── Reusable utilities for staged pipelines ──


async def summarize_stage_output(
    text: str,
    stage_name: str,
    llm: Any,
    model: str | None = None,
    max_len: int = 2000,
) -> str:
    """Summarize stage output for passing to the next stage.

    Short output (<max_len) is passed through unchanged.
    Longer output is LLM-summarized, with truncation as fallback.
    """
    if len(text) <= max_len:
        return text

    sample = text[:8000] if len(text) > 8000 else text

    try:
        resp = await llm.chat(
            messages=[
                Message(
                    role="user",
                    content=(
                        f"Summarize the output of stage '{stage_name}' for the next stage.\n"
                        f"Keep: key decisions, files created/modified, test results, errors, exact IDs.\n"
                        f"Drop: raw code, verbose tool output, boilerplate.\n"
                        f"Max {max_len - 200} chars.\n\n"
                        f"Stage output:\n{sample}"
                    ),
                )
            ],
            model=model,
            temperature=0,
        )
        summary = (resp.message.content or "").strip()
        if summary:
            return f"[{stage_name} summary]\n{summary}"
    except Exception:
        pass

    return text[:max_len] + f"\n\n[... truncated from {len(text)} chars]"


def parse_verdict(text: str) -> str | None:
    """Extract structured verdict from stage output.

    Returns "ACCEPT", "FAIL", or "REJECT" if a VERDICT: tag is found.
    Returns None if no structured verdict is present.
    """
    import re

    match = re.search(r"VERDICT:(ACCEPT|FAIL|REJECT)", text)
    return match.group(1) if match else None


async def judge_stage_failure(
    text: str,
    gate: str,
    llm: Any,
    model: str | None = None,
) -> bool:
    """Detect stage failure: structured verdict first, LLM judge as fallback.

    Fast path: VERDICT:FAIL or VERDICT:REJECT → immediate failure (no LLM call).
    Fast path: VERDICT:ACCEPT → immediate success (no LLM call).
    Slow path: no verdict found → LLM judge decides.

    Returns True if the stage failed, False if it passed.
    On judge error, assumes success (avoids infinite retries).
    """
    if not text.strip():
        return False

    # Fast path: structured verdict from stage_verdict tool
    verdict = parse_verdict(text)
    if verdict is not None:
        return verdict in ("FAIL", "REJECT")

    # Slow path: LLM judge for unstructured output
    if len(text) > 3000:
        sample = text[:1500] + "\n\n[...middle truncated...]\n\n" + text[-1500:]
    else:
        sample = text

    gate_line = f"\nCompletion criteria: {gate}\n" if gate else ""

    prompt = (
        "Judge whether the agent completed its current stage.\n"
        f"{gate_line}\n"
        "Rules:\n"
        "- SUCCESS if the agent's work passed its criteria\n"
        "- SUCCESS if failures are pre-existing or unrelated\n"
        "- FAIL only if the agent's own output is broken or incomplete\n"
        "- When in doubt: SUCCESS\n\n"
        f"Agent output:\n{sample}\n\n"
        "Respond with ONLY one word: SUCCESS or FAIL"
    )

    try:
        resp = await llm.chat(
            messages=[Message(role="user", content=prompt)],
            model=model,
            temperature=0,
        )
        result = (resp.message.content or "").strip().upper()
        return result.startswith("FAIL")
    except Exception:
        return False
