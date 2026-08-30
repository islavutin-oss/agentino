"""Agent — the async core tool-calling loop with built-in resilience."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from agentino.core.context import get_context
from agentino.core.llm import LLMClient, LLMResponse
from agentino.core.message import Event, EventType, Message, ToolCall, Usage
from agentino.core.session import Session
from agentino.core.tool import FinalResult, Tool
from agentino.extras.knowledge import KnowledgeBase
from agentino.reliability.resilience import (
    compact_history,
    estimate_tokens,
    repair_messages,
    retry_with_backoff,
    strip_think_tags,
    truncate_result,
)
from agentino.safety.sanitize import sanitize_tool_args


class Agent:
    """An async LLM agent with tools and built-in resilience.

    Tool execution chain (unified):
        hooks.PreToolUse → gates.check → tool.validate_input → tool.check_permission → tool.fn → hooks.PostToolUse

    The agent runs a loop:
    1. Send messages + tools to the LLM (with retry on transient failures)
    2. If the LLM calls tools → execute them (async-native) → truncate results → loop to 1
    3. If the LLM returns text → strip think tags → done
    4. If context is filling up → compact history automatically

    All methods are async-native. Use run() for one-shot, stream() for streaming.
    """

    def __init__(
        self,
        model: str | None = None,
        instructions: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 20,
        temperature: float = 0.7,
        on_event: Callable[[Event], None] | None = None,
        # Resilience settings
        max_retries: int = 3,
        tool_result_cap: int = 4000,
        think_filter: bool = True,
        auto_compact: bool = True,
        context_window: int = 128_000,
        max_continuations: int = 2,
        provider: str | None = None,
        name: str = "",
        # Knowledge base (auto-set by config loader from skills/*/knowledge/)
        knowledge: KnowledgeBase | None = None,
        # Fallback models — tried in order if primary fails
        fallback_models: list[str] | None = None,
        # Tool call style guidance — appended to system prompt when tools exist.
        # Set to "" to disable. Configurable via agents.yml `tool_instructions`.
        tool_instructions: str = (
            "Default: do not narrate routine, low-risk tool calls (just call the tool). "
            "Narrate only when it helps: multi-step work, sensitive actions, or when the user explicitly asks. "
            "Keep narration brief and value-dense; avoid repeating obvious steps. "
            "When a first-class tool exists for an action, use the tool directly instead of asking the user."
        ),
        # When True, text-only responses (no tool calls) are retried.
        # Use for agents that must always act via tools (orchestrators, workflows).
        require_tool_use: bool = False,
        # Optional hook: (text, turn, max_turns, had_tool_calls) → nudge message to retry, or None
        # to accept.
        # Agents use this for domain-specific handling (refusals, forced tool use, etc.)
        # Can be sync or async — Agent auto-detects and awaits if needed.
        response_filter: Callable[[str, int, int, bool], str | None] | None = None,
        # Sanitization: list of parameter names treated as file paths (cleaned before execution)
        sanitize_path_params: list[str] | None = None,
    ):
        self.instructions = instructions
        self.tools = list(tools or [])
        self.max_turns = max_turns
        self.temperature = temperature
        self.on_event = on_event
        self.name = name
        self.knowledge = knowledge
        self.fallback_models = fallback_models or []
        self._tool_instructions = tool_instructions

        # Auto-register search_knowledge tool when knowledge base exists
        if self.knowledge and self.knowledge.entries:
            search_tool = self.knowledge.make_search_tool()
            # Add only if not already present (user may have overridden)
            if not any(t.name == search_tool.name for t in self.tools):
                self.tools.append(search_tool)

        # Resilience
        self._max_retries = max_retries
        self._tool_result_cap = tool_result_cap
        self._think_filter = think_filter
        self._auto_compact = auto_compact
        self._require_tool_use = require_tool_use
        self._response_filter = response_filter
        self._context_window = context_window
        self._max_continuations = max_continuations

        self._sanitize_path_params = sanitize_path_params or []

        self._llm = LLMClient(
            base_url=base_url,
            api_key=api_key,
            default_model=model,
            provider=provider,
        )

        # Use LLM client's resolved model (handles Codex auto-detection)
        self.model = model or self._llm.default_model

        # Tool lookup
        self._tool_map: dict[str, Tool] = {t.name: t for t in self.tools}

        # Usage tracking
        self.last_usage = Usage()
        self.total_usage = Usage()

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to this agent (updates both tools list and lookup map)."""
        if not any(t.name == tool.name for t in self.tools):
            self.tools.append(tool)
        self._tool_map[tool.name] = tool

    def add_tools(self, tools: list[Tool]) -> None:
        """Add multiple tools (use this instead of tools.extend)."""
        for t in tools:
            self.add_tool(t)

    # ------------------------------------------------------------------
    # Text response handling — shared by run() and stream()
    # ------------------------------------------------------------------

    def _reset_retries(self) -> None:
        """Reset per-run retry counters."""
        pass  # Kept for compatibility; filters manage their own state

    async def _check_text_response(
        self, text: str, messages: list[Message], turn: int, had_tool_calls: bool = False
    ) -> str | None:
        """Check if a text response should be retried. Returns nudge message or None to accept.

        1. Empty (thinking-only) → silent retry (universal)
        2. Agent response_filter hook → domain-specific handling
        """
        at_limit = turn >= self.max_turns - 1

        # 1. Empty response (thinking tokens stripped to nothing)
        if not text and not at_limit:
            return "You returned an empty response. Please proceed with the task — call the appropriate tools."

        # 1b. At limit with empty response after tool calls — force summarization
        if at_limit and not text.strip() and had_tool_calls:
            return (
                "You have exhausted your tool call limit. DO NOT call any more tools. "
                "Summarize ALL results you have already collected and give the user "
                "a comprehensive, well-structured answer."
            )

        # 2. Agent-provided filter (refusals, forced tool use, etc.)
        if self._response_filter and not at_limit:
            result = self._response_filter(text, turn, self.max_turns, had_tool_calls)
            # Support both sync and async filters
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                return result

        return None  # Accept the response

    async def _llm_call_with_fallback(
        self, messages: list[Message], tool_choice: str | None = None
    ) -> LLMResponse:
        """Try primary model, then fallbacks in order on failure.

        On 401/403 errors, attempts token refresh before falling through to next model.
        """
        import httpx

        models_to_try = [self.model] + self.fallback_models

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                response = await retry_with_backoff(
                    lambda m=model, tc=tool_choice: self._llm.chat(
                        messages=messages,
                        tools=self.tools or None,
                        model=m,
                        temperature=self.temperature,
                        tool_choice=tc,
                    ),
                    max_attempts=self._max_retries,
                    initial_delay=1.0,
                )
                if model != self.model:
                    self._emit(
                        Event(type=EventType.FALLBACK, data=f"Using fallback model: {model}")
                    )
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    # Token may have expired mid-session — try refresh once
                    refreshed = await self._try_token_refresh()
                    if refreshed:
                        try:
                            return await retry_with_backoff(
                                lambda m=model: self._llm.chat(
                                    messages=messages,
                                    tools=self.tools or None,
                                    model=m,
                                    temperature=self.temperature,
                                ),
                                max_attempts=1,
                                initial_delay=0,
                            )
                        except Exception:
                            pass  # Fall through to next model
                last_error = e
                self._emit(Event(type=EventType.ERROR, data=f"Model {model} failed: {e}"))
                continue
            except Exception as e:
                last_error = e
                self._emit(Event(type=EventType.ERROR, data=f"Model {model} failed: {e}"))
                continue

        raise last_error or RuntimeError("All models failed")

    async def _try_token_refresh(self) -> bool:
        """Attempt to refresh the API token. Returns True if successful."""
        try:
            from agentino.safety.auth import (
                load_codex_credentials,
                load_credentials,
                refresh_openai_token,
                save_credentials,
            )

            creds = load_credentials("openai")
            if creds and creds.refresh_token:
                refreshed = refresh_openai_token(creds)
                if refreshed and not refreshed.is_expired:
                    save_credentials(refreshed)
                    # Update the HTTP client headers with new token
                    self._llm.api_key = refreshed.access_token
                    self._llm._client.headers["Authorization"] = f"Bearer {refreshed.access_token}"
                    return True
            # Try codex credentials
            creds = load_codex_credentials()
            if creds and not creds.is_expired:
                self._llm.api_key = creds.access_token
                self._llm._client.headers["Authorization"] = f"Bearer {creds.access_token}"
                return True
        except Exception:
            pass
        return False

    async def run(
        self, message: str, session: Session | None = None, images: list[str] | None = None
    ) -> str:
        """Run the agent to completion. Returns the final text response."""
        messages = self._prepare_messages(message, session, images=images)
        prev_turn_calls: set[str] = set()
        continuations = 0
        had_tool_calls = False
        self._reset_retries()
        response: LLMResponse | None = None

        for turn in range(self.max_turns):
            # Auto-compact if context is filling up
            if self._auto_compact:
                pre_tokens = estimate_tokens(messages)
                messages = await compact_history(
                    messages,
                    self._llm,
                    max_tokens=self._context_window,
                )
                post_tokens = estimate_tokens(messages)
                self._emit(
                    Event(
                        type="context",
                        data={
                            "tokens": post_tokens,
                            "max": self._context_window,
                            "compacted": pre_tokens != post_tokens,
                        },
                    )
                )

            # Repair before sending (fix orphaned tool results, etc.)
            messages = repair_messages(messages)

            # LLM call with fallback
            # Force tool calls on first turn when require_tool_use is set
            tc = "required" if self._require_tool_use and not had_tool_calls else None
            response = await self._llm_call_with_fallback(messages, tool_choice=tc)

            self.last_usage = response.usage
            self.total_usage = self.total_usage + response.usage
            messages.append(response.message)
            # Optional verbose LLM trace — when AGENTINO_LLM_TRACE=1 is set,
            # the LLM_RESPONSE event carries the actual prompt messages and
            # completion content alongside token counts. This is what the
            # harness dashboard's trial-detail "drill into the LLM call" view
            # consumes. Off by default because for chatty agents the
            # per-event payload can hit hundreds of KB.
            llm_data: dict[str, Any] | None = None
            if os.environ.get("AGENTINO_LLM_TRACE") == "1":
                llm_data = {
                    "prompt": [
                        {
                            "role": m.role,
                            "content": (m.content or "")[:8000],
                            "tool_calls": [
                                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                for tc in (m.tool_calls or [])
                            ]
                            if m.tool_calls
                            else None,
                            "tool_call_id": getattr(m, "tool_call_id", None),
                        }
                        for m in messages[:-1]  # exclude the response we just appended
                    ],
                    "completion": {
                        "content": (response.message.content or "")[:8000],
                        "tool_calls": [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in (response.message.tool_calls or [])
                        ]
                        if response.message.tool_calls
                        else None,
                    },
                    "model": getattr(response, "model", None) or self.model,
                }
            self._emit(Event(type=EventType.LLM_RESPONSE, usage=response.usage, data=llm_data))

            # Handle max_tokens continuation — reactive compact to free context
            if response.finish_reason == "length" and continuations < self._max_continuations:
                if self._auto_compact:
                    messages = await compact_history(
                        messages,
                        self._llm,
                        max_tokens=self._context_window,
                        reactive=True,
                    )
                    self._emit(Event(type="context", data={"reactive_compact": True}))
                messages.append(Message(role="user", content="Continue."))
                continuations += 1
                continue

            if not response.message.tool_calls:
                text = response.message.content or ""
                if self._think_filter:
                    text = strip_think_tags(text).strip()

                nudge = await self._check_text_response(text, messages, turn, had_tool_calls)
                if nudge:
                    messages.append(Message(role="user", content=nudge))
                    continue
                break  # Accept the response

            had_tool_calls = True

            # Execute tool calls (with truncation)
            # Only flag duplicates from the immediately previous turn (not all-time)
            messages, final = await self._execute_tools(
                response.message.tool_calls, messages, prev_turn_calls
            )

            # Track this turn's calls for next-turn dedup
            prev_turn_calls = {self._call_hash(c) for c in response.message.tool_calls}

            # A tool delivered a ready-made response — return it directly
            if final is not None:
                if session:
                    session.save(messages)
                return final.text

        # Save session
        if session:
            session.save(messages)

        # Apply think filter
        text = (response.message.content if response else "") or ""
        if self._think_filter and text:
            text = strip_think_tags(text)
        return text

    async def stream(self, message: str, session: Session | None = None) -> AsyncIterator[Event]:
        """Stream agent execution. Yields Events."""
        messages = self._prepare_messages(message, session)
        prev_turn_calls: set[str] = set()
        had_tool_calls = False
        self._reset_retries()
        final_message: Message | None = None

        for turn in range(self.max_turns):
            if self._auto_compact:
                pre_tokens = estimate_tokens(messages)
                messages = await compact_history(
                    messages,
                    self._llm,
                    max_tokens=self._context_window,
                )
                post_tokens = estimate_tokens(messages)
                yield Event(
                    type="context",
                    data={
                        "tokens": post_tokens,
                        "max": self._context_window,
                        "compacted": pre_tokens != post_tokens,
                    },
                )
            messages = repair_messages(messages)

            assistant_msg: Message | None = None

            async for event in self._llm.chat_stream(
                messages=messages,
                tools=self.tools or None,
                model=self.model,
                temperature=self.temperature,
            ):
                if event.type == EventType.TEXT:
                    yield event
                elif event.type == EventType.LLM_RESPONSE:
                    assistant_msg = event.data
                    self.last_usage = event.usage or Usage()
                    self.total_usage = self.total_usage + self.last_usage

            if assistant_msg is None:
                break

            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                text = assistant_msg.content or ""
                if self._think_filter:
                    text = strip_think_tags(text).strip()

                nudge = await self._check_text_response(text, messages, turn, had_tool_calls)
                if nudge:
                    messages.append(Message(role="user", content=nudge))
                    # Only emit error for non-empty retries after turn 0
                    # Turn 0 retries are normal (session restore orientation) — stay silent
                    if text and turn > 0:
                        yield Event(type=EventType.ERROR, data="response_filter_retry")
                    continue

                final_message = assistant_msg
                break

            had_tool_calls = True

            # Execute tool calls, yielding events
            # Only flag duplicates from the immediately previous turn
            final_result: FinalResult | None = None
            current_turn_calls: set[str] = set()
            for call in assistant_msg.tool_calls:
                call_hash = self._call_hash(call)
                if call_hash in prev_turn_calls:
                    result_text = "Error: duplicate tool call detected, breaking loop"
                    messages.append(
                        Message(
                            role="tool",
                            content=result_text,
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
                    yield Event(type=EventType.TOOL_RESULT, data=result_text, name=call.name)
                    continue
                current_turn_calls.add(call_hash)

                yield Event(type=EventType.TOOL_START, name=call.name, args=call.arguments)
                result = await self._execute_one_tool(call)

                if isinstance(result, FinalResult):
                    final_result = result
                    result_text = result.text
                else:
                    result_text = result

                messages.append(
                    Message(
                        role="tool",
                        content=result_text,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
                yield Event(type=EventType.TOOL_RESULT, data=result_text, name=call.name)

            # Track this turn's calls for next-turn dedup
            prev_turn_calls = current_turn_calls

            if final_result is not None:
                if session:
                    session.save(messages)
                yield Event(type=EventType.TEXT, data=final_result.text)
                yield Event(type=EventType.DONE, usage=self.last_usage, data=final_result.text)
                return

            final_message = assistant_msg

        if session:
            session.save(messages)

        yield Event(type=EventType.DONE, usage=self.last_usage, data=final_message)

    def _prepare_messages(
        self, user_message: str, session: Session | None, images: list[str] | None = None
    ) -> list[Message]:
        """Build the message list: system + history + new user message.

        Knowledge retrieval is handled by the search_knowledge tool —
        the LLM calls it when it needs facts, avoiding token waste on
        requests that don't need factual information.
        """
        system_content = self.instructions

        # Tool call style guidance (configurable via agents.yml or Agent constructor)
        if self.tools and self._tool_instructions:
            system_content += f"\n\n## Tool Call Style\n{self._tool_instructions}"

        messages: list[Message] = [Message(role="system", content=system_content)]

        if session:
            history = session.load()
            messages.extend(history)

        messages.append(
            Message(role="user", content=user_message, timestamp=time.time(), images=images)
        )
        return messages

    # Default read-only tools (fallback when is_read_only not set on Tool)
    _DEFAULT_CONCURRENT_SAFE = frozenset(
        {
            "read_file",
            "list_files",
            "grep",
            "search_files",
            "web_search",
            "web_fetch",
            "tree",
            "find",
            "search",
            "list_dir",
            "read",
            "get_time",
        }
    )

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        messages: list[Message],
        prev_turn_calls: set[str],
    ) -> tuple[list[Message], FinalResult | None]:
        """Execute tool calls and append results to messages.

        Read-only tools run in parallel for speed. Write tools run sequentially.
        Returns (messages, terminal) — if a tool returns FinalResult,
        the agent loop should terminate and return its text immediately.
        """
        terminal: FinalResult | None = None

        # Partition into batches: consecutive read-only → parallel, others → sequential
        batches: list[list[ToolCall]] = []
        current_batch: list[ToolCall] = []
        current_is_safe = True

        for call in tool_calls:
            tool_obj = self._tool_map.get(call.name)
            # Borrow #4: per-tool execution_mode overrides the global policy.
            mode = getattr(tool_obj, "execution_mode", None) if tool_obj else None
            if mode == "parallel":
                is_safe = True
            elif mode == "sequential":
                is_safe = False
            else:
                is_safe = (
                    tool_obj.is_read_only
                    if tool_obj and hasattr(tool_obj, "is_read_only")
                    else False
                ) or call.name in self._DEFAULT_CONCURRENT_SAFE
            if current_batch and is_safe != current_is_safe:
                batches.append((current_batch, current_is_safe))
                current_batch = []
            current_is_safe = is_safe
            current_batch.append(call)
        if current_batch:
            batches.append((current_batch, current_is_safe))

        for batch, is_safe in batches:
            if is_safe and len(batch) > 1:
                # Run read-only tools in parallel
                results = await self._execute_batch_parallel(batch, prev_turn_calls)
            else:
                # Run sequentially
                results = []
                for call in batch:
                    r = await self._execute_single(call, prev_turn_calls)
                    results.append(r)

            for call, (result_text, is_final) in zip(batch, results):
                if is_final and isinstance(result_text, FinalResult):
                    terminal = result_text
                    result_text = result_text.text
                if self._tool_result_cap:
                    result_text = truncate_result(result_text, self._tool_result_cap)
                messages.append(
                    Message(
                        role="tool",
                        content=result_text,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
                self._emit(Event(type=EventType.TOOL_RESULT, data=result_text, name=call.name))

        return messages, terminal

    async def _execute_single(
        self, call: ToolCall, prev_turn_calls: set[str]
    ) -> tuple[str | FinalResult, bool]:
        """Execute one tool call. Returns (result, is_final)."""
        call_hash = self._call_hash(call)
        if call_hash in prev_turn_calls:
            self._emit(Event(type=EventType.ERROR, data="loop_guard", name=call.name))
            return "Error: duplicate tool call detected, breaking loop", False

        self._emit(Event(type=EventType.TOOL_START, name=call.name, args=call.arguments))
        result = await self._execute_one_tool(call)
        is_final = isinstance(result, FinalResult)
        return result, is_final

    async def _execute_batch_parallel(
        self, calls: list[ToolCall], prev_turn_calls: set[str]
    ) -> list[tuple[str | FinalResult, bool]]:
        """Execute multiple read-only tools in parallel."""
        tasks = [self._execute_single(call, prev_turn_calls) for call in calls]
        return await asyncio.gather(*tasks)

    async def _execute_one_tool(self, call: ToolCall) -> str:
        """Execute a single tool call through the full chain:

        1. PreToolUse hooks → can block
        2. Gate check → precondition enforcement
        3. Sanitize arguments
        4. tool.execute (validate_input → check_permission → fn)
        5. PostToolUse hooks → observe result
        """
        tool = self._tool_map.get(call.name)
        if not tool:
            return f"Error: unknown tool '{call.name}'"

        # 1. PreToolUse hooks — may block OR rewrite arguments (Borrow #3)
        hook_mgr = get_context("_hook_manager")
        args = dict(call.arguments)  # copy so hook rewrites don't leak into the message log
        if hook_mgr:
            hook_result = await hook_mgr.fire(
                "PreToolUse",
                {
                    "tool_name": call.name,
                    "arguments": args,
                },
            )
            if hook_result.blocked:
                return hook_result.message
            if hook_result.arguments_override is not None:
                args = hook_result.arguments_override

        # 2. Gate check
        gm = get_context("_gate_manager")
        if gm:
            rejection = gm.check(call.name)
            if rejection:
                return rejection

        # 3. Sanitize path arguments
        if self._sanitize_path_params:
            args = sanitize_tool_args(args, self._sanitize_path_params)

        # 4. Execute (tool's own validate → permission → fn chain)
        result = await tool.execute(args)

        # 5. PostToolUse hooks — may override result (Borrow #3)
        if hook_mgr:
            post = await hook_mgr.fire(
                "PostToolUse",
                {
                    "tool_name": call.name,
                    "arguments": args,
                    "result": str(result) if not isinstance(result, FinalResult) else result.text,
                },
            )
            if post.result_override is not None and not isinstance(result, FinalResult):
                result = post.result_override

        return result

    def _call_hash(self, call: ToolCall) -> str:
        """Hash a tool call for loop guard dedup."""
        raw = json.dumps({"n": call.name, "a": call.arguments}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _emit(self, event: Event) -> None:
        """Emit an event to the callback if registered."""
        if self.on_event:
            self.on_event(event)

    async def close(self) -> None:
        """Close the underlying LLM client and release resources."""
        await self._llm.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        await self.close()
