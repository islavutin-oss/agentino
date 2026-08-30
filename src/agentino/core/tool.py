"""@tool decorator — turns Python functions into LLM-callable tools.

Supports both sync and async tool functions:
    @tool
    def search(query: str) -> str:
        return results

    @tool
    async def search_async(query: str) -> str:
        return await db.search(query)
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

# Python type → JSON Schema type
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolError:
    """Structured error from tool execution.

    Categories: timeout, invalid_args, validation, not_found, permission, internal.
    """

    category: str
    tool_name: str
    detail: str

    def format(self) -> str:
        """Format as a string for the LLM to interpret."""
        return f"Error: [{self.category}] {self.tool_name}: {self.detail}"


class FinalResult:
    """Return this from a tool to deliver a ready-made response directly.

    The agent loop completes immediately and returns this text as-is,
    skipping the final LLM call. Perfect for deterministic outputs like
    confirmations, templated messages, or pre-formatted responses.

        @tool
        async def process_order(...) -> str:
            order = await service.create(...)
            return FinalResult(render_confirmation(order))
    """

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


@dataclass
class Tool:
    """A tool that can be called by an LLM.

    Three-stage execution chain (inspired by Claude Code):
    1. validate_input() — check arguments before execution (fast, no side effects)
    2. check_permission() — verify the caller is allowed to do this
    3. fn() — execute the actual tool logic

    Stages 1-2 are optional. If not set, tool executes directly.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]
    timeout: float | None = None
    is_read_only: bool = False  # Safe for parallel execution
    validate_input: Callable[..., str | None] | None = None  # returns error msg or None
    check_permission: Callable[..., str | None] | None = None  # returns rejection msg or None
    # Optional arg-rewriter run before validate_input. Receives the raw args dict the LLM produced
    # and returns a new dict (e.g. coerce "" → None, normalise paths, fill defaults).
    # Sync only: runs in the synchronous part of the chain, must be cheap and pure.
    prepare_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # Borrow #4 (pi): per-tool override of the agent's batching policy.
    #   "parallel"   — fan out with sibling parallel calls even if not is_read_only
    #                  (e.g. rate-limited reads, slow API calls, independent tenants).
    #   "sequential" — never batch with siblings (e.g. shared mutable resource).
    # None falls back to the global policy (is_read_only or agent's _DEFAULT_CONCURRENT_SAFE).
    execution_mode: str | None = None

    @property
    def schema(self) -> dict[str, Any]:
        """OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def is_async(self) -> bool:
        """Check if the tool function is async."""
        return asyncio.iscoroutinefunction(self.fn)

    async def execute(self, arguments: dict[str, Any]) -> str | FinalResult:
        """Execute the tool with the 3-stage chain.

        1. validate_input — reject bad arguments early
        2. check_permission — reject unauthorized calls
        3. fn — execute the tool

        If any stage returns a string, it's returned as an error to the LLM.
        """
        # Stage 0: prepare_arguments — coerce/normalize raw LLM args before validation
        if self.prepare_arguments is not None:
            try:
                arguments = self.prepare_arguments(arguments)
            except Exception as e:
                return ToolError("invalid_args", self.name, str(e)).format()

        # Stage 1: Input validation (fast, no side effects)
        if self.validate_input is not None:
            try:
                error = (
                    self.validate_input(**arguments)
                    if not asyncio.iscoroutinefunction(self.validate_input)
                    else await self.validate_input(**arguments)
                )
                if error:
                    return error
            except Exception as e:
                return ToolError("validation", self.name, str(e)).format()

        # Stage 2: Permission check
        if self.check_permission is not None:
            try:
                rejection = (
                    self.check_permission(**arguments)
                    if not asyncio.iscoroutinefunction(self.check_permission)
                    else await self.check_permission(**arguments)
                )
                if rejection:
                    return rejection
            except Exception as e:
                return ToolError("permission", self.name, str(e)).format()

        # Stage 3: Execute
        try:
            if self.is_async:
                coro = self.fn(**arguments)
                if self.timeout is not None:
                    result = await asyncio.wait_for(coro, timeout=self.timeout)
                else:
                    result = await coro
            else:
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self.fn, **arguments),
                        timeout=self.timeout,
                    )
                else:
                    result = self.fn(**arguments)
            if isinstance(result, FinalResult):
                return result
            return str(result) if result is not None else ""
        except asyncio.TimeoutError:
            return ToolError(
                "timeout",
                self.name,
                f"timed out after {self.timeout}s — do NOT retry the same call, try a different approach",
            ).format()
        except TypeError as e:
            return ToolError("invalid_args", self.name, str(e)).format()
        except ValueError as e:
            return ToolError("validation", self.name, str(e)).format()
        except FileNotFoundError as e:
            return ToolError("not_found", self.name, str(e)).format()
        except PermissionError as e:
            return ToolError("permission", self.name, str(e)).format()
        except Exception as e:
            return ToolError("internal", self.name, f"{type(e).__name__}: {e}").format()


def _python_type_to_json(py_type: Any) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    if py_type is type(None):
        return {"type": "null"}

    if py_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[py_type]}

    origin = get_origin(py_type)
    args = get_args(py_type)

    # Optional[T] → T (with nullable note)
    if origin is Union and len(args) == 2 and type(None) in args:
        inner = [a for a in args if a is not type(None)][0]
        return _python_type_to_json(inner)

    # list[T]
    if origin is list and args:
        return {"type": "array", "items": _python_type_to_json(args[0])}

    # dict[str, T]
    if origin is dict:
        return {"type": "object"}

    # Literal["a", "b", "c"]
    try:
        from typing import Literal

        if get_origin(py_type) is Literal:
            return {"type": "string", "enum": list(get_args(py_type))}
    except ImportError:
        pass

    # Fallback
    return {"type": "string"}


def _build_schema(fn: Callable) -> dict[str, Any]:
    """Build JSON Schema parameters from function signature."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        py_type = hints.get(name, str)
        if py_type is inspect.Parameter.empty:
            py_type = str

        prop = _python_type_to_json(py_type)

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    is_read_only: bool = False,
    validate_input: Callable | None = None,
    check_permission: Callable | None = None,
    prepare_arguments: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    execution_mode: str | None = None,
) -> Tool | Callable[..., Tool]:
    """Decorator that turns a function into an LLM-callable Tool.

    Supports both sync and async functions:
        @tool
        def my_func(query: str) -> str:
            '''Search for something.'''
            return results

        @tool(is_read_only=True)
        def read_data(path: str) -> str:
            '''Read data (safe for parallel execution).'''
            ...

        @tool(validate_input=check_args, check_permission=verify_access)
        def write_data(path: str, content: str) -> str:
            '''Write with validation and permission checks.'''
            ...
    """

    def _wrap(f: Callable) -> Tool:
        tool_name = name or f.__name__
        tool_desc = description or inspect.getdoc(f) or f.__name__
        params = _build_schema(f)
        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=params,
            fn=f,
            timeout=timeout,
            is_read_only=is_read_only,
            validate_input=validate_input,
            check_permission=check_permission,
            prepare_arguments=prepare_arguments,
            execution_mode=execution_mode,
        )

    if fn is not None:
        return _wrap(fn)

    return _wrap
