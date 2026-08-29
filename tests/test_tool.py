"""Tests for the @tool decorator and schema generation."""

import pytest

from agentino import Tool, tool


@pytest.mark.asyncio
async def test_basic_tool():
    @tool
    def greet(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"

    assert isinstance(greet, Tool)
    assert greet.name == "greet"
    assert greet.description == "Say hello."
    assert await greet.execute({"name": "Alex"}) == "Hello, Alex!"


def test_tool_with_defaults():
    @tool
    def search(query: str, limit: int = 5) -> str:
        """Search for items."""
        return f"Found {limit} results for '{query}'"

    schema = search.parameters
    assert schema["required"] == ["query"]
    assert "limit" not in schema.get("required", [])
    assert schema["properties"]["limit"]["default"] == 5
    assert schema["properties"]["limit"]["type"] == "integer"


def test_tool_with_custom_name():
    @tool(name="custom_search", description="Custom description")
    def search(query: str) -> str:
        return query

    assert search.name == "custom_search"
    assert search.description == "Custom description"


def test_tool_schema_format():
    @tool
    def my_tool(text: str, count: int, flag: bool = False) -> str:
        """A tool."""
        return ""

    schema = my_tool.schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "my_tool"
    assert schema["function"]["parameters"]["properties"]["text"]["type"] == "string"
    assert schema["function"]["parameters"]["properties"]["count"]["type"] == "integer"
    assert schema["function"]["parameters"]["properties"]["flag"]["type"] == "boolean"
    assert set(schema["function"]["parameters"]["required"]) == {"text", "count"}


@pytest.mark.asyncio
async def test_tool_error_handling():
    @tool
    def bad_tool(x: str) -> str:
        """Fails."""
        raise ValueError("boom")

    result = await bad_tool.execute({"x": "test"})
    assert "Error:" in result
    assert "boom" in result


def test_tool_with_list_type():
    @tool
    def process(items: list[str]) -> str:
        """Process items."""
        return str(len(items))

    props = process.parameters["properties"]
    assert props["items"]["type"] == "array"
    assert props["items"]["items"]["type"] == "string"


def test_tool_with_optional():

    @tool
    def maybe(name: str, title: str | None = None) -> str:
        """Maybe titled."""
        return f"{title or ''} {name}"

    props = maybe.parameters["properties"]
    assert props["title"]["type"] == "string"
    assert maybe.parameters["required"] == ["name"]


def test_tool_with_timeout():
    @tool(timeout=30)
    def slow(query: str) -> str:
        """Slow search."""
        return query

    assert slow.timeout == 30


@pytest.mark.asyncio
async def test_async_tool():
    @tool
    async def async_greet(name: str) -> str:
        """Greet async."""
        return f"Hi, {name}!"

    assert isinstance(async_greet, Tool)
    assert async_greet.is_async
    assert await async_greet.execute({"name": "Bob"}) == "Hi, Bob!"


def test_sync_tool_not_async():
    @tool
    def sync_fn(x: str) -> str:
        """Sync."""
        return x

    assert not sync_fn.is_async


# ----------------------------------------------------------------------
# Borrow #2 — prepare_arguments hook
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_arguments_coerces_empty_string_to_default():
    """Classic 'LLM emits empty string for optional int' bug — fixed in one place."""

    def coerce(args: dict) -> dict:
        if args.get("limit") == "":
            args["limit"] = 10
        return args

    @tool(prepare_arguments=coerce)
    def search(q: str, limit: int = 10) -> str:
        return f"{q}/{limit}"

    assert await search.execute({"q": "cats", "limit": ""}) == "cats/10"


@pytest.mark.asyncio
async def test_prepare_arguments_runs_before_validate_input():
    """prepare_arguments must run first so validate_input sees normalised args."""
    seen: dict = {}

    def coerce(args: dict) -> dict:
        return {**args, "name": args["name"].strip().lower()}

    def validate(name: str) -> str | None:
        seen["validate_saw"] = name
        return None

    @tool(prepare_arguments=coerce, validate_input=validate)
    def greet(name: str) -> str:
        return f"hi {name}"

    out = await greet.execute({"name": "  ALICE  "})
    assert out == "hi alice"
    assert seen["validate_saw"] == "alice"


@pytest.mark.asyncio
async def test_prepare_arguments_exception_returns_invalid_args_error():
    """A raise inside prepare_arguments is reported as invalid_args, not propagated."""

    def boom(args: dict) -> dict:
        raise ValueError("nope")

    @tool(prepare_arguments=boom)
    def t(x: str) -> str:
        return x

    out = await t.execute({"x": "anything"})
    assert "invalid_args" in out
    assert "nope" in out


@pytest.mark.asyncio
async def test_prepare_arguments_passes_through_when_unset():
    """No prepare_arguments → behavior identical to before."""

    @tool
    def echo(x: str) -> str:
        return x

    assert await echo.execute({"x": "hi"}) == "hi"
