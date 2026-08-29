"""Tests for message types and serialization."""

from agentino import Message, ToolCall, Usage


def test_user_message_to_api():
    msg = Message(role="user", content="Hello")
    api = msg.to_api()
    assert api == {"role": "user", "content": "Hello"}


def test_assistant_message_with_tool_calls():
    msg = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "test"})],
    )
    api = msg.to_api()
    assert api["role"] == "assistant"
    assert len(api["tool_calls"]) == 1
    assert api["tool_calls"][0]["function"]["name"] == "search"


def test_tool_result_message():
    msg = Message(role="tool", content="result data", tool_call_id="c1", name="search")
    api = msg.to_api()
    assert api["role"] == "tool"
    assert api["tool_call_id"] == "c1"
    assert api["name"] == "search"


def test_jsonl_roundtrip():
    msg = Message(
        role="assistant",
        content="text",
        tool_calls=[ToolCall(id="c1", name="fn", arguments={"a": 1})],
        timestamp=1234567890.0,
    )
    data = msg.to_jsonl()
    restored = Message.from_jsonl(data)

    assert restored.role == msg.role
    assert restored.content == msg.content
    assert restored.tool_calls[0].name == "fn"
    assert restored.tool_calls[0].arguments == {"a": 1}
    assert restored.timestamp == 1234567890.0


def test_usage_arithmetic():
    u1 = Usage(prompt_tokens=100, completion_tokens=20)
    u2 = Usage(prompt_tokens=50, completion_tokens=10)
    total = u1 + u2
    assert total.prompt_tokens == 150
    assert total.completion_tokens == 30
    assert total.total_tokens == 180
