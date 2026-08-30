"""Tests for JSONL session persistence."""

import tempfile
from pathlib import Path

from agentino import Message, Session


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(Path(tmp) / "test.jsonl")

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        session.save(messages)

        loaded = session.load()
        assert len(loaded) == 3
        assert loaded[0].role == "user"
        assert loaded[0].content == "Hello"
        assert loaded[2].content == "How are you?"


def test_system_messages_not_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(Path(tmp) / "test.jsonl")

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
        session.save(messages)

        loaded = session.load()
        assert len(loaded) == 2
        assert all(m.role != "system" for m in loaded)


def test_tool_calls_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(Path(tmp) / "test.jsonl")

        from agentino import ToolCall

        messages = [
            Message(role="user", content="Search for cats"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "cats"})],
            ),
            Message(role="tool", content="Found 5 cats", tool_call_id="c1", name="search"),
            Message(role="assistant", content="I found 5 cats!"),
        ]
        session.save(messages)

        loaded = session.load()
        assert len(loaded) == 4
        assert loaded[1].tool_calls is not None
        assert loaded[1].tool_calls[0].name == "search"
        assert loaded[2].role == "tool"
        assert loaded[2].tool_call_id == "c1"


def test_max_messages_trim():
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(Path(tmp) / "test.jsonl", max_messages=4)

        messages = [Message(role="user", content=f"msg-{i}") for i in range(10)]
        session.save(messages)

        loaded = session.load()
        assert len(loaded) == 4
        assert loaded[0].content == "msg-6"  # kept last 4


def test_clear():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jsonl"
        session = Session(path)

        session.save([Message(role="user", content="hi")])
        assert path.exists()

        session.clear()
        assert not path.exists()


def test_append():
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(Path(tmp) / "test.jsonl")

        session.save([Message(role="user", content="first")])
        session.append([Message(role="assistant", content="second")])

        loaded = session.load()
        assert len(loaded) == 2
        assert loaded[1].content == "second"


def test_empty_session():
    with tempfile.TemporaryDirectory() as tmp:
        session = Session(Path(tmp) / "nonexistent.jsonl")
        assert session.load() == []


def test_ephemeral_never_touches_disk():
    """Ephemeral sessions load nothing and persist nothing — no file is created."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ephemeral.jsonl"
        session = Session(path, ephemeral=True)

        session.save([Message(role="user", content="hello")])
        session.append([Message(role="assistant", content="world")])

        assert session.load() == []
        assert not path.exists()


def test_ephemeral_ignores_existing_history():
    """An ephemeral session does not read a pre-existing file on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.jsonl"
        Session(path).save([Message(role="user", content="prior")])
        assert path.exists()

        assert Session(path, ephemeral=True).load() == []
