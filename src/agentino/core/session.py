"""JSONL session manager — conversation persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agentino.core.message import Message


class Session:
    """Conversation history stored as a JSONL file.

    Each line is a JSON-serialized message. Human-readable, appendable, git-friendly.

    Usage:
        session = Session("./chats/user-123.jsonl")
        messages = session.load()
        # ... agent loop adds messages ...
        session.save(messages)
    """

    def __init__(self, path: str | Path, max_messages: int = 100, ephemeral: bool = False):
        self.path = Path(path)
        self.max_messages = max_messages
        # ephemeral: never read or write disk — load() returns [], save()/
        # append() are no-ops. For one-shot/benchmark runs that must not
        # accumulate or share history.
        self.ephemeral = ephemeral

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB guard

    def load(self) -> list[Message]:
        """Load conversation history from disk."""
        if self.ephemeral:
            return []
        if not self.path.exists():
            return []

        # Guard against oversized session files
        if self.path.stat().st_size > self.MAX_FILE_SIZE:
            import logging

            logging.getLogger(__name__).warning(
                f"Session file too large ({self.path.stat().st_size} bytes), truncating"
            )
            self.path.unlink()
            return []

        messages: list[Message] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    messages.append(Message.from_jsonl(data))
                except (json.JSONDecodeError, KeyError):
                    continue

        return messages

    def save(self, messages: list[Message]) -> None:
        """Save conversation history to disk (overwrites)."""
        if self.ephemeral:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Trim to max_messages (keep most recent, skip system messages in count)
        non_system = [m for m in messages if m.role != "system"]
        if len(non_system) > self.max_messages:
            # Keep the last max_messages non-system messages
            keep = set(id(m) for m in non_system[-self.max_messages :])
            messages = [m for m in messages if m.role == "system" or id(m) in keep]

        with open(self.path, "w") as f:
            for msg in messages:
                if msg.role == "system":
                    continue  # don't persist system prompt
                d = msg.to_jsonl()
                if msg.timestamp is None:
                    d["ts"] = time.time()
                f.write(json.dumps(d) + "\n")

    def append(self, messages: list[Message]) -> None:
        """Append new messages to existing history."""
        if self.ephemeral:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for msg in messages:
                if msg.role == "system":
                    continue
                d = msg.to_jsonl()
                if msg.timestamp is None:
                    d["ts"] = time.time()
                f.write(json.dumps(d) + "\n")

    def clear(self) -> None:
        """Delete conversation history."""
        if self.ephemeral:
            return
        if self.path.exists():
            self.path.unlink()

    def __repr__(self) -> str:
        return f"Session({self.path})"
