"""Audit log — tracks agent decisions for traceability.

Every agent action is logged: who decided, what was decided, why, context.
Stored as JSONL (appendable, human-readable, git-friendly).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = os.path.expanduser("~/.agentino/audit.jsonl")


class AuditLog:
    """Structured audit log for multi-agent decisions."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or _DEFAULT_LOG_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        agent: str,
        action: str,
        input_text: str = "",
        output_text: str = "",
        tools_used: list[str] | None = None,
        decision: str = "",
        context: dict[str, Any] | None = None,
        channel: str = "",
        peer_id: str = "",
    ) -> None:
        """Log an agent action.

        Args:
            agent: Agent name/role (e.g. "po", "tech")
            action: What happened (e.g. "responded", "delegated", "approved")
            input_text: User message that triggered this
            output_text: Agent's response
            tools_used: List of tool names called
            decision: Key decision made (if any)
            context: Additional context dict
            channel: Channel name (slack, telegram, etc.)
            peer_id: User/thread ID
        """
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "agent": agent,
            "action": action,
            "input": input_text[:500],
            "output": output_text[:500],
            "tools": tools_used or [],
            "decision": decision,
            "channel": channel,
            "peer_id": peer_id,
        }
        if context:
            entry["context"] = context

        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error("Audit log write failed: %s", e)

    def query(
        self,
        agent: str = "",
        action: str = "",
        since_hours: int = 24,
        limit: int = 50,
    ) -> list[dict]:
        """Query recent audit log entries."""
        if not self.path.exists():
            return []

        from datetime import timedelta

        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        results = []
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("ts", "") < cutoff:
                            continue
                        if agent and entry.get("agent") != agent:
                            continue
                        if action and entry.get("action") != action:
                            continue
                        results.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        return results[-limit:]
