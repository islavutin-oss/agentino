"""Generic webhook transport — expose an agent as an HTTP endpoint.

Works with any ASGI/WSGI framework. Zero framework dependency.

Usage with FastAPI:
    from fastapi import FastAPI, Request
    from agentino import Agent
    from agentino.transport import WebhookHandler

    app = FastAPI()
    handler = WebhookHandler(agent=my_agent, session_dir="./sessions")

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        return handler.handle(
            message=body["message"],
            session_id=body.get("session_id", "default"),
        )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentino.core.agent import Agent
from agentino.core.session import Session


class WebhookHandler:
    """Handles incoming webhook requests for an agent.

    Framework-agnostic: returns a dict that you serialize however you want.
    """

    def __init__(
        self,
        agent: Agent,
        session_dir: str | Path | None = None,
    ):
        self.agent = agent
        self.session_dir = Path(session_dir) if session_dir else None

    async def handle(
        self,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a message and return the response.

        Args:
            message: The user's message.
            session_id: Optional session ID for conversation continuity.

        Returns:
            Dict with "reply", "usage", and optional "session_id".
        """
        session = None
        if self.session_dir and session_id:
            session = Session(self.session_dir / f"{session_id}.jsonl")

        try:
            reply = await self.agent.run(message, session=session)
            return {
                "reply": reply,
                "usage": {
                    "prompt_tokens": self.agent.last_usage.prompt_tokens,
                    "completion_tokens": self.agent.last_usage.completion_tokens,
                },
                "session_id": session_id,
            }
        except Exception as e:
            return {
                "error": str(e),
                "session_id": session_id,
            }
