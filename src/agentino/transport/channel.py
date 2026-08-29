"""Channel — abstract base for messaging platform adapters.

A channel receives messages from an external platform (Telegram, Slack, etc.),
routes them to an agent, and sends the response back.

Implementing a new channel requires only start() and stop().
Message handling, session management, audio transcription, command routing,
and error recovery are provided by the base.

Commands:
    Register command handlers via on_command() decorator or register_command().
    Commands are auto-detected from /slash messages. Unhandled commands pass
    through to the agent.

    channel.register_command("help", "Show help", my_help_handler)

    # Or via decorator:
    @channel.on_command("help", "Show help")
    async def my_help(peer_id, args):
        return "Help text here"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agentino.core.agent import Agent
from agentino.core.session import Session
from agentino.extras.audio import AudioTranscriber

logger = logging.getLogger(__name__)

# Type for command handlers: async (channel, peer_id, args) -> reply_text
CommandHandler = Callable[["Channel", str, str], Awaitable[str]]


class Channel(ABC):
    """Base class for messaging platform channels."""

    name: str  # channel type identifier: "telegram", "slack", etc.

    def __init__(
        self,
        agent: Agent,
        session_dir: Path,
        config: dict[str, Any] | None = None,
        transcriber: AudioTranscriber | None = None,
        chat_history_size: int = 10,
        pipeline: Any = None,
    ):
        self.agent = agent
        self.session_dir = session_dir
        self.config = config or {}
        self.transcriber = transcriber
        self.chat_history_size = chat_history_size
        self.pipeline = pipeline  # StagedPipeline if stages.yml detected
        # Command registry: name → (description, handler)
        self._commands: dict[str, tuple[str, CommandHandler]] = {}
        # Callback registry: prefix → handler (for inline buttons)
        self._callbacks: dict[str, CommandHandler] = {}
        # Message handler hook — intercepts before agent
        self._message_handler: Callable | None = None

    def register_command(
        self,
        name: str,
        description: str,
        handler: CommandHandler,
    ) -> None:
        """Register a slash command handler.

        Args:
            name: Command name without slash (e.g. "help")
            description: Short description for menu
            handler: async (channel, peer_id, args) -> reply_text
        """
        self._commands[name.lstrip("/")] = (description, handler)

    def on_command(self, name: str, description: str = ""):
        """Decorator to register a command handler."""

        def decorator(fn: CommandHandler) -> CommandHandler:
            """Register the decorated function as a command handler."""
            self.register_command(name, description or fn.__doc__ or name, fn)
            return fn

        return decorator

    def register_callback(self, prefix: str, handler: CommandHandler) -> None:
        """Register a callback handler for inline button presses.

        Args:
            prefix: Callback data prefix (e.g. "proj" matches "proj:value")
            handler: async (channel, peer_id, data) -> reply_text or reply_dict
        """
        self._callbacks[prefix] = handler

    async def _handle_command(self, text: str, peer_id: str) -> str | dict | None:
        """Route /command messages to handlers.

        Returns:
            str — plain text reply
            dict — rich reply: {"text": "...", "buttons": [{"label": "...", "data": "..."}]}
            None — not a command, pass to agent
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped.split(maxsplit=1)
        cmd_raw = parts[0][1:].lower()  # strip /
        # Strip @botname suffix (Telegram sends /project@BotName)
        cmd = cmd_raw.split("@")[0]
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in self._commands:
            _, handler = self._commands[cmd]
            try:
                return await handler(self, peer_id, args)
            except Exception as e:
                logger.error("Command /%s error: %s", cmd, e)
                return f"Error: {e}"
        return None  # not a registered command — pass to agent

    async def _handle_callback(self, peer_id: str, data: str) -> str | dict | None:
        """Route callback button presses to handlers."""
        prefix = data.split(":", 1)[0] if ":" in data else data
        if prefix in self._callbacks:
            try:
                return await self._callbacks[prefix](self, peer_id, data)
            except Exception as e:
                logger.error("Callback %s error: %s", prefix, e)
                return f"Error: {e}"
        return None

    def get_session(self, peer_id: str) -> Session:
        """Get or create a session for a peer (user/chat).

        Session key format: {agent_name}--{channel}--{peer_id}
        This ensures unique sessions per agent+channel+user combination.
        """
        agent_name = self.agent.name or "agent"
        key = f"{agent_name}--{self.name}--{peer_id}"
        return Session(self.session_dir / f"{key}.jsonl")

    async def transcribe_audio(
        self,
        audio: bytes,
        mime: str = "audio/ogg",
        language: str | None = None,
    ) -> str | None:
        """Transcribe audio to text using configured STT provider.

        Returns transcript text, or None if no transcriber configured.
        """
        if not self.transcriber:
            logger.debug("No audio transcriber configured, skipping voice message")
            return None
        try:
            result = await self.transcriber.transcribe(audio, mime=mime, language=language)
            return result.text
        except Exception as e:
            logger.error("Audio transcription failed: %s", e)
            return None

    def set_message_handler(
        self,
        handler: Callable[[Channel, str, str], Awaitable[str | None]],
    ) -> None:
        """Set a message handler that intercepts messages before the agent.

        Handler: async (channel, text, peer_id) -> reply or None.
        If handler returns a string, it's sent as the reply (agent skipped).
        If handler returns None, the message proceeds to the agent normally.
        """
        self._message_handler = handler

    async def handle_message(
        self,
        text: str,
        peer_id: str,
        metadata: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> str:
        """Process an incoming message: route to agent, return reply.

        If a message_handler is set, it gets first shot at the message.
        If it returns a reply, the agent is skipped (fast path for casual/question).
        """
        # Message handler hook — fast path for intent routing
        if self._message_handler:
            try:
                reply = await self._message_handler(self, text, peer_id)
                if reply is not None:
                    return reply
            except Exception as e:
                logger.error(f"[{self.name}] Message handler error: {e}")

        session = self.get_session(peer_id)
        try:
            # If pipeline exists, run it. Intent routing is app-level, not framework.
            if self.pipeline:
                results = await self.pipeline.run(self.agent, text, session=session)
                wf = getattr(self.pipeline, "last_working_file", None)
                if wf:
                    from pathlib import Path as P

                    if P(wf).exists():
                        content = P(wf).read_text(encoding="utf-8").strip()
                        if content:
                            return content
                for r in reversed(results):
                    if r.output:
                        return r.output

            reply = await self.agent.run(text, session=session, images=images)
            return reply
        except Exception as e:
            logger.error(f"[{self.name}] Error handling message from {peer_id}: {e}")
            return "Sorry, something went wrong. Please try again."

    @abstractmethod
    async def start(self) -> None:
        """Start receiving messages. Blocks until stop() is called or error."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop receiving messages and release resources."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(agent={self.agent.name!r})"
