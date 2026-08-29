"""Slack channel — Socket Mode via slack_bolt.

Requires: pip install agentino[slack]  (slack_bolt + aiohttp)

Socket Mode doesn't need a public URL — connects to Slack via WebSocket.
Requires an app-level token (xapp-...) in addition to the bot token (xoxb-...).

Features:
- Thread support: replies in threads, follow-ups in same thread share session
- Markdown conversion: **bold** → *bold*, ## headers → *Title*
- Dedup: tracks processed events to avoid double-responses
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from agentino.core.agent import Agent

from .channel import Channel

logger = logging.getLogger(__name__)


def _md_to_mrkdwn(text: str) -> str:
    """Convert Markdown to Slack mrkdwn format."""
    # Headers: ## Title → *Title*
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    # Bold: **text** → *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # Horizontal rules: --- → ───
    text = re.sub(r"^-{3,}$", "───", text, flags=re.MULTILINE)
    return text


class SlackChannel(Channel):
    """Slack bot using Socket Mode (no public URL needed)."""

    name = "slack"

    def __init__(
        self,
        agent: Agent,
        session_dir: Path,
        bot_token: str,
        app_token: str,
        config: dict[str, Any] | None = None,
        listen_to_bots: list[str] | None = None,
        max_bot_replies: int = 3,
        **kwargs: Any,
    ):
        super().__init__(agent, session_dir, config, **kwargs)
        self.bot_token = bot_token
        self.app_token = app_token
        self._listen_to_bots: set[str] = set(listen_to_bots or [])
        self._max_bot_replies = max_bot_replies
        # Track consecutive bot messages per thread to prevent infinite loops
        self._bot_chains: dict[str, int] = {}
        # Resolved at start() via auth.test
        self._own_user_id: str = ""
        # Cache: user_id → display name
        self._name_cache: dict[str, str] = {}

    async def _resolve_name(self, user_id: str) -> str:
        """Resolve Slack user_id to display name (cached)."""
        if user_id in self._name_cache:
            return self._name_cache[user_id]
        try:
            info = await self._app.client.users_info(user=user_id)
            user = info.get("user", {})
            name = (
                user.get("profile", {}).get("display_name")
                or user.get("real_name")
                or user.get("name")
                or user_id
            )
            self._name_cache[user_id] = name
            return name
        except Exception:
            self._name_cache[user_id] = user_id
            return user_id

    async def _fetch_thread_context(
        self,
        channel_id: str,
        thread_ts: str,
        exclude_ts: str,
    ) -> list:
        """Fetch Slack thread and convert to Message history.

        Returns list of Messages representing the full thread (excluding
        the current message identified by exclude_ts).
        Own previous replies → assistant role.
        Everyone else (humans + other bots) → user role with [Name] prefix.
        """
        from agentino.core.message import Message

        try:
            result = await self._app.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=50,
            )
        except Exception as e:
            logger.warning("Failed to fetch thread %s: %s", thread_ts, e)
            return []

        messages: list[Message] = []
        for msg in result.get("messages", []):
            ts = msg.get("ts", "")
            if ts == exclude_ts:
                continue  # skip the message we're about to process
            text = msg.get("text", "")
            if not text or msg.get("subtype"):
                continue
            # Strip bot mentions
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
            if not text:
                continue

            user_id = msg.get("user", "")
            if user_id == self._own_user_id:
                # Our own previous reply
                messages.append(Message(role="assistant", content=text))
            else:
                # Human or another bot — label with name
                name = await self._resolve_name(user_id)
                messages.append(
                    Message(
                        role="user",
                        content=f"[{name}]: {text}",
                    )
                )

        return messages

    async def start(self) -> None:
        """Start the Slack bot in Socket Mode.

        Connects to Slack's Socket Mode API for real-time message handling.
        Blocks until stop() is called.

        Raises:
            ImportError: If slack_bolt is not installed.
        """
        try:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
            from slack_bolt.async_app import AsyncApp
        except ImportError:
            raise ImportError("Slack channel requires slack_bolt: pip install agentino[slack]")

        self._app = AsyncApp(token=self.bot_token)

        # Resolve own identity for thread context (self vs others)
        if self._listen_to_bots:
            try:
                auth = await self._app.client.auth_test()
                self._own_user_id = auth.get("user_id", "")
                logger.info("Slack bot identity: user_id=%s", self._own_user_id)
            except Exception as e:
                logger.warning("Failed to resolve bot identity: %s", e)

        channel = self
        _processed: set[str] = set()

        async def _handle(event: dict, say: Any, is_dm: bool = False) -> None:
            text = event.get("text", "")
            if not text or event.get("subtype"):
                return

            # Bot message handling: allow trusted bots, ignore others
            bot_id = event.get("bot_id")
            if bot_id:
                if bot_id not in channel._listen_to_bots:
                    return
                # Loop guard: check consecutive bot messages in this thread
                thread_ts = event.get("thread_ts") or event.get("ts")
                chain = channel._bot_chains.get(thread_ts, 0)
                if chain >= channel._max_bot_replies:
                    logger.debug(
                        "Bot chain limit (%d) reached in thread %s, skipping",
                        channel._max_bot_replies,
                        thread_ts,
                    )
                    return
                channel._bot_chains[thread_ts] = chain + 1
            else:
                # Human message — reset bot chain counter
                thread_ts = event.get("thread_ts") or event.get("ts")
                channel._bot_chains.pop(thread_ts, None)

            event_ts = event.get("event_ts", "")
            if event_ts in _processed:
                return
            _processed.add(event_ts)
            if len(_processed) > 1000:
                _processed.clear()

            # Strip bot mention from text
            text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()
            if not text:
                return

            # Thread support: use thread_ts for session continuity
            thread_ts = event.get("thread_ts") or event.get("ts")
            slack_channel = event.get("channel", "dm")
            user = event.get("user", "unknown")

            # Session key includes thread — same thread = same conversation
            peer_id = f"{slack_channel}:{user}:{thread_ts}"

            # Multi-bot threads: build session from Slack thread history
            # so every participant (humans + bots) is visible to the LLM
            if channel._listen_to_bots and event.get("thread_ts"):
                thread_msgs = await channel._fetch_thread_context(
                    slack_channel,
                    thread_ts,
                    exclude_ts=event_ts,
                )
                if thread_msgs:
                    session = channel.get_session(peer_id)
                    session.save(thread_msgs)

            reply = await channel.handle_message(text, peer_id)
            reply_mrkdwn = _md_to_mrkdwn(reply)

            # Reply in thread: if message was in a thread, reply there
            # If new message in channel, start a new thread
            if event.get("thread_ts"):
                # Follow-up in existing thread
                await say(reply_mrkdwn, thread_ts=event["thread_ts"])
            elif not is_dm:
                # New message in channel — start a thread with the reply
                await say(reply_mrkdwn, thread_ts=event["ts"])
            else:
                # DM — no threads needed
                await say(reply_mrkdwn)

        @self._app.event("message")
        async def on_message(event: dict, say: Any) -> None:
            """Handle incoming DM messages from users."""
            if event.get("channel_type") != "im":
                return
            await _handle(event, say, is_dm=True)

        @self._app.event("app_mention")
        async def on_mention(event: dict, say: Any) -> None:
            """Handle @bot mentions in channels."""
            await _handle(event, say, is_dm=False)

        self._handler = AsyncSocketModeHandler(self._app, self.app_token)
        logger.info("Slack channel starting (Socket Mode)...")
        await self._handler.start_async()

    async def stop(self) -> None:
        """Stop the Slack bot and close connections."""
        if hasattr(self, "_handler"):
            await self._handler.close_async()
        logger.info("Slack channel stopped.")
