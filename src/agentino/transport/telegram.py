"""Telegram channel — long polling via aiogram 3.x.

Requires: pip install agentino[telegram]  (aiogram>=3.10)

Features:
- Text, photo, voice/audio messages
- Audio auto-transcribed via configured STT provider
- Slash commands registered via channel.register_command()
- Bot menu auto-set from registered commands
- Typing indicator while agent processes
- Markdown → Telegram HTML conversion
- Long message chunking (4096 char limit)

Usage in agents.yml:
    gateway:
      audio:
        base_url: https://api.groq.com/openai/v1
        api_key: ${GROQ_API_KEY}
      telegram:
        token: ${TELEGRAM_BOT_TOKEN}
        agent: max
"""

from __future__ import annotations

import asyncio
import base64
import html
import io
import logging
import re
from pathlib import Path
from typing import Any

from agentino.core.agent import Agent
from agentino.extras.audio import AudioTranscriber

from .channel import Channel

logger = logging.getLogger(__name__)


def _md_to_telegram_html(text: str) -> str:
    """Convert basic markdown to Telegram-compatible HTML.

    Supports: **bold**, *italic*, `code`, ```code blocks```, [links](url).
    Telegram HTML subset: <b>, <i>, <code>, <pre>, <a href>.
    """
    text = html.escape(text, quote=False)
    text = re.sub(r"```(\w*)\n?(.*?)```", r"<pre>\2</pre>", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


class TelegramChannel(Channel):
    """Telegram bot using long polling.

    Supports registered commands, voice transcription, photos, and typing indicator.
    """

    name = "telegram"

    def __init__(
        self,
        agent: Agent,
        session_dir: Path,
        token: str,
        config: dict[str, Any] | None = None,
        transcriber: AudioTranscriber | None = None,
        chat_history_size: int = 10,
        allowed_users: list[int] | None = None,
    ):
        super().__init__(
            agent, session_dir, config, transcriber=transcriber, chat_history_size=chat_history_size
        )
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._running_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the Telegram bot and begin polling for messages.

        Connects to Telegram via aiogram, registers command handlers,
        and enters the polling loop. Blocks until stop() is called.

        Raises:
            ImportError: If aiogram is not installed.
        """
        try:
            from aiogram import Bot, Dispatcher, types
            from aiogram.filters import Command, CommandStart  # noqa: F401
            from aiogram.types import BotCommand
        except ImportError:
            raise ImportError("Telegram channel requires aiogram: pip install agentino[telegram]")

        self._bot = Bot(token=self.token)
        self._dp = Dispatcher()
        channel = self

        # Register bot menu commands from registered handlers
        if self._commands:
            bot_commands = [
                BotCommand(command=name, description=desc)
                for name, (desc, _) in self._commands.items()
                if name != "start"
            ]
            if bot_commands:
                await self._bot.set_my_commands(bot_commands)

        async def _reply(message: types.Message, response: str | dict) -> None:
            """Send reply — plain text or rich response with inline keyboard."""
            from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

            if isinstance(response, dict):
                text = response.get("text", "")
                buttons = response.get("buttons", [])
                keyboard = None
                if buttons:
                    rows = []
                    for btn in buttons:
                        rows.append(
                            [
                                InlineKeyboardButton(
                                    text=btn["label"],
                                    callback_data=btn["data"],
                                )
                            ]
                        )
                    keyboard = InlineKeyboardMarkup(inline_keyboard=rows)
                formatted = _md_to_telegram_html(text)
                try:
                    await message.answer(formatted, parse_mode="HTML", reply_markup=keyboard)
                except Exception:
                    await message.answer(text, reply_markup=keyboard)
                return

            text = response
            formatted = _md_to_telegram_html(text)
            for i in range(0, len(formatted), 4096):
                chunk = formatted[i : i + 4096]
                try:
                    await message.answer(chunk, parse_mode="HTML")
                except Exception:
                    plain = text[i : i + 4096] if i < len(text) else chunk
                    await message.answer(plain)

        async def _send_typing_until_done(chat_id: int, done: asyncio.Event) -> None:
            """Send 'typing' action every 4s until the agent responds."""
            while not done.is_set():
                try:
                    await self._bot.send_chat_action(chat_id, "typing")
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(done.wait(), timeout=4.0)
                except asyncio.TimeoutError:
                    pass

        @self._dp.message(CommandStart())
        async def on_start(message: types.Message) -> None:
            """Handle /start command from user."""
            peer_id = str(message.chat.id)
            # Try registered /start handler first
            cmd_reply = await channel._handle_command("/start", peer_id)
            if cmd_reply:
                await _reply(message, cmd_reply)
                return
            reply = await channel.handle_message("/start", peer_id)
            await _reply(message, reply)

        # Inline button callbacks
        @self._dp.callback_query()
        async def on_callback(callback: types.CallbackQuery) -> None:
            """Handle inline button callback queries."""
            peer_id = str(callback.message.chat.id)
            data = callback.data or ""
            result = await channel._handle_callback(peer_id, data)
            if result is not None:
                text = result if isinstance(result, str) else result.get("text", "")
                formatted = _md_to_telegram_html(text)
                try:
                    await callback.message.edit_text(formatted, parse_mode="HTML")
                except Exception:
                    await callback.message.edit_text(text)
            await callback.answer()

        @self._dp.message()
        async def on_message(message: types.Message) -> None:
            """Handle incoming text or voice messages from users."""
            # Whitelist check
            if channel.allowed_users and message.from_user.id not in channel.allowed_users:
                return

            text = message.text or message.caption or ""
            images: list[str] = []

            # Photos → base64
            if message.photo:
                photo = message.photo[-1]
                file = await self._bot.get_file(photo.file_id)
                buf = io.BytesIO()
                await self._bot.download_file(file.file_path, buf)
                b64 = base64.b64encode(buf.getvalue()).decode()
                images.append(f"data:image/jpeg;base64,{b64}")
                if not text:
                    text = "[User sent an image]"

            # Voice/audio → transcribe
            voice = message.voice or message.audio
            if voice and not text:
                file = await self._bot.get_file(voice.file_id)
                buf = io.BytesIO()
                await self._bot.download_file(file.file_path, buf)
                mime = voice.mime_type or "audio/ogg"
                transcript = await channel.transcribe_audio(buf.getvalue(), mime=mime)
                if transcript:
                    text = transcript
                else:
                    await _reply(
                        message, "Couldn't transcribe voice message. Please type your message."
                    )
                    return

            if not text:
                return

            peer_id = str(message.chat.id)

            # Check registered commands first
            cmd_reply = await channel._handle_command(text, peer_id)
            if cmd_reply is not None:
                await _reply(message, cmd_reply)
                return

            # Agent processing with typing indicator
            done = asyncio.Event()
            typing_task = asyncio.create_task(_send_typing_until_done(message.chat.id, done))
            try:
                channel._running_task = asyncio.current_task()
                reply = await channel.handle_message(text, peer_id, images=images or None)
            except asyncio.CancelledError:
                reply = "Task cancelled."
            finally:
                channel._running_task = None
                done.set()
                await typing_task

            await _reply(message, reply)

        logger.info("Telegram channel starting (long polling)...")
        try:
            await self._dp.start_polling(self._bot)
        finally:
            await self._bot.session.close()

    async def stop(self) -> None:
        """Stop the Telegram bot and close connections."""
        if hasattr(self, "_dp"):
            await self._dp.stop_polling()
        if hasattr(self, "_bot"):
            await self._bot.session.close()
        logger.info("Telegram channel stopped.")
