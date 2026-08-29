"""WebSocket channel — lightweight server for local clients (VS Code, web UI).

Requires: pip install agentino[serve]  (aiohttp)

Features:
- JSON message protocol over WebSocket
- Slash commands routed like other channels
- Streaming text chunks + tool call notifications
- Audio transcription (base64 audio → text)
- Multiple concurrent clients, each with a peer_id

Usage in agents.yml:
    gateway:
      websocket:
        port: 8765
        host: 127.0.0.1
        agent: coder

Protocol (client → server):
    {"type": "message", "text": "...", "peer_id": "vscode-1"}
    {"type": "command", "name": "project", "args": "myapp", "peer_id": "vscode-1"}
    {"type": "audio", "data": "<base64>", "mime": "audio/webm", "peer_id": "vscode-1"}

Protocol (server → client):
    {"type": "chunk", "text": "...", "id": 1}
    {"type": "tool_call", "name": "read_file", "id": 1}
    {"type": "done", "text": "full reply", "id": 1}
    {"type": "error", "text": "...", "id": 1}
    {"type": "stage", "event": "start|complete|fail", "name": "IMPLEMENT", "id": 1}
    {"type": "transcript", "text": "transcribed text"}
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from agentino.core.agent import Agent
from agentino.extras.audio import AudioTranscriber

from .channel import Channel

logger = logging.getLogger(__name__)


class WebSocketChannel(Channel):
    """WebSocket server channel for local IDE/UI clients."""

    name = "websocket"

    def __init__(
        self,
        agent: Agent,
        session_dir: Path,
        port: int = 8765,
        host: str = "127.0.0.1",
        config: dict[str, Any] | None = None,
        transcriber: AudioTranscriber | None = None,
        chat_history_size: int = 10,
        **kwargs: Any,
    ):
        super().__init__(
            agent, session_dir, config, transcriber=transcriber, chat_history_size=chat_history_size
        )
        self.host = host
        self.port = port
        self._site: Any = None
        self._app: Any = None
        self._runner: Any = None
        self._clients: set = set()
        self._msg_counter = 0

    async def start(self) -> None:
        """Start the WebSocket server.

        Starts an HTTP server with WebSocket endpoint at /ws.
        Blocks until stop() is called.

        Raises:
            ImportError: If aiohttp is not installed.
        """
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError("WebSocket channel requires aiohttp: pip install agentino[serve]")

        self._app = web.Application()
        self._app.router.add_get("/ws", self._handle_ws)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        logger.info("WebSocket channel listening on ws://%s:%d/ws", self.host, self.port)

        # Keep running until cancelled
        try:
            await asyncio.Future()  # block forever
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the WebSocket server and close all client connections."""
        # Close all client connections
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()

        if self._runner:
            await self._runner.cleanup()
        logger.info("WebSocket channel stopped.")

    async def _handle_health(self, request: Any) -> Any:
        from aiohttp import web

        return web.json_response({"status": "ok", "channel": "websocket"})

    async def _handle_ws(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.info("WebSocket client connected (%d total)", len(self._clients))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._dispatch(ws, data)
                    except json.JSONDecodeError:
                        await self._send(ws, {"type": "error", "text": "Invalid JSON"})
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self._clients.discard(ws)
            logger.info("WebSocket client disconnected (%d remaining)", len(self._clients))

        return ws

    async def _dispatch(self, ws: Any, data: dict) -> None:
        """Route incoming message to the appropriate handler."""
        msg_type = data.get("type", "message")
        peer_id = data.get("peer_id", "vscode-default")
        # Use client-provided id so responses match pending callbacks
        msg_id = data.get("id") or self._next_id()

        if msg_type == "message":
            text = data.get("text", "").strip()
            if not text:
                return
            await self._handle_text(ws, text, peer_id, msg_id)

        elif msg_type == "command":
            name = data.get("name", "")
            args = data.get("args", "")
            await self._handle_cmd(ws, name, args, peer_id, msg_id)

        elif msg_type == "callback":
            callback_data = data.get("data", "")
            await self._handle_callback_click(ws, callback_data, peer_id, msg_id)

        elif msg_type == "audio":
            audio_b64 = data.get("data", "")
            mime = data.get("mime", "audio/webm")
            await self._handle_audio(ws, audio_b64, mime, peer_id)

        else:
            await self._send(ws, {"type": "error", "text": f"Unknown type: {msg_type}"})

    def _next_id(self) -> int:
        self._msg_counter += 1
        return self._msg_counter

    async def _handle_text(self, ws: Any, text: str, peer_id: str, msg_id: Any = None) -> None:
        """Handle a text message — commands, message handler, or direct agent."""
        if msg_id is None:
            msg_id = self._next_id()

        # Check slash commands first
        cmd_reply = await self._handle_command(text, peer_id)
        if cmd_reply is not None:
            reply_text = cmd_reply if isinstance(cmd_reply, str) else json.dumps(cmd_reply)
            await self._send(ws, {"type": "done", "text": reply_text, "id": msg_id})
            return

        # Wire up real-time event forwarding (tool calls, progress)
        old_on_event = self.agent.on_event

        def _forward_event(event: Any) -> None:
            """Forward agent events to WebSocket client in real-time."""
            from agentino.core.message import EventType

            msg = None
            if event.type == EventType.TOOL_START:
                logger.info("[ws] tool_start: %s", event.name)
                msg = {"type": "tool_call", "name": event.name or "?", "id": msg_id}
            elif event.type == EventType.TOOL_RESULT:
                result_preview = (event.data or "")[:200]
                logger.info("[ws] tool_result: %s → %s", event.name, result_preview[:80])
                msg = {
                    "type": "tool_result",
                    "name": event.name or "?",
                    "text": result_preview,
                    "id": msg_id,
                }
            elif event.type == EventType.ERROR:
                logger.error("[ws] agent error event: %s", event.data)
                msg = {"type": "error", "text": str(event.data), "id": msg_id}
            elif event.type == EventType.LLM_RESPONSE:
                tokens = event.usage.total_tokens if event.usage else 0
                logger.info("[ws] llm_response: %d tokens", tokens)
            else:
                logger.debug(
                    "[ws] event: type=%s data=%s",
                    event.type,
                    str(event.data)[:100] if event.data else "",
                )
            if msg:
                asyncio.ensure_future(self._send(ws, msg))
            if old_on_event:
                old_on_event(event)

        self.agent.on_event = _forward_event

        try:
            # Message handler (intent routing) — may return string or _TaskRequest
            if self._message_handler:
                logger.info("[ws] routing msg from %s: %s", peer_id, text[:80])
                try:
                    result = await self._message_handler(self, text, peer_id)
                    logger.info(
                        "[ws] handler returned: type=%s len=%d",
                        type(result).__name__,
                        len(str(result)) if result else 0,
                    )
                    if result is not None:
                        if isinstance(result, str):
                            await self._send(ws, {"type": "done", "text": result, "id": msg_id})
                            return
                        if isinstance(result, dict):
                            # Rich response with buttons — send as JSON string
                            await self._send(
                                ws, {"type": "done", "text": json.dumps(result), "id": msg_id}
                            )
                            return
                        await self._send(ws, {"type": "done", "text": str(result), "id": msg_id})
                        return
                except Exception as e:
                    logger.error("[ws] message handler error: %s", e, exc_info=True)
                    await self._send(ws, {"type": "error", "text": str(e), "id": msg_id})
                    return

            # Fallback: direct agent
            logger.info("[ws] fallback to direct agent for: %s", text[:80])
            try:
                session = self.get_session(peer_id)
                reply = await self.agent.run(text, session=session)
                logger.info("[ws] agent reply len=%d", len(reply or ""))
                await self._send(ws, {"type": "done", "text": reply or "", "id": msg_id})
            except Exception as e:
                logger.error("[ws] agent error: %s", e, exc_info=True)
                await self._send(ws, {"type": "error", "text": str(e), "id": msg_id})
        finally:
            self.agent.on_event = old_on_event

    async def _handle_cmd(
        self, ws: Any, name: str, args: str, peer_id: str, msg_id: Any = None
    ) -> None:
        """Handle an explicit command request."""
        if msg_id is None:
            msg_id = self._next_id()

        cmd_reply = await self._handle_command(f"/{name} {args}".strip(), peer_id)
        if cmd_reply is not None:
            reply_text = cmd_reply if isinstance(cmd_reply, str) else json.dumps(cmd_reply)
            await self._send(ws, {"type": "done", "text": reply_text, "id": msg_id})
        else:
            await self._send(
                ws, {"type": "error", "text": f"Unknown command: {name}", "id": msg_id}
            )

    async def _handle_callback_click(
        self, ws: Any, data: str, peer_id: str, msg_id: Any = None
    ) -> None:
        """Handle inline button callback."""
        if msg_id is None:
            msg_id = self._next_id()

        result = await self._handle_callback(peer_id, data)
        if result is not None:
            reply_text = result if isinstance(result, str) else json.dumps(result)
            await self._send(ws, {"type": "done", "text": reply_text, "id": msg_id})
        else:
            await self._send(
                ws, {"type": "error", "text": f"Unknown callback: {data}", "id": msg_id}
            )

    async def _handle_audio(self, ws: Any, audio_b64: str, mime: str, peer_id: str) -> None:
        """Handle audio transcription request."""
        import base64

        if not audio_b64:
            await self._send(ws, {"type": "error", "text": "No audio data"})
            return

        audio_bytes = base64.b64decode(audio_b64)
        transcript = await self.transcribe_audio(audio_bytes, mime=mime)

        if transcript:
            await self._send(ws, {"type": "transcript", "text": transcript})
        else:
            await self._send(ws, {"type": "error", "text": "Transcription failed"})

    async def _send(self, ws: Any, data: dict) -> None:
        """Send JSON to a single client."""
        try:
            await ws.send_json(data)
        except Exception:
            self._clients.discard(ws)

    async def send_to_peer(self, peer_id: str, text: str) -> None:
        """Send a progress message to all connected clients.

        Used by gateway_commands to forward stage and tool events.
        Parses markers into typed messages for the VS Code UI.
        """
        msg: dict
        if text.startswith("▰"):
            name = text.split("**")[1] if "**" in text else text
            msg = {"type": "stage", "event": "start", "name": name}
        elif text.startswith("✓"):
            name = text.split("**")[1] if "**" in text else text
            msg = {"type": "stage", "event": "complete", "name": name}
        elif text.startswith("✗"):
            name = text.split("**")[1] if "**" in text else text
            msg = {"type": "stage", "event": "fail", "name": name}
        elif text.startswith("↻"):
            name = text.split("**")[1] if "**" in text else text
            msg = {"type": "stage", "event": "retry", "name": name}
        elif text.startswith("▸"):
            # Tool call start
            msg = {"type": "tool_call", "name": text[2:].strip()}
        elif text.startswith("  →"):
            # Tool result preview
            msg = {"type": "tool_result", "name": "", "text": text[4:].strip()}
        else:
            msg = {"type": "stage_message", "text": text}
        await self.broadcast(msg)

    async def broadcast(self, data: dict) -> None:
        """Send JSON to all connected clients."""
        for ws in list(self._clients):
            await self._send(ws, data)
