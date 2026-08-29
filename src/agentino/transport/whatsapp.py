"""WhatsApp channel — HTTP adapter for a Baileys (Node.js) bridge.

WhatsApp Web protocol requires a Node.js runtime (Baileys library).
This channel acts as the Python side: it starts an HTTP server that receives
messages forwarded by the bridge, routes them to the agent, and sends
replies back via the bridge's /send endpoint.

Architecture:
    WhatsApp ↔ Baileys bridge (Node.js) ↔ WhatsAppChannel (Python/agentino)

The bridge is a lightweight Node.js process that:
- Connects to WhatsApp Web via @whiskeysockets/baileys
- Forwards incoming messages as HTTP POSTs to this channel
- Exposes POST /send for outbound messages
- Handles QR/pairing code auth, reconnection, whitelist/blacklist

Usage in agents.yml:
    gateway:
      whatsapp:
        bridge_url: http://localhost:3001   # where the Baileys bridge runs
        port: 8080                          # port this channel listens on
        agent: max

Requires: pip install agentino[serve]  (starlette + uvicorn for the webhook server)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentino.core.agent import Agent
from agentino.extras.audio import AudioTranscriber

from .channel import Channel

logger = logging.getLogger(__name__)


class WhatsAppChannel(Channel):
    """WhatsApp via Baileys bridge — webhook receiver + outbound sender."""

    name = "whatsapp"

    def __init__(
        self,
        agent: Agent,
        session_dir: Path,
        bridge_url: str = "http://localhost:3001",
        port: int = 8080,
        host: str = "0.0.0.0",
        config: dict[str, Any] | None = None,
        transcriber: AudioTranscriber | None = None,
        chat_history_size: int = 10,
    ):
        super().__init__(
            agent, session_dir, config, transcriber=transcriber, chat_history_size=chat_history_size
        )
        self.bridge_url = bridge_url.rstrip("/")
        self.port = port
        self.host = host
        self._http: Any = None  # persistent httpx.AsyncClient

    async def start(self) -> None:
        """Start the WhatsApp webhook server.

        Starts an HTTP server that receives messages from the WhatsApp bridge
        and sends replies back. Blocks until stop() is called.

        Raises:
            ImportError: If starlette/uvicorn are not installed.
        """
        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            from starlette.routing import Route
        except ImportError:
            raise ImportError(
                "WhatsApp channel requires starlette + uvicorn: pip install agentino[serve]"
            )

        import httpx

        self._http = httpx.AsyncClient(timeout=15)
        channel = self

        async def handle_message(request: Request) -> JSONResponse:
            """Receive a message forwarded by the Baileys bridge.

            Expected body (text):
                { "sender_id": "357...@s.whatsapp.net",
                  "phone": "35799123456",
                  "message": "Hello!",
                  "sender_name": "John" }

            Expected body (audio):
                { "sender_id": "357...@s.whatsapp.net",
                  "phone": "35799123456",
                  "message": "",
                  "message_type": "audio",
                  "media_base64": "...",
                  "media_mime": "audio/ogg; codecs=opus",
                  "sender_name": "John" }
            """
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON"}, status_code=400)

            text = body.get("message", "").strip()
            sender_id = body.get("sender_id", "")
            if not sender_id:
                return JSONResponse({"error": "missing sender_id"}, status_code=400)

            # Audio message → transcribe
            msg_type = body.get("message_type", "text")
            if msg_type == "audio" and not text:
                media_b64 = body.get("media_base64", "")
                if media_b64:
                    import base64

                    audio_bytes = base64.b64decode(media_b64)
                    mime = body.get("media_mime", "audio/ogg")
                    transcript = await channel.transcribe_audio(audio_bytes, mime=mime)
                    if transcript:
                        text = transcript
                    else:
                        text = "[Voice message — transcription unavailable]"

            if not text:
                return JSONResponse({"error": "missing message"}, status_code=400)

            # Route to agent
            reply = await channel.handle_message(text, sender_id)

            # Send reply back via bridge
            try:
                # Use sender_id as the recipient — bridge handles JID → phone conversion
                phone = body.get("phone") or sender_id.split("@")[0].replace(":", "")
                await channel._http.post(
                    f"{channel.bridge_url}/send",
                    json={"phone": phone, "message": reply},
                )
            except Exception as e:
                logger.error(f"Failed to send reply via bridge: {e}")

            return JSONResponse({"response": reply})

        async def handle_health(request: Request) -> JSONResponse:
            """Health check endpoint for monitoring."""
            return JSONResponse(
                {
                    "status": "running",
                    "agent": channel.agent.name or "agent",
                    "bridge_url": channel.bridge_url,
                }
            )

        app = Starlette(
            routes=[
                Route("/message", handle_message, methods=["POST"]),
                Route("/health", handle_health, methods=["GET"]),
            ],
        )

        logger.info(
            f"WhatsApp channel listening on {self.host}:{self.port} (bridge at {self.bridge_url})"
        )

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the WhatsApp webhook server and close connections."""
        if hasattr(self, "_server"):
            self._server.should_exit = True
        if self._http:
            await self._http.aclose()
        logger.info("WhatsApp channel stopped.")
