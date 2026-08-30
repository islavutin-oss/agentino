"""Gateway — multi-channel message router.

Runs multiple channels (Telegram, Slack, etc.) concurrently, each routing
messages to its configured agent. Handles graceful shutdown and channel restarts.

Usage:
    from agentino import load_config
    from agentino.transport import build_gateway

    config = load_config("agents.yml")
    gateway = build_gateway(config)
    gateway.run()  # blocking
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentino.core.agent import Agent

from .channel import Channel

logger = logging.getLogger(__name__)


@dataclass
class GatewayConfig:
    """Parsed gateway section from agents.yml.

    Each channel type maps to a list of instance configs, supporting
    multiple bots per platform (e.g. two Telegram bots for different agents).

    commands: dotted module path with register(channel) function.
    """

    channels: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    session_dir: str = "./sessions"
    commands: str = ""  # e.g. "ag.gateway_commands"
    chat_history: int = 10  # messages to keep per user for context


class Gateway:
    """Multi-channel gateway — runs channels concurrently, routes to agents."""

    def __init__(self, channels: list[Channel]):
        self.channels = channels
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start all channels concurrently. Blocks until stopped."""
        if not self.channels:
            logger.warning("No channels configured.")
            return

        names = ", ".join(ch.name for ch in self.channels)
        logger.info(f"Gateway starting {len(self.channels)} channel(s): {names}")

        for ch in self.channels:
            task = asyncio.create_task(
                self._run_channel(ch),
                name=f"channel-{ch.name}",
            )
            self._tasks.append(task)

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def _run_channel(self, channel: Channel) -> None:
        """Run a channel with exponential backoff restart on crash."""
        max_retries = 5
        delay = 2.0

        for attempt in range(max_retries):
            try:
                await channel.start()
                return  # clean exit
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[{channel.name}] Crashed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"[{channel.name}] Restarting in {delay:.0f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    logger.error(f"[{channel.name}] Max retries reached.")

    async def stop(self) -> None:
        """Gracefully stop all channels."""
        logger.info("Gateway shutting down...")
        for task in self._tasks:
            task.cancel()
        for ch in self.channels:
            try:
                await ch.stop()
            except Exception as e:
                logger.error(f"[{ch.name}] Shutdown error: {e}")
        self._tasks.clear()
        logger.info("Gateway stopped.")

    def run(self) -> None:
        """Run the gateway (blocking). Handles SIGINT/SIGTERM."""
        loop = asyncio.new_event_loop()
        gateway = self

        async def _main() -> None:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(gateway.stop()))
            await gateway.start()

        try:
            loop.run_until_complete(_main())
        except KeyboardInterrupt:
            loop.run_until_complete(self.stop())
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Builder — creates Gateway from Config
# ---------------------------------------------------------------------------

# Registry of channel type → constructor
CHANNEL_TYPES: dict[str, type[Channel]] = {}


def register_channel(name: str, cls: type[Channel]) -> None:
    """Register a channel type for config-driven gateway building."""
    CHANNEL_TYPES[name] = cls


def build_gateway(config: Any, session_dir: str | Path | None = None) -> Gateway:
    """Build a Gateway from a Config object.

    Reads the `gateway` section from config and instantiates channels.
    Each channel entry maps to a channel type with its own constructor args.

    Supports both single and multi-bot configs:

        # Single bot per platform:
        gateway:
          telegram:
            token: ${TELEGRAM_BOT_TOKEN}
            agent: max

        # Multiple bots per platform:
        gateway:
          telegram:
            - token: ${TG_GEORGE_TOKEN}
              agent: max
            - token: ${TG_MARIA_TOKEN}
              agent: maria
    """
    gw_cfg = config.gateway
    if not gw_cfg:
        raise ValueError("No gateway section in config")

    sdir = Path(session_dir or gw_cfg.session_dir)
    sdir.mkdir(parents=True, exist_ok=True)

    # Lazy-register built-in channel types
    _ensure_registered()

    # Build audio transcriber from gateway-level config (shared by all channels)
    from agentino.extras.audio import build_transcriber

    audio_cfg = gw_cfg.channels.pop("audio", None)
    # audio config is a single dict, not a list of channel instances
    transcriber = build_transcriber(audio_cfg[0] if isinstance(audio_cfg, list) else audio_cfg)

    channels: list[Channel] = []
    for ch_type, instances in gw_cfg.channels.items():
        if ch_type not in CHANNEL_TYPES:
            logger.warning(f"Unknown channel type: {ch_type} (skipping)")
            continue

        cls = CHANNEL_TYPES[ch_type]
        for ch_cfg in instances:
            ch_cfg = dict(ch_cfg)  # don't mutate original
            agent_name = ch_cfg.pop("agent", None)
            agent = _resolve_agent(agent_name, config.agents)

            try:
                channel = cls(
                    agent=agent,
                    session_dir=sdir,
                    transcriber=transcriber,
                    chat_history_size=gw_cfg.chat_history,
                    pipeline=config.pipeline,
                    **ch_cfg,
                )
            except TypeError as e:
                raise ValueError(
                    f"Invalid config for {ch_type} channel: {e}\n"
                    f"  Check required fields in agents.yml gateway.{ch_type}"
                ) from e
            channels.append(channel)

    # Load and register commands from configured module
    commands_module = gw_cfg.commands
    if commands_module:
        try:
            import importlib

            mod = importlib.import_module(commands_module)
            register_fn = getattr(mod, "register", None)
            if register_fn:
                for ch in channels:
                    register_fn(ch)
                logger.info("Loaded commands from %s", commands_module)
            else:
                logger.warning("Commands module %s has no register() function", commands_module)
        except Exception as e:
            logger.error("Failed to load commands module %s: %s", commands_module, e)

    # Load and register message_hook on channels (intent routing)
    hook_module = config.raw.get("message_hook") if hasattr(config, "raw") else None
    if hook_module:
        try:
            import importlib

            mod = importlib.import_module(hook_module)
            hook_fn = getattr(mod, "classify_and_route", None) or getattr(mod, "handle", None)
            if hook_fn:
                # Wrap to match channel._message_handler signature (channel, text, peer_id)
                async def _channel_hook(channel, text, peer_id, _fn=hook_fn):
                    session = channel.get_session(peer_id)
                    return await _fn(channel, channel.agent, text, session)

                for ch in channels:
                    ch.set_message_handler(_channel_hook)
                logger.info(
                    "Registered message_hook %s on %d channel(s)", hook_module, len(channels)
                )
        except Exception as e:
            logger.warning("Failed to load message_hook %s: %s", hook_module, e)

    return Gateway(channels)


def _resolve_agent(name: str | None, agents: dict[str, Agent]) -> Agent:
    """Resolve agent by name, defaulting to the first agent."""
    if name and name in agents:
        agent = agents[name]
        agent.name = agent.name or name
        return agent
    if agents:
        first_name = next(iter(agents))
        first = agents[first_name]
        first.name = first.name or first_name
        if name:
            logger.warning(f"Agent '{name}' not found, using '{first.name}'")
        return first
    raise ValueError("No agents defined — gateway needs at least one agent")


def _ensure_registered() -> None:
    """Lazy-register built-in channel types."""
    if CHANNEL_TYPES:
        return
    try:
        from .telegram import TelegramChannel

        register_channel("telegram", TelegramChannel)
    except Exception:
        pass  # aiogram not installed
    try:
        from .slack import SlackChannel

        register_channel("slack", SlackChannel)
    except Exception:
        pass  # slack_bolt not installed
    try:
        from .whatsapp import WhatsAppChannel

        register_channel("whatsapp", WhatsAppChannel)
    except Exception:
        pass  # starlette/uvicorn not installed
    try:
        from .websocket import WebSocketChannel

        register_channel("websocket", WebSocketChannel)
    except Exception:
        pass  # aiohttp not installed
