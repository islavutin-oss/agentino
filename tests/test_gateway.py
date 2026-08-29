"""Tests for gateway — channel base, config parsing, gateway orchestration."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentino import Agent, load_config
from agentino.transport.channel import Channel
from agentino.transport.gateway import (
    CHANNEL_TYPES,
    Gateway,
    build_gateway,
    register_channel,
)

# ---------------------------------------------------------------------------
# Stub channel for testing
# ---------------------------------------------------------------------------


class StubChannel(Channel):
    """A channel that records messages instead of talking to a platform."""

    name = "stub"

    def __init__(self, agent, session_dir, **kwargs):
        super().__init__(agent, session_dir, kwargs)
        self.started = False
        self.stopped = False
        self.messages: list[tuple[str, str]] = []
        self._stop_event: asyncio.Event | None = None

    async def start(self) -> None:
        self.started = True
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def stop(self) -> None:
        self.stopped = True
        if self._stop_event:
            self._stop_event.set()


# ---------------------------------------------------------------------------
# Channel base class
# ---------------------------------------------------------------------------


class TestChannel:
    def test_session_key_format(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        ch = StubChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        session = ch.get_session("user-123")
        assert "max" in str(session.path)
        assert "stub" in str(session.path)
        assert "user-123" in str(session.path)

    def test_session_key_unique_per_channel(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        ch1 = StubChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        ch1.name = "telegram"
        ch2 = StubChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        ch2.name = "slack"
        s1 = ch1.get_session("user-1")
        s2 = ch2.get_session("user-1")
        assert s1.path != s2.path

    @pytest.mark.asyncio
    async def test_handle_message_calls_agent(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        agent.run = AsyncMock(return_value="Hello back!")
        ch = StubChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        reply = await ch.handle_message("Hi", "user-1")
        assert reply == "Hello back!"
        agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message_error_returns_friendly(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        ch = StubChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        reply = await ch.handle_message("Hi", "user-1")
        assert "sorry" in reply.lower()

    def test_repr(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        ch = StubChannel(agent=agent, session_dir=Path("/tmp"))
        assert "StubChannel" in repr(ch)
        assert "max" in repr(ch)


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class TestGateway:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        ch = StubChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        gw = Gateway([ch])

        # Start in background, then stop
        task = asyncio.create_task(gw.start())
        await asyncio.sleep(0.05)  # let channels start
        assert ch.started
        await gw.stop()
        assert ch.stopped
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_empty_gateway(self):
        gw = Gateway([])
        await gw.start()  # should return immediately

    @pytest.mark.asyncio
    async def test_channel_crash_restarts(self):
        """Channel that crashes should be retried."""
        agent = MagicMock(spec=Agent)
        agent.name = "max"

        call_count = 0

        class CrashOnceChannel(Channel):
            name = "crasher"

            async def start(self):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("first crash")
                # Second attempt succeeds and blocks
                self._event = asyncio.Event()
                await self._event.wait()

            async def stop(self):
                if hasattr(self, "_event"):
                    self._event.set()

        ch = CrashOnceChannel(agent=agent, session_dir=Path("/tmp"))
        gw = Gateway([ch])

        task = asyncio.create_task(gw.start())
        await asyncio.sleep(3)  # wait for restart (2s backoff)
        assert call_count >= 2
        await gw.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# GatewayConfig parsing
# ---------------------------------------------------------------------------


_GATEWAY_YAML = """\
agents:
  max:
    instructions: "You are Max, a booking assistant."
    model: gpt-4o

gateway:
  session_dir: ./my-sessions
  telegram:
    token: test-telegram-token
    agent: max
  slack:
    bot_token: xoxb-test
    app_token: xapp-test
    agent: max
"""


class TestGatewayConfig:
    def test_parses_gateway_section(self):
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(_GATEWAY_YAML)
            f.flush()
            try:
                config = load_config(f.name)
                assert config.gateway is not None
                assert config.gateway.session_dir == "./my-sessions"
                assert "telegram" in config.gateway.channels
                assert "slack" in config.gateway.channels
                assert config.gateway.channels["telegram"][0]["token"] == "test-telegram-token"
                assert config.gateway.channels["slack"][0]["bot_token"] == "xoxb-test"
            finally:
                os.unlink(f.name)

    def test_no_gateway_section(self):
        yaml = """\
agents:
  bot:
    instructions: "Hello"
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                assert config.gateway is None
            finally:
                os.unlink(f.name)

    def test_default_session_dir(self):
        yaml = """\
agents:
  bot:
    instructions: "Hello"

gateway:
  telegram:
    token: abc
    agent: bot
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                assert config.gateway.session_dir == "./sessions"
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# build_gateway
# ---------------------------------------------------------------------------


class TestBuildGateway:
    def test_build_with_stub_channel(self):
        # Register our stub channel type
        old_types = dict(CHANNEL_TYPES)
        try:
            CHANNEL_TYPES.clear()
            register_channel("stub", StubChannel)

            yaml = """\
agents:
  max:
    instructions: "Booking assistant."

gateway:
  stub:
    agent: max
"""
            with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
                f.write(yaml)
                f.flush()
                try:
                    config = load_config(f.name)
                    gw = build_gateway(config)
                    assert len(gw.channels) == 1
                    assert gw.channels[0].name == "stub"
                    assert gw.channels[0].agent.name is not None
                finally:
                    os.unlink(f.name)
        finally:
            CHANNEL_TYPES.clear()
            CHANNEL_TYPES.update(old_types)

    def test_build_default_agent(self):
        """Channel without explicit agent gets the first agent."""
        old_types = dict(CHANNEL_TYPES)
        try:
            CHANNEL_TYPES.clear()
            register_channel("stub", StubChannel)

            yaml = """\
agents:
  first_bot:
    instructions: "I am the first bot."

gateway:
  stub: {}
"""
            with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
                f.write(yaml)
                f.flush()
                try:
                    config = load_config(f.name)
                    gw = build_gateway(config)
                    assert len(gw.channels) == 1
                finally:
                    os.unlink(f.name)
        finally:
            CHANNEL_TYPES.clear()
            CHANNEL_TYPES.update(old_types)

    def test_build_unknown_channel_skipped(self):
        old_types = dict(CHANNEL_TYPES)
        try:
            CHANNEL_TYPES.clear()
            register_channel("stub", StubChannel)

            yaml = """\
agents:
  bot:
    instructions: "Hello"

gateway:
  stub:
    agent: bot
  discord:
    token: abc
    agent: bot
"""
            with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
                f.write(yaml)
                f.flush()
                try:
                    config = load_config(f.name)
                    gw = build_gateway(config)
                    assert len(gw.channels) == 1  # discord skipped
                    assert gw.channels[0].name == "stub"
                finally:
                    os.unlink(f.name)
        finally:
            CHANNEL_TYPES.clear()
            CHANNEL_TYPES.update(old_types)

    def test_no_gateway_raises(self):
        yaml = """\
agents:
  bot:
    instructions: "Hello"
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                with pytest.raises(ValueError, match="No gateway"):
                    build_gateway(config)
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# Environment variable resolution in gateway config
# ---------------------------------------------------------------------------


class TestGatewayEnvVars:
    def test_token_resolved_from_env(self):
        yaml = """\
agents:
  bot:
    instructions: "Hello"

gateway:
  telegram:
    token: ${TEST_TG_TOKEN}
    agent: bot
"""
        with patch.dict(os.environ, {"TEST_TG_TOKEN": "real-token-123"}):
            with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
                f.write(yaml)
                f.flush()
                try:
                    config = load_config(f.name)
                    assert config.gateway.channels["telegram"][0]["token"] == "real-token-123"
                finally:
                    os.unlink(f.name)


# ---------------------------------------------------------------------------
# Multi-bot support
# ---------------------------------------------------------------------------


class TestMultiBot:
    def test_multi_bot_config_parsed(self):
        yaml = """\
agents:
  max:
    instructions: "Booking assistant."
  maria:
    instructions: "Wine sommelier."

gateway:
  telegram:
    - token: token-max
      agent: max
    - token: token-maria
      agent: maria
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                assert len(config.gateway.channels["telegram"]) == 2
                assert config.gateway.channels["telegram"][0]["token"] == "token-max"
                assert config.gateway.channels["telegram"][1]["token"] == "token-maria"
            finally:
                os.unlink(f.name)

    def test_multi_bot_builds_separate_channels(self):
        old_types = dict(CHANNEL_TYPES)
        try:
            CHANNEL_TYPES.clear()
            register_channel("stub", StubChannel)

            yaml = """\
agents:
  max:
    instructions: "Booking."
  maria:
    instructions: "Wine."

gateway:
  stub:
    - agent: max
    - agent: maria
"""
            with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
                f.write(yaml)
                f.flush()
                try:
                    config = load_config(f.name)
                    gw = build_gateway(config)
                    assert len(gw.channels) == 2
                    agents = {ch.agent.name for ch in gw.channels}
                    assert "max" in agents or "maria" in agents
                finally:
                    os.unlink(f.name)
        finally:
            CHANNEL_TYPES.clear()
            CHANNEL_TYPES.update(old_types)

    def test_single_dict_still_works(self):
        """Backward compat: single dict config normalizes to list of one."""
        yaml = """\
agents:
  bot:
    instructions: "Hello"

gateway:
  telegram:
    token: single-token
    agent: bot
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                assert len(config.gateway.channels["telegram"]) == 1
                assert config.gateway.channels["telegram"][0]["token"] == "single-token"
            finally:
                os.unlink(f.name)
