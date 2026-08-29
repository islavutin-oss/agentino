"""Tests for WhatsApp channel — HTTP webhook adapter."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentino import Agent, load_config
from agentino.transport.whatsapp import WhatsAppChannel


class TestWhatsAppChannel:
    def test_init(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        ch = WhatsAppChannel(
            agent=agent,
            session_dir=Path("/tmp/sessions"),
            bridge_url="http://localhost:3001",
            port=8080,
        )
        assert ch.name == "whatsapp"
        assert ch.bridge_url == "http://localhost:3001"
        assert ch.port == 8080

    def test_bridge_url_trailing_slash_stripped(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        ch = WhatsAppChannel(
            agent=agent,
            session_dir=Path("/tmp"),
            bridge_url="http://localhost:3001/",
        )
        assert ch.bridge_url == "http://localhost:3001"

    def test_session_key_includes_whatsapp(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        ch = WhatsAppChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        session = ch.get_session("35799123456@s.whatsapp.net")
        assert "whatsapp" in str(session.path)
        assert "max" in str(session.path)

    @pytest.mark.asyncio
    async def test_handle_message_routes_to_agent(self):
        agent = MagicMock(spec=Agent)
        agent.name = "max"
        agent.run = AsyncMock(return_value="Booking confirmed!")
        ch = WhatsAppChannel(agent=agent, session_dir=Path("/tmp/sessions"))
        reply = await ch.handle_message("Book a table", "35799123456@s.whatsapp.net")
        assert reply == "Booking confirmed!"
        agent.run.assert_called_once()


class TestWhatsAppConfig:
    def test_whatsapp_gateway_config(self):
        yaml = """\
agents:
  max:
    instructions: "Booking assistant."

gateway:
  whatsapp:
    bridge_url: http://localhost:3001
    port: 9090
    agent: max
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                assert config.gateway is not None
                assert "whatsapp" in config.gateway.channels
                assert (
                    config.gateway.channels["whatsapp"][0]["bridge_url"] == "http://localhost:3001"
                )
                assert config.gateway.channels["whatsapp"][0]["port"] == 9090
            finally:
                os.unlink(f.name)

    def test_whatsapp_multi_instance(self):
        yaml = """\
agents:
  max:
    instructions: "Booking."
  maria:
    instructions: "Wine."

gateway:
  whatsapp:
    - bridge_url: http://localhost:3001
      port: 8080
      agent: max
    - bridge_url: http://localhost:3002
      port: 8081
      agent: maria
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                assert len(config.gateway.channels["whatsapp"]) == 2
                assert config.gateway.channels["whatsapp"][0]["port"] == 8080
                assert config.gateway.channels["whatsapp"][1]["port"] == 8081
            finally:
                os.unlink(f.name)

    def test_mixed_channels(self):
        """WhatsApp alongside Telegram in same gateway config."""
        yaml = """\
agents:
  max:
    instructions: "Booking."

gateway:
  telegram:
    token: tg-token
    agent: max
  whatsapp:
    bridge_url: http://localhost:3001
    port: 8080
    agent: max
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                config = load_config(f.name)
                assert "telegram" in config.gateway.channels
                assert "whatsapp" in config.gateway.channels
            finally:
                os.unlink(f.name)
