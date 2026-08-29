"""Transport adapters — connect agents to external systems.

Built-in channels:
  - telegram: Long polling via aiogram (pip install agentino[telegram])
  - slack: Socket Mode via slack_bolt (pip install agentino[slack])
  - whatsapp: HTTP adapter for Baileys bridge (pip install agentino[serve])
  - webhook: Generic HTTP endpoint (pip install agentino[serve])
"""

from .channel import Channel
from .gateway import Gateway, GatewayConfig, build_gateway, register_channel
from .webhook import WebhookHandler

__all__ = [
    "Channel",
    "Gateway",
    "GatewayConfig",
    "WebhookHandler",
    "build_gateway",
    "register_channel",
]
