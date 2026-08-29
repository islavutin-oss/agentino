"""DeliverySink protocol + DeliveryRouter + SilentSink.

Concrete sinks that talk to a workspace's MessagingService, Telegram,
WhatsApp, etc. live in the consumer project — this module ships only
the contract, the router, and the no-op silent sink that's useful
everywhere.

See ADR-16 for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SenderPersona:
    """Who the message appears to come from. Sinks map it to whatever
    shape their backend expects."""

    sender_id: str
    name: str
    avatar: str = "🤖"
    color: str = "#6B7280"
    metadata: dict | None = None


class DeliverySink(Protocol):
    """Posts text to a single target.

    Contract:
      - `kind` is the discriminator the router uses.
      - `target` is sink-specific: a channel slug for channel sinks,
        a user id for DM sinks, a chat id for Telegram, etc. May be
        None for sinks that have a single fixed destination.
      - `send` returns True iff the message reached the backend.
        Sinks that drop a message intentionally (silent mode) return
        True.
    """

    kind: str

    async def send(
        self,
        target: str | None,
        text: str,
        *,
        sender: SenderPersona,
        tenant_id: str,
    ) -> bool: ...


class DeliveryRouter:
    """Lookup table from delivery kind → sink. One per scheduler."""

    def __init__(self) -> None:
        self._sinks: dict[str, DeliverySink] = {}

    def register(self, sink: DeliverySink) -> None:
        self._sinks[sink.kind] = sink

    def get(self, kind: str) -> DeliverySink | None:
        return self._sinks.get(kind)

    def kinds(self) -> list[str]:
        return sorted(self._sinks.keys())

    async def send(
        self,
        kind: str,
        target: str | None,
        text: str,
        *,
        sender: SenderPersona,
        tenant_id: str,
    ) -> bool:
        sink = self.get(kind)
        if sink is None:
            print(f"[DeliveryRouter] no sink for kind={kind!r}; dropping")
            return False
        return await sink.send(target, text, sender=sender, tenant_id=tenant_id)


class SilentSink:
    """No-op delivery — succeeds without announcing anything. Useful
    when an executor wants to record success without posting (e.g. a
    routine where the agent acted via tools)."""

    kind = "silent"

    async def send(
        self,
        target: str | None,
        text: str,
        *,
        sender: SenderPersona,
        tenant_id: str,
    ) -> bool:
        return True
