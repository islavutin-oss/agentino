"""JobExecutor protocol + registry — ADR-16 layer 1.

A `JobExecutor` knows how to run jobs of one `payload.kind`. The
scheduler dispatches by kind; executors compose adapters (Store,
DeliverySink, AppRegistry, …) but the *protocol* itself is a plain
async callable.

This module is intentionally tiny: protocol + context + registry +
nothing else. Concrete executors live in `services/cron/executors/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from .core import CronJob


@dataclass
class ExecutorContext:
    """Cross-cutting things an executor may want at run-time.

    Kept narrow on purpose — executor-specific dependencies (an
    `AppRegistry`, a `DeliverySink`) belong in the executor's
    constructor, not in the context.
    """

    tenant_id: str
    now: datetime = field(default_factory=datetime.now)


class JobExecutor(Protocol):
    """Runs a job of one specific `payload.kind`.

    Contract:
      - `kind` is a string discriminator owned by the executor; the
        scheduler dispatches via `executor.kind == job.payload.kind`.
      - `run` returns True for success, False for failure (the
        scheduler then drives retry/backoff via `mark_failure`).
      - `run` must not raise for *expected* failure modes — return
        False instead. Unexpected exceptions bubble up; the scheduler
        catches them and counts as a failure.
    """

    kind: str

    async def run(self, job: CronJob, ctx: ExecutorContext) -> bool: ...


class ExecutorRegistry:
    """Looks up executors by payload.kind. One per CronService."""

    def __init__(self) -> None:
        self._by_kind: dict[str, JobExecutor] = {}

    def register(self, executor: JobExecutor) -> None:
        self._by_kind[executor.kind] = executor

    def get(self, kind: str) -> JobExecutor | None:
        return self._by_kind.get(kind)

    def __contains__(self, kind: str) -> bool:
        return kind in self._by_kind

    def kinds(self) -> list[str]:
        return sorted(self._by_kind.keys())
