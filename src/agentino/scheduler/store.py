"""JobStore protocol + InMemoryJobStore.

Concrete persistence backends (Supabase, SQLite, file) live in the
consumer project — this module ships only the contract and the
in-memory impl that's useful for tests everywhere.

See ADR-16 for the rationale.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .core import CronJob


class JobStore(Protocol):
    """Persistence backend for cron jobs.

    Contract:
      - `list` returns *all* jobs the scheduler should consider on
        load. Filtering by `tenant_id` is a hint backends may push
        down; otherwise the scheduler filters in memory.
      - `upsert` is idempotent — same `job.id` overwrites.
      - `delete` is idempotent — a missing id is not an error.
      - `record_run` is *optional* persistence of an audit trail. A
        backend that doesn't track history can no-op. The scheduler
        upserts the whole row after every run anyway, so
        `record_run` exists purely for centralised audit needs.
    """

    def list(self, tenant_id: str | None = None) -> list[CronJob]: ...

    def upsert(self, job: CronJob) -> None: ...

    def delete(self, job_id: str) -> None: ...

    def record_run(
        self,
        job_id: str,
        ok: bool,
        ts: datetime,
        error: str | None = None,
    ) -> None: ...


class InMemoryJobStore:
    """No-persistence backend. Jobs live for the process lifetime.

    Used by:
      - unit tests (no DB needed)
      - sandbox / dev runs that want a clean slate per restart
      - environments where no real backend is configured

    The scheduler also keeps an in-memory copy of every job, so this
    store's only job is to "successfully forget" — `list` returns
    whatever has been upserted in this process, and after a restart
    the list is empty.
    """

    def __init__(self) -> None:
        self._rows: dict[str, CronJob] = {}

    def list(self, tenant_id: str | None = None) -> list[CronJob]:
        rows = list(self._rows.values())
        if tenant_id:
            rows = [r for r in rows if r.tenant_id == tenant_id]
        return rows

    def upsert(self, job: CronJob) -> None:
        self._rows[job.id] = job

    def delete(self, job_id: str) -> None:
        self._rows.pop(job_id, None)

    def record_run(
        self,
        job_id: str,
        ok: bool,
        ts: datetime,
        error: str | None = None,
    ) -> None:
        return None
