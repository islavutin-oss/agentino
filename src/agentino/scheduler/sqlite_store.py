"""SqliteJobStore — durable, single-process JobStore backed by SQLite.

Right pick when you want jobs to survive restarts but don't want to
run a network DB. One file on disk; no daemons.

Schema is created on first use. Rows mirror `CronJob.to_dict()` —
exactly what `JobStore` already serialises — so backends are
interchangeable: dump from Sqlite, load into Supabase, etc.

Thread-safety: a single `Connection` with `check_same_thread=False`
plus a `threading.RLock` around every call. The scheduler runs all
ticks on one event loop in one process, so this is enough.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from .core import CronJob

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cron_jobs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    row_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cron_jobs_tenant_id_idx ON cron_jobs(tenant_id);

CREATE TABLE IF NOT EXISTS cron_runs (
    job_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS cron_runs_job_idx ON cron_runs(job_id, ts);
"""


class SqliteJobStore:
    """JobStore backed by a SQLite file.

    `path` may be a string or Path; ":memory:" works for tests but
    note that a `:memory:` connection is per-instance — a second
    `SqliteJobStore(":memory:")` won't see the first's rows.
    """

    def __init__(self, path: str | Path = "scheduler.sqlite") -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur

    def list(self, tenant_id: str | None = None) -> list[CronJob]:
        if tenant_id:
            cur = self._exec(
                "SELECT row_json FROM cron_jobs WHERE tenant_id = ?",
                (tenant_id,),
            )
        else:
            cur = self._exec("SELECT row_json FROM cron_jobs")
        out: list[CronJob] = []
        for row in cur.fetchall():
            try:
                out.append(CronJob.from_dict(json.loads(row["row_json"])))
            except Exception as e:
                # Bad row → skip + log. Don't poison the scheduler.
                print(f"[SqliteJobStore] skipping bad row: {e}")
        return out

    def upsert(self, job: CronJob) -> None:
        payload = json.dumps(job.to_dict())
        with self._lock:
            self._conn.execute(
                "INSERT INTO cron_jobs (id, tenant_id, row_json) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  tenant_id = excluded.tenant_id, "
                "  row_json = excluded.row_json",
                (job.id, job.tenant_id, payload),
            )
            self._conn.commit()

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def record_run(
        self,
        job_id: str,
        ok: bool,
        ts: datetime,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO cron_runs (job_id, ts, ok, error) VALUES (?, ?, ?, ?)",
                (job_id, ts.isoformat(), 1 if ok else 0, error),
            )
            self._conn.commit()
