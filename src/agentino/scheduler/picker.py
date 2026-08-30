"""`pick_job_store(env=os.environ)` — env-driven JobStore selector.

One env var, four backends:

    SCHEDULER_STORE=memory    → InMemoryJobStore (default)
    SCHEDULER_STORE=sqlite    → SqliteJobStore(SCHEDULER_SQLITE_PATH or
                                "scheduler.sqlite")
    SCHEDULER_STORE=file      → FileJobStore(SCHEDULER_FILE_DIR or ".")
    SCHEDULER_STORE=custom    → not picked here; the consumer wires its
                                own (e.g. SupabaseJobStore in acme)

Consumers with a project-specific backend (Supabase, …) build their
own `pick_*` that falls through to this for the generic backends.
See `acme/services/cron/store.py::pick_job_store` for an example.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from .store import InMemoryJobStore, JobStore


def pick_job_store(env: Mapping[str, str] | None = None) -> JobStore:
    if env is None:
        env = os.environ
    kind = (env.get("SCHEDULER_STORE") or "memory").strip().lower()

    if kind in ("", "memory", "inmemory", "in-memory"):
        return InMemoryJobStore()

    if kind == "sqlite":
        from .sqlite_store import SqliteJobStore

        path = env.get("SCHEDULER_SQLITE_PATH", "scheduler.sqlite")
        return SqliteJobStore(path)

    if kind == "file":
        from .file_store import FileJobStore

        d = env.get("SCHEDULER_FILE_DIR", ".")
        return FileJobStore(d)

    raise ValueError(
        f"SCHEDULER_STORE={kind!r} not recognised. "
        f"Supported: memory | sqlite | file. "
        f"For project-specific backends, build your own picker that "
        f"falls through to agentino.scheduler.pick_job_store for the "
        f"generic options."
    )
