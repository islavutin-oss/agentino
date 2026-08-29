"""FileJobStore — flat-file JobStore, definitions in YAML, state in JSON.

Right pick when you want routine *definitions* in git (reviewable,
deployable) and *runtime state* in a sibling file the runtime owns.

Layout on disk:

    routines.yaml        ← human-edited, git-tracked definitions
    routines.state.json  ← machine-written, last_run / next_run /
                            consecutive_failures, ignored by git

Definitions yaml is a list of dicts shaped like `CronJob.to_dict()`,
minus the runtime-state fields. State json is `{job_id: {...state}}`.
On load, the two are merged.

This store is intentionally write-light: definitions are only
re-read when `list()` is called *and* the YAML file's mtime
changed since last load. `upsert` writes both files; `delete`
removes from both.

Dependencies: `pyyaml`. If unavailable, `__init__` raises a clear
ImportError.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .core import CronJob

_RUNTIME_KEYS = {
    "status",
    "next_run_at",
    "last_run_at",
    "last_error",
    "consecutive_failures",
    "retry_after",
}


@dataclass
class FileJobStorePaths:
    definitions: Path
    state: Path


class FileJobStore:
    """JobStore backed by a YAML+JSON pair.

    `dir` is the directory holding both files; defaults to the cwd.
    File names are configurable via the constructor for projects
    that already have conventions.
    """

    def __init__(
        self,
        dir: str | Path = ".",
        *,
        definitions_filename: str = "routines.yaml",
        state_filename: str = "routines.state.json",
    ) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FileJobStore requires `pyyaml`. Install with "
                "`pip install pyyaml` or use SqliteJobStore."
            ) from e
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        self._paths = FileJobStorePaths(
            definitions=d / definitions_filename,
            state=d / state_filename,
        )

    # ── public protocol methods ───────────────────────────────────
    def list(self, tenant_id: str | None = None) -> list[CronJob]:
        merged = self._merged_rows()
        if tenant_id:
            merged = [r for r in merged if r.get("tenant_id") == tenant_id]
        out: list[CronJob] = []
        for row in merged:
            try:
                out.append(CronJob.from_dict(row))
            except Exception as e:
                print(f"[FileJobStore] skipping bad row {row.get('id')!r}: {e}")
        return out

    def upsert(self, job: CronJob) -> None:
        full = job.to_dict()
        defn = {k: v for k, v in full.items() if k not in _RUNTIME_KEYS}
        state = {k: v for k, v in full.items() if k in _RUNTIME_KEYS}

        # Update definitions YAML.
        defs = self._read_definitions()
        defs = [d for d in defs if d.get("id") != job.id] + [defn]
        self._write_definitions(defs)

        # Update state JSON.
        states = self._read_state()
        states[job.id] = state
        self._write_state(states)

    def delete(self, job_id: str) -> None:
        defs = [d for d in self._read_definitions() if d.get("id") != job_id]
        self._write_definitions(defs)
        states = self._read_state()
        states.pop(job_id, None)
        self._write_state(states)

    def record_run(
        self,
        job_id: str,
        ok: bool,
        ts: datetime,
        error: str | None = None,
    ) -> None:
        # Append-only audit trail next to state. Keep small —
        # truncate to 500 entries per job.
        path = self._paths.state.with_suffix(".runs.json")
        try:
            blob = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            blob = {}
        runs = blob.get(job_id, [])
        runs.append({"ts": ts.isoformat(), "ok": bool(ok), "error": error})
        if len(runs) > 500:
            runs = runs[-500:]
        blob[job_id] = runs
        path.write_text(json.dumps(blob, indent=2))

    # ── helpers ───────────────────────────────────────────────────
    def _merged_rows(self) -> list[dict]:
        defs = self._read_definitions()
        states = self._read_state()
        out: list[dict] = []
        for d in defs:
            jid = d.get("id")
            row = dict(d)
            if jid and jid in states:
                row.update(states[jid])
            out.append(row)
        return out

    def _read_definitions(self) -> list[dict]:
        if not self._paths.definitions.exists():
            return []
        import yaml

        try:
            data = yaml.safe_load(self._paths.definitions.read_text()) or []
        except Exception as e:
            print(f"[FileJobStore] definitions YAML parse failed: {e}")
            return []
        if isinstance(data, dict):
            data = data.get("routines", [])
        return list(data) if isinstance(data, list) else []

    def _write_definitions(self, defs: list[dict]) -> None:
        import yaml

        # Stable ordering for clean diffs.
        defs_sorted = sorted(defs, key=lambda d: d.get("id", ""))
        self._paths.definitions.write_text(yaml.safe_dump(defs_sorted, sort_keys=True))

    def _read_state(self) -> dict[str, dict]:
        if not self._paths.state.exists():
            return {}
        try:
            return json.loads(self._paths.state.read_text())
        except Exception:
            return {}

    def _write_state(self, states: dict[str, dict]) -> None:
        # Atomic-ish write: write temp + rename. State is
        # short-lived, so a torn write is recoverable but ugly.
        tmp = self._paths.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(states, indent=2, sort_keys=True))
        os.replace(tmp, self._paths.state)
