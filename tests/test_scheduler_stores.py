"""Tests for the JobStore implementations shipped with agentino.scheduler.

Three stores share the same `JobStore` protocol; the contract test
below runs each through the same paces (upsert/list/delete/round-trip
state). Sqlite + file get extra tests for durability across instances.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from agentino.scheduler import (
    CronJob,
    InMemoryJobStore,
    Payload,
    Schedule,
    ScheduleKind,
    SqliteJobStore,
    pick_job_store,
)


def _job(name: str = "j", tenant_id: str = "t") -> CronJob:
    return CronJob(
        id=f"job_{name}",
        tenant_id=tenant_id,
        name=name,
        schedule=Schedule(kind=ScheduleKind.CRON, cron_expr="0 8 * * *"),
        payload=Payload(kind="routine", data={"x": 1}),
    )


# ── Contract ────────────────────────────────────────────────────────────
@pytest.fixture(params=["memory", "sqlite", "file"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryJobStore()
    if request.param == "sqlite":
        return SqliteJobStore(tmp_path / "sched.sqlite")
    if request.param == "file":
        try:
            from agentino.scheduler import FileJobStore
        except ImportError:
            pytest.skip("pyyaml not installed; FileJobStore unavailable")
        return FileJobStore(tmp_path)
    raise AssertionError(request.param)


def test_upsert_then_list(store):
    a, b = _job("a"), _job("b", tenant_id="other")
    store.upsert(a)
    store.upsert(b)
    rows = store.list()
    assert {r.id for r in rows} == {"job_a", "job_b"}
    assert {r.id for r in store.list(tenant_id="t")} == {"job_a"}
    assert {r.id for r in store.list(tenant_id="other")} == {"job_b"}


def test_upsert_overwrites(store):
    j = _job("a")
    store.upsert(j)
    j.name = "renamed"
    store.upsert(j)
    rows = store.list()
    assert len(rows) == 1 and rows[0].name == "renamed"


def test_delete_idempotent(store):
    store.upsert(_job("a"))
    store.delete("job_a")
    store.delete("job_a")  # second delete must not raise
    assert store.list() == []


def test_record_run_does_not_raise(store):
    store.upsert(_job("a"))
    store.record_run("job_a", ok=True, ts=datetime.now())
    store.record_run("job_a", ok=False, ts=datetime.now(), error="boom")


# ── Sqlite-specific: durability across instances ────────────────────────
def test_sqlite_persists_across_instances(tmp_path):
    path = tmp_path / "p.sqlite"
    s1 = SqliteJobStore(path)
    s1.upsert(_job("a"))
    del s1
    s2 = SqliteJobStore(path)
    rows = s2.list()
    assert [r.id for r in rows] == ["job_a"]


# ── File-specific: defs + state separation ──────────────────────────────
def test_file_writes_yaml_and_json(tmp_path):
    try:
        from agentino.scheduler import FileJobStore
    except ImportError:
        pytest.skip("pyyaml not installed")
    s = FileJobStore(tmp_path)
    s.upsert(_job("a"))
    assert (tmp_path / "routines.yaml").exists()
    assert (tmp_path / "routines.state.json").exists()


# ── Picker ──────────────────────────────────────────────────────────────
def test_picker_default_is_memory():
    assert isinstance(pick_job_store({}), InMemoryJobStore)
    assert isinstance(pick_job_store({"SCHEDULER_STORE": ""}), InMemoryJobStore)
    assert isinstance(pick_job_store({"SCHEDULER_STORE": "memory"}), InMemoryJobStore)


def test_picker_sqlite(tmp_path):
    env = {
        "SCHEDULER_STORE": "sqlite",
        "SCHEDULER_SQLITE_PATH": str(tmp_path / "x.sqlite"),
    }
    s = pick_job_store(env)
    assert isinstance(s, SqliteJobStore)


def test_picker_file(tmp_path):
    try:
        from agentino.scheduler import FileJobStore
    except ImportError:
        pytest.skip("pyyaml not installed")
    env = {"SCHEDULER_STORE": "file", "SCHEDULER_FILE_DIR": str(tmp_path)}
    s = pick_job_store(env)
    assert isinstance(s, FileJobStore)


def test_picker_unknown_kind_raises():
    with pytest.raises(ValueError, match="not recognised"):
        pick_job_store({"SCHEDULER_STORE": "redis"})
