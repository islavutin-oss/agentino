"""
Cron Scheduler - Core scheduling logic

Data types (Schedule, Payload, Delivery, CronJob, ...) — agentino owns
its own definitions here. The runspace ecosystem ALSO declares the
same shapes in `runspace-contracts/contracts/scheduling.py`; the two
are intentional parallel copies (structurally equivalent, two physical
class objects). Single-class-identity is NOT preserved across the
package boundary because that would force agentino to depend on
runspace-contracts, which would re-create the back-import the IP-hedge
redesign was meant to eliminate.

A runspace-side test (`runspace/contracts/tests/test_scheduling.py::
test_agentino_shape_matches_contracts`) asserts the two definitions
share field names + types so drift is caught at PR time.

The runtime piece — `CronScheduler` — stays in this file.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from croniter import croniter


class JobStatus(Enum):
    """Job execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


class ScheduleKind(Enum):
    """Type of schedule"""

    AT = "at"  # One-shot at specific time
    CRON = "cron"  # Recurring cron expression
    EVERY = "every"  # Fixed interval (ms)


@dataclass
class Schedule:
    """Job schedule definition"""

    kind: ScheduleKind
    at: datetime | None = None
    cron_expr: str | None = None
    every_ms: int | None = None
    timezone: str = "Europe/Nicosia"

    def next_run(self, after: datetime | None = None) -> datetime | None:
        after = after or datetime.now()
        if self.kind == ScheduleKind.AT:
            if self.at and self.at > after:
                return self.at
            return None
        if self.kind == ScheduleKind.CRON:
            if self.cron_expr:
                try:
                    cron = croniter(self.cron_expr, after)
                    return cron.get_next(datetime)
                except Exception:
                    return None
        if self.kind == ScheduleKind.EVERY:
            if self.every_ms:
                return after + timedelta(milliseconds=self.every_ms)
        return None


@dataclass
class Delivery:
    """Job delivery configuration"""

    channel: str = "whatsapp"
    to: str | None = None
    best_effort: bool = True


@dataclass
class Payload:
    """Job payload — what to execute"""

    kind: str
    skill: str | None = None
    template: str | None = None
    message: str | None = None
    data: dict = field(default_factory=dict)


@dataclass
class CronJob:
    """A scheduled job"""

    id: str
    tenant_id: str
    name: str
    schedule: Schedule
    payload: Payload
    delivery: Delivery | None = None

    enabled: bool = True
    status: JobStatus = JobStatus.PENDING
    delete_after_run: bool = False

    created_at: datetime = field(default_factory=datetime.now)
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_error: str | None = None

    consecutive_failures: int = 0
    retry_after: datetime | None = None

    def __post_init__(self):
        if self.next_run_at is None:
            self.next_run_at = self.schedule.next_run()

    def calculate_retry_delay(self) -> timedelta:
        delays = [30, 60, 300, 900, 3600]
        idx = min(self.consecutive_failures, len(delays) - 1)
        return timedelta(seconds=delays[idx])

    def mark_success(self):
        self.status = JobStatus.COMPLETED
        self.last_run_at = datetime.now()
        self.last_error = None
        self.consecutive_failures = 0
        self.retry_after = None
        if self.schedule.kind != ScheduleKind.AT:
            self.next_run_at = self.schedule.next_run(self.last_run_at)
            self.status = JobStatus.PENDING
        else:
            if not self.delete_after_run:
                self.enabled = False

    def mark_failure(self, error: str):
        self.status = JobStatus.FAILED
        self.last_run_at = datetime.now()
        self.last_error = error
        self.consecutive_failures += 1
        delay = self.calculate_retry_delay()
        self.retry_after = datetime.now() + delay
        if self.schedule.kind != ScheduleKind.AT:
            self.next_run_at = self.retry_after
            self.status = JobStatus.PENDING

    def is_due(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        if not self.enabled:
            return False
        if self.status == JobStatus.RUNNING:
            return False

        def to_naive(dt):
            if dt is None:
                return None
            return dt.replace(tzinfo=None) if dt.tzinfo else dt

        now_naive = to_naive(now)
        if self.retry_after and now_naive < to_naive(self.retry_after):
            return False
        if self.next_run_at and now_naive >= to_naive(self.next_run_at):
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "schedule_kind": self.schedule.kind.value,
            "schedule_at": self.schedule.at.isoformat() if self.schedule.at else None,
            "schedule_cron": self.schedule.cron_expr,
            "schedule_every_ms": self.schedule.every_ms,
            "schedule_timezone": self.schedule.timezone,
            "payload_kind": self.payload.kind,
            "payload_skill": self.payload.skill,
            "payload_template": self.payload.template,
            "payload_message": self.payload.message,
            "payload_data": self.payload.data,
            "delivery_channel": self.delivery.channel if self.delivery else None,
            "delivery_to": self.delivery.to if self.delivery else None,
            "delivery_best_effort": self.delivery.best_effort if self.delivery else True,
            "enabled": self.enabled,
            "status": self.status.value,
            "delete_after_run": self.delete_after_run,
            "created_at": self.created_at.isoformat(),
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "retry_after": self.retry_after.isoformat() if self.retry_after else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CronJob":
        def _parse_dt(val):
            if not val:
                return None
            dt = datetime.fromisoformat(val) if isinstance(val, str) else val
            return dt.replace(tzinfo=None) if dt.tzinfo else dt

        schedule = Schedule(
            kind=ScheduleKind(data["schedule_kind"]),
            at=_parse_dt(data.get("schedule_at")),
            cron_expr=data.get("schedule_cron"),
            every_ms=data.get("schedule_every_ms"),
            timezone=data.get("schedule_timezone", "Europe/Nicosia"),
        )
        payload = Payload(
            kind=data["payload_kind"],
            skill=data.get("payload_skill"),
            template=data.get("payload_template"),
            message=data.get("payload_message"),
            data=data.get("payload_data", {}),
        )
        delivery = None
        if data.get("delivery_channel"):
            delivery = Delivery(
                channel=data["delivery_channel"],
                to=data.get("delivery_to"),
                best_effort=data.get("delivery_best_effort", True),
            )
        return cls(
            id=data["id"],
            tenant_id=data["tenant_id"],
            name=data["name"],
            schedule=schedule,
            payload=payload,
            delivery=delivery,
            enabled=data.get("enabled", True),
            status=JobStatus(data.get("status", "pending")),
            delete_after_run=data.get("delete_after_run", False),
            created_at=_parse_dt(data.get("created_at")) or datetime.now(),
            next_run_at=_parse_dt(data.get("next_run_at")),
            last_run_at=_parse_dt(data.get("last_run_at")),
            last_error=data.get("last_error"),
            consecutive_failures=data.get("consecutive_failures", 0),
            retry_after=_parse_dt(data.get("retry_after")),
        )


class CronScheduler:
    """Manages cron jobs — loading, saving, ticking.

    Persistence is delegated to a `JobStore` (ADR-16, Layer 1↔2). The
    scheduler itself depends only on the `JobStore` protocol — no
    Supabase, no FastAPI, no agentino imports.

    Usage:
        from services.cron.store import pick_job_store
        scheduler = CronScheduler(store=pick_job_store())
        await scheduler.start()
    """

    def __init__(self, store=None):
        # Lazy-import keeps this module importable from places that
        # don't want to wire a store (e.g. dataclass-only callers
        # reading `CronJob`/`Schedule`).
        if store is None:
            from .store import InMemoryJobStore

            store = InMemoryJobStore()
        self._store = store
        self._jobs: dict[str, CronJob] = {}
        self._running = False
        self._executor: Callable[[CronJob], Awaitable[bool]] | None = None

    def set_executor(self, executor: Callable[[CronJob], Awaitable[bool]]):
        """Set the job executor function"""
        self._executor = executor

    async def load_from_db(self):
        """Load all jobs from the configured `JobStore`.

        Method name kept for backwards-compat with callers that
        already invoke `load_from_db()` at startup; the store
        decides whether "the DB" is Supabase, SQLite, or memory.
        """
        try:
            rows = self._store.list()
        except Exception as e:
            print(f"[CronScheduler] store.list failed: {e}; starting empty")
            return

        self._jobs.clear()
        for job in rows:
            self._jobs[job.id] = job

        print(
            f"[CronScheduler] Loaded {len(self._jobs)} jobs from store "
            f"({type(self._store).__name__})"
        )

    async def save_job(self, job: CronJob):
        """Persist a job via the store."""
        try:
            self._store.upsert(job)
        except Exception as e:
            print(f"[CronScheduler] store.upsert failed for {job.id}: {e}")

    async def delete_job(self, job_id: str):
        """Drop a job from memory and the store."""
        self._jobs.pop(job_id, None)
        try:
            self._store.delete(job_id)
        except Exception as e:
            print(f"[CronScheduler] store.delete failed for {job_id}: {e}")

    async def add(
        self,
        tenant_id: str,
        name: str,
        schedule: Schedule,
        payload: Payload,
        delivery: Delivery | None = None,
        delete_after_run: bool = True,
    ) -> CronJob:
        """Add a new job"""
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        job = CronJob(
            id=job_id,
            tenant_id=tenant_id,
            name=name,
            schedule=schedule,
            payload=payload,
            delivery=delivery,
            delete_after_run=delete_after_run,
        )

        self._jobs[job_id] = job
        await self.save_job(job)

        print(f"[CronScheduler] Added job: {name} (next run: {job.next_run_at})")
        return job

    async def get(self, job_id: str) -> CronJob | None:
        """Get a job by ID"""
        return self._jobs.get(job_id)

    async def list_jobs(self, tenant_id: str | None = None) -> list[CronJob]:
        """List all jobs, optionally filtered by tenant"""
        jobs = list(self._jobs.values())
        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        return jobs

    async def get_due_jobs(self) -> list[CronJob]:
        """Get all jobs that are due for execution"""
        now = datetime.now()
        return [job for job in self._jobs.values() if job.is_due(now)]

    async def execute_job(self, job: CronJob) -> bool:
        """Execute a single job"""
        if not self._executor:
            print(f"[CronScheduler] No executor set, skipping job: {job.name}")
            return False

        job.status = JobStatus.RUNNING
        await self.save_job(job)

        try:
            success = await self._executor(job)

            if success:
                job.mark_success()
                print(f"[CronScheduler] Job completed: {job.name}")
                self._record_run(job.id, ok=True)

                # Delete one-shot jobs if configured
                if job.delete_after_run and job.schedule.kind == ScheduleKind.AT:
                    await self.delete_job(job.id)
                    return True
            else:
                job.mark_failure("Executor returned False")
                print(f"[CronScheduler] Job failed: {job.name}")
                self._record_run(job.id, ok=False, error="executor returned False")

            await self.save_job(job)
            return success

        except Exception as e:
            job.mark_failure(str(e))
            await self.save_job(job)
            self._record_run(job.id, ok=False, error=str(e))
            print(f"[CronScheduler] Job error: {job.name} - {e}")
            return False

    def _record_run(self, job_id: str, ok: bool, error: str | None = None) -> None:
        """Notify the store of a run for audit (no-op in stores that
        don't track history). Failures here must never break the
        scheduler — audit is best-effort."""
        try:
            self._store.record_run(job_id, ok=ok, ts=datetime.now(), error=error)
        except Exception as e:
            print(f"[CronScheduler] store.record_run failed for {job_id}: {e}")

    async def run_due_jobs(self):
        """Execute all due jobs"""
        due_jobs = await self.get_due_jobs()

        for job in due_jobs:
            await self.execute_job(job)

    async def start(self, interval_seconds: int = 60):
        """Start the scheduler loop"""
        self._running = True
        await self.load_from_db()

        print(f"[CronScheduler] Starting scheduler (check every {interval_seconds}s)")

        while self._running:
            try:
                await self.run_due_jobs()
            except Exception as e:
                print(f"[CronScheduler] Error in scheduler loop: {e}")

            await asyncio.sleep(interval_seconds)

    def stop(self):
        """Stop the scheduler loop"""
        self._running = False
        print("[CronScheduler] Scheduler stopped")
