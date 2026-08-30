"""Project-agnostic scheduling primitives — ADR-16 layer 1.

Three protocols + the tick loop. Persistence, executors, and delivery
sinks are pluggable. Concrete backends live in the consumer
(acme, harness, globex), not here — except for the four shipped
JobStore impls (InMemory, Sqlite, File, plus whatever the consumer
adds), which are framework-level enough to ship together.

Public API:

    from agentino.scheduler import (
        Schedule, ScheduleKind, Payload, Delivery, CronJob, JobStatus,
        CronScheduler,
        JobStore, InMemoryJobStore, SqliteJobStore, FileJobStore,
        JobExecutor, ExecutorContext, ExecutorRegistry,
        DeliverySink, DeliveryRouter, SenderPersona, SilentSink,
        pick_job_store,
    )
"""

from .core import (
    CronJob,
    CronScheduler,
    Delivery,
    JobStatus,
    Payload,
    Schedule,
    ScheduleKind,
)
from .delivery import (
    DeliveryRouter,
    DeliverySink,
    SenderPersona,
    SilentSink,
)
from .executor import ExecutorContext, ExecutorRegistry, JobExecutor
from .picker import pick_job_store
from .sqlite_store import SqliteJobStore
from .store import InMemoryJobStore, JobStore

# FileJobStore depends on pyyaml — import lazily. Consumers without
# pyyaml can still use everything else.
try:
    from .file_store import FileJobStore  # noqa: F401

    _has_file_store = True
except ImportError:
    _has_file_store = False

__all__ = [
    "CronJob",
    "CronScheduler",
    "Delivery",
    "JobStatus",
    "Payload",
    "Schedule",
    "ScheduleKind",
    "JobStore",
    "InMemoryJobStore",
    "SqliteJobStore",
    "JobExecutor",
    "ExecutorContext",
    "ExecutorRegistry",
    "DeliverySink",
    "DeliveryRouter",
    "SenderPersona",
    "SilentSink",
    "pick_job_store",
]
if _has_file_store:
    __all__.append("FileJobStore")
