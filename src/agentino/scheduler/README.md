# `agentino.scheduler` — cron-style routines

The scheduler runs YAML-declared routines (cron expression + agent prompt + delivery)
on a 60-second tick loop. It's deliberately small — most of the structure is the
JobStore protocol so apps can decide where routine state lives.

## Public surface

```python
from agentino.scheduler import (
    JobStore,                 # Protocol — apps implement or use a built-in
    InMemoryJobStore,         # No persistence. Process-local. For tests.
    FileJobStore,             # JSON file on disk. For single-host deploys.
    SqliteJobStore,           # SQLite. For deploys that want concurrency safety.
    CronScheduler,            # The 60s tick loop wrapper around JobStore.
    Job,                      # Dataclass: id, schedule, agent_id, prompt, status.
    Schedule,                 # Cron expression validation + parsing.
    Routine,                  # User-facing helper for declaring routines in code.
)
```

## Three impls — pick by deploy shape

| Impl | When | Trade-off |
|------|------|-----------|
| `InMemoryJobStore` | Tests, REPL, ephemeral demos | Loses everything on restart |
| `FileJobStore` | Single-host single-process deploys | Simple. Don't run two writers against the same file. |
| `SqliteJobStore` | Single-host but parallel processes | Adds an SQLite dep; safe under concurrent writers |

For multi-host deploys, write your own `JobStore` impl (e.g. against Postgres or
Supabase). The protocol is small (5 methods) and contract tests in
`agentino/tests/test_scheduler_*.py` enforce the behaviour.

For per-tenant file-as-truth (the tenant-app pattern), see
Runspace's `WorkspaceRoutinesStore`. That's a
JobStore impl that reads/writes `tenants/<id>/routines.yml` directly — yaml is the
source of truth, not a DB row. The consuming app's ADRs cover the rationale.

## How a tenant app wires it

```python
# In your FastAPI lifespan:
from agentino.scheduler import CronScheduler, FileJobStore

scheduler = CronScheduler(
    store=FileJobStore(path="data/jobs.json"),
    app_registry=app_registry,        # how to dispatch agent.chat()
)
await scheduler.start()
# … on shutdown:
await scheduler.stop()
```

The scheduler is decoupled from the agent loop — it's just "every 60s, check the
JobStore for due jobs, dispatch via the app registry, log the result". You can swap
`AppRegistry` for any callable shape (see `executor` parameter).

## See also

- `tests/test_scheduler_stores.py` — the contract tests every JobStore impl must
  pass. Copy them when adding a new impl.
- Runspace's `routines_store.py` — reference impl for
  file-as-truth tenant routines.
