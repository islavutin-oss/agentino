# Agentino

**An agent is a prompt, a model, and a list of tools. A tool is an async
function with a decorator. There is no third concept.**

No graphs, no DSLs, no DAG editors. You write the function; the runtime reads
its signature, offers it to the model, calls it when the model asks, and hands
the result back — until there is an answer.

```bash
pip install agentino-framework
```

The distribution is `agentino-framework` because the `agentino` name on PyPI
belongs to an unrelated project. It imports as `agentino` regardless:

```python
from agentino import Agent, tool
```

Agents need somewhere to live. [Runspace](https://github.com/islavutin-oss/runspace)
is the open-source **LLM workspace** around this one — channels, schedules,
gateways and a UI — and it runs Agentino in-process.

---

## An agent that does something

```python
import httpx
from agentino import Agent, tool

@tool
async def failed_runs(since_hours: int = 24) -> str:
    """Jobs that failed in the last N hours, newest first."""
    async with httpx.AsyncClient(timeout=10) as http:
        r = await http.get(f"{CI}/api/runs", params={"status": "failed", "since_hours": since_hours})
        r.raise_for_status()
    return r.text

agent = Agent(
    instructions="You are the on-call desk. Lead with the answer, then the evidence.",
    tools=[failed_runs],
)
print(await agent.run("Anything break overnight?"))
# → "Three failures, all in the nightly export — a statement timeout at 02:14,
#    02:47 and 03:31. Everything else passed."
```

That's the whole API: define tools as plain async functions, hand them to an `Agent`, call `.run()`. The framework handles the LLM round-trip, tool dispatch, retries, and the final-text extraction.

---

## What's in the box

```
src/agentino/
├── core/              Agent, Runner, LLM, Tool, Message, Context, State, Session
├── config/            YAML loaders for agents, pipelines, tools
├── pipeline/          Pipeline, StagedPipeline (multi-stage flows with verdicts)
├── safety/            GateManager, HookManager, security, sanitizers
├── reliability/       resilience (retry/backoff), compaction, error taxonomy
├── extras/            knowledge (TF-IDF + embeddings), memory, audio, skills
├── providers/         Codex, Anthropic — pluggable LLM backends
├── scheduler/         CronScheduler + JobStore protocol (file/sqlite/in-memory)
├── tools/std/         Built-in tools: files, shell, grep, web search and fetch,
│                      weather, document generation (pdf/docx/xlsx/pptx/csv),
│                      agent memory. `BUILTIN_TOOLS` is the ten a coding-style
│                      agent gets by default; the rest are opt-in.
├── transport/         Outbound channel adapters (Telegram, Slack, WhatsApp, WebSocket)
├── workers/           fork_agent, make_spawn_tool — multi-agent spawning
└── cli/               REPL renderer
```

Top-level `from agentino import …` exports the curated public API. Deeper paths
like `from agentino.safety.gates import GateManager` are how internal packages
talk to each other.

---

## Configure agents from YAML

```yaml
# agents.yml
agents:
  reviewer:
    model: gpt-5.4-codex
    instructions_file: prompts/reviewer.md
    tools: [read_file, grep, shell]            # auto-discovered from tools/
    knowledge:
      dir: ./knowledge                          # TF-IDF + dense embeddings
```

```bash
agentino run agents.yml                  # one-shot REPL
agentino run agents.yml --agent reviewer # specific agent
agentino run agents.yml --serve 8080     # HTTP server
agentino run agents.yml -m "Review PR #42"
agentino run agents.yml -m "Review PR #42" --mode json   # machine-readable (one envelope)
agentino run agents.yml -m "Review PR #42" --mode jsonl  # streaming events + final envelope
```

### Headless / foreign-harness mode

`--mode json|jsonl` makes `agentino run --message …` emit a structured contract
on stdout instead of ANSI-prettified markdown — the same shape `pi --print`,
`codex exec --json`, and `claude -p --output-format stream-json` provide. Lets
non-Python harnesses (IDE extensions, polyglot stacks)
shell out to agentino and parse the result programmatically.

```bash
$ agentino run agents.yml -m "List open invoices" --mode json
{"type":"final","text":"…","tools_used":["list_invoices"],
 "tool_outputs":["…"],"usage":{"prompt_tokens":1200,"completion_tokens":85},
 "model":"gpt-5.4-codex","elapsed_ms":2254}
```

---

## What you can do beyond a single tool call

### Pipelines

`StagedPipeline` runs multi-stage flows where each stage produces a verdict
the next stage can read. A benchmark harness can use it for *security check → execute →
report*; the security stage rejects unsafe inputs before the execute stage
ever runs.

```python
from agentino import StagedPipeline, StageDef
pipeline = StagedPipeline(stages=[
    StageDef(name="security", agent=security_agent, verdict_required=True),
    StageDef(name="execute", agent=worker_agent, on_reject="report_threat"),
])
```

### Gates — declarative tool preconditions

`GateManager` rejects tool calls whose preconditions haven't been met.
Useful when you want guarantees beyond the LLM following its instructions.

```python
from agentino.safety.gates import GateRule, GateManager
rules = [GateRule(
    gate="invoice_listed",
    tools=["set_invoice_status"],
    message="Run list_invoices first so you've actually seen the IDs.",
)]
```

When the agent loop encounters `set_invoice_status` and `invoice_listed`
isn't marked, the tool returns the rejection message instead of running.

### Hooks — observe + block tool calls without touching the tool

Two flavours of [`HookManager`](docs/cookbook/hooks.md): Python callbacks
(in-process, fast — for audit logs, history mirroring, metric emission)
and shell commands (subprocess — for ops integrations, external validators).

```python
from agentino.safety.hooks import HookManager
hooks = HookManager()
hooks.register("PostToolUse", matcher={"tool_name": "chat"},
               callback=lambda ctx: audit_db.insert(ctx))
```

### Scheduler — cron-style routine execution

```python
from agentino.scheduler import CronScheduler, FileJobStore
scheduler = CronScheduler(store=FileJobStore("data/jobs.json"))
await scheduler.start()
```

`JobStore` is a protocol — ship `InMemoryJobStore`, `SqliteJobStore`,
`FileJobStore`, or write your own (e.g. file-as-truth tenant routines, see
`runspace`'s tenant routine store).

### Knowledge base

Hybrid TF-IDF + dense-embedding retrieval with one tool: `search_knowledge`.
Drop markdown files in a directory, point an agent at it, the LLM gets a
search tool and reaches into the corpus when it needs to.

### Multi-channel gateway

```bash
agentino run agents.yml --gateway
```

Maps Slack, Telegram, WhatsApp, and WebSocket transports onto the same agent
config. One agent serves users from any channel without changing its code.

---

## Pointing it at a model

```bash
export AGENTINO_BASE_URL=https://api.openai.com/v1   # or vLLM, Ollama, OpenRouter…
export AGENTINO_API_KEY=sk-…
```

The wire protocol is inferred from the URL: Anthropic for an Anthropic
endpoint, Codex for `chatgpt.com/backend-api` or a `/codex` path, and plain
OpenAI-compatible `/chat/completions` for everything else — which is what vLLM,
Ollama, LM Studio, OpenRouter and api.openai.com all speak.

Set `AGENTINO_PROVIDER` to `openai`, `openai-codex` or `anthropic` to override
the guess. A `sk-ant-` key implies Anthropic, and a ChatGPT subscription token
implies Codex, whatever the URL says.

## Why a new framework

Agentino was built around a few opinions other frameworks make hard:

- **Functions, not classes**: tools are `@tool`-decorated `async def`. No
  `BaseTool.execute()` ceremony.
- **YAML for shape, code for behaviour**: agent identity (model, prompt,
  available tools) is config. Logic stays in Python.
- **No graph editor**: complex flows are just `Pipeline` / `StagedPipeline`
  composed in code. If you can read a function call, you can read your flow.
- **Async-first, batteries included**: retry-with-backoff, context
  compaction, tool-output truncation, error taxonomy — all built in.
- **Provider-agnostic**: Codex, OpenAI, Anthropic, anything OpenAI-compatible.

If you've fought a framework's abstractions to get a simple agent working,
agentino is the one with the smallest surface that still grows with you.

---

## Cookbook

Concrete recipes for the patterns above:

- [`docs/cookbook/hooks.md`](docs/cookbook/hooks.md) — Python callbacks +
  shell hooks; `PostToolUse` audit trails; `PreToolUse` blockers
- ADRs in [`docs/architecture/`](docs/architecture/) — design rationale for
  the agent loop, tool chain, async-first core, JSONL sessions
- [`docs/integration-guide.md`](docs/integration-guide.md) — wiring agentino
  into an existing FastAPI app

---

## Used by

Agentino powers:

- **[Runspace](https://github.com/islavutin-oss/runspace)** — a multi-agent
  workspace: channels, @mention routing, scheduled routines and a protocol
  layer (Store, Vision, Transport, FileStorage, Embeddings). Agentino is one
  of the runtimes it drives.
- **Multi-agent back offices** — agents grouped by role (booking, finance,
  inventory, analytics), reached by @mention in a shared channel
- **Single-pane chat shells** — the same gateway configured down to one agent
- **Agentic benchmarks** — staged pipelines with security/execute/report
  stages and an LLM-gate pattern (the gates cookbook is built from it)

---

## Stability

- Public API at `from agentino import …` is stable as of v1.0
- Internal layout (subpackages) was reshaped in v1.0 — deep imports like
  `agentino.context` moved to `agentino.core.context`
- Async-first throughout. There is no synchronous wrapper: call it with
  `asyncio.run(agent.run(...))` from sync code

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
