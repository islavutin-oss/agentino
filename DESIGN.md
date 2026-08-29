# Design Document

## Vision

Lightweight Python agent framework. YAML config → agents → run.

Apache-2.0 licensed. Provider-agnostic. Core deps: httpx + pyyaml.

## Core Principle

**An agent is a loop.** Send messages + tools to an LLM. If the LLM calls a tool, execute it and loop. If the LLM returns text, you're done.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Layer 4: Transport                         │
│  Telegram, Slack, WhatsApp, WebSocket       │
├─────────────────────────────────────────────┤
│  Layer 3: Orchestration                     │
│  StagedPipeline, RouterPipeline, Parallel   │
├─────────────────────────────────────────────┤
│  Layer 2: Persistence                       │
│  Session (JSONL), KnowledgeBase (TF-IDF +   │
│  dense embeddings, SQLite cache)            │
├─────────────────────────────────────────────┤
│  Layer 1: Core                              │
│  Agent, @tool, LLMClient, Config, Events    │
└─────────────────────────────────────────────┘
```

Each layer independently usable. All layers optional except Core.

## Config-Driven Agents

```yaml
# agents.yml
providers:
  router:
    base_url: https://router.example.com/v1
    api_key: ${AGENTINO_API_KEY}
    provider: openai-codex

agents:
  coder:
    model: router/gpt-5.4-codex
    soul: ./SOUL.md
    tools_dir: ./tools
    tools: [read_file, write_file, shell, grep, stage_verdict]
    temperature: 0.3
    max_turns: 50

gateway:
  slack:
    bot_token: ${SLACK_BOT_TOKEN}
    app_token: ${SLACK_APP_TOKEN}
    agent: coder
```

`load_config("agents.yml")` auto-detects `stages.yml` → creates StagedPipeline.

## Agent Pattern

```
my-agent/
├── SOUL.md          # identity + rules
├── agents.yml       # model, tools, provider config
├── stages.yml       # pipeline stages (optional)
└── tools/           # custom tools (optional)
```

## Staged Pipeline

Multi-stage execution with deterministic control flow:

```yaml
# stages.yml
global_max_cycles: 20

stages:
  - name: IMPLEMENT
    prompt: "Implement the plan..."
    verdict_tool: stage_verdict
    repeatable: true
    on_fail: retry

  - name: TEST
    prompt: "Run tests..."
    verdict_tool: stage_verdict
    repeatable: true
    on_fail: IMPLEMENT    # jump back on failure
```

- `verdict_tool` — structured ACCEPT/REJECT/FAIL signals
- `repeatable` + `max_cycles` — retry stages
- `on_fail: "STAGE_NAME"` — jump to named stage (loops)
- `global_max_cycles` — hard budget
- Fresh Agent context per stage
- `on_event` callback for all pipeline events

## Router Pipeline

LLM-based intent classification → route to specialist agents:

```yaml
pipeline:
  type: router
  router: classifier
  routes:
    support: support_agent
    billing: billing_agent
  default: support
```

## Message Hook

App-level intent routing:

```yaml
message_hook: my_module    # loads my_module.classify_and_route
```

`async (runner, agent, message, session) -> str | None`
Return str = reply (skip pipeline). Return None = run pipeline.

## Built-in Tools

`shell`, `read_file`, `write_file`, `edit_file`, `list_files`, `grep`, `search_files`, `stage_verdict`

Shell timeout: `AGENTINO_SHELL_TIMEOUT` env var (default: no limit).

## Knowledge Base

Hybrid TF-IDF + dense embeddings. Auto-indexed from `.agentino/`.
Tools: `search_knowledge`, `save_knowledge`, `delete_knowledge`.

## Gateway

Single process, multiple channels: Telegram, Slack, WhatsApp, WebSocket.

```
agentino run agents.yml --gateway
```

## File Layout

```
src/agentino/           ~12,000 LOC
├── agent.py            Core loop + resilience
├── tool.py             @tool decorator, schema gen
├── llm.py              OpenAI-compatible client
├── config.py           YAML loader, auto-detect stages.yml
├── staged.py           StagedPipeline
├── pipeline.py         RouterPipeline, sequence, parallel
├── runner.py           Runner, message_hook, HTTP, REPL
├── cli_renderer.py     Terminal rendering
├── builtin_tools.py    shell, read_file, grep, etc.
├── knowledge.py        Hybrid search KB
├── session.py          JSONL persistence
├── resilience.py       Retry, compaction, truncation
├── message.py          Message, ToolCall, Event
├── auth.py             Multi-provider auth
├── context.py          Thread-local context
├── spawn.py            Subagent spawning
├── audio.py            STT transcription
├── usage.py            Token tracking
└── transport/          Channel adapters
    ├── gateway.py
    ├── telegram.py
    ├── slack.py
    ├── websocket.py
    ├── webhook.py
    └── whatsapp.py
```

27 test files. Python >=3.10.

## Dependencies

| Install | Adds |
|---------|------|
| `pip install agentino` | httpx, pyyaml |
| `agentino[memory]` | + numpy |
| `agentino[telegram]` | + aiogram |
| `agentino[slack]` | + slack-bolt, aiohttp |
| `agentino[serve]` | + uvicorn, starlette |
| `agentino[all]` | everything |

## Design Decisions

- YAML-only config (see ADR-003)
- OpenAI-compatible API (Codex, OpenAI, Anthropic, any endpoint)
- Async-first architecture (see ADR-001)
- Tool execution chain: validate → permission → execute (see ADR-002)
- FinalResult for deterministic structured outputs
- JSONL sessions (see ADR-004)
- Framework provides mechanisms, not policies — no app logic in core
- stages.yml = pure pipeline config, app routing via message_hook

## Architecture Decision Records

See `docs/architecture/` for detailed ADRs on major design choices:

| ADR | Title | Summary |
|-----|-------|---------|
| ADR-001 | Async-First | All core APIs are async for concurrent I/O efficiency |
| ADR-002 | Tool Execution Chain | 3-stage validation → permission → execution pattern |
| ADR-003 | YAML-Only | Single configuration format for simplicity |
| ADR-004 | JSONL Sessions | Line-based conversation persistence for git-friendliness |
