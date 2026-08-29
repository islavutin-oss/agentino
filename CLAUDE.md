# CLAUDE.md — Agentino

Notes for coding agents (and humans) working in this repository.

## What this is

A lightweight Python agent framework — 88 modules, about 12,000 lines of
code. An agent is a loop:
send messages and tool schemas to an LLM; if it calls a tool, run the tool and
loop; if it returns text, stop. Everything else here is in service of that.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

```bash
pytest tests/ -q                  # the suite CI runs
ruff check src/ tests/            # lint — must be clean
ruff format --check src/ tests/   # formatting — must be clean
python -m build                   # build the wheel

agentino chat                     # REPL
agentino run agents.yml           # run agents from config
agentino run agents.yml -m "hi"   # one-shot message
agentino run agents.yml --gateway # multi-channel gateway
```

## Layout

```
src/agentino/
├── core/         Agent, Runner, LLMClient, Tool, Message, Context, State, Session
├── config/       YAML loaders for agents, pipelines, tools
├── pipeline/     Pipeline and StagedPipeline (multi-stage flows with verdicts)
├── safety/       GateManager, HookManager, sanitizers
├── reliability/  retry and backoff, history compaction, the error taxonomy
├── extras/       knowledge base, memory, audio, skills, usage tracking
├── providers/    pluggable LLM backends
├── scheduler/    CronScheduler and the JobStore protocol
├── transport/    outbound channels — Telegram, Slack, WhatsApp, WebSocket
├── workers/      fork_agent and make_spawn_tool for multi-agent work
├── tools/std/    the standard tool catalogue
└── cli/          REPL renderer and the JSON event emitter
```

`from agentino import ...` is the public API and is what `__all__` in
`src/agentino/__init__.py` declares. Deeper paths such as
`from agentino.safety.gates import GateManager` are how internal packages talk
to each other; treat them as unstable.

## Conventions that matter here

- **The framework provides mechanisms, not policies.** Anything that encodes
  one application's behaviour — prompt wording, thresholds, a specific tool's
  name, a natural language — belongs to that application, not here. If you
  find yourself adding a rule that only makes sense for one deployment, add a
  seam instead and let config supply the rule.
- **No hardcoded endpoints or credentials.** Configuration comes from the
  environment; `agentino.tools.std._llm_env` resolves the LLM endpoint and is
  the only place that should know a default host.
- **Tests come with the change.** A bug fix without a test that fails before
  it is not finished.
- **Lint is enforced.** CI runs `ruff check` and `ruff format --check`, so a
  formatting argument is never worth having.

## Releasing

Version lives in `pyproject.toml`. The distribution is published as
`agentino-framework` because the `agentino` name on PyPI belongs to an
unrelated project; the import name is unaffected.
