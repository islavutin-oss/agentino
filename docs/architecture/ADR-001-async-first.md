# ADR-001: Async-First Architecture

## Status
Accepted

## Context
Agentino is an LLM agent framework that involves:
- HTTP API calls to LLM providers (OpenAI, Anthropic, Codex)
- I/O-bound tool execution (file system, shell commands, web requests)
- Potentially long-running conversations with multiple turns
- Gateway mode handling concurrent connections from multiple users/channels

The question was whether to use synchronous or asynchronous programming as the default.

## Decision
We chose **async-first** architecture with `asyncio` as the core concurrency model.

### Key aspects:
1. **All core APIs are async** (`async def`)
2. **Sync functions are wrapped** using `asyncio.to_thread()` when needed
3. **HTTP client is `httpx.AsyncClient`**
4. **Gateway runs multiple channels concurrently** via `asyncio.gather()`

## Consequences

### Positive
- **Concurrent I/O**: Multiple agent runs, tool executions, and LLM calls can overlap
- **Gateway scalability**: Single process handles many concurrent Telegram/Slack/WhatsApp connections
- **Resource efficiency**: No thread-per-connection overhead
- **Modern Python**: Aligns with FastAPI, Starlette, aiogram patterns

### Negative
- **Learning curve**: Users must understand `async/await`
- **Sync integration**: Wrapping sync tools adds slight overhead
- **Debugging**: Async stack traces can be harder to follow
- **Library constraints**: Some libraries lack async support (requires thread wrapping)

## Alternatives Considered

### 1. Sync-first with threading
- Rejected: Thread-per-request doesn't scale for gateway mode
- GIL limits true parallelism for CPU-bound work anyway

### 2. Hybrid (sync core, async gateway)
- Rejected: Creates two APIs to maintain, confusing for users
- Sync core would still block during LLM calls

### 3. Trio or other async libraries
- Rejected: `asyncio` is standard library, widest ecosystem
- `httpx`, `aiogram`, `slack-bolt` all support asyncio

## Implementation Notes

### Pattern for sync tools:
```python
@tool
def sync_tool(x: str) -> str:
    return x.upper()

# Framework handles this internally:
if not tool.is_async:
    result = await asyncio.to_thread(tool.fn, **args)
```

### Pattern for users:
```python
# Direct async usage
reply = await agent.run("hello")

# Or use the CLI/Runner which handles the event loop
runner.repl()  # Runs asyncio internally
```
