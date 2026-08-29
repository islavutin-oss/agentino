# Agentino API Examples

Common patterns and usage examples for the Agentino framework.

## Table of Contents

1. [Basic Agent Usage](#basic-agent-usage)
2. [Custom Tools](#custom-tools)
3. [Configuration-Driven Agents](#configuration-driven-agents)
4. [Staged Pipelines](#staged-pipelines)
5. [Gateway (Multi-Channel)](#gateway-multi-channel)
6. [Sessions and Persistence](#sessions-and-persistence)
7. [Error Handling and Resilience](#error-handling-and-resilience)

---

## Basic Agent Usage

### Simple One-Shot

```python
import asyncio
from agentino import Agent

async def main():
    agent = Agent(
        model="gpt-4o",
        instructions="You are a helpful assistant.",
    )
    reply = await agent.run("What is the capital of France?")
    print(reply)
    await agent.close()

asyncio.run(main())
```

### With Built-in Tools

```python
from agentino import Agent, BUILTIN_TOOLS

agent = Agent(
    model="gpt-4o",
    instructions="You are a file analyzer.",
    tools=BUILTIN_TOOLS,  # read_file, grep, shell, etc.
)

reply = await agent.run("Count the lines of Python code in the current directory")
```

### Streaming Responses

```python
from agentino import Agent, EventType

agent = Agent(model="gpt-4o", instructions="You are a poet.")

async for event in agent.stream("Write a haiku about coding"):
    if event.type == EventType.TEXT:
        print(event.data, end="", flush=True)
    elif event.type == EventType.TOOL_START:
        print(f"\n[Tool: {event.name}]")
```

---

## Custom Tools

### Basic Tool

```python
from agentino import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Implementation here
    return f"Sunny and 22°C in {city}"

agent = Agent(
    model="gpt-4o",
    tools=[get_weather],
)
```

### Async Tool

```python
import httpx
from agentino import tool

@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10)
        return resp.text[:1000]
```

### Read-Only Tool (Parallel Execution)

```python
from agentino import tool

@tool(is_read_only=True)
def search_docs(query: str) -> list[str]:
    """Search documentation. Safe for parallel execution."""
    return ["result1", "result2"]
```

### Tool with Validation

```python
from agentino import tool

def validate_path(path: str) -> str | None:
    """Return error message if path is invalid."""
    if ".." in path:
        return "Path cannot contain parent directory references"
    return None

def check_access(path: str) -> str | None:
    """Return rejection message if access denied."""
    if path.startswith("/etc/"):
        return "Access to /etc/ is restricted"
    return None

@tool(validate_input=validate_path, check_permission=check_access)
def read_config(path: str) -> str:
    """Read a configuration file."""
    with open(path) as f:
        return f.read()
```

### Deterministic Output with FinalResult

```python
from agentino import tool, FinalResult

@tool
def confirm_order(order_id: str) -> str:
    """Confirm an order and return formatted result."""
    # Skip final LLM call, return directly
    return FinalResult(f"✓ Order {order_id} confirmed!")
```

---

## Configuration-Driven Agents

### agents.yml

```yaml
agents:
  reviewer:
    model: gpt-5.4-codex
    instructions_file: prompts/reviewer.md
    tools: [read_file, grep, shell, stage_verdict]
    temperature: 0.3
    max_turns: 50
    
  helper:
    model: gpt-4o-mini
    instructions: "You are a helpful assistant."
    tools: [web_search, web_fetch]

gateway:
  telegram:
    token: ${TELEGRAM_BOT_TOKEN}
    agent: helper
```

### Loading Config

```python
from agentino import load_config, create_runner

# Load config
config = load_config("agents.yml")

# Access agents
reviewer = config.agents["reviewer"]
reply = await reviewer.run("Review this code: ...")

# Or use Runner for full features
runner = create_runner("agents.yml")
runner.repl()  # Interactive mode
```

---

## Staged Pipelines

### stages.yml

```yaml
global_max_cycles: 20

stages:
  - name: PLAN
    prompt: "Create a plan for the task."
    verdict_tool: stage_verdict
    max_turns: 10
    repeatable: true
    on_fail: retry

  - name: IMPLEMENT
    prompt: "Implement according to the plan."
    verdict_tool: stage_verdict
    max_turns: 20
    repeatable: true
    on_fail: PLAN  # Jump back to plan on failure

  - name: TEST
    prompt: "Run tests and verify."
    verdict_tool: stage_verdict
    max_turns: 10
```

### Running Staged Pipeline

```python
from agentino import load_config

config = load_config("agents.yml")  # Auto-detects stages.yml

# Pipeline runs automatically when using runner
runner = create_runner("agents.yml")
reply = await runner.send("Add user authentication")
```

### Custom FactStore

```python
from agentino import StagedPipeline, StageDef, FactStore
from dataclasses import dataclass

@dataclass
class MyFacts(FactStore):
    task_type: str = ""
    complexity: str = "medium"
    
    def to_context(self) -> str:
        return f"Task: {self.task_type} (complexity: {self.complexity})"

stages = [
    StageDef(name="analyze", prompt="Analyze the task.", verdict_tool="stage_verdict"),
    StageDef(name="execute", prompt="Execute the task."),
]

pipeline = StagedPipeline(stages=stages, facts=MyFacts())
results = await pipeline.run(agent, "Build a login form")
```

---

## Gateway (Multi-Channel)

### Running Gateway

```bash
agentino run agents.yml --gateway
```

### Programmatic Gateway

```python
from agentino import load_config
from agentino.transport import build_gateway

config = load_config("agents.yml")
gateway = build_gateway(config)
gateway.run()  # Blocks, handles SIGINT gracefully
```

### Custom Message Hook

```python
# my_hooks.py
async def classify_and_route(channel, agent, message, session):
    """Route messages based on intent.
    
    Return a string to skip the agent (fast reply).
    Return None to let the agent handle it.
    """
    if "status" in message.lower():
        return "✓ System is operational"
    return None  # Let agent handle
```

```yaml
# agents.yml
message_hook: my_hooks  # Loads my_hooks.classify_and_route
```

---

## Sessions and Persistence

### Basic Session Usage

```python
from agentino import Agent, Session

session = Session("./sessions/user-123.jsonl")
agent = Agent(model="gpt-4o")

# First conversation
reply1 = await agent.run("Hello", session=session)

# Later... session persists
reply2 = await agent.run("What did I just say?", session=session)
```

### Session with Runner

```python
from agentino import create_runner

runner = create_runner("agents.yml", session_dir="./my-sessions")

# Each user gets their own session
reply = await runner.send("Hello", agent_name="helper", session_id="user-456")
```

### Clearing Sessions

```python
session = Session("./sessions/user-123.jsonl")
session.clear()  # Delete history
```

---

## Error Handling and Resilience

### Retry with Fallback Models

```python
agent = Agent(
    model="gpt-4o",
    fallback_models=["gpt-4o-mini", "claude-sonnet-4-20250514"],
    max_retries=3,
)

# Automatically retries on transient errors
# Falls back to next model on persistent failures
reply = await agent.run("Important task...")
```

### Handling Timeouts

```python
from agentino import tool

@tool(timeout=30.0)
def slow_operation(data: str) -> str:
    """Operation that might take a while."""
    import time
    time.sleep(25)
    return "Done"
```

### Custom Response Filter

```python
def my_filter(text: str, turn: int, max_turns: int, had_tool_calls: bool) -> str | None:
    """Return nudge message to retry, or None to accept."""
    if "I cannot" in text and not had_tool_calls:
        return "Please try using the available tools instead."
    return None

agent = Agent(
    model="gpt-4o",
    response_filter=my_filter,
)
```

---

## Advanced Patterns

### Parallel Tool Execution

Read-only tools automatically run in parallel:

```python
from agentino import Agent, tool

@tool(is_read_only=True)
def read_a() -> str:
    return "A"

@tool(is_read_only=True)
def read_b() -> str:
    return "B"

agent = Agent(tools=[read_a, read_b])
# If LLM calls both, they execute concurrently
```

### Context Compaction

```python
agent = Agent(
    model="gpt-4o",
    auto_compact=True,
    context_window=128_000,
)

# Automatically summarizes old messages when context fills up
```

### Usage Tracking

```python
from agentino import create_runner

runner = create_runner("agents.yml", usage_file="./usage.jsonl")

# After running...
print(runner.usage_tracker.summary())
# Total: 15,234 tokens · ~$0.023
```
