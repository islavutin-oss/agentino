# Integration Guide — Two Ways to Use Agentino

Agentino supports two integration patterns depending on your project's architecture.

---

## Option 1: Gateway Mode (Standalone)

Use agentino as the **entire backend**. The gateway handles messaging channels, session management, and agent routing — no external server needed.

```
Telegram / Slack / WhatsApp
    ↕
Agentino Gateway (multi-channel)
    ↕
Agent (tool-calling loop)
    ↕
Tools (@tool functions)
```

### When to use

- **New project** — no existing backend
- **Simple bots** — message in, response out, no complex middleware
- **Multi-channel** — same agent serves Telegram, Slack, WhatsApp simultaneously
- **Rapid prototyping** — one config file, one command

### Setup

**1. Define agents and gateway in `agents.yml`:**

```yaml
agents:
  assistant:
    model: gpt-4o
    instructions: "You are a helpful assistant."
    skills: [customer_support]
    skills_dir: skills

gateway:
  telegram:
    token: ${TELEGRAM_BOT_TOKEN}
    agent: assistant
  slack:
    bot_token: ${SLACK_BOT_TOKEN}
    app_token: ${SLACK_APP_TOKEN}
    agent: assistant
  whatsapp:
    bridge_url: http://localhost:3001
    port: 8080
    agent: assistant
```

**2. Run:**

```bash
agentino run agents.yml --gateway
```

**3. Multiple bots (different agents on different channels):**

```yaml
agents:
  support:
    instructions: "You handle support tickets."
  sales:
    instructions: "You handle sales inquiries."

gateway:
  telegram:
    - token: ${TG_SUPPORT_TOKEN}
      agent: support
    - token: ${TG_SALES_TOKEN}
      agent: sales
  slack:
    bot_token: ${SLACK_TOKEN}
    app_token: ${SLACK_APP_TOKEN}
    agent: support
```

### What the gateway handles

| Concern | How |
|---------|-----|
| Session persistence | JSONL files, keyed by `{agent}--{channel}--{peer_id}` |
| Channel reconnection | Exponential backoff (2s → 60s), auto-restart on crash |
| Multi-channel | Each channel runs as an independent async task |
| Graceful shutdown | SIGINT/SIGTERM → stops all channels |

### What the gateway does NOT handle

- Multi-tenant routing
- Database persistence (bookings, orders, user profiles)
- Admin dashboards / management APIs
- Custom authentication middleware
- Complex business logic beyond the agent's tools

If you need any of the above, use **Option 2**.

---

## Option 2: Library Mode (Embedded in Existing Backend)

Use agentino as the **AI brain** inside your existing application. Your backend handles routing, sessions, persistence, and admin APIs. Agentino handles the agent loop.

```
Messaging Channel (WhatsApp, Telegram, etc.)
    ↕
Your Backend (FastAPI, Django, Express, etc.)
    ├─ Routing, auth, tenant config
    ├─ Session management (Redis, DB, etc.)
    ├─ Business logic, CRUD operations
    └─ Calls agentino Agent.run()  ← agent here
         ├─ Loads config from agents.yml
         ├─ Injects dynamic context
         ├─ Runs tool-calling loop
         └─ Returns text response
    ↕
Database (bookings, orders, analytics)
```

### When to use

- **Existing backend** — you already have FastAPI/Django/Flask/Express
- **Multi-tenant** — different tenants need different configs, routing, middleware
- **Complex domain logic** — booking systems, e-commerce, CRM with DB operations
- **Custom sessions** — Redis, Supabase, DynamoDB (not JSONL files)
- **Admin APIs** — dashboards, settings editors, analytics alongside the bot

### Setup

**1. Install agentino:**

```bash
pip install agentino
# or vendor a wheel:
pip install vendor/agentino-0.6.0-py3-none-any.whl
```

**2. Define agent config in `agents.yml`:**

```yaml
providers:
  codex:
    # Auto-detected from ~/.codex/auth.json
  openai:
    api_key: ${OPENAI_API_KEY}

agents:
  assistant:
    model:
      primary: codex/gpt-5.4-codex
      fallbacks:
        - openai/gpt-4o
    skills: [customer_support]
    skills_dir: skills
    knowledge:
      embedding_base_url: ${EMBEDDING_URL}
      embedding_model: ${EMBEDDING_MODEL}
```

**3. Create a wrapper that injects your domain context:**

```python
from agentino import Agent, load_config, context
from agentino import Message

_config = load_config("agents.yml")
_template = _config.agents["assistant"]


async def handle_message(user_text: str, user_id: str, tenant_id: str) -> str:
    """Called by your backend for each incoming message."""

    # Build dynamic context (your domain data)
    dynamic = build_context_for_user(user_id, tenant_id)

    # Create per-request agent from template
    agent = Agent(
        model=_template.model,
        instructions=_template.instructions + "\n\n" + dynamic,
        tools=_template.tools,
        knowledge=_template.knowledge,
        base_url=_template._llm.base_url,
        api_key=_template._llm.api_key,
        fallback_models=_template.fallback_models,
    )

    # Load conversation history from YOUR session store
    history = await your_session_store.get(user_id)
    messages = agent._prepare_messages(user_text, session=None)

    # Inject history
    if history:
        user_msg = messages.pop()
        for msg in history:
            messages.append(Message(role=msg["role"], content=msg["content"]))
        messages.append(user_msg)

    # Run agent loop
    response = await run_agent_loop(agent, messages)

    # Save to YOUR session store
    await your_session_store.save(user_id, messages)

    return response
```

**4. Your backend calls this from its message endpoint:**

```python
# FastAPI example
@app.post("/api/{tenant_id}/message")
async def on_message(tenant_id: str, body: MessageRequest):
    reply = await handle_message(body.text, body.sender_id, tenant_id)
    return {"response": reply}
```

### What you handle

| Concern | Your backend |
|---------|-------------|
| Transport | Your existing bridge/webhook/polling |
| Routing | Multi-tenant, auth, rate limiting |
| Sessions | Your session store (Redis, DB) |
| Persistence | Messages to DB, booking CRUD |
| Admin APIs | Dashboards, settings, analytics |

### What agentino handles

| Concern | Agentino |
|---------|----------|
| LLM calls | Retry, fallback models, context compaction |
| Tool execution | Async-native, timeout enforcement, dedup |
| Knowledge search | TF-IDF + optional dense embeddings |
| Skill loading | SKILL.md → instructions, tools/*.py → @tool discovery |
| Config | YAML-driven model/provider/skills |

---

## Comparison

| | Gateway Mode | Library Mode |
|-|-------------|-------------|
| **Setup** | One file (`agents.yml`) | Your app + `agents.yml` |
| **Sessions** | JSONL files (automatic) | Your session store |
| **Channels** | Built-in (Telegram, Slack, WhatsApp) | Your transport layer |
| **Multi-tenant** | No | Yes (your routing) |
| **Admin UI** | No | Yes (your endpoints) |
| **DB persistence** | No | Yes (your database) |
| **Complexity** | Minimal | More code, full control |
| **Best for** | Standalone bots, prototypes | Production apps with existing backends |

---

## Channel Reference (Gateway Mode)

### Telegram

Long polling via aiogram. No public URL needed.

```yaml
gateway:
  telegram:
    token: ${TELEGRAM_BOT_TOKEN}
    agent: assistant
```

```bash
pip install agentino[telegram]
```

### Slack

Socket Mode via slack_bolt. No public URL needed. Requires an app-level token (`xapp-...`) for Socket Mode, plus a bot token (`xoxb-...`).

```yaml
gateway:
  slack:
    bot_token: ${SLACK_BOT_TOKEN}
    app_token: ${SLACK_APP_TOKEN}
    agent: assistant
```

```bash
pip install agentino[slack]
```

### WhatsApp

HTTP adapter for a Baileys (Node.js) bridge. The bridge handles WhatsApp Web protocol; agentino handles the AI.

```yaml
gateway:
  whatsapp:
    bridge_url: http://localhost:3001  # where the bridge runs
    port: 8080                         # port agentino listens on
    agent: assistant
```

```bash
pip install agentino[serve]

# Start the bridge (Node.js, separate process)
cd src/agentino/transport/whatsapp-bridge
npm install && AGENTINO_URL=http://localhost:8080 node bridge.js
```

### Custom Channel

Register your own channel type:

```python
from agentino.transport import Channel, register_channel

class DiscordChannel(Channel):
    name = "discord"

    def __init__(self, agent, session_dir, token, **kwargs):
        super().__init__(agent, session_dir, kwargs)
        self.token = token

    async def start(self):
        # Connect to Discord, receive messages
        # Call self.handle_message(text, peer_id) for each message
        ...

    async def stop(self):
        ...

register_channel("discord", DiscordChannel)
```

Then use in config:

```yaml
gateway:
  discord:
    token: ${DISCORD_TOKEN}
    agent: assistant
```
