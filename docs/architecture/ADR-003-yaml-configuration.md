# ADR-003: YAML-Only Configuration

## Status
Accepted

## Context
Agentino needs a configuration format for:
- Agent definitions (model, instructions, tools)
- Pipeline definitions (stages, routing)
- Gateway configuration (channels, tokens)
- Provider settings (base URLs, API keys)

The question was which format(s) to support.

## Decision
We chose **YAML as the sole configuration format**.

```yaml
# agents.yml
agents:
  reviewer:
    model: gpt-5.4-codex
    instructions_file: prompts/reviewer.md
    tools: [read_file, grep, shell]
    
gateway:
  telegram:
    token: ${TELEGRAM_BOT_TOKEN}
    agent: reviewer
```

## Consequences

### Positive
- **Single format to learn**: No confusion about which format for what
- **Comments supported**: Unlike JSON
- **String interpolation**: `${ENV_VAR}` syntax for secrets
- **Multi-document friendly**: stages.yml separate from agents.yml
- **Human-readable**: Less noise than TOML's `key = "value"`

### Negative
- **Speed**: YAML parsing is slower than JSON (irrelevant for config files)
- **Strictness**: No schema validation by default (we validate manually)
- **Python dependency**: Requires `pyyaml` package

## Alternatives Considered

### 1. TOML
```toml
[agents.reviewer]
model = "gpt-5.4-codex"
tools = ["read_file", "grep"]
```
- Rejected: Tables syntax confusing for nested structures

### 2. JSON
- Rejected: No comments, too verbose, trailing comma issues

### 3. Python (code as config)
- Rejected: Too powerful (arbitrary code execution), harder to generate

### 4. Multiple formats
- Rejected: Documentation complexity, testing burden

## Related
- `src/agentino/config/` - config loaders
- `src/agentino/config/utils.py` - YAML parsing, env var resolution
