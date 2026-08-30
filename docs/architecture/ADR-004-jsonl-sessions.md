# ADR-004: JSONL Session Persistence

## Status
Accepted

## Context
Agentino needs to persist conversation history for:
- User session continuity (chatbots)
- Context across agent restarts
- Debugging and audit trails

Requirements:
- Human-readable format
- Append-friendly
- Git-friendly (line-based)
- Recoverable on corruption

## Decision
We chose **JSON Lines (JSONL)** for session persistence.

```jsonl
{"role": "user", "content": "Hello", "ts": 1234567890}
{"role": "assistant", "content": "Hi there", "ts": 1234567891}
{"role": "tool", "content": "result", "tool_call_id": "call_123", "ts": 1234567892}
```

## Consequences

### Positive
- **Human-readable**: Plain text, easy to inspect
- **Append-only**: New messages just append to file
- **Line-based**: Git diffs show individual message changes
- **Partial recovery**: Corrupt line doesn't prevent reading rest of file
- **Streaming**: Can process large files line-by-line

### Negative
- **No indexing**: Must scan entire file to find specific message
- **No compression**: Larger than binary formats
- **No transactions**: Partial writes possible on crash

## Alternatives Considered

### 1. SQLite
- Rejected: Overkill for simple append-only log
- Schema migration complexity
- Binary format (not git-friendly)

### 2. Plain JSON array
```text
[{"role": "user", ...}, {"role": "assistant", ...}]
```
- Rejected: Must rewrite entire file on every append
- Merge conflicts in git

### 3. Binary formats (pickle, msgpack)
- Rejected: Not human-readable
- Python-specific (pickle)

### 4. Structured logging format
- Rejected: Additional dependency, no clear benefit

## Implementation Details

### File Structure
- One file per session: `{agent_name}--{channel}--{peer_id}.jsonl`
- Stored in configurable `session_dir` (default: `./sessions`)
- System messages excluded from persistence

### Size Limits
- Max file size: 10 MB (configurable)
- Max messages: 100 per session (configurable, trims oldest)

### Recovery
If file is corrupted or oversized:
1. Log warning
2. Delete file (start fresh session)
3. Continue without history

## Related
- `src/agentino/core/session.py` - Session implementation
