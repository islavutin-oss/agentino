# ADR-002: Tool Execution Chain

## Status
Accepted

## Context
Tools in an agent framework need:
- Input validation before execution (fail fast)
- Permission checking (security)
- Actual execution
- Error handling with categorization

The question was how to structure this flow and where to allow customization.

## Decision
We implemented a **3-stage execution chain** inspired by Claude Code:

```
validate_input() → check_permission() → fn()
     (fast)           (authz)         (execute)
```

Each stage is optional and can be sync or async.

## Consequences

### Positive
- **Fail fast**: Validation catches errors before side effects
- **Security boundary**: Clear separation for permission checks
- **Composability**: Each stage can be added independently
- **Error categorization**: Each stage has distinct error types (validation vs permission)

### Negative
- **Cognitive overhead**: Three concepts to understand vs one
- **Indirection**: Stack traces go through `Tool.execute()` wrapper
- **Naming confusion**: "Tool" refers to both the class and the decorated function

## Details

### Stage 1: validate_input()
```python
def validate_input(path: str) -> str | None:
    """Return error message if invalid, None if OK."""
    if not path.endswith('.py'):
        return "Only Python files allowed"
    return None
```

Characteristics:
- No side effects (should not modify state)
- Fast execution (no I/O)
- Called before permission check

### Stage 2: check_permission()
```python
def check_permission(path: str) -> str | None:
    """Return rejection message if not allowed, None if OK."""
    if path.startswith('/etc/'):
        return "Access to /etc/ denied"
    return None
```

Characteristics:
- May involve I/O (database lookup, auth service)
- Called after validation, before execution
- Used for gates and hooks integration

### Stage 3: fn()
The actual tool implementation.

Characteristics:
- Can be sync or async
- May have side effects
- Timeout enforced via `asyncio.wait_for()`

## Error Handling

Each stage produces structured errors:

| Stage | Error Category | Example |
|-------|---------------|---------|
| validate_input | validation | "Invalid file extension" |
| check_permission | permission | "Access denied" |
| fn | timeout, not_found, internal | "Timed out after 30s" |

## Alternatives Considered

### 1. Decorator-based middleware
```python
@validate(some_validator)
@require_permission(some_checker)
@tool
def my_tool(): ...
```
- Rejected: More verbose, harder to introspect

### 2. Single execute() method override
```python
class MyTool(Tool):
    def execute(self, args):
        ...  # validation, permissions and the work, all in one place
```
- Rejected: Verbose for simple tools, encourages mixing concerns

### 3. Pre/post hooks only
- Rejected: No clear validation vs permission distinction

## Related
- `src/agentino/core/tool.py` - Tool class implementation
- `src/agentino/builtin_tools.py` - Examples of tool definitions
