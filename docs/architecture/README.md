# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for Agentino.

An ADR documents a significant architectural decision, including:
- The context and problem
- The decision that was made
- The consequences (positive and negative)
- Alternatives that were considered

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-async-first.md) | Async-First Architecture | Accepted |
| [ADR-002](ADR-002-tool-execution-chain.md) | Tool Execution Chain | Accepted |
| [ADR-003](ADR-003-yaml-configuration.md) | YAML-Only Configuration | Accepted |
| [ADR-004](ADR-004-jsonl-sessions.md) | JSONL Session Persistence | Accepted |

## What Warrants an ADR?

Create an ADR when:
- Introducing a new dependency or technology
- Changing a core abstraction (Agent, Tool, Session, etc.)
- Making a choice that affects the public API
- Deciding between significantly different approaches

Don't create an ADR for:
- Bug fixes
- Performance optimizations (unless changing architecture)
- Documentation improvements
- Internal refactoring with no API change

## Template

See existing ADRs for the format. Key sections:
1. **Status**: Proposed, Accepted, Deprecated, Superseded
2. **Context**: What problem are we solving?
3. **Decision**: What did we decide?
4. **Consequences**: What are the trade-offs?
5. **Alternatives Considered**: What else did we evaluate?
