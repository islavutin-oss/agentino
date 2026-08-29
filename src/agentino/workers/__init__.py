"""Workers — three modes of parallel agent execution.

spawn: Full agent copy, blocking, with depth tracking. Use for subagent delegation.
fork: Lightweight background worker, shares parent's prompt cache. Use for verification, memory extraction.
coordinator: Orchestrate multiple workers with synthesis. Use for complex multi-step tasks.
"""

from .coordinator import Coordinator
from .fork import ForkConfig, ForkResult, fork_agent
from .spawn import make_spawn_tool

__all__ = [
    "Coordinator",
    "ForkConfig",
    "ForkResult",
    "fork_agent",
    "make_spawn_tool",
]
