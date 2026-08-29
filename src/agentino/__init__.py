"""Agentino — lightweight Python agent framework.

Config → agents → run. That's it.

    agentino run agents.yml                     # REPL
    agentino run agents.yml --agent reviewer    # specific agent
    agentino run agents.yml --serve 8080        # HTTP server
    agentino run agents.yml -m "Review PR #42"  # one-shot
"""

from agentino.builtin_tools import BUILTIN_TOOLS
from agentino.cli.renderer import CLIRenderer
from agentino.config import Config, load_agents, load_config
from agentino.core.agent import Agent
from agentino.core.extensions import ExtensionLoader, ReloadResult
from agentino.core.message import Attachment, Event, EventType, Message, ToolCall, Usage
from agentino.core.models import (
    ModelCost,
    ModelInfo,
    all_models,
    lookup_model,
    register_model,
)
from agentino.core.runner import Runner, create_runner
from agentino.core.session import Session
from agentino.core.state import (
    get_session_id,
    get_state,
    record_model_usage,
    record_skill,
    reset_state,
)
from agentino.core.tool import FinalResult, Tool, tool
from agentino.extras.audio import AudioTranscriber, build_transcriber
from agentino.extras.audit import AuditLog
from agentino.extras.knowledge import KnowledgeBase
from agentino.extras.memory import MemoryEntry, MemoryStore
from agentino.extras.skills import SkillMeta, SkillRegistry
from agentino.extras.usage import UsageTracker
from agentino.pipeline.core import ParallelPipeline, Pipeline, RouterPipeline, Step
from agentino.pipeline.staged import (
    FactStore,
    StageDef,
    StagedPipeline,
    StageResult,
    judge_stage_failure,
    parse_verdict,
    summarize_stage_output,
)
from agentino.reliability.errors import (
    ErrorClass,
    ToolError,
    classify_error,
    error_blocked,
    error_duplicate,
    error_internal,
    error_invalid_args,
    error_not_found,
    error_permission,
    error_timeout,
    error_unavailable,
    error_unknown_tool,
    error_validation,
    format_error,
    get_overflow_tokens,
    get_retry_delay,
    get_ssl_hint,
)
from agentino.reliability.resilience import (
    compact_history,
    estimate_tokens,
    repair_messages,
    retry_with_backoff,
    strip_think_tags,
    truncate_result,
)
from agentino.safety.gates import GateManager, GateRule
from agentino.safety.hooks import HOOK_EVENTS, HookManager, HookResult
from agentino.safety.sanitize import clean_path, normalize_text, sanitize_tool_args
from agentino.safety.security import INJECTION_PATTERNS, check_security, make_security_scan_tool
from agentino.workers import make_spawn_tool

from .core import context

__version__ = "1.1.0"


__all__ = [
    "Agent",
    "Attachment",
    "AudioTranscriber",
    "AuditLog",
    "BUILTIN_TOOLS",
    "CLIRenderer",
    "Config",
    "ErrorClass",
    "Event",
    "EventType",
    "ExtensionLoader",
    "FactStore",
    "FinalResult",
    "GateManager",
    "GateRule",
    "HOOK_EVENTS",
    "HookManager",
    "HookResult",
    "INJECTION_PATTERNS",
    "KnowledgeBase",
    "MemoryEntry",
    "MemoryStore",
    "Message",
    "ModelCost",
    "ModelInfo",
    "ParallelPipeline",
    "Pipeline",
    "ReloadResult",
    "RouterPipeline",
    "Runner",
    "Session",
    "SkillMeta",
    "SkillRegistry",
    "StageDef",
    "StageResult",
    "StagedPipeline",
    "Step",
    "Tool",
    "ToolCall",
    "ToolError",
    "Usage",
    "UsageTracker",
    "all_models",
    "build_transcriber",
    "check_security",
    "classify_error",
    "clean_path",
    "compact_history",
    "context",
    "create_runner",
    "error_blocked",
    "error_duplicate",
    "error_internal",
    "error_invalid_args",
    "error_not_found",
    "error_permission",
    "error_timeout",
    "error_unavailable",
    "error_unknown_tool",
    "error_validation",
    "estimate_tokens",
    "format_error",
    "get_overflow_tokens",
    "get_retry_delay",
    "get_session_id",
    "get_ssl_hint",
    "get_state",
    "judge_stage_failure",
    "load_agents",
    "load_config",
    "lookup_model",
    "make_security_scan_tool",
    "make_spawn_tool",
    "normalize_text",
    "parse_verdict",
    "record_model_usage",
    "record_skill",
    "register_model",
    "repair_messages",
    "reset_state",
    "retry_with_backoff",
    "sanitize_tool_args",
    "strip_think_tags",
    "summarize_stage_output",
    "tool",
    "truncate_result",
]
