"""Config loader — define agents and pipelines in YAML files.

Split into focused modules:
- config_utils.py: YAML loading, env vars, markdown parsing
- config_tools.py: tool and skill loading
- config_agent.py: agent building, model resolution, knowledge
- config_pipeline.py: pipeline and gateway building

This file is the public API — load_config(), load_agents(), Config class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentino.builtin_tools import BUILTIN_TOOLS
from agentino.config.agents_yaml import build_agent, load_project_knowledge, resolve_model_ref
from agentino.config.pipeline_yaml import build_gateway_config, build_pipeline
from agentino.config.tools_yaml import discover_tools_from_dir, import_tools_from_file, load_skills

# Sub-modules
from agentino.config.utils import load_yaml, parse_instructions_md
from agentino.core.agent import Agent
from agentino.core.tool import Tool

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Loaded configuration — agents, pipeline, gateway, and raw data."""

    agents: dict[str, Agent] = field(default_factory=dict)
    pipeline: Any = None  # StagedPipeline or Pipeline
    gateway: Any = None  # GatewayConfig
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path, tools: list[Tool] | None = None) -> Config:
    """Load a full config from agents.yml — agents, pipeline, gateway."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".yml", ".yaml"):
        raise ValueError(f"Unsupported config format: {suffix}. Use .yml or .yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config_dir = path.parent

    data = load_yaml(path)
    config = Config(raw=data)

    # Defaults: explicit "defaults:" section or top-level non-reserved keys
    defaults = data.get("defaults", {}) or {}
    for k, v in data.items():
        if k not in (
            "agents",
            "pipeline",
            "gateway",
            "providers",
            "stages",
            "message_hook",
            "defaults",
        ):
            if k not in defaults:
                defaults[k] = v
    providers = data.get("providers", {})

    # Build tool map: builtins + any extras passed in
    tool_map: dict[str, Tool] = {t.name: t for t in BUILTIN_TOOLS}
    if tools:
        for t in tools:
            tool_map[t.name] = t

    # Build agents
    agents_cfg = data.get("agents", {})
    for name, agent_cfg in agents_cfg.items():
        agent_tool_map = dict(tool_map)  # Copy — each agent may add its own tools
        config.agents[name] = build_agent(
            name,
            agent_cfg,
            defaults,
            agent_tool_map,
            config_dir,
            providers,
        )

    # Staged pipeline (from stages.yml — explicit or auto-detected)
    stages_file = data.get("stages")
    if not stages_file and (config_dir / "stages.yml").exists():
        stages_file = "stages.yml"  # Auto-detect convention
    if stages_file:
        stages_path = Path(stages_file)
        if not stages_path.is_absolute():
            stages_path = config_dir / stages_file
        if stages_path.exists():
            from agentino.pipeline.staged import StageDef, StagedPipeline

            stages_data = load_yaml(stages_path)
            stages = []
            for s in stages_data.get("stages", []):
                kwargs = {
                    "name": s["name"],
                    "prompt": s.get("prompt", ""),
                    "tools": s.get("tools", []),
                    "verdict_tool": s.get("verdict_tool", ""),
                    "max_turns": s.get("max_turns", 10),
                    "on_fail": s.get("on_fail", ""),
                }
                # Optional fields
                for opt in ("skip_if", "repeatable", "max_cycles"):
                    if opt in s:
                        kwargs[opt] = s[opt]
                stages.append(StageDef(**kwargs))
            config.pipeline = StagedPipeline(
                stages=stages,
                global_max_cycles=stages_data.get("global_max_cycles", 3),
            )

    # Regular pipeline
    if not config.pipeline and "pipeline" in data:
        config.pipeline = build_pipeline(data["pipeline"], config.agents)

    # Gateway
    if "gateway" in data:
        config.gateway = build_gateway_config(dict(data["gateway"]))

    return config


def load_agents(path: str | Path, tools: list[Tool] | None = None) -> dict[str, Agent]:
    """Load only agents from a config file (convenience wrapper)."""
    return load_config(path, tools).agents


# Backward compat: private names used by tests and apps.
from agentino.config.pipeline_yaml import (  # noqa: E402,F401
    build_condition as _build_condition,
)
from agentino.config.utils import (  # noqa: E402,F401
    resolve_config_values as _resolve_config_values,
)
from agentino.config.utils import (  # noqa: E402,F401
    resolve_env_vars as _resolve_env_vars,
)

_resolve_model_ref = resolve_model_ref
_import_tools_from_file = import_tools_from_file
_discover_tools_from_dir = discover_tools_from_dir
_load_skills = load_skills
_build_agent = build_agent
_build_pipeline = build_pipeline
_build_gateway_config = build_gateway_config
_parse_instructions_md = parse_instructions_md
_load_project_knowledge = load_project_knowledge


__all__ = [
    "Agent",
    "Any",
    "BUILTIN_TOOLS",
    "Path",
    "StageDef",
    "StagedPipeline",
    "Tool",
    "annotations",
    "build_agent",
    "build_gateway_config",
    "build_pipeline",
    "dataclass",
    "discover_tools_from_dir",
    "field",
    "import_tools_from_file",
    "load_project_knowledge",
    "load_skills",
    "load_yaml",
    "logging",
    "parse_instructions_md",
    "resolve_model_ref",
]
