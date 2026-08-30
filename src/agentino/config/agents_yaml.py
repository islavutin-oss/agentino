"""Config — agent building from YAML config dicts."""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from agentino.config.tools_yaml import discover_tools_from_dir, load_skills
from agentino.config.utils import parse_instructions_md
from agentino.core.agent import Agent
from agentino.core.tool import Tool
from agentino.extras.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

# Well-known providers with built-in base URLs
WELL_KNOWN_PROVIDERS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


def resolve_model_ref(
    ref: str,
    providers: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Resolve 'provider/model' to (model, base_url, api_key)."""
    if "/" not in ref:
        return ref, None, None
    provider_name, model = ref.split("/", 1)
    pcfg = providers.get(provider_name, {}) or {}
    base_url = pcfg.get("base_url") or WELL_KNOWN_PROVIDERS.get(provider_name)
    api_key = pcfg.get("api_key")
    return model, base_url, api_key


def load_project_knowledge(
    agent_name: str,
    knowledge_cfg: dict | None,
    config_dir: Path | None,
    existing_kb: KnowledgeBase | None,
) -> tuple[str, KnowledgeBase | None]:
    """Load per-project knowledge from .agentino/ folder.

    Sets AGENTINO_PROJECT_STATUS env var: bootstrap | refresh | current.
    """
    bridge_dir = os.environ.get("AGENTINO_PROJECT_DIR")
    if not bridge_dir:
        return agent_name, existing_kb

    project_path = Path(bridge_dir)
    slug = project_path.name
    qualified_name = f"{agent_name}_{slug}"
    docs_dir = project_path / ".agentino"

    status = _detect_project_status(docs_dir, bridge_dir)
    os.environ["AGENTINO_PROJECT_STATUS"] = status

    # Build knowledge base
    kcfg = knowledge_cfg or {}
    kb = existing_kb
    tool_desc = "Search for project knowledge — architecture, commands, patterns, gotchas."

    if kb is None:
        kb = KnowledgeBase(
            embedding_base_url=kcfg.get("embedding_base_url"),
            embedding_model=kcfg.get("embedding_model"),
            embedding_api_key=kcfg.get("embedding_api_key"),
            tool_description=kcfg.get("tool_description") or tool_desc,
            agent_name=qualified_name,
            dense_weight=kcfg.get("dense_weight"),
            language_boost=kcfg.get("language_boost"),
            min_score=kcfg.get("min_score"),
            top_k=kcfg.get("top_k"),
        )
    elif kb._agent_name != qualified_name:
        kb = KnowledgeBase(
            embedding_base_url=kb._embedding_base_url,
            embedding_model=kb._embedding_model,
            embedding_api_key=kb._embedding_api_key,
            tool_description=kb._tool_description or tool_desc,
            agent_name=qualified_name,
            dense_weight=kb.DENSE_WEIGHT,
            language_boost=kb.LANGUAGE_BOOST,
            min_score=kb.MIN_SCORE,
            top_k=kb.TOP_K,
        )

    try:
        count = kb.index_directory(docs_dir)
        if count:
            logger.debug("Project knowledge: %d entries from %s", count, docs_dir)
    except Exception as e:
        logger.warning("Failed to index project knowledge: %s", e)

    return qualified_name, kb


def _detect_project_status(docs_dir: Path, bridge_dir: str) -> str:
    """Detect project knowledge status: bootstrap | refresh | current."""
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        return "bootstrap"

    meta_path = docs_dir / "_meta.yml"
    md_files = list(docs_dir.glob("*.md"))
    if not meta_path.exists() or not md_files:
        return "bootstrap"

    try:
        import yaml

        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception:
        meta = {}

    stored_hash = meta.get("last_commit_hash", "")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=bridge_dir,
        ).stdout.strip()
    except Exception:
        head = ""

    if stored_hash != head and head:
        if stored_hash:
            try:
                diff_stat = subprocess.run(
                    ["git", "diff", "--stat", f"{stored_hash}..{head}"],
                    capture_output=True,
                    text=True,
                    cwd=bridge_dir,
                ).stdout.strip()
                if diff_stat:
                    os.environ["AGENTINO_PROJECT_DIFF"] = diff_stat
            except Exception:
                pass
        return "refresh"

    return "current"


def build_agent(
    name: str,
    cfg: dict[str, Any],
    defaults: dict[str, Any],
    tool_map: dict[str, Tool],
    config_dir: Path | None = None,
    providers: dict[str, Any] | None = None,
) -> Agent:
    """Build an Agent from config dict + defaults."""
    providers = providers or {}

    # Model resolution
    model, base_url, api_key = _resolve_model(cfg, defaults, providers)
    base_url = cfg.get("base_url", defaults.get("base_url")) or base_url
    api_key = cfg.get("api_key", defaults.get("api_key")) or api_key
    provider = cfg.get("provider", defaults.get("provider"))

    # Instructions
    instructions = _load_instructions(cfg, config_dir)
    workspace_dir = config_dir

    # Skills
    instructions, tool_map, skill_kb, has_skill_tools = _load_agent_skills(
        name,
        cfg,
        defaults,
        tool_map,
        config_dir,
        workspace_dir,
        instructions,
    )

    # Direct tools_dir
    _load_direct_tools(cfg, defaults, config_dir, tool_map)

    # Workspace files (SOUL.md, RULES.md, etc.)
    instructions = _inject_workspace_files(workspace_dir, instructions)

    if not instructions:
        instructions = f"You are {name}."

    # Context files
    instructions = _inject_context_files(cfg, defaults, config_dir, instructions)

    # Runtime data
    now = datetime.now()
    instructions += (
        f"\n\n# Runtime\n"
        f"Current date: {now.strftime('%A, %d %B %Y')} ({now.strftime('%Y-%m-%d')})\n"
        f"Current time: {now.strftime('%H:%M')}\n"
    )

    # Resolve tools list
    tools = _resolve_tools(name, cfg, tool_map, has_skill_tools)

    # Knowledge
    knowledge = _resolve_knowledge(name, cfg, defaults, config_dir, skill_kb)

    # Auth
    if cfg.get("auth", defaults.get("auth")) == "setup-token" and not api_key:
        api_key = os.getenv("ANTHROPIC_SETUP_TOKEN", "")
        if not provider:
            provider = "anthropic"
        if not base_url:
            base_url = "https://api.anthropic.com"

    # Tool instructions
    tool_instructions_kwargs: dict[str, Any] = {}
    ti = cfg.get("tool_instructions", defaults.get("tool_instructions"))
    if ti:
        tool_instructions_kwargs["tool_instructions"] = ti

    sanitize_cfg = cfg.get("sanitize", defaults.get("sanitize", {}))
    sanitize_path_params = sanitize_cfg.get("path_args", []) if sanitize_cfg else []

    return Agent(
        model=model,
        instructions=instructions,
        tools=tools,
        api_key=api_key,
        base_url=base_url,
        max_turns=cfg.get("max_turns", defaults.get("max_turns", 20)),
        temperature=cfg.get("temperature", defaults.get("temperature", 0.7)),
        tool_result_cap=cfg.get("tool_result_cap", defaults.get("tool_result_cap", 4000)),
        provider=provider,
        knowledge=knowledge,
        fallback_models=cfg.get("model", {}).get("fallbacks", [])
        if isinstance(cfg.get("model"), dict)
        else [],
        require_tool_use=cfg.get("require_tool_use", defaults.get("require_tool_use", False)),
        sanitize_path_params=sanitize_path_params,
        **tool_instructions_kwargs,
    )


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _resolve_model(cfg, defaults, providers):
    model_cfg = cfg.get("model", defaults.get("model"))
    model_ref = (
        model_cfg.get("primary")
        if isinstance(model_cfg, dict)
        else model_cfg
        if isinstance(model_cfg, str)
        else None
    )
    if model_ref:
        return resolve_model_ref(model_ref, providers)
    return None, None, None


def _load_instructions(cfg, config_dir):
    instructions = cfg.get("instructions")
    if not instructions and cfg.get("instructions_file"):
        p = Path(cfg["instructions_file"])
        if config_dir and not p.is_absolute():
            p = config_dir / p
        instructions = p.read_text().strip()
    return instructions or ""


def _load_agent_skills(name, cfg, defaults, tool_map, config_dir, workspace_dir, instructions):
    skill_names = cfg.get("skills", defaults.get("skills", []))
    has_skill_tools = False
    skill_kb = None
    if skill_names:
        skills_dir_str = cfg.get("skills_dir", defaults.get("skills_dir", "skills"))
        skills_dir = Path(skills_dir_str)
        if config_dir and not skills_dir.is_absolute():
            skills_dir = config_dir / skills_dir_str
        knowledge_cfg = cfg.get("knowledge", defaults.get("knowledge"))
        soul_parts, skill_tools, skill_kb = load_skills(
            skill_names,
            skills_dir,
            knowledge_cfg,
            config_dir,
            agent_name=name,
            workspace_dir=workspace_dir,
        )
        if soul_parts:
            soul_text = "\n\n".join(soul_parts)
            instructions = f"{soul_text}\n\n{instructions}" if instructions else soul_text
        if skill_tools:
            has_skill_tools = True
        for t in skill_tools:
            tool_map[t.name] = t
    return instructions, tool_map, skill_kb, has_skill_tools


def _load_direct_tools(cfg, defaults, config_dir, tool_map):
    tools_dir_raw = cfg.get("tools_dir", defaults.get("tools_dir"))
    if tools_dir_raw:
        dirs = tools_dir_raw if isinstance(tools_dir_raw, list) else [tools_dir_raw]
        for td in dirs:
            p = Path(td)
            if config_dir and not p.is_absolute():
                p = config_dir / td
            if p.is_dir():
                for t in discover_tools_from_dir(p):
                    tool_map[t.name] = t


def _inject_workspace_files(workspace_dir, instructions):
    if not workspace_dir:
        return instructions
    for wf_name in ["SOUL.md", "RULES.md", "USER.md", "TOOLS.md"]:
        wf_path = workspace_dir / wf_name
        if wf_path.exists():
            content = wf_path.read_text(encoding="utf-8").strip()
            if content:
                parsed = parse_instructions_md(content)
                if parsed:
                    section = f"# {wf_name.removesuffix('.md')}\n\n{parsed}"
                    instructions = f"{instructions}\n\n{section}" if instructions else section
    return instructions


def _inject_context_files(cfg, defaults, config_dir, instructions):
    context_files = cfg.get("context_files", defaults.get("context_files", []))
    if not context_files:
        return instructions
    parts = []
    for cf in context_files:
        p = Path(cf)
        if config_dir and not p.is_absolute():
            p = config_dir / p
        if p.exists():
            content = p.read_text().strip()
            if content:
                parts.append(f"## {p.name}\n\n{content}")
    if parts:
        instructions += "\n\n# Context\n\n" + "\n\n".join(parts)
    return instructions


def _resolve_tools(name, cfg, tool_map, has_skill_tools):
    tool_names = cfg.get("tools", [])
    if tool_names:
        missing = [t for t in tool_names if t not in tool_map]
        if missing:
            import warnings

            warnings.warn(
                f"Agent '{name}': {len(missing)} tool(s) not loaded: {', '.join(missing)}."
            )
        return [tool_map[t] for t in tool_names if t in tool_map]
    elif has_skill_tools:
        from agentino.builtin_tools import BUILTIN_TOOLS

        builtin_names = {t.name for t in BUILTIN_TOOLS}
        return [t for t in tool_map.values() if t.name not in builtin_names]
    return list(tool_map.values())


def _resolve_knowledge(name, cfg, defaults, config_dir, skill_kb):
    knowledge = skill_kb if cfg.get("skills") else None
    knowledge_cfg = cfg.get("knowledge", defaults.get("knowledge"))
    knowledge_dir_str = knowledge_cfg.get("dir") if knowledge_cfg else None
    if knowledge_dir_str and not knowledge:
        p = Path(knowledge_dir_str)
        if config_dir and not p.is_absolute():
            p = config_dir / knowledge_dir_str
        if p.is_dir():
            kb = KnowledgeBase(
                embedding_base_url=knowledge_cfg.get("embedding_base_url"),
                embedding_api_key=knowledge_cfg.get("embedding_api_key"),
                embedding_model=knowledge_cfg.get("embedding_model"),
                tool_description=knowledge_cfg.get("tool_description"),
                top_k=knowledge_cfg.get("top_k"),
                agent_name=name,
            )
            td_file = knowledge_cfg.get("tool_description_file")
            if td_file:
                td_path = Path(td_file)
                if config_dir and not td_path.is_absolute():
                    td_path = config_dir / td_file
                if td_path.exists():
                    kb._tool_description = td_path.read_text().strip()
            kb.index_directory(p)
            knowledge = kb

    if os.environ.get("AGENTINO_PROJECT_DIR"):
        _, project_kb = load_project_knowledge(name, knowledge_cfg, config_dir, knowledge)
        if project_kb:
            knowledge = project_kb

    return knowledge
