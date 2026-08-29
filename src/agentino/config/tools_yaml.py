"""Config — tool and skill loading from directories and files."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from agentino.config.utils import parse_instructions_md
from agentino.core.tool import Tool
from agentino.extras.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


def import_tools_from_file(path: Path) -> list[Tool]:
    """Import a Python file and collect all Tool instances from it."""
    spec = importlib.util.spec_from_file_location(
        f"_agentino_skill_tool_{path.stem}",
        path,
    )
    if not spec or not spec.loader:
        return []
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        import traceback

        logger.warning("Failed to import tool file %s: %s", path, e)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return []

    tools: list[Tool] = []
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, Tool):
            tools.append(attr)
    return tools


def discover_tools_from_dir(tools_dir: str | Path) -> list[Tool]:
    """Discover tools from a directory.

    Package mode (__init__.py present): import as Python package, relative imports work.
    Flat mode: import each .py file individually.
    Recursive: subdirectories are scanned too.

    Accepts a string as well as a Path: this is a public export, and every
    caller inside the package happened to pass a Path, so a string used to
    fail on `.is_dir()` with an AttributeError.
    """
    tools: list[Tool] = []
    tools_dir = Path(tools_dir)
    if not tools_dir.is_dir():
        return tools

    resolved = tools_dir.resolve()
    is_package = (resolved / "__init__.py").exists()

    if is_package:
        parent_str = str(resolved.parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
        pkg_name = resolved.name

        for py_file in sorted(resolved.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            # Build dotted module name from relative path
            rel = py_file.relative_to(resolved)
            parts = list(rel.parent.parts) + [py_file.stem]
            module_name = f"{pkg_name}.{'.'.join(parts)}"
            try:
                mod = importlib.import_module(module_name)
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name)
                    if isinstance(obj, Tool):
                        tools.append(obj)
            except Exception as e:
                logger.warning("Failed to import tool file %s: %s", py_file.name, e)
    else:
        tools_str = str(resolved)
        if tools_str not in sys.path:
            sys.path.insert(0, tools_str)
        for py_file in sorted(resolved.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            tools.extend(import_tools_from_file(py_file))

    return tools


def load_skills(
    skill_names: list[str],
    skills_dir: Path,
    knowledge_cfg: dict[str, Any] | None = None,
    config_dir: Path | None = None,
    agent_name: str | None = None,
    workspace_dir: Path | None = None,
) -> tuple[list[str], list[Tool], KnowledgeBase | None]:
    """Load skills by name: SKILL.md → instructions, tools/*.py → Tools, knowledge/ → KB.

    Resolution order: workspace override > shared > bundled.
    """
    instruction_parts: list[str] = []
    tools: list[Tool] = []
    kb: KnowledgeBase | None = None

    _pkg_root = Path(__file__).resolve().parent.parent.parent
    bundled_skills_dir = _pkg_root / "skills"

    for name in skill_names:
        workspace_skill = workspace_dir / "skills" / name if workspace_dir else None
        shared_skill = skills_dir / name
        bundled_skill = bundled_skills_dir / name

        if workspace_skill and workspace_skill.is_dir():
            skill_path = workspace_skill
        elif shared_skill.is_dir():
            skill_path = shared_skill
        elif bundled_skill.is_dir():
            skill_path = bundled_skill
        else:
            skill_path = shared_skill

        # Instructions
        skill_md = skill_path / "SKILL.md"
        if skill_md.exists():
            raw = skill_md.read_text(encoding="utf-8")
            parsed = parse_instructions_md(raw)
            if parsed:
                instruction_parts.append(parsed)

        # Tools (workspace overrides shared by name)
        seen_tool_names: set[str] = set()
        for tools_root in [workspace_skill, shared_skill]:
            if not tools_root:
                continue
            tools_path = tools_root / "tools"
            if tools_path.is_dir():
                for py_file in sorted(tools_path.glob("*.py")):
                    if py_file.name.startswith("_"):
                        continue
                    imported = import_tools_from_file(py_file)
                    for t in imported:
                        if t.name not in seen_tool_names:
                            tools.append(t)
                            seen_tool_names.add(t.name)

        # Knowledge
        knowledge_path = skill_path / "knowledge"
        if knowledge_path.is_dir():
            if kb is None:
                kb = _build_knowledge_base(knowledge_cfg or {}, config_dir, agent_name)
            try:
                count = kb.index_directory(knowledge_path)
                if count:
                    logger.debug("Indexed %d knowledge entries from %s", count, knowledge_path)
            except Exception as e:
                logger.warning("Failed to index knowledge from %s: %s", knowledge_path, e)

    return instruction_parts, tools, kb


def _build_knowledge_base(
    kcfg: dict[str, Any], config_dir: Path | None, agent_name: str | None
) -> KnowledgeBase:
    """Build a KnowledgeBase from config dict."""
    tool_desc = kcfg.get("tool_description")
    tool_desc_file = kcfg.get("tool_description_file")
    if tool_desc_file:
        td_path = Path(tool_desc_file)
        if config_dir and not td_path.is_absolute():
            td_path = config_dir / tool_desc_file
        if td_path.exists():
            tool_desc = td_path.read_text(encoding="utf-8").strip()

    return KnowledgeBase(
        embedding_base_url=kcfg.get("embedding_base_url"),
        embedding_model=kcfg.get("embedding_model"),
        embedding_api_key=kcfg.get("embedding_api_key"),
        tool_description=tool_desc,
        agent_name=agent_name,
        dense_weight=kcfg.get("dense_weight"),
        language_boost=kcfg.get("language_boost"),
        min_score=kcfg.get("min_score"),
        top_k=kcfg.get("top_k"),
    )
