"""Skill system — lazy-loaded markdown prompts with frontmatter metadata.

Skills are discoverable prompt templates that agents can invoke on demand.
Loaded from directories, with token estimation from headers only.

Ported from Claude Code's skill system pattern.

Usage:
    registry = SkillRegistry()
    registry.scan("~/.agentino/skills")
    registry.scan(".agentino/skills")

    # List available skills (cheap — frontmatter only)
    for skill in registry.list():
        print(f"{skill.name}: {skill.description}")

    # Invoke skill (loads full content)
    content = registry.load("commit")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillMeta:
    """Skill metadata from frontmatter — loaded eagerly, content loaded lazily."""

    name: str
    description: str = ""
    when_to_use: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    path: str = ""
    estimated_tokens: int = 0  # Rough estimate from frontmatter only

    @property
    def id(self) -> str:
        return self.name.lower().replace(" ", "-")


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter from markdown file."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}
    try:
        import yaml

        return yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}


def _estimate_tokens_from_meta(meta: SkillMeta) -> int:
    """Rough token estimate from frontmatter fields only (not full content)."""
    text = f"{meta.name} {meta.description} {meta.when_to_use}"
    return len(text) // 4  # ~4 chars per token


class SkillRegistry:
    """Registry of available skills with lazy loading.

    Skills are markdown files with YAML frontmatter:
        ---
        name: commit
        description: Stage and commit changes with AI-generated message
        whenToUse: When user asks to commit or says /commit
        allowedTools: [shell, read_file]
        ---

        ## Instructions
        ...full skill content loaded on invoke...
    """

    def __init__(self):
        self._skills: dict[str, SkillMeta] = {}
        self._scanned_dirs: set[str] = set()

    def scan(self, directory: str | Path) -> int:
        """Scan a directory for skill files. Returns count of skills found.

        Deduplicates by realpath (catches symlinks).
        Only reads frontmatter — content loaded lazily on invoke.
        """
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            return 0

        real = str(directory.resolve())
        if real in self._scanned_dirs:
            return 0
        self._scanned_dirs.add(real)

        count = 0
        for f in directory.rglob("*.md"):
            if f.name.startswith("_") or f.name.startswith("."):
                continue
            try:
                # Read only first 2KB for frontmatter
                with open(f) as fh:
                    header = fh.read(2048)
                fm = _parse_frontmatter(header)
                if not fm.get("name"):
                    fm["name"] = f.stem

                meta = SkillMeta(
                    name=fm.get("name", f.stem),
                    description=fm.get("description", ""),
                    when_to_use=fm.get("whenToUse", fm.get("when_to_use", "")),
                    allowed_tools=fm.get("allowedTools", fm.get("allowed_tools", [])),
                    path=str(f),
                )
                meta.estimated_tokens = _estimate_tokens_from_meta(meta)

                # Deduplicate by name (first wins)
                if meta.id not in self._skills:
                    self._skills[meta.id] = meta
                    count += 1
            except Exception:
                continue

        return count

    def list(self) -> list[SkillMeta]:
        """List all registered skills (metadata only, no content loaded)."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> SkillMeta | None:
        """Get skill metadata by name/id."""
        return self._skills.get(name.lower().replace(" ", "-"))

    def load(self, name: str) -> str | None:
        """Load full skill content (lazy — reads file on demand)."""
        meta = self.get(name)
        if not meta or not meta.path:
            return None
        try:
            with open(meta.path) as f:
                content = f.read()
            # Strip frontmatter, return content only
            stripped = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)
            return stripped.strip()
        except Exception:
            return None

    def total_estimated_tokens(self) -> int:
        """Total estimated tokens for all skill metadata (not content)."""
        return sum(s.estimated_tokens for s in self._skills.values())
