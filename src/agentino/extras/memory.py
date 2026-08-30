"""Memory system — persisted facts with relevance selection.

Two-layer index: MEMORY.md (capped at 200 lines) as entrypoint,
topic files as detail. Relevance selected by keyword matching
(upgradeable to LLM classifier).

Ported from Claude Code's memdir pattern.

Usage:
    mem = MemoryStore("~/.agentino/memory")
    mem.save("user_role", "User is a senior Python developer", type="user")
    relevant = mem.find_relevant("How should I structure this API?")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agentino.core.session import safe_segment

_MAX_INDEX_LINES = 200
_MAX_INDEX_BYTES = 25_000


@dataclass
class MemoryEntry:
    """A memory file with frontmatter metadata."""

    name: str
    description: str
    type: str  # user, feedback, project, reference
    path: str
    content: str = ""


def _parse_memory_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from a memory file."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}
    result = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


class MemoryStore:
    """Persistent memory with relevance selection.

    Memory files live in a directory with MEMORY.md as the index.
    Each memory file has frontmatter: name, description, type.
    """

    def __init__(self, directory: str | Path):
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, MemoryEntry] = {}
        self._scan()

    def _scan(self) -> None:
        """Scan directory for memory files."""
        self._entries.clear()
        for f in self._dir.glob("*.md"):
            if f.name == "MEMORY.md":
                continue
            try:
                text = f.read_text()
                fm = _parse_memory_frontmatter(text)
                entry = MemoryEntry(
                    name=fm.get("name", f.stem),
                    description=fm.get("description", ""),
                    type=fm.get("type", "reference"),
                    path=str(f),
                    content=text,
                )
                self._entries[f.stem] = entry
            except Exception:
                continue

    def save(
        self, key: str, content: str, type: str = "reference", description: str = "", name: str = ""
    ) -> Path:
        """Save a memory entry and update the index."""
        name = name or key.replace("_", " ").title()
        description = description or content[:100]

        text = f"---\nname: {name}\ndescription: {description}\ntype: {type}\n---\n\n{content}\n"
        path = self._dir / f"{safe_segment(key, fallback='memory')}.md"
        path.write_text(text)

        self._entries[key] = MemoryEntry(
            name=name,
            description=description,
            type=type,
            path=str(path),
            content=text,
        )
        self._update_index()
        return path

    def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        if key in self._entries:
            path = Path(self._entries[key].path)
            if path.exists():
                path.unlink()
            del self._entries[key]
            self._update_index()
            return True
        return False

    def get(self, key: str) -> MemoryEntry | None:
        """Get a memory entry by key."""
        return self._entries.get(key)

    def list_all(self) -> list[MemoryEntry]:
        """List all memory entries."""
        return sorted(self._entries.values(), key=lambda e: e.name)

    def find_relevant(self, query: str, max_results: int = 5) -> list[MemoryEntry]:
        """Find memory entries relevant to a query.

        Uses keyword overlap scoring. Can be upgraded to LLM classifier
        (like Claude Code's Sonnet selector) for better relevance.
        """
        query_words = set(query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries.values():
            # Score by keyword overlap with name + description + content
            entry_words = set(
                (entry.name + " " + entry.description + " " + entry.content[:500]).lower().split()
            )
            overlap = len(query_words & entry_words)
            if overlap > 0:
                # Boost by type relevance
                boost = {"feedback": 1.5, "user": 1.3, "project": 1.2, "reference": 1.0}
                score = overlap * boost.get(entry.type, 1.0)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def load_index(self) -> str:
        """Load MEMORY.md content, capped at 200 lines / 25KB."""
        index_path = self._dir / "MEMORY.md"
        if not index_path.exists():
            return ""
        text = index_path.read_text()
        # Apply caps (from Claude Code: 200 lines, 25KB)
        lines = text.split("\n")
        if len(lines) > _MAX_INDEX_LINES:
            text = "\n".join(lines[:_MAX_INDEX_LINES])
            text += f"\n\n[...truncated at {_MAX_INDEX_LINES} lines]"
        if len(text.encode()) > _MAX_INDEX_BYTES:
            text = text[:_MAX_INDEX_BYTES]
            # Cut at last newline
            last_nl = text.rfind("\n")
            if last_nl > 0:
                text = text[:last_nl]
            text += f"\n\n[...truncated at {_MAX_INDEX_BYTES} bytes]"
        return text

    def _update_index(self) -> None:
        """Regenerate MEMORY.md index file."""
        lines = []
        for key, entry in sorted(self._entries.items()):
            lines.append(f"- [{entry.name}]({key}.md) — {entry.description[:80]}")

        index_path = self._dir / "MEMORY.md"
        index_path.write_text("\n".join(lines) + "\n")
