"""Agent memory — file-backed durable facts an agent commits about a user/tenant
across conversations.

Inspired by OpenClaw's `memory/YYYY-MM-DD.md` + curated `MEMORY.md` pattern,
plus Claude Code's per-memory-file frontmatter convention.

Layout
======
    <memory_root>/<agent_id>/<user_key>/
        MEMORY.md            # index, one line per fact (auto-maintained)
        <slug>.md            # one fact per file, with YAML frontmatter
        <slug>.md
        ...

Where `memory_root` is `<tenant_config_dir>/memories/`. The tenant config dir
is the same disk root the SOULs and workspace.yml live under (mounted from
`/var/lib/acme/<tenant>/config/` on the VPS, or local repo equivalent
during dev).

Each fact file:

    ---
    name: prefers-window-seat
    description: User prefers window seats
    kind: preference
    session_id: analytics-sam_rivera_example_com-1745520000
    created: 2026-04-20T14:32:00Z
    updated: 2026-04-20T14:32:00Z
    ---

    Optional longer prose. The frontmatter `description` is what gets shown
    when memories are auto-injected into the system prompt — keep it tight.

Tenant + user boundaries
========================
The memory root is rooted in the tenant's config dir. Cross-tenant access is
impossible because each VPS only has its own tenant's dir mounted, and the
helper resolves the dir from the agent context (which is set by the framework
at chat-handler entry — same place tenant_id is set).

User identity (`user_key`) comes from the agent context's `sender_id` (the
session_id, e.g. `analytics-sam_rivera_example_com`) — so memories are
per-user-per-agent.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# Strict slug — same restrictions as filenames; no path separators, no `..`.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_AGENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_USER_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.@:+-]{0,127}$", re.IGNORECASE)

MAX_MEMORIES_PER_LOAD = 50  # cap for system-prompt injection
MAX_DESCRIPTION_LEN = 200  # one-liner shown in the index + prompt
MAX_BODY_LEN = 4000  # full prose body of a memory file


# ----------------------------------------------------------------------------
# Path / context resolution
# ----------------------------------------------------------------------------


def _resolve_tenant_id() -> str:
    try:
        from agentino.core.context import get_context
    except ImportError:
        raise RuntimeError("agentino.core.context unavailable — cannot resolve tenant_id")
    tenant = get_context("tenant")
    tid = getattr(tenant, "id", None) if tenant else None
    if not tid:
        tid = get_context("tenant_id")
    if not tid:
        raise RuntimeError("tenant context missing — agent memory requires tenant_id")
    tid = str(tid)
    if not _TENANT_ID_RE.match(tid):
        raise RuntimeError(f"invalid tenant_id {tid!r}")
    return tid


def _resolve_agent_id() -> str:
    try:
        from agentino.core.context import get_context
    except ImportError:
        raise RuntimeError("agentino.core.context unavailable — cannot resolve agent_id")
    aid = get_context("agent_id")
    if not aid:
        raise RuntimeError("agent_id context missing")
    aid = str(aid)
    if not _AGENT_ID_RE.match(aid):
        raise RuntimeError(f"invalid agent_id {aid!r}")
    return aid


def _resolve_user_key() -> str:
    try:
        from agentino.core.context import get_context
    except ImportError:
        return "anonymous"
    uk = get_context("sender_id") or get_context("session_id")
    if not uk:
        return "anonymous"
    uk = str(uk)
    if not _USER_KEY_RE.match(uk):
        # Sanitize aggressively rather than fail — sender_id can be any user
        # input. Replace anything funky with `_`.
        uk = re.sub(r"[^A-Za-z0-9_.@:+-]", "_", uk)[:128]
    return uk


def _resolve_memory_root() -> Path:
    """Where this tenant's memories live on disk.

    The path is ALWAYS tenant-scoped — tenant_id is the top segment under the
    chosen base. Two bases are supported:

      - `WORKSPACE_MEMORY_ROOT` env (tests / dev): memory dir is
        `<MEMORY_ROOT>/<tenant_id>/<agent>/<user>/`.
      - Production default: `<WORKSPACE_TENANTS_ROOT>/<tenant_id>/memories/
        <agent>/<user>/`. WORKSPACE_TENANTS_ROOT defaults to `/app/tenants`.
        This matches the SOUL/workspace.yml mount layout used elsewhere.

    `_resolve_tenant_id()` runs FIRST so a malformed tenant_id is rejected
    even when the override is in play — the override never bypasses tenant
    boundary enforcement.
    """
    tenant_id = _resolve_tenant_id()
    override = os.environ.get("WORKSPACE_MEMORY_ROOT")
    if override:
        return Path(override) / tenant_id
    base = Path(os.environ.get("WORKSPACE_TENANTS_ROOT", "/app/tenants"))
    return base / tenant_id / "memories"


def _user_memory_dir(*, agent_id: str | None = None, user_key: str | None = None) -> Path:
    aid = agent_id or _resolve_agent_id()
    uk = user_key or _resolve_user_key()
    d = _resolve_memory_root() / aid / uk
    return d


def _validate_slug(slug: str) -> str:
    if not slug:
        raise ValueError("slug required")
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"invalid slug {slug!r}: lowercase letters/digits/hyphen only, max 64 chars"
        )
    return slug


# ----------------------------------------------------------------------------
# Frontmatter helpers
# ----------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


def _parse_memory_file(text: str) -> tuple[dict, str]:
    """Split a memory file into (frontmatter dict, body string).

    Frontmatter is a small subset of YAML: `key: value` lines, scalar values
    only. This avoids a yaml dependency and keeps memory files trivially
    inspectable.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text, body = m.group(1), m.group(2)
    fm: dict = {}
    for line in fm_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm, body.lstrip("\n")


def _serialize_memory_file(fm: dict, body: str) -> str:
    lines = ["---"]
    for k, v in fm.items():
        if v is None or v == "":
            continue
        # Single-line scalar; agents writing newlines into descriptions get
        # them collapsed (we control description via API).
        lines.append(f"{k}: {str(v).replace(chr(10), ' ').strip()}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), suffix=".tmp", encoding="utf-8"
    )
    try:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------
# CRUD primitives (used by the @tool wrappers in shared_tools/remember.py etc.)
# ----------------------------------------------------------------------------


def list_memories(
    *, agent_id: str | None = None, user_key: str | None = None, limit: int = MAX_MEMORIES_PER_LOAD
) -> list[dict]:
    """Return active memories sorted newest-first. Each item:
        {slug, description, kind, created, updated, session_id, body}
    Index file `MEMORY.md` is ignored; we read the actual *.md files."""
    d = _user_memory_dir(agent_id=agent_id, user_key=user_key)
    if not d.exists():
        return []
    out = []
    for fp in sorted(d.glob("*.md")):
        if fp.name == "MEMORY.md":
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("failed to read memory %s: %s", fp, e)
            continue
        fm, body = _parse_memory_file(text)
        out.append(
            {
                "slug": fp.stem,
                "description": fm.get("description", ""),
                "kind": fm.get("kind", ""),
                "created": fm.get("created", ""),
                "updated": fm.get("updated", ""),
                "session_id": fm.get("session_id", ""),
                "body": body,
            }
        )
    # Sort by created desc (newest first) — frontmatter is ISO-formatted so lex sort works.
    out.sort(key=lambda m: m.get("created", ""), reverse=True)
    return out[:limit]


def read_memory(
    slug: str, *, agent_id: str | None = None, user_key: str | None = None
) -> dict | None:
    slug = _validate_slug(slug)
    fp = _user_memory_dir(agent_id=agent_id, user_key=user_key) / f"{slug}.md"
    if not fp.exists():
        return None
    text = fp.read_text(encoding="utf-8")
    fm, body = _parse_memory_file(text)
    return {
        "slug": slug,
        "description": fm.get("description", ""),
        "kind": fm.get("kind", ""),
        "created": fm.get("created", ""),
        "updated": fm.get("updated", ""),
        "session_id": fm.get("session_id", ""),
        "body": body,
    }


def write_memory(
    *,
    slug: str,
    description: str,
    body: str = "",
    kind: str | None = None,
    agent_id: str | None = None,
    user_key: str | None = None,
) -> dict:
    """Create-or-replace a memory file. If the slug already exists, body +
    description + kind are overwritten; `created` is preserved, `updated` set."""
    slug = _validate_slug(slug)
    if not description.strip():
        raise ValueError("description required")
    if len(description) > MAX_DESCRIPTION_LEN:
        raise ValueError(f"description too long (max {MAX_DESCRIPTION_LEN})")
    if len(body) > MAX_BODY_LEN:
        raise ValueError(f"body too long (max {MAX_BODY_LEN})")

    aid = agent_id or _resolve_agent_id()
    uk = user_key or _resolve_user_key()
    fp = _user_memory_dir(agent_id=aid, user_key=uk) / f"{slug}.md"

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = read_memory(slug, agent_id=aid, user_key=uk)
    created = existing["created"] if (existing and existing["created"]) else now

    try:
        from agentino.core.context import get_context

        session_id = get_context("sender_id") or get_context("session_id") or ""
    except ImportError:
        session_id = ""

    fm = {
        "name": slug,
        "description": description.strip(),
        "kind": (kind or "").strip(),
        "session_id": session_id,
        "created": created,
        "updated": now,
    }
    _atomic_write(fp, _serialize_memory_file(fm, body))
    _refresh_index(_user_memory_dir(agent_id=aid, user_key=uk))
    return {"slug": slug, "path": str(fp), "created": created, "updated": now}


def update_memory(
    slug: str,
    *,
    description: str | None = None,
    body: str | None = None,
    kind: str | None = None,
    agent_id: str | None = None,
    user_key: str | None = None,
) -> dict:
    """Patch an existing memory in place. Fields you don't pass are preserved."""
    slug = _validate_slug(slug)
    existing = read_memory(slug, agent_id=agent_id, user_key=user_key)
    if not existing:
        raise FileNotFoundError(f"memory {slug!r} does not exist — use remember() to create")
    return write_memory(
        slug=slug,
        description=(description if description is not None else existing["description"]),
        body=(body if body is not None else existing["body"]),
        kind=(kind if kind is not None else existing["kind"]),
        agent_id=agent_id,
        user_key=user_key,
    )


def forget_memory(slug: str, *, agent_id: str | None = None, user_key: str | None = None) -> bool:
    """Soft-delete: rename `<slug>.md` → `archive/<slug>.<timestamp>.md`."""
    slug = _validate_slug(slug)
    d = _user_memory_dir(agent_id=agent_id, user_key=user_key)
    fp = d / f"{slug}.md"
    if not fp.exists():
        return False
    archive = d / "archive"
    archive.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive / f"{slug}.{ts}.md"
    fp.rename(target)
    _refresh_index(d)
    return True


def _refresh_index(memory_dir: Path) -> None:
    """Rewrite MEMORY.md as a one-line index. Useful for human inspection +
    grep. Not authoritative — list_memories() reads the actual files."""
    if not memory_dir.exists():
        return
    lines = ["# Memory Index\n"]
    for fp in sorted(memory_dir.glob("*.md")):
        if fp.name == "MEMORY.md":
            continue
        try:
            fm, _ = _parse_memory_file(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        prefix = f"[{fm.get('kind')}] " if fm.get("kind") else ""
        lines.append(f"- {prefix}**{fp.stem}** — {fm.get('description', '')}")
    try:
        _atomic_write(memory_dir / "MEMORY.md", "\n".join(lines) + "\n")
    except Exception as e:
        log.warning("MEMORY.md refresh failed in %s: %s", memory_dir, e)


# ----------------------------------------------------------------------------
# Read-side helper used by AppRegistry to inject into the system prompt
# ----------------------------------------------------------------------------


def format_memories_for_prompt(memories: list[dict]) -> str:
    """One-line-per-memory block for system-prompt injection. Empty list →
    empty string so the prompt stays clean for new users."""
    if not memories:
        return ""
    lines = [
        "## Things you remember about this user (from past sessions)",
        "",
    ]
    for m in memories:
        prefix = f"[{m['kind']}] " if m.get("kind") else ""
        lines.append(f"- **{m['slug']}** — {prefix}{m['description']}")
    lines.append("")
    lines.append(
        "Use the `read_memory(slug)` tool to fetch full context for any of "
        "the above. Use `remember(...)`, `update_memory(...)`, `forget(slug)` "
        "to maintain this list as the conversation evolves."
    )
    return "\n".join(lines)


def load_and_format_for_session(*, agent_id: str | None = None, user_key: str | None = None) -> str:
    """Convenience: list + format in one call. Used by AppRegistry."""
    try:
        return format_memories_for_prompt(list_memories(agent_id=agent_id, user_key=user_key))
    except Exception as e:
        log.warning("memory load for session failed: %s", e)
        return ""
