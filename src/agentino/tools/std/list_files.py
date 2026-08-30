"""List files in the tenant's workspace.

Reads through `protocols.FileStorage` — Supabase Storage in
production, LocalFileStorage in dev. Tenant-scoped via
agentino.context.

History: previously read a flat workspace-files directory directly,
which broke tenant isolation and required a hardcoded local-only
path. Migrated 2026-05-02.
"""

from __future__ import annotations

from pathlib import Path

from agentino.core.tool import tool


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


@tool(is_read_only=True)
async def list_files() -> str:
    """List all files currently in the workspace (uploaded or generated).
    Shows filename, size, and type. Tenant-scoped automatically."""
    from agentino.core.context import get_context
    from agentino.tools.std._file_storage import get_file_storage

    tenant_id = get_context("tenant_id") or "default"
    storage = get_file_storage()

    try:
        items = storage.list(tenant_id)
    except Exception as e:
        return f"Could not list files: {e}"

    if not items:
        return "No files in workspace."

    # Sort newest first by created_at (ISO 8601 — string sort works)
    items = sorted(items, key=lambda m: m.created_at or "", reverse=True)

    lines = []
    for meta in items:
        suffix = Path(meta.original_name).suffix.lower() or "unknown"
        lines.append(f"- {meta.original_name} ({suffix}, {_fmt_size(meta.size_bytes)})")
    return f"Workspace files ({len(lines)}):\n" + "\n".join(lines)
