"""File-storage façade for generated artifacts (PDF, XLSX, DOCX, …).

Thin wrapper over `protocols.FileStorage` (the canonical adapter
protocol — see ADR-0001). Callers (create_pdf, create_csv, etc.) get
a stable `StoredFile` shape regardless of which backend FileStorage is
wired to (Supabase Storage in production, LocalFileStorage in dev).

History: this module used to host its own `LocalFileStore` +
`SupabaseFileStore` classes that wrote directly to disk or Supabase.
Replaced 2026-05-02 — duplicate of protocols.FileStorage. The
backwards-compatible `get_default_store()` API stays so the 5 file-
generation tools (create_pdf, create_csv, etc.) need no changes.

Tenant scoping: `save()` reads `tenant_id` from agentino.core.context. If
context is missing (sandbox, tests), uses `default` as a fallback —
storage segregates by tenant_id, so no leak risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class StoredFile:
    """Stable callsite-facing result. Same shape as before the refactor."""

    url: str  # ready to embed in chat as `[label](url)`
    file_id: str  # storage-layer file id
    size_bytes: int
    mime: str
    filename: str

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "file_id": self.file_id,
            "size_bytes": self.size_bytes,
            "mime": self.mime,
            "filename": self.filename,
        }


def _resolve_tenant_id() -> str:
    """Pull tenant_id from agentino.core.context. Sandbox/tests get `default`."""
    try:
        from agentino.core.context import get_context

        return get_context("tenant_id") or "default"
    except Exception:
        return "default"


class _FileStorageStore:
    """Adapter — all `save()` calls go through `protocols.FileStorage`.

    URL semantics:
      - LocalFileStorage  → `/api/workspace/files/<file_id>` (gateway-served)
      - SupabaseFileStorage → signed URL with default 1-hour TTL
    """

    URL_PREFIX = "/api/workspace/files"
    SIGNED_URL_TTL_SECONDS = 3600

    def save(
        self,
        *,
        content_bytes: bytes,
        filename: str,
        mime: str,
        title: str | None = None,
        description: str | None = None,
        extracted_text: str | None = None,
    ) -> StoredFile:
        # `title`, `description`, `extracted_text` are legacy args from the
        # old SupabaseFileStore impl. Kept for Protocol parity; ignored —
        # FileStorage doesn't surface them. Add a separate `documents`-table
        # adapter if those become load-bearing.
        del title, description, extracted_text

        if not content_bytes:
            raise ValueError("content_bytes must be non-empty")

        from agentino.tools.std._file_storage import get_file_storage

        storage = get_file_storage()
        tenant_id = _resolve_tenant_id()

        meta = storage.put(
            tenant_id,
            filename,
            content_bytes,
            content_type=mime,
        )

        # Build the URL. For local backend we want the gateway-served path
        # so the chat UI's existing /api/workspace/files/{id} handler picks
        # it up. For Supabase, ask the storage for a signed URL.
        impl = type(storage).__name__
        if impl == "SupabaseFileStorage":
            try:
                url = storage.signed_url(
                    tenant_id,
                    meta.file_id,
                    ttl_seconds=self.SIGNED_URL_TTL_SECONDS,
                )
            except Exception as e:
                log.warning("signed_url failed (%s); falling back to gateway path", e)
                url = f"{self.URL_PREFIX}/{meta.file_id}"
        else:
            url = f"{self.URL_PREFIX}/{meta.file_id}"

        return StoredFile(
            url=url,
            file_id=meta.file_id,
            size_bytes=meta.size_bytes,
            mime=meta.content_type or mime,
            filename=meta.original_name,
        )


# ---------------------------------------------------------------------------
# Factory — single cached instance per process
# ---------------------------------------------------------------------------

_default: _FileStorageStore | None = None


def get_default_store() -> _FileStorageStore:
    """Return the file-storage façade. Cached process-wide."""
    global _default
    if _default is None:
        _default = _FileStorageStore()
        log.info("[agentino.tools.std.storage] using protocols.FileStorage backend")
    return _default


def _reset_for_tests() -> None:
    """Tests that swap STORAGE_BACKEND mid-process call this to bust the cache.
    Real callers shouldn't need it."""
    global _default
    _default = None
