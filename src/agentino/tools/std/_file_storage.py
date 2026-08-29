"""Where the standard tools read and write workspace files.

The tools that produce or consume files need somewhere to put them. Agentino
ships a local-filesystem implementation so a fresh install works with no
configuration; a host application that has its own storage — object storage,
a database, a multi-tenant bucket — registers that instead:

    from agentino.tools.std import set_file_storage_provider

    set_file_storage_provider(my_get_storage)

The provider is a zero-argument callable returning an object with `put`,
`get`, `metadata`, `list` and `signed_url`. Registration replaces the default
for the whole process.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe(part: str, fallback: str) -> str:
    """One path segment, with no way out of the directory it is joined to.

    A tenant id is attacker-influenced in a multi-tenant host. Substituting
    the unsafe characters is not enough on its own: "." and ".." survive that
    pass untouched and still traverse, so anything that is only dots is
    replaced outright.
    """
    cleaned = _UNSAFE.sub("_", (part or "").strip())
    if not cleaned or set(cleaned) <= {"."}:
        cleaned = fallback
    return cleaned[:200]


@dataclass(frozen=True)
class FileMetadata:
    """Provenance and addressability for one stored file."""

    file_id: str
    tenant_id: str
    original_name: str
    size_bytes: int
    content_type: str
    created_at: str
    sha256: str


class LocalFileStorage:
    """Files on disk, one directory per tenant, with a JSON sidecar each.

    Root comes from ``AGENTINO_FILES_DIR``, falling back to
    ``./.agentino/files`` so an unconfigured install still works.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(
            root or os.environ.get("AGENTINO_FILES_DIR") or Path.cwd() / ".agentino" / "files"
        )

    def _tenant_dir(self, tenant_id: str) -> Path:
        d = self.root / _safe(tenant_id, "default")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put(
        self,
        tenant_id: str,
        original_name: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> FileMetadata:
        digest = hashlib.sha256(content).hexdigest()
        suffix = Path(original_name or "").suffix
        file_id = f"{digest[:16]}{suffix}"
        if content_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(original_name or "")
            content_type = guessed or content_type
        meta = FileMetadata(
            file_id=file_id,
            tenant_id=tenant_id,
            original_name=original_name,
            size_bytes=len(content),
            content_type=content_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            sha256=digest,
        )
        d = self._tenant_dir(tenant_id)
        (d / _safe(file_id, "file")).write_bytes(content)
        (d / f"{_safe(file_id, 'file')}.json").write_text(json.dumps(asdict(meta)))
        return meta

    def get(self, tenant_id: str, file_id: str) -> bytes:
        path = self._tenant_dir(tenant_id) / _safe(file_id, "file")
        if not path.is_file():
            raise FileNotFoundError(f"no such file for tenant {tenant_id!r}: {file_id!r}")
        return path.read_bytes()

    def metadata(self, tenant_id: str, file_id: str) -> FileMetadata:
        path = self._tenant_dir(tenant_id) / f"{_safe(file_id, 'file')}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no metadata for tenant {tenant_id!r}: {file_id!r}")
        return FileMetadata(**json.loads(path.read_text()))

    def list(self, tenant_id: str) -> list[FileMetadata]:
        out: list[FileMetadata] = []
        for sidecar in sorted(self._tenant_dir(tenant_id).glob("*.json")):
            try:
                out.append(FileMetadata(**json.loads(sidecar.read_text())))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def signed_url(self, tenant_id: str, file_id: str, ttl_seconds: int = 300) -> str:
        """A tenant-scoped relative path. Local storage has no expiry, so
        `ttl_seconds` is accepted and ignored."""
        del ttl_seconds
        return f"/api/files/{_safe(tenant_id, 'default')}/{_safe(file_id, 'file')}"


_provider: Callable[[], Any] | None = None
_default: Any | None = None


def set_file_storage_provider(factory: Callable[[], Any] | None) -> None:
    """Register the factory the standard tools call to obtain storage.

    Pass None to restore the built-in local-filesystem default.
    """
    global _provider, _default
    _provider = factory
    _default = None


def get_file_storage() -> Any:
    """The storage the standard tools should use."""
    global _default
    if _provider is not None:
        return _provider()
    if _default is None:
        _default = LocalFileStorage()
    return _default


def _reset_for_tests() -> None:
    """Drop the cached default so a test can change the root mid-process."""
    global _default
    _default = None
