"""The std tools' storage seam: a working default, and a host override."""

from __future__ import annotations

import pytest

from agentino.tools.std import (
    LocalFileStorage,
    get_file_storage,
    set_file_storage_provider,
)
from agentino.tools.std._file_storage import _reset_for_tests


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTINO_FILES_DIR", str(tmp_path))
    _reset_for_tests()
    yield
    set_file_storage_provider(None)
    _reset_for_tests()


def test_default_is_local_and_needs_no_configuration():
    assert isinstance(get_file_storage(), LocalFileStorage)


def test_round_trip(tmp_path):
    s = get_file_storage()
    meta = s.put("acme", "notes.txt", b"hello world", content_type="text/plain")
    assert s.get("acme", meta.file_id) == b"hello world"
    assert meta.size_bytes == 11
    assert meta.original_name == "notes.txt"
    assert meta.tenant_id == "acme"


def test_content_type_is_guessed_from_the_name():
    meta = get_file_storage().put("acme", "report.csv", b"a,b\n1,2\n")
    assert meta.content_type == "text/csv"


def test_metadata_round_trips():
    s = get_file_storage()
    meta = s.put("acme", "notes.txt", b"hello")
    assert s.metadata("acme", meta.file_id) == meta


def test_list_is_scoped_to_the_tenant():
    s = get_file_storage()
    s.put("acme", "a.txt", b"a")
    s.put("globex", "b.txt", b"b")
    assert [m.original_name for m in s.list("acme")] == ["a.txt"]
    assert [m.original_name for m in s.list("globex")] == ["b.txt"]


def test_one_tenant_cannot_read_another_by_file_id():
    s = get_file_storage()
    meta = s.put("acme", "secret.txt", b"classified")
    with pytest.raises(FileNotFoundError):
        s.get("globex", meta.file_id)


@pytest.mark.parametrize("tenant", ["../../etc", "a/../../b", "..", "/absolute"])
def test_a_traversing_tenant_id_cannot_escape_the_root(tmp_path, tenant):
    """A tenant id is attacker-influenced in a multi-tenant host, so it must
    never resolve outside the configured root."""
    s = get_file_storage()
    meta = s.put(tenant, "x.txt", b"x")
    written = list(tmp_path.rglob(meta.file_id))
    assert written, "the file was written outside the configured root"
    for path in written:
        assert tmp_path.resolve() in path.resolve().parents


def test_signed_url_is_tenant_scoped():
    s = get_file_storage()
    meta = s.put("acme", "notes.txt", b"hello")
    assert s.signed_url("acme", meta.file_id) == f"/api/files/acme/{meta.file_id}"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        get_file_storage().get("acme", "nope.txt")


def test_a_host_can_replace_the_storage():
    sentinel = object()
    set_file_storage_provider(lambda: sentinel)
    assert get_file_storage() is sentinel


def test_passing_none_restores_the_default():
    set_file_storage_provider(lambda: object())
    set_file_storage_provider(None)
    assert isinstance(get_file_storage(), LocalFileStorage)
