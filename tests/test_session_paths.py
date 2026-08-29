"""Session and memory identifiers must not choose where the process writes.

`session_id` and `peer_id` arrive from outside — an HTTP request body, a
Telegram chat id, a webhook payload — and were interpolated straight into a
filename. `session_id="../../../tmp/x"` wrote `/tmp/x.jsonl`, outside the
session directory, so whoever could name a session could pick the path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentino.core.session import safe_segment

TRAVERSALS = [
    "../../../tmp/pwned",
    "..",
    "../..",
    "/etc/passwd",
    "a/b/c",
    "..\\..\\windows",
    ".hidden",
    "with\x00null",
]


@pytest.mark.parametrize("evil", TRAVERSALS)
def test_a_hostile_identifier_stays_one_segment(evil):
    out = safe_segment(evil)
    assert "/" not in out and "\\" not in out
    assert "\x00" not in out
    assert out not in (".", "..")
    assert not out.startswith(".")


@pytest.mark.parametrize("evil", TRAVERSALS)
def test_the_resulting_path_stays_inside_the_directory(tmp_path, evil):
    """The property that actually matters, stated against a real path."""
    root = tmp_path / "sessions"
    root.mkdir()
    p = (root / f"{safe_segment(evil)}.jsonl").resolve()
    assert root.resolve() in p.parents, f"{evil} escaped to {p}"


def test_an_ordinary_identifier_is_left_alone():
    """Sanitising must not churn the common case — these are filenames a human
    reads when debugging."""
    for ok in ("default", "user-123", "agent.main", "abc_DEF-9"):
        assert safe_segment(ok) == ok


def test_two_different_identifiers_do_not_collide():
    """The mapping is lossy, so without a disambiguator `a/b` and `a_b` would
    become one file and two callers would share one conversation."""
    assert safe_segment("a/b") != safe_segment("a_b")
    assert safe_segment("../x") != safe_segment("__x")


def test_an_empty_or_fully_stripped_identifier_still_yields_a_name():
    for empty in ("", "...", "///"):
        out = safe_segment(empty)
        assert out and not out.startswith(".") and "/" not in out


def test_the_runner_applies_it(tmp_path):
    """The guard is only worth anything at the call sites."""
    from agentino.core.runner import Runner

    class _R(Runner):
        def __init__(self, d):
            self.session_dir = Path(d)
            self._sessions = {}
            self.no_session = True

    r = _R(tmp_path)
    s = r.get_session("agent", "../../../tmp/pwned")
    assert tmp_path.resolve() in Path(s.path).resolve().parents
