"""Documentation claims checked against the package.

Three kinds of drift, all silent, all found in this repository:

- an import that no longer resolves (`from agentino.message import Message`)
- a method that never existed (`Runner.run_sync`, promised in the README)
- a file named in prose after it was renamed
  (`tests/test_scheduler_protocol.py`)

None of them fails anything until a reader copies it.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PROSE_PREFIXES = ("src/", "docs/", "examples/", "tests/", "scripts/", ".github/")


def _markdown() -> list[Path]:
    return sorted(p for p in _ROOT.rglob("*.md") if ".git" not in p.parts)


def _import_claims() -> list[tuple[str, str, str]]:
    out = []
    for f in _markdown():
        for m in re.finditer(
            r"^from (agentino[\w.]*) import ([\w, ]+)", f.read_text(errors="replace"), re.M
        ):
            for name in (n.split("#")[0].strip() for n in m.group(2).split(",")):
                if name:
                    out.append((str(f.relative_to(_ROOT)), m.group(1), name))
    return sorted(set(out))


def _prose_paths() -> list[tuple[str, str]]:
    out = []
    for f in _markdown():
        for m in re.finditer(
            r"`([A-Za-z0-9_./-]+\.(?:py|md|ts|tsx|yml|yaml|toml|mjs))`",
            f.read_text(errors="replace"),
        ):
            if m.group(1).startswith(_PROSE_PREFIXES):
                out.append((str(f.relative_to(_ROOT)), m.group(1)))
    return sorted(set(out))


def test_the_scan_found_something():
    """Guards the parametrised tests from passing because a glob broke."""
    assert _markdown(), "no markdown found"
    assert _import_claims(), "no agentino imports parsed"
    assert _prose_paths(), "no first-party file paths parsed"


@pytest.mark.parametrize("page,module,name", _import_claims(), ids=lambda v: str(v)[:44])
def test_a_documented_import_resolves(page, module, name):
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        pytest.skip(f"{module} needs an extra not installed here: {e}")
    assert hasattr(mod, name), f"{page} shows `from {module} import {name}`, which does not exist"


@pytest.mark.parametrize("page,target", _prose_paths(), ids=lambda v: str(v)[:44])
def test_a_file_named_in_prose_exists(page, target):
    assert (_ROOT / target).exists(), f"{page} names {target}, which is not in the repository"


# Names that were promised and never existed, or existed and were removed.
# Listed rather than inferred, because a claim about something absent cannot
# be checked by resolving it — it simply finds nothing and reports success.
_NEVER_EXISTED = {
    "Runner.run_sync": "there is no sync wrapper; use asyncio.run(agent.run(...))",
    "Config.from_file": "the loader is load_config()",
    "agentino.message": "the module is agentino.core.message, re-exported from agentino",
    "agentino.tool": "the decorator is imported from agentino directly",
}


@pytest.mark.parametrize("phantom,correction", sorted(_NEVER_EXISTED.items()))
def test_no_document_promises_something_that_does_not_exist(phantom, correction):
    # Bounded, or `agentino.tool` matches inside `agentino.tools.std`, which is
    # a real module — a false positive that would make the check untrustworthy
    # and then ignored.
    pattern = re.compile(re.escape(phantom) + r"(?![\w])")
    hits = [
        str(f.relative_to(_ROOT))
        for f in _markdown()
        # A changelog's job is to record what changed, which means naming
        # things that no longer exist. Every other document naming one is
        # telling a reader to use it.
        if f.name != "CHANGELOG.md" and pattern.search(f.read_text(errors="replace"))
    ]
    assert not hits, f"{phantom} appears in {', '.join(hits)} — {correction}"


def test_the_package_is_marked_as_typed():
    """PEP 561: a type checker ignores every annotation in an installed
    package that has no `py.typed` marker. Seventy annotated modules here
    would do nothing for a consumer, silently."""
    try:
        import tomllib
    except ModuleNotFoundError:  # tomllib is 3.11+; the project supports 3.10
        import tomli as tomllib

    assert (_ROOT / "src" / "agentino" / "py.typed").exists(), "py.typed marker missing"
    meta = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "Typing :: Typed" in meta["project"]["classifiers"]


def test_the_declared_maturity_matches_what_the_readme_promises():
    """A README promising a stable public API beside a `3 - Alpha` classifier
    tells a reader two different things about whether to depend on this."""
    import re

    try:
        import tomllib
    except ModuleNotFoundError:  # tomllib is 3.11+; the project supports 3.10
        import tomli as tomllib

    meta = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    status = next(c for c in meta["project"]["classifiers"] if c.startswith("Development Status"))
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    claims_stable = bool(re.search(r"public api.*stable", readme, re.I))
    if claims_stable:
        assert "Alpha" not in status, (
            f"README says the public API is stable, but the classifier says {status!r}"
        )


def test_the_security_policy_does_not_contradict_the_version():
    """The supported-versions section said "Agentino is pre-1.0" at version
    1.1.0. That is the section a reader consults to learn which releases get
    patched, so being wrong about which era the project is in is not cosmetic.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # tomllib is 3.11+; the project supports 3.10
        import tomli as tomllib

    version = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    policy = (_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    if not version.startswith("0."):
        assert "pre-1.0" not in policy, f"SECURITY.md says pre-1.0 but the package is {version}"


def test_contributing_and_ci_run_the_same_commands():
    """CONTRIBUTING says "CI runs exactly these". If it stops being true, a
    contributor gets a green local run and a red pull request."""
    contributing = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for cmd in ("ruff check .", "ruff format --check .", "pytest tests/ -q"):
        assert cmd in contributing, f"CONTRIBUTING no longer shows `{cmd}`"
        assert cmd in ci, f"CI no longer runs `{cmd}`, but CONTRIBUTING promises it does"
