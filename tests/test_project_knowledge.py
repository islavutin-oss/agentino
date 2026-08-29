"""Reading a project's `.agentino/` knowledge folder.

The writing half of these tests targeted tools that lived in a coding-agent
example, and went with it — they skipped permanently, which is worse than
absent: twelve tests that can never run train everyone to ignore the skip
count, and a real skip then hides in the noise.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from agentino.config import _load_project_knowledge
from agentino.extras.knowledge import KnowledgeBase

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _init_git_repo(path: Path) -> str:
    """Initialize a git repo at path and return the HEAD commit hash."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True)
    (path / "dummy.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, capture_output=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# -------------------------------------------------------------------
# Knowledge tools — save, delete, update_docs_meta
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Project knowledge loading — config.py integration
# -------------------------------------------------------------------


class TestLoadProjectKnowledge:
    def test_no_bridge_dir_returns_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGENTINO_PROJECT_DIR", None)
            name, kb = _load_project_knowledge("coder", None, None, None)
            assert name == "coder"
            assert kb is None

    def test_bootstrap_status_on_empty_docs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(tmp_path)}):
            name, kb = _load_project_knowledge("coder", None, None, None)
            assert name == f"coder_{tmp_path.name}"
            assert os.environ["AGENTINO_PROJECT_STATUS"] == "bootstrap"
            assert kb is not None
            assert len(kb.entries) == 0

    def test_current_status_when_commit_matches(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Init git repo
        head = _init_git_repo(tmp_path)

        # Create .agentino/ with meta pointing to HEAD
        docs_dir = tmp_path / ".agentino"
        docs_dir.mkdir()
        import yaml

        (docs_dir / "_meta.yml").write_text(yaml.dump({"last_commit_hash": head}))
        (docs_dir / "architecture.md").write_text("## service_layout.en\nSome layout info.\n")

        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(tmp_path)}):
            name, kb = _load_project_knowledge("coder", None, None, None)
            assert os.environ["AGENTINO_PROJECT_STATUS"] == "current"
            assert len(kb.entries) == 1
            assert kb.entries[0].topic == "service_layout"

    def test_refresh_status_when_commit_differs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Init git repo
        _init_git_repo(tmp_path)

        # Create .agentino/ with old commit hash
        docs_dir = tmp_path / ".agentino"
        docs_dir.mkdir()
        import yaml

        (docs_dir / "_meta.yml").write_text(yaml.dump({"last_commit_hash": "old_hash_abc"}))
        (docs_dir / "architecture.md").write_text("## layout.en\nOld layout.\n")

        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(tmp_path)}):
            name, kb = _load_project_knowledge("coder", None, None, None)
            assert os.environ["AGENTINO_PROJECT_STATUS"] == "refresh"
            # Should still have indexed existing entries
            assert len(kb.entries) == 1

    def test_refresh_sets_diff_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Init git repo with a commit
        head1 = _init_git_repo(tmp_path)

        # Make another commit
        (tmp_path / "new_file.py").write_text("print('hello')")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add file"], cwd=tmp_path, capture_output=True)

        # Set up .agentino/ pointing to first commit
        docs_dir = tmp_path / ".agentino"
        docs_dir.mkdir()
        import yaml

        (docs_dir / "_meta.yml").write_text(yaml.dump({"last_commit_hash": head1}))
        (docs_dir / "arch.md").write_text("## test.en\nTest.\n")

        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(tmp_path)}):
            _load_project_knowledge("coder", None, None, None)
            assert os.environ["AGENTINO_PROJECT_STATUS"] == "refresh"
            diff = os.environ.get("AGENTINO_PROJECT_DIFF", "")
            assert "new_file.py" in diff

    def test_qualified_name_uses_slug(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(project_dir)}):
            name, kb = _load_project_knowledge("coder", None, None, None)
            assert name == "coder_my_project"

    def test_indexes_multiple_docs_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        # Init git repo so HEAD matches meta
        head = _init_git_repo(tmp_path)

        docs_dir = tmp_path / ".agentino"
        docs_dir.mkdir()
        import yaml

        (docs_dir / "_meta.yml").write_text(yaml.dump({"last_commit_hash": head}))
        (docs_dir / "architecture.md").write_text(
            "## layout.en\nService layout.\n\n## data_flow.en\nData flow info.\n"
        )
        (docs_dir / "commands.md").write_text("## build.en\nRun make build.\n")
        (docs_dir / "learnings.md").write_text("## redis_gotcha.en\nUnset REDIS_URL for tests.\n")

        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(tmp_path)}):
            name, kb = _load_project_knowledge("coder", None, None, None)
            assert len(kb.entries) == 4
            topics = {e.topic for e in kb.entries}
            assert topics == {"layout", "data_flow", "build", "redis_gotcha"}

    def test_meta_yml_not_indexed_as_knowledge(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        head = _init_git_repo(tmp_path)

        docs_dir = tmp_path / ".agentino"
        docs_dir.mkdir()
        import yaml

        (docs_dir / "_meta.yml").write_text(yaml.dump({"last_commit_hash": head}))
        (docs_dir / "facts.md").write_text("## test.en\nA fact.\n")

        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(tmp_path)}):
            _, kb = _load_project_knowledge("coder", None, None, None)
            # _meta.yml is YAML, not markdown — should not be indexed
            topics = {e.topic for e in kb.entries}
            assert "last_commit_hash" not in topics
            assert len(kb.entries) == 1

    def test_reuses_existing_kb_if_same_agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        slug = tmp_path.name
        existing_kb = KnowledgeBase(agent_name=f"coder_{slug}")

        with patch.dict(os.environ, {"AGENTINO_PROJECT_DIR": str(tmp_path)}):
            _, kb = _load_project_knowledge("coder", None, None, existing_kb)
            # Should reuse the same KB instance
            assert kb is existing_kb


# -------------------------------------------------------------------
# End-to-end: save → load → search
# -------------------------------------------------------------------
