"""Tests for config loader — YAML/TOML agent definitions."""

import os
import tempfile
from unittest.mock import patch

from agentino import Agent, load_agents, load_config
from agentino.config import Config, _build_condition, _resolve_env_vars
from agentino.core.tool import tool
from agentino.pipeline import Pipeline, RouterPipeline


@tool
def mock_search(query: str) -> str:
    """Search for things."""
    return f"Results for: {query}"


@tool
def mock_book(date: str, party_size: int) -> str:
    """Book a table."""
    return f"Booked for {party_size} on {date}"


# ---------------------------------------------------------------------------
# Environment variable resolution
# ---------------------------------------------------------------------------


class TestEnvVarResolution:
    def test_resolves_env_var(self):
        with patch.dict(os.environ, {"MY_KEY": "secret123"}):
            assert _resolve_env_vars("key=${MY_KEY}") == "key=secret123"

    def test_unset_var_unchanged(self):
        result = _resolve_env_vars("${NONEXISTENT_VAR_XYZ}")
        assert result == "${NONEXISTENT_VAR_XYZ}"

    def test_multiple_vars(self):
        with patch.dict(os.environ, {"A": "1", "B": "2"}):
            assert _resolve_env_vars("${A}-${B}") == "1-2"

    def test_no_vars(self):
        assert _resolve_env_vars("plain text") == "plain text"


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

_SIMPLE_YAML = """\
defaults:
  model: gpt-4o
  temperature: 0.5

agents:
  maria:
    instructions: "You are Maria, a booking assistant."
    tools:
      - mock_search
      - mock_book
    max_turns: 10

  router:
    model: gpt-4o-mini
    instructions: "Classify intent: booking or general. ONE word."
    max_turns: 1
    temperature: 0
"""

_PIPELINE_YAML = """\
agents:
  checker:
    model: gpt-4o-mini
    instructions: "Check CI status."
  reporter:
    model: gpt-4o
    instructions: "Write a report."
  notifier:
    model: gpt-4o-mini
    instructions: "Send notification."

pipeline:
  type: sequence
  steps:
    - agent: checker
      message: "Check all repos"
    - agent: reporter
      message: "Report on {previous}"
    - agent: notifier
      message: "Notify: {previous}"
      condition: "failure"
"""

_ROUTER_YAML = """\
agents:
  router:
    model: gpt-4o-mini
    instructions: "Classify: booking or wine. ONE word."
    max_turns: 1

  booking:
    instructions: "You handle bookings."

  wine:
    instructions: "You are a sommelier."

pipeline:
  type: router
  router: router
  routes:
    booking: booking
    wine: wine
  default: booking
"""


class TestLoadAgents:
    def test_loads_agents_from_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(_SIMPLE_YAML)
            f.flush()
            try:
                agents = load_agents(f.name, tools=[mock_search, mock_book])
                assert "maria" in agents
                assert "router" in agents
                assert isinstance(agents["maria"], Agent)
                assert agents["maria"].model == "gpt-4o"
                assert agents["maria"].temperature == 0.5
                assert agents["maria"].max_turns == 10
                assert len(agents["maria"].tools) == 2
                assert agents["router"].model == "gpt-4o-mini"
                assert agents["router"].max_turns == 1
            finally:
                os.unlink(f.name)

    def test_missing_tool_skipped(self):
        yaml = """\
agents:
  bot:
    instructions: "Hello"
    tools:
      - nonexistent_tool
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                agents = load_agents(f.name, tools=[mock_search])
                assert len(agents["bot"].tools) == 0  # nonexistent tool skipped
            finally:
                os.unlink(f.name)

    def test_defaults_applied(self):
        yaml = """\
defaults:
  model: custom-model
  temperature: 0.3
  base_url: http://localhost:8000/v1

agents:
  bot:
    instructions: "Hello"
"""
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                agents = load_agents(f.name)
                assert agents["bot"].model == "custom-model"
                assert agents["bot"].temperature == 0.3
            finally:
                os.unlink(f.name)


class TestLoadConfig:
    def test_sequence_pipeline(self):
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(_PIPELINE_YAML)
            f.flush()
            try:
                config = load_config(f.name)
                assert isinstance(config, Config)
                assert isinstance(config.pipeline, Pipeline)
                assert len(config.pipeline.steps) == 3
            finally:
                os.unlink(f.name)

    def test_router_pipeline(self):
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(_ROUTER_YAML)
            f.flush()
            try:
                config = load_config(f.name)
                assert isinstance(config.pipeline, RouterPipeline)
                assert "booking" in config.pipeline.routes
                assert "wine" in config.pipeline.routes
                assert config.pipeline.default == "booking"
            finally:
                os.unlink(f.name)

    def test_unsupported_format(self):
        try:
            load_config("test.json")
            assert False, "Should have raised"
        except ValueError as e:
            assert ".json" in str(e)

    def test_toml_rejected(self):
        try:
            load_config("test.toml")
            assert False, "Should have raised"
        except ValueError as e:
            assert ".toml" in str(e)


# ---------------------------------------------------------------------------
# Workspace features
# ---------------------------------------------------------------------------


class TestWorkspace:
    def test_workspace_bootstrap_files_injected(self, tmp_path):
        """Workspace SOUL.md and RULES.md are appended to agent instructions."""
        # Shared skill
        skill_dir = tmp_path / "skills" / "coding"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("You are a coding agent.\n")

        # Workspace = agents.yml dir
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "agents.yml").write_text(f"""
agents:
  coder:
    skills: [coding]
    skills_dir: {skill_dir.parent}
""")
        (ws / "SOUL.md").write_text("Always check migrations before DB changes.\n")
        (ws / "RULES.md").write_text("Never commit secrets.\n")

        agents = load_agents(str(ws / "agents.yml"))
        instructions = agents["coder"].instructions
        assert "coding agent" in instructions
        assert "migrations" in instructions
        assert "Never commit secrets" in instructions

    def test_workspace_skill_override(self, tmp_path):
        """Workspace skills/ overrides shared skills by name."""
        # Shared skill
        shared = tmp_path / "shared" / "coding"
        shared.mkdir(parents=True)
        (shared / "SKILL.md").write_text("Shared instructions.\n")

        # Workspace with override
        ws = tmp_path / "workspace"
        ws.mkdir()
        ws_skill = ws / "skills" / "coding"
        ws_skill.mkdir(parents=True)
        (ws_skill / "SKILL.md").write_text("Workspace override instructions.\n")

        (ws / "agents.yml").write_text(f"""
agents:
  coder:
    skills: [coding]
    skills_dir: {shared.parent}
""")

        agents = load_agents(str(ws / "agents.yml"))
        assert "Workspace override" in agents["coder"].instructions
        assert "Shared instructions" not in agents["coder"].instructions

    def test_workspace_tool_override(self, tmp_path):
        """Workspace tools override shared tools by name, shared tools still available."""
        # Shared skill with two tools
        shared = tmp_path / "shared" / "myskill"
        (shared / "tools").mkdir(parents=True)
        (shared / "SKILL.md").write_text("Instructions.\n")
        (shared / "tools" / "tools.py").write_text("""
from agentino.core.tool import tool
@tool
def tool_x(a: str) -> str:
    \"\"\"Shared X.\"\"\"
    return a
@tool
def tool_y(a: str) -> str:
    \"\"\"Shared Y.\"\"\"
    return a
""")

        # Workspace overrides tool_x
        ws = tmp_path / "workspace"
        ws.mkdir()
        ws_skill = ws / "skills" / "myskill"
        (ws_skill / "tools").mkdir(parents=True)
        (ws_skill / "tools" / "tools.py").write_text("""
from agentino.core.tool import tool
@tool
def tool_x(a: str) -> str:
    \"\"\"Workspace X.\"\"\"
    return a
""")

        (ws / "agents.yml").write_text(f"""
agents:
  coder:
    skills: [myskill]
    skills_dir: {shared.parent}
    tools: [tool_x, tool_y]
""")

        agents = load_agents(str(ws / "agents.yml"))
        tool_map = {t.name: t for t in agents["coder"].tools}
        assert tool_map["tool_x"].description == "Workspace X."
        assert tool_map["tool_y"].description == "Shared Y."

    def test_no_workspace_files_is_fine(self, tmp_path):
        """Agent works without any workspace bootstrap files."""
        skill_dir = tmp_path / "skills" / "coding"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("Base instructions.\n")

        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "agents.yml").write_text(f"""
agents:
  coder:
    skills: [coding]
    skills_dir: {skill_dir.parent}
""")
        # No SOUL.md, RULES.md, etc. in workspace

        agents = load_agents(str(ws / "agents.yml"))
        assert "Base instructions" in agents["coder"].instructions


# ---------------------------------------------------------------------------
# Condition builder
# ---------------------------------------------------------------------------


class TestBuildCondition:
    def test_always(self):
        cond = _build_condition("always")
        assert cond({}) is True

    def test_contains_pattern(self):
        cond = _build_condition("{checker} contains 'failure'")
        assert cond({"checker": "Found a failure in CI"}) is True
        assert cond({"checker": "All good"}) is False

    def test_keyword_match(self):
        cond = _build_condition("failure")
        assert cond({"step1": "CI failure detected"}) is True
        assert cond({"step1": "All passing"}) is False


class TestDiscoverToolsFromDirAcceptsAString:
    """`discover_tools_from_dir` is a public export, so it has to take what a
    caller outside the package would naturally hand it. Every internal caller
    passes a Path, which is how a string got to fail on `.is_dir()`."""

    @staticmethod
    def _write_tool(tmp_path):
        d = tmp_path / "tools"
        d.mkdir()
        (d / "ping.py").write_text(
            "from agentino import tool\n\n\n"
            "@tool\n"
            "async def ping() -> str:\n"
            '    """Return a fixed string."""\n'
            '    return "pong"\n'
        )
        return d

    def test_a_string_path_discovers_the_same_tools_as_a_path(self, tmp_path):
        from agentino.config import discover_tools_from_dir

        d = self._write_tool(tmp_path)
        from_path = discover_tools_from_dir(d)
        from_str = discover_tools_from_dir(str(d))
        assert [t.name for t in from_path] == ["ping"]
        assert [t.name for t in from_str] == ["ping"]

    def test_a_missing_directory_is_empty_rather_than_an_error(self, tmp_path):
        from agentino.config import discover_tools_from_dir

        assert discover_tools_from_dir(str(tmp_path / "nope")) == []


class TestCredentialResolutionMatchesWhatIsDocumented:
    """The documented resolution order has to be the real one.

    The LLMClient docstring listed `OPENAI_BASE_URL` as a legacy fallback for
    the endpoint. It is not read at all, so anyone setting it got the default
    endpoint and no indication why. These pin what actually happens.
    """

    @staticmethod
    def _clear(monkeypatch):
        for var in (
            "AGENTINO_BASE_URL",
            "AGENTINO_API_KEY",
            "AGENTINO_PROVIDER",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_SETUP_TOKEN",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_agentino_base_url_selects_the_endpoint(self, monkeypatch):
        from agentino.core.llm import LLMClient

        self._clear(monkeypatch)
        monkeypatch.setenv("AGENTINO_BASE_URL", "http://chosen.test/v1")
        monkeypatch.setenv("AGENTINO_API_KEY", "sk-test")
        assert LLMClient().base_url == "http://chosen.test/v1"

    def test_openai_base_url_is_not_read(self, monkeypatch):
        """Pinned deliberately. If this ever starts working, the docstring and
        the environment reference both have to change with it."""
        from agentino.core.llm import LLMClient

        self._clear(monkeypatch)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://ignored.test/v1")
        monkeypatch.setenv("AGENTINO_API_KEY", "sk-test")
        assert LLMClient().base_url != "http://ignored.test/v1"
        assert LLMClient().base_url == "https://api.openai.com/v1"

    def test_a_constructor_argument_beats_the_environment(self, monkeypatch):
        from agentino.core.llm import LLMClient

        self._clear(monkeypatch)
        monkeypatch.setenv("AGENTINO_BASE_URL", "http://from-env.test/v1")
        monkeypatch.setenv("AGENTINO_API_KEY", "sk-test")
        assert LLMClient(base_url="http://from-arg.test/v1").base_url == "http://from-arg.test/v1"

    def test_a_trailing_slash_does_not_produce_a_double_slash(self, monkeypatch):
        from agentino.core.llm import LLMClient

        self._clear(monkeypatch)
        monkeypatch.setenv("AGENTINO_BASE_URL", "http://chosen.test/v1/")
        monkeypatch.setenv("AGENTINO_API_KEY", "sk-test")
        assert LLMClient().base_url == "http://chosen.test/v1"

    def test_the_runspace_variable_names_do_not_configure_agentino(self, monkeypatch):
        """They are separate packages with separate names. Someone setting only
        the runspace pair and calling agentino directly gets the default
        endpoint, and the install page says so."""
        from agentino.core.llm import LLMClient

        self._clear(monkeypatch)
        monkeypatch.setenv("AI_BASE_URL", "http://runspace-name.test/v1")
        monkeypatch.setenv("AGENTINO_API_KEY", "sk-test")
        assert LLMClient().base_url != "http://runspace-name.test/v1"


class TestTheReadmeLayoutMatchesTheTree:
    """The README's layout block is the map a reader navigates by.

    It omitted `tools/` entirely — the built-in tools package, which is most
    of what "batteries included" refers to — because nothing checks a prose
    tree against the directory it describes.
    """

    @staticmethod
    def _claimed_and_actual():
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        text = (root / "README.md").read_text(encoding="utf-8")
        m = re.search(r"```\nsrc/agentino/\n(.*?)```", text, re.S)
        assert m, "the README no longer has a src/agentino/ layout block"
        claimed = {
            mm.group(1)
            for line in m.group(1).splitlines()
            if (mm := re.match(r"[├└│─\s]+([a-z_]+)/", line))
        }
        actual = {
            p.name
            for p in (root / "src" / "agentino").iterdir()
            if p.is_dir() and not p.name.startswith(("_", "."))
        }
        return claimed, actual

    def test_every_package_on_disk_is_in_the_readme(self):
        claimed, actual = self._claimed_and_actual()
        missing = sorted(actual - claimed)
        assert not missing, f"packages shipped but absent from the README layout: {missing}"

    def test_the_readme_names_no_package_that_does_not_exist(self):
        claimed, actual = self._claimed_and_actual()
        phantom = sorted(claimed - actual)
        assert not phantom, f"README describes packages that are not there: {phantom}"
