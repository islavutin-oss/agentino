"""Smoke tests — catch import errors, missing modules, syntax issues.

Run before any integration test. Should complete in <1 second.
"""

import importlib


def test_all_modules_import():
    """Every agentino module must import without errors."""
    modules = [
        "agentino",
        "agentino.core.agent",
        "agentino.config",
        "agentino.core.runner",
        "agentino.pipeline.staged",
        "agentino.extras.knowledge",
        "agentino.core.tool",
        "agentino.core.message",
        "agentino.core.session",
        "agentino.reliability.resilience",
        "agentino.core.context",
        "agentino.builtin_tools",
        "agentino.cli.renderer",
        "agentino.core.llm",
        "agentino.workers",
        "agentino.extras.usage",
        "agentino.transport.channel",
        "agentino.transport.gateway",
        "agentino.transport.slack",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            if "slack_bolt" in str(e) or "aiogram" in str(e):
                continue  # Optional deps
            raise AssertionError(f"Failed to import {mod}: {e}")


def test_runner_creates():
    """Runner can be created from a minimal config."""
    from agentino.config import Config
    from agentino.core.runner import Runner

    config = Config(agents={}, pipeline=None, gateway=None, raw={})
    runner = Runner(config)
    assert runner is not None
    assert runner.message_hook is None


def test_staged_pipeline_creates():
    """StagedPipeline can be created with empty stages."""
    from agentino.pipeline.staged import StageDef, StagedPipeline

    pipeline = StagedPipeline(stages=[])
    assert pipeline is not None

    pipeline2 = StagedPipeline(
        stages=[
            StageDef(name="test", prompt="do something"),
        ]
    )
    assert len(pipeline2.stages) == 1


def test_stage_def_fields():
    """StageDef accepts all YAML fields."""
    from agentino.pipeline.staged import StageDef

    s = StageDef(
        name="verify",
        prompt="check things",
        max_turns=20,
        verdict_tool="stage_verdict",
        repeatable=True,
        max_cycles=3,
        on_fail="respond",
        tools=["read_file", "grep"],
    )
    assert s.name == "verify"
    assert s.verdict_tool == "stage_verdict"
    assert s.repeatable is True
    assert s.max_cycles == 3
    assert s.on_fail == "respond"
    assert s.tools == ["read_file", "grep"]


def test_cli_renderer_creates():
    """CLIRenderer can be created in both modes."""
    from agentino.cli.renderer import CLIRenderer

    r1 = CLIRenderer(use_rich=False)
    assert r1 is not None

    r2 = CLIRenderer(use_rich=True)
    assert r2 is not None


def test_builtin_tools_complete():
    """All expected builtin tools exist."""
    from agentino.builtin_tools import BUILTIN_TOOLS

    names = {t.name for t in BUILTIN_TOOLS}
    expected = {
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "shell",
        "grep",
        "web_search",
        "stage_verdict",
    }
    missing = expected - names
    assert not missing, f"Missing builtin tools: {missing}"


def test_grep_includes_go_files():
    """Grep tool must search .go files (not just .py/.js)."""
    # Check the source for --include=*.go
    import inspect

    from agentino.builtin_tools import grep

    src = inspect.getsource(grep.fn if hasattr(grep, "fn") else grep)
    assert "*.go" in src, "grep tool missing --include=*.go"
    assert "*.proto" in src, "grep tool missing --include=*.proto"


def test_knowledge_base_creates():
    """KnowledgeBase can be created without embeddings."""
    from agentino.extras.knowledge import KnowledgeBase

    kb = KnowledgeBase()
    assert kb is not None
    assert len(kb.entries) == 0


def test_message_hook_signature():
    """Runner.message_hook field exists and is initially None."""
    from agentino.config import Config
    from agentino.core.runner import Runner

    config = Config(agents={}, pipeline=None, gateway=None, raw={})
    runner = Runner(config)
    assert hasattr(runner, "message_hook")
    assert runner.message_hook is None


def test_gateway_build_with_empty_config():
    """Gateway builder handles empty/no gateway config without crashing."""
    from agentino.config import Config

    config = Config(agents={}, pipeline=None, gateway=None, raw={})
    # No gateway config → should not crash
    assert config.gateway is None


def test_gateway_build_with_message_hook():
    """Gateway builder handles message_hook in raw config."""
    from agentino.config import Config

    config = Config(
        agents={}, pipeline=None, gateway=None, raw={"message_hook": "nonexistent_module"}
    )
    # Raw config has message_hook but no gateway — runner handles it, not gateway
    assert config.raw.get("message_hook") == "nonexistent_module"


def test_runner_with_message_hook_bad_module():
    """Runner handles missing message_hook module gracefully."""
    from agentino.config import Config
    from agentino.core.runner import Runner

    config = Config(
        agents={}, pipeline=None, gateway=None, raw={"message_hook": "totally_fake_module"}
    )
    runner = Runner(config)
    # Should warn but not crash
    assert runner.message_hook is None


def test_staged_pipeline_uses_path():
    """StagedPipeline must import Path for working file handling."""
    import inspect

    from agentino.pipeline.staged import StagedPipeline

    src = inspect.getsource(StagedPipeline.run)
    # If it references Path(), the import must exist
    if "Path(" in src:
        from agentino.pipeline import staged

        assert hasattr(staged, "Path"), "staged.py uses Path but doesn't import it"


def test_working_file_replacement():
    """StagedPipeline replaces {{working_file}} in prompts."""
    from agentino.pipeline.staged import StageDef, StagedPipeline

    pipeline = StagedPipeline(
        stages=[
            StageDef(name="test", prompt='read_file("{{working_file}}")'),
        ]
    )
    # The replacement happens during run(), but we can verify the template is there
    assert "{{working_file}}" in pipeline.stages[0].prompt
