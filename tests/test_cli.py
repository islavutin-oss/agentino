"""Tests for CLI entry point."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agentino.__main__ import _discover_tools, main


class TestDiscoverTools:
    def test_discovers_tools_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools_dir.mkdir()
            (tools_dir / "my_tools.py").write_text(
                "from agentino import tool\n\n"
                "@tool\n"
                "def hello(name: str) -> str:\n"
                '    """Say hi."""\n'
                '    return f"Hi {name}"\n'
            )
            tools = _discover_tools(Path(tmp))
            assert len(tools) == 1
            assert tools[0].name == "hello"

    def test_empty_tools_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools_dir.mkdir()
            tools = _discover_tools(Path(tmp))
            assert tools == []

    def test_no_tools_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = _discover_tools(Path(tmp))
            assert tools == []

    def test_skips_underscore_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools_dir.mkdir()
            (tools_dir / "__init__.py").write_text("")
            (tools_dir / "_private.py").write_text(
                'from agentino import tool\n\n@tool\ndef secret() -> str:\n    return "x"\n'
            )
            tools = _discover_tools(Path(tmp))
            assert tools == []


class TestCLIVersion:
    def test_version(self):
        with patch("sys.argv", ["agentino", "version"]):
            # Should not crash
            try:
                main()
            except SystemExit:
                pass


class TestCLIAgents:
    def test_agents_command(self, capsys):
        yaml = "agents:\n  bot:\n    model: gpt-4o\n    instructions: 'Hello'\n"
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            f.write(yaml)
            f.flush()
            try:
                with patch("sys.argv", ["agentino", "agents", f.name]):
                    main()
                out = capsys.readouterr().out
                assert "bot" in out
                assert "gpt-4o" in out
            finally:
                os.unlink(f.name)
