"""Tests for the file-based extension loader — Borrow #6."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

from agentino.core.agent import Agent
from agentino.core.extensions import ExtensionLoader


def _agent() -> Agent:
    """A test agent — no real LLM calls."""
    return Agent(model="x", api_key="x", base_url="http://localhost")


def _write(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


class TestImplicitToolDiscovery:
    """Pattern 1: extension file exports @tool-decorated callables at module top level."""

    def test_loads_one_implicit_tool(self, tmp_path: Path):
        _write(
            tmp_path / "search.py",
            """
            from agentino import tool

            @tool
            def search(q: str) -> str:
                '''Search for things.'''
                return f'found {q}'
        """,
        )

        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        result = loader.reload()

        assert any(t.name == "search" for t in agent.tools)
        assert "search" in agent._tool_map
        assert result.loaded[0].tool_names == ["search"]
        assert not result.errors

    def test_underscore_prefixed_files_skipped(self, tmp_path: Path):
        _write(tmp_path / "_helper.py", "X = 1\n")
        _write(
            tmp_path / "real.py",
            """
            from agentino import tool

            @tool
            def real_tool() -> str:
                return 'ok'
        """,
        )

        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        result = loader.reload()

        assert any(t.name == "real_tool" for t in agent.tools)
        assert all("_helper" not in (e.path.name) for e in result.loaded)


class TestExplicitRegister:
    """Pattern 2: extension file defines `def register(agent): ...` and installs itself."""

    def test_register_function_called(self, tmp_path: Path):
        _write(
            tmp_path / "pair.py",
            """
            from agentino import tool

            def register(agent):
                @tool
                def alpha() -> str: return 'A'
                @tool
                def beta() -> str: return 'B'
                agent.add_tools([alpha, beta])
        """,
        )

        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        result = loader.reload()

        names = {t.name for t in agent.tools}
        assert {"alpha", "beta"} <= names
        assert set(result.loaded[0].tool_names) == {"alpha", "beta"}


class TestReloadIdempotency:
    def test_reloading_same_file_does_not_duplicate_tools(self, tmp_path: Path):
        _write(
            tmp_path / "x.py",
            """
            from agentino import tool

            @tool
            def hello() -> str: return 'hi'
        """,
        )

        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        loader.reload()
        loader.reload()

        names = [t.name for t in agent.tools]
        assert names.count("hello") == 1

    def test_edit_on_disk_picked_up_after_reload(self, tmp_path: Path):
        f = tmp_path / "v.py"
        _write(
            f,
            """
            from agentino import tool

            @tool
            def v() -> str: return 'v1'
        """,
        )

        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        loader.reload()
        v_tool = next(t for t in agent.tools if t.name == "v")
        assert asyncio.run(v_tool.execute({})) == "v1"

        # Mutate the file and reload — second load should rebind to the new fn.
        _write(
            f,
            """
            from agentino import tool

            @tool
            def v() -> str: return 'v2'
        """,
        )
        loader.reload()
        v_tool = next(t for t in agent.tools if t.name == "v")
        assert asyncio.run(v_tool.execute({})) == "v2"

    def test_deleting_file_removes_tool_on_next_reload(self, tmp_path: Path):
        f = tmp_path / "go.py"
        _write(
            f,
            """
            from agentino import tool

            @tool
            def go() -> str: return 'go'
        """,
        )

        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        loader.reload()
        assert any(t.name == "go" for t in agent.tools)

        f.unlink()
        loader.reload()
        assert not any(t.name == "go" for t in agent.tools)
        assert "go" not in agent._tool_map


class TestErrorHandling:
    def test_broken_extension_does_not_crash_loader(self, tmp_path: Path):
        _write(tmp_path / "broken.py", "this is not valid python {{")
        _write(
            tmp_path / "good.py",
            """
            from agentino import tool

            @tool
            def good() -> str: return 'g'
        """,
        )

        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        result = loader.reload()

        assert any(t.name == "good" for t in agent.tools)
        assert any("broken" in p for p, _ in result.errors)

    def test_missing_extension_dir_is_silent(self, tmp_path: Path):
        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path / "nonexistent")
        result = loader.reload()
        assert result.loaded == []
        assert result.errors == []


class TestSelfExtension:
    """The 'agent writes its own tool, then calls reload_extensions' loop."""

    def test_make_reload_tool_invokes_loader(self, tmp_path: Path):
        agent = _agent()
        loader = ExtensionLoader(agent, extensions_dir=tmp_path)
        agent.add_tool(loader.make_reload_tool())
        assert "reload_extensions" in agent._tool_map

        # Drop a new tool file as if the agent itself had just written it.
        _write(
            tmp_path / "newt.py",
            """
            from agentino import tool

            @tool
            def newt() -> str: return 'fresh'
        """,
        )

        # Agent invokes its reload tool — new tool appears.
        out = asyncio.run(agent._tool_map["reload_extensions"].execute({}))
        assert "newt" in out
        assert any(t.name == "newt" for t in agent.tools)
