"""Extension loader — file-based hot-reloadable tools (Borrow #6 from pi).

Pattern: drop a .py file in a watched directory, the agent sees a new tool on
the next `reload()`. Two file shapes are supported:

    # 1. Implicit — module exports `Tool` instances at top level
    # ~/.agentino/extensions/my_search.py
    from agentino import tool

    @tool
    def search(q: str) -> str:
        '''Search the project knowledge base.'''
        ...

    # 2. Explicit register() — module installs into the agent itself
    # ~/.agentino/extensions/my_pair.py
    from agentino import tool

    def register(agent):
        @tool
        def pair_a(...): ...
        @tool
        def pair_b(...): ...
        agent.add_tools([pair_a, pair_b])

The "self-extensible" loop (pi's pitch): the agent has a `reload_extensions`
tool. It uses bash/write to drop a new file in the extension dir, then calls
its own `reload_extensions` tool. From the next turn the new tool is in its
toolbox.

Safety:
- Reload is idempotent: tools from a previous reload are removed before the
  new set is installed (matched by extension file path).
- A failing extension file is skipped with a logged error; other extensions
  still load.
- Module names are namespaced under `agentino_ext.` so they don't collide
  with project imports.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agentino.core.tool import Tool

if TYPE_CHECKING:
    from agentino.core.agent import Agent

log = logging.getLogger(__name__)

_MODULE_NS = "agentino_ext"


@dataclass
class _LoadedExtension:
    """Bookkeeping for one loaded extension file."""

    path: Path
    module_name: str
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ReloadResult:
    """Outcome of one reload pass — useful for the reload_extensions tool."""

    loaded: list[_LoadedExtension] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (path, error)

    def summary(self) -> str:
        ok = len([x for x in self.loaded if not x.error])
        n_tools = sum(len(x.tool_names) for x in self.loaded)
        out = [f"Reloaded {ok} extension(s), {n_tools} tool(s)."]
        for ext in self.loaded:
            out.append(f"  - {ext.path.name}: {', '.join(ext.tool_names) or '(no tools exported)'}")
        for path, err in self.errors:
            out.append(f"  ! {path}: {err}")
        return "\n".join(out)


class ExtensionLoader:
    """Loads and reloads .py extensions from a directory, attaching tools to an Agent."""

    def __init__(self, agent: Agent, extensions_dir: str | os.PathLike | None = None):
        self.agent = agent
        self.extensions_dir = Path(extensions_dir or "~/.agentino/extensions").expanduser()
        # Path → loaded ext bookkeeping. Used to remove stale tools on reload.
        self._loaded: dict[Path, _LoadedExtension] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(self) -> ReloadResult:
        """Scan the extension dir, import each .py file, install its tools.

        Idempotent: tools from a previous reload of the *same file* are removed
        first. Tools from extension files that no longer exist are also removed.
        """
        result = ReloadResult()
        if not self.extensions_dir.exists():
            return result

        seen: set[Path] = set()
        for path in sorted(self.extensions_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            seen.add(path)
            ext = self._reload_one(path)
            result.loaded.append(ext)
            if ext.error:
                result.errors.append((str(path), ext.error))

        # Drop extensions whose files were deleted between reloads.
        for stale_path in list(self._loaded.keys()):
            if stale_path not in seen:
                self._remove_tools(self._loaded[stale_path])
                self._purge_module(self._loaded[stale_path].module_name)
                self._loaded.pop(stale_path, None)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reload_one(self, path: Path) -> _LoadedExtension:
        # Remove tools from previous load of this file (if any).
        prev = self._loaded.get(path)
        if prev:
            self._remove_tools(prev)
            self._purge_module(prev.module_name)

        # Include mtime_ns in the module name so reloads land in distinct
        # sys.modules entries. We also bypass Python's bytecode (.pyc) cache by
        # compiling the source directly: pyc lookup is path-keyed, not module-name
        # keyed, so two edits within the same 1s window can otherwise re-execute
        # the stale cached bytecode even after sys.modules.pop().
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        module_name = f"{_MODULE_NS}.{path.stem}_{abs(hash((str(path), mtime))) & 0xFFFFFF:06x}"
        ext = _LoadedExtension(path=path, module_name=module_name)
        try:
            import types

            source = path.read_text(encoding="utf-8")
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            module.__name__ = module_name
            sys.modules[module_name] = module
            code = compile(source, str(path), "exec")
            exec(code, module.__dict__)

            # Two install styles: (1) explicit register(agent), (2) collect Tool instances.
            installed: list[Tool] = []
            if hasattr(module, "register") and callable(module.register):
                # Pattern 1: register(agent) — the module installs itself.
                # Snapshot tool list before/after so we can track ownership.
                before = {t.name for t in self.agent.tools}
                module.register(self.agent)
                after = {t.name for t in self.agent.tools}
                installed = [t for t in self.agent.tools if t.name in (after - before)]
            else:
                # Pattern 2: collect every top-level Tool instance.
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue
                    obj = getattr(module, attr_name)
                    if isinstance(obj, Tool):
                        installed.append(obj)
                self.agent.add_tools(installed)

            ext.tool_names = [t.name for t in installed]
            self._loaded[path] = ext
        except Exception as e:
            log.warning("Failed to load extension %s: %s", path, e)
            ext.error = f"{type(e).__name__}: {e}"
            self._loaded[path] = ext
        return ext

    def _remove_tools(self, ext: _LoadedExtension) -> None:
        if not ext.tool_names:
            return
        keep = [t for t in self.agent.tools if t.name not in set(ext.tool_names)]
        self.agent.tools = keep
        for n in ext.tool_names:
            self.agent._tool_map.pop(n, None)

    @staticmethod
    def _purge_module(module_name: str) -> None:
        # Drop from sys.modules so the next load re-runs module-level code.
        sys.modules.pop(module_name, None)

    # ------------------------------------------------------------------
    # Self-extension hook: build an LLM-callable tool that drives reload.
    # ------------------------------------------------------------------

    def make_reload_tool(self) -> Tool:
        """Return a Tool the LLM can call to re-scan its extension dir.

        Pair this with the built-in write_file tool and you have pi's
        "self-extensible coding agent" pattern: model writes a new .py file,
        then calls reload_extensions().
        """

        from agentino.core.tool import tool as _tool_dec

        loader = self

        @_tool_dec(
            name="reload_extensions",
            description="Re-scan the agent's extension directory and (re)install any "
            "tools defined there. Use after writing a new tool file.",
        )
        async def reload_extensions() -> str:
            return loader.reload().summary()

        return reload_extensions
