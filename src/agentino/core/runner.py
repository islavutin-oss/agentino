"""Runner — the async framework execution engine.

Loads config, creates agents, manages sessions, dispatches messages.

Usage:
    agentino run agents.yml                     # interactive REPL with default agent
    agentino run agents.yml --agent reviewer    # pick a specific agent
    agentino run agents.yml --serve 8080        # HTTP server
    agentino run agents.yml --message "hello"   # one-shot, print reply and exit
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from agentino.config import Config, load_config
from agentino.core.agent import Agent
from agentino.core.session import Session, safe_segment
from agentino.core.tool import Tool
from agentino.extras.usage import UsageTracker

_HAS_RICH = importlib.util.find_spec("rich") is not None


class Runner:
    """Async framework execution engine.

    Loads a config file, manages multiple agents, routes messages,
    handles sessions and usage tracking automatically.
    """

    def __init__(
        self,
        config: Config,
        session_dir: str | Path = "./sessions",
        usage_file: str | Path = "./usage.jsonl",
        verbose: bool = False,
        no_session: bool = False,
    ):
        self.config = config
        self.session_dir = Path(session_dir)
        self.usage_tracker = UsageTracker(path=usage_file)
        self.verbose = verbose
        # no_session: every session is ephemeral — no history loaded or
        # persisted. For one-shot / benchmark runs that must each start clean.
        self.no_session = no_session
        self._sessions: dict[str, Session] = {}
        self.message_hook = None  # App-level hook: async (runner, agent, msg, session) -> str|None

        # Load message_hook from config if specified
        hook_module = config.raw.get("message_hook")
        if hook_module:
            try:
                import importlib

                # Add config dir + tools dirs to sys.path so hook modules are importable
                config_dir = os.environ.get("AGENTINO_CONFIG_DIR", "")
                if config_dir and config_dir not in sys.path:
                    sys.path.insert(0, config_dir)
                # Also add tools_dir paths (hook may live in shared tools)
                tools_dirs = config.raw.get("agents", {})
                for agent_cfg in tools_dirs.values() if isinstance(tools_dirs, dict) else []:
                    td = agent_cfg.get("tools_dir") if isinstance(agent_cfg, dict) else None
                    if td:
                        dirs = [td] if isinstance(td, str) else td
                        for d in dirs:
                            resolved = (
                                str((Path(config_dir) / d).resolve())
                                if config_dir and not Path(d).is_absolute()
                                else d
                            )
                            if resolved not in sys.path:
                                sys.path.insert(0, resolved)
                mod = importlib.import_module(hook_module)
                self.message_hook = getattr(mod, "classify_and_route", None) or getattr(
                    mod, "handle", None
                )
                if self.message_hook:
                    import logging

                    logging.getLogger(__name__).info(f"Loaded message hook from {hook_module}")
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to load message_hook {hook_module}: {e}"
                )

        for name, agent in self.config.agents.items():
            agent.name = agent.name or name
            self._wire_usage(agent, verbose=verbose)

    def _wire_usage(self, agent: Agent, verbose: bool = False) -> None:
        original_on_event = agent.on_event
        tracker = self.usage_tracker
        model = agent.model
        agent_name = agent.name

        def _on_event(event: Any) -> None:
            if event.type == "llm_response" and event.usage:
                tracker.record(model=model, usage=event.usage, agent_name=agent_name)
            if verbose:
                _cli_render_event(event)
            if original_on_event:
                original_on_event(event)

        agent.on_event = _on_event

    def get_session(self, agent_name: str, session_id: str = "default") -> Session:
        """Get or create a session for the given agent and session ID."""
        key = f"{safe_segment(agent_name, fallback='agent')}--{safe_segment(session_id)}"
        if key not in self._sessions:
            path = self.session_dir / f"{key}.jsonl"
            self._sessions[key] = Session(path, ephemeral=self.no_session)
        return self._sessions[key]

    async def send(
        self, message: str, agent_name: str | None = None, session_id: str = "default", **kwargs
    ) -> str:
        """Send a message to an agent. Runs pipeline if configured, otherwise direct chat.

        Apps can set self.message_hook to intercept messages before pipeline.
        Hook signature: async (runner, agent, message, session_id) -> str | None
        Return str = reply (skip pipeline). Return None = run pipeline.
        """
        agent = self._resolve_agent(agent_name)

        # App-level message hook (intent classification, follow-up handling)
        if hasattr(self, "message_hook") and self.message_hook:
            session = self.get_session(agent.name or "default", session_id)
            result = await self.message_hook(self, agent, message, session)
            if result is not None:
                return result

        if self.config.pipeline and hasattr(self.config.pipeline, "run"):
            from agentino.pipeline.staged import StagedPipeline

            if isinstance(self.config.pipeline, StagedPipeline):
                from agentino.cli.renderer import CLIRenderer

                renderer = CLIRenderer(use_rich=True)
                renderer.set_total_stages(len(self.config.pipeline.stages))
                results = await self.config.pipeline.run(
                    agent,
                    message,
                    on_event=renderer.handle,
                )
                renderer.print_summary(results)
                # Read working document if pipeline created one
                wf = getattr(self.config.pipeline, "last_working_file", None)
                if wf and Path(wf).exists():
                    content = Path(wf).read_text(encoding="utf-8").strip()
                    if content:
                        return content
                # Fallback to last stage's text output
                for r in reversed(results):
                    if r.output:
                        return r.output
                return "Pipeline completed but produced no output."
            return await self.config.pipeline.run(message)

        session = self.get_session(agent.name or "default", session_id)
        return await agent.run(message, session=session)

    def _print_header(self, agent: Agent, agents: dict) -> None:
        """Print fancy REPL header."""
        import shutil

        cols = shutil.get_terminal_size().columns
        name = agent.name or "agent"
        model = agent.model or "?"

        # Check for stages
        has_stages = self.config.pipeline is not None
        from agentino.pipeline.staged import StagedPipeline

        stage_names = ""
        if has_stages and isinstance(self.config.pipeline, StagedPipeline):
            stage_names = " → ".join(s.name.upper() for s in self.config.pipeline.stages)

        # Clear screen
        import os

        os.system("cls" if os.name == "nt" else "clear")

        if _HAS_RICH:
            from rich.console import Console
            from rich.text import Text

            c = Console()

            # Title bar
            title = f" {name} "
            pad = cols - len(title)
            left = pad // 2
            right = pad - left
            bar = Text()
            bar.append("─" * left + title + "─" * right, style="bold reverse")
            c.print(bar)
            c.print()

            # Info
            c.print(f"  [dim]model[/]   {model}")
            c.print(f"  [dim]tools[/]   {len(agent.tools)} [dim](/help for commands)[/]")
            if agent.knowledge:
                c.print(f"  [dim]KB[/]      {len(agent.knowledge.entries)} entries")
            if stage_names:
                c.print(f"  [dim]stages[/]  {stage_names}")
            if len(agents) > 1:
                c.print(f"  [dim]agents[/]  {', '.join(agents.keys())} [dim](/agent <name>)[/]")
            c.print()
            c.print("  [dim]Enter to submit · /exit to quit · /help for commands[/]")
            c.print()
            c.print("─" * cols, style="dim")
            c.print()
        else:
            print(f"\n{'─' * cols}")
            print(f"  {name} ({model})")
            print(
                f"  Tools: {len(agent.tools)} | {'KB: ' + str(len(agent.knowledge.entries)) + ' entries | ' if agent.knowledge else ''}{stage_names}"
            )
            print(f"{'─' * cols}\n")

    def _get_input(self) -> str:
        """Get user input — prompt_toolkit if available, fallback to input()."""
        try:
            from prompt_toolkit import prompt
            from prompt_toolkit.formatted_text import HTML

            return prompt(HTML("<b>❯ </b>"))
        except ImportError:
            return input("❯ ")

    def _resolve_agent(self, name: str | None = None) -> Agent:
        if name and name in self.config.agents:
            return self.config.agents[name]

        default_name = self.config.raw.get("default_agent")
        if default_name and default_name in self.config.agents:
            return self.config.agents[default_name]

        if self.config.agents:
            return next(iter(self.config.agents.values()))

        raise RuntimeError("No agents defined in config")

    def list_agents(self) -> list[dict[str, str]]:
        """Return a list of all agents with their basic info."""
        result = []
        for name, agent in self.config.agents.items():
            result.append(
                {
                    "name": name,
                    "model": agent.model,
                    "tools": [t.name for t in agent.tools],
                    "instructions": (agent.instructions[:80] + "...")
                    if len(agent.instructions) > 80
                    else agent.instructions,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------

    def repl(self, agent_name: str | None = None, session_id: str = "default") -> None:
        """Interactive REPL mode (runs async loop internally)."""
        asyncio.run(self._repl_async(agent_name, session_id))

    async def _repl_async(self, agent_name: str | None, session_id: str) -> None:
        agent = self._resolve_agent(agent_name)
        agents = self.config.agents

        self._print_header(agent, agents)

        current_agent_name = agent.name or next(iter(agents.keys()), "default")

        while True:
            try:
                user_input = await asyncio.to_thread(self._get_input)
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if self._handle_command(user_input, current_agent_name, session_id):
                        if user_input.startswith("/agent "):
                            new_name = user_input.split(maxsplit=1)[1].strip()
                            if new_name in agents:
                                current_agent_name = new_name
                                agent = agents[new_name]
                                print(f"\n→ Switched to {new_name} ({agent.model})\n")
                            else:
                                print(
                                    f"\n→ Unknown agent: {new_name}. Available: {', '.join(agents.keys())}\n"
                                )
                    continue

                reply = await self.send(
                    user_input, agent_name=current_agent_name, session_id=session_id
                )
                # For staged pipelines, TUI already rendered progress — just print the answer
                if reply:
                    if _HAS_RICH:
                        from rich.console import Console
                        from rich.markdown import Markdown

                        _c = Console()
                        _c.print()
                        _c.print(Markdown(reply))
                        _c.print()
                    else:
                        print(f"\n{reply}\n")

            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{self.usage_tracker.summary()}")
                print("Goodbye!")
                break
            except Exception as e:
                err = str(e)
                if "401" in err or "Unauthorized" in err or "403" in err:
                    print(
                        f"\n  ✗ Authentication failed. Check your API key and .env file.\n    {err[:200]}\n"
                    )
                elif "400" in err:
                    print(f"\n  ✗ Bad request: {err[:200]}\n")
                else:
                    print(f"\n  ✗ Error: {err[:200]}\n")

    def _handle_command(self, cmd: str, agent_name: str, session_id: str) -> bool:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()

        if command in ("/exit", "/quit", "/q"):
            print(f"\n{self.usage_tracker.summary()}")
            print("Bye!")
            raise SystemExit(0)

        if command == "/help":
            print("""
Commands:
  /agent <name>  — switch to a different agent
  /agents        — list available agents
  /usage         — show token usage and cost
  /session       — show current session info
  /clear         — clear current session
  /exit          — exit the REPL
  /help          — show this help
""")
            return True

        if command == "/agents":
            for info in self.list_agents():
                tools = ", ".join(info["tools"]) if info["tools"] else "none"
                print(f"  {info['name']:15s} model={info['model']:20s} tools=[{tools}]")
            print()
            return True

        if command == "/usage":
            print(f"\n{self.usage_tracker.summary()}\n")
            return True

        if command == "/session":
            session = self.get_session(agent_name, session_id)
            msgs = session.load()
            print(f"\n  Session: {session.path}")
            print(f"  Messages: {len(msgs)}\n")
            return True

        if command == "/clear":
            session = self.get_session(agent_name, session_id)
            session.clear()
            print("\n  Session cleared.\n")
            return True

        if command == "/agent":
            return True

        print(f"\n  Unknown command: {command}. Type /help\n")
        return True

    async def one_shot(self, message: str, agent_name: str | None = None) -> str:
        """Send a single message and return the reply.

        Uses the "default" session — so history IS loaded/persisted unless the
        runner was created with no_session=True (CLI: --no-session). One-shot
        callers that must start clean every time should pass --no-session.
        """
        return await self.send(message, agent_name=agent_name)

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Start an HTTP server. Requires uvicorn + starlette.

        Binds loopback by default. The endpoint runs an agent with whatever
        message it is given and has no authentication of its own, so binding
        every interface published that to the network — on a shared host, to
        everyone on it. Pass `host="0.0.0.0"` deliberately, behind something
        that authenticates.
        """
        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            from starlette.routing import Route
        except ImportError:
            print("HTTP server requires: pip install uvicorn starlette")
            sys.exit(1)

        runner = self

        async def handle_chat(request: Request) -> JSONResponse:
            """Handle POST /chat endpoint."""
            try:
                body = await request.json()
            except Exception:
                # Malformed JSON is the caller's mistake, not a server fault,
                # and a 500 carrying the parser's exception text says more
                # about the process than the client needs to know.
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
            msg = body.get("message", "")
            agent_name = body.get("agent")
            session_id = body.get("session_id", "default")
            if agent_name and agent_name not in runner.list_agents():
                return JSONResponse({"error": f"unknown agent: {agent_name}"}, status_code=404)
            try:
                reply = await runner.send(msg, agent_name=agent_name, session_id=session_id)
                return JSONResponse(
                    {
                        "reply": reply,
                        "agent": agent_name or "default",
                        "session_id": session_id,
                        "usage": {
                            "prompt_tokens": runner.usage_tracker.total.prompt_tokens,
                            "completion_tokens": runner.usage_tracker.total.completion_tokens,
                        },
                    }
                )
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def handle_agents(request: Request) -> JSONResponse:
            """Handle GET /agents endpoint."""
            return JSONResponse({"agents": runner.list_agents()})

        async def handle_usage(request: Request) -> JSONResponse:
            """Handle GET /usage endpoint."""
            return JSONResponse(
                {
                    "total": {
                        "prompt_tokens": runner.usage_tracker.total.prompt_tokens,
                        "completion_tokens": runner.usage_tracker.total.completion_tokens,
                        "total_tokens": runner.usage_tracker.total.total_tokens,
                    },
                    "cost_estimate": runner.usage_tracker.cost_estimate,
                    "by_model": {
                        m: {
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                        }
                        for m, u in runner.usage_tracker.by_model.items()
                    },
                }
            )

        app = Starlette(
            routes=[
                Route("/chat", handle_chat, methods=["POST"]),
                Route("/agents", handle_agents, methods=["GET"]),
                Route("/usage", handle_usage, methods=["GET"]),
            ]
        )

        print(f"\n🤖 Agentino server on http://{host}:{port}")
        print("   POST /chat    — send message")
        print("   GET  /agents  — list agents")
        print("   GET  /usage   — token usage\n")
        uvicorn.run(app, host=host, port=port, log_level="info")


_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _cli_render_event(event: Any) -> None:
    """Render agent events to terminal for live progress."""
    from agentino.core.message import EventType

    if event.type == EventType.TOOL_START:
        name = event.name
        # Compact args: show first meaningful value only
        hint = ""
        if event.args:
            for k, v in event.args.items():
                s = str(v).replace("\n", " ")
                if len(s) > 70:
                    s = s[:67] + "…"
                hint = f" {_DIM}{s}{_RESET}"
                break
        print(f"  {_CYAN}▸{_RESET} {_BOLD}{name}{_RESET}{hint}", flush=True)

    elif event.type == EventType.TOOL_RESULT:
        text = str(event.data or "")
        if text:
            first_line = text.split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "…"
            # Color based on content
            if text.startswith("Error"):
                print(f"    {_RED}✗ {first_line}{_RESET}", flush=True)
            elif (
                text.startswith("Saved") or text.startswith("Updated") or text.startswith("Deleted")
            ):
                print(f"    {_GREEN}✓ {first_line}{_RESET}", flush=True)
            else:
                print(f"    {_DIM}→ {first_line}{_RESET}", flush=True)

    elif event.type == EventType.ERROR:
        print(f"  {_RED}✗ {event.name}: {event.data}{_RESET}", flush=True)

    elif event.type == EventType.FALLBACK:
        print(f"  {_YELLOW}⚠ {event.data}{_RESET}", flush=True)

    elif event.type == EventType.LLM_RESPONSE and event.usage:
        tokens = event.usage.total_tokens
        if tokens:
            print(f"  {_DIM}⟡ {tokens} tokens{_RESET}", flush=True)


def create_runner(
    config_path: str | Path,
    tools: list[Tool] | None = None,
    session_dir: str | Path = "./sessions",
    usage_file: str | Path = "./usage.jsonl",
    verbose: bool = False,
    no_session: bool = False,
) -> Runner:
    """Create a Runner from a config file.

    Usage:
        runner = create_runner("agents.yml", tools=[my_tool])
        runner.repl()                    # interactive
        await runner.one_shot("hi")      # single message
        runner.serve(port=8080)          # HTTP server
    """
    config = load_config(config_path, tools=tools)
    return Runner(
        config,
        session_dir=session_dir,
        usage_file=usage_file,
        verbose=verbose,
        no_session=no_session,
    )
