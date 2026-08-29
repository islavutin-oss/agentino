"""CLI entry point — python -m agentino, or `agentino` command.

Usage:
    agentino run agents.yml                       # REPL with default agent
    agentino run agents.yml --agent reviewer      # REPL with specific agent
    agentino run agents.yml --message "hello"     # one-shot
    agentino run agents.yml --serve 8080          # HTTP server
    agentino chat                                 # quick REPL, no config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """CLI entry point for the agentino command."""
    parser = argparse.ArgumentParser(
        prog="agentino",
        description="Agentino — lightweight Python agent framework",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_parser = sub.add_parser("run", help="Run agents from a config file")
    run_parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to agents config file (.yml, .yaml). Auto-discovers agents.yml in current directory if omitted.",
    )
    run_parser.add_argument(
        "--agent", "-a", help="Agent name (default: first agent or default_agent)"
    )
    run_parser.add_argument("--message", "-m", help="One-shot message (print reply and exit)")
    run_parser.add_argument("--serve", type=int, metavar="PORT", help="Start HTTP server on PORT")
    run_parser.add_argument("--session-dir", default="./sessions", help="Session storage directory")
    run_parser.add_argument("--usage-file", default="./usage.jsonl", help="Usage log file")
    run_parser.add_argument("--session-id", default="default", help="Session ID for REPL/one-shot")
    run_parser.add_argument(
        "--no-session",
        action="store_true",
        help="Ephemeral run — don't load or persist any session history",
    )
    run_parser.add_argument(
        "--project",
        "-p",
        help="Project directory (sets AGENTINO_PROJECT_DIR for file tools, coder, knowledge)",
    )
    run_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress live tool call output"
    )
    run_parser.add_argument(
        "--gateway",
        action="store_true",
        help="Start gateway (multi-channel: Telegram, Slack, etc.)",
    )
    run_parser.add_argument(
        "--iterate",
        "-i",
        action="store_true",
        help="After one-shot, drop into REPL to iterate on the result",
    )
    run_parser.add_argument(
        "--mode",
        choices=["text", "json", "jsonl"],
        default="text",
        help="Output mode for one-shot (--message) runs. text=ANSI-pretty (default), "
        "json=single envelope at end, jsonl=streaming events + final envelope. "
        "json/jsonl force --quiet to keep stdout machine-readable.",
    )

    # --- chat ---
    chat_parser = sub.add_parser("chat", help="Quick chat REPL (no config file needed)")
    chat_parser.add_argument(
        "--model", "-M", default=None, help="Model to use (auto-detected from auth)"
    )
    chat_parser.add_argument(
        "--instructions", "-i", default="You are a helpful assistant.", help="System instructions"
    )
    chat_parser.add_argument("--base-url", help="API base URL")
    chat_parser.add_argument("--api-key", help="API key")

    # --- agents ---
    agents_parser = sub.add_parser("agents", help="List agents in a config file")
    agents_parser.add_argument("config", help="Path to agents config file")

    # --- login ---
    login_parser = sub.add_parser(
        "login", help="Authenticate with OpenAI (Codex subscription) or Anthropic"
    )
    login_parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider to authenticate with (default: openai)",
    )
    login_parser.add_argument(
        "--token", "-t", help="Paste a token directly (Anthropic setup-token or API key)"
    )

    # --- logout ---
    sub.add_parser("logout", help="Clear stored credentials")

    # --- status ---
    sub.add_parser("status", help="Show auth status and stored credentials")

    # --- version ---
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        from . import __version__

        print(f"agentino {__version__}")
        return

    if args.command == "login":
        _cmd_login(args)
        return

    if args.command == "logout":
        _cmd_logout()
        return

    if args.command == "status":
        _cmd_status()
        return

    if args.command == "agents":
        _cmd_agents(args)
        return

    if args.command == "chat":
        _cmd_chat(args)
        return

    if args.command == "run":
        _cmd_run(args)
        return

    parser.print_help()


def _cmd_login(args: argparse.Namespace) -> None:
    """Authenticate with a provider."""
    from agentino.safety.auth import (
        AuthCredentials,
        login_openai,
        save_anthropic_token,
        save_credentials,
    )

    if args.provider == "openai":
        if args.token:
            # Direct API key
            creds = AuthCredentials(provider="openai", access_token=args.token)
            save_credentials(creds)
            print("Saved OpenAI API key.")
        else:
            # OAuth PKCE flow
            print("Logging in with OpenAI (Codex subscription)...")
            try:
                creds = login_openai()
                email = f" ({creds.email})" if creds.email else ""
                print(f"Logged in to OpenAI{email}")
                print(f"Token expires: {_format_expiry(creds.expires_at)}")
                print("Stored in: ~/.agentino/auth.json")
            except Exception as e:
                print(f"Login failed: {e}", file=sys.stderr)
                sys.exit(1)

    elif args.provider == "anthropic":
        if args.token:
            save_anthropic_token(args.token)
            print("Saved Anthropic token.")
        else:
            print("For Anthropic, run `claude setup-token` first, then:")
            print("  agentino login --provider anthropic --token <your-token>")
            print()
            print("Or paste your API key:")
            print("  agentino login --provider anthropic --token sk-ant-...")
            sys.exit(1)

    print("\nReady! Run: agentino run agents.yml")


def _cmd_logout() -> None:
    """Clear stored credentials."""
    from agentino.safety.auth import AUTH_FILE

    if AUTH_FILE.exists():
        AUTH_FILE.unlink()
        print("Credentials cleared.")
    else:
        print("No stored credentials found.")


def _cmd_status() -> None:
    """Show auth status."""
    from agentino.safety.auth import (
        AUTH_FILE,
        CODEX_AUTH_FILE,
        load_codex_credentials,
        load_credentials,
    )

    print(f"Auth file: {AUTH_FILE}")
    print(f"Exists: {AUTH_FILE.exists()}\n")

    for provider in ("openai", "anthropic"):
        creds = load_credentials(provider)
        source = "agentino"

        # Fall back to Codex CLI credentials for OpenAI
        if not creds and provider == "openai":
            creds = load_codex_credentials()
            if creds:
                source = "codex"

        if creds:
            # Mask token
            token = creds.access_token
            masked = token[:8] + "..." + token[-4:] if len(token) > 16 else "***"
            print(f"  {provider}:")
            if source == "codex":
                print(f"    Source: Codex CLI ({CODEX_AUTH_FILE})")
            print(f"    Token: {masked}")
            if creds.email:
                print(f"    Email: {creds.email}")
            if creds.expires_at:
                print(f"    Expires: {_format_expiry(creds.expires_at)}")
                if creds.is_expired:
                    print("    Status: EXPIRED (will auto-refresh)")
                else:
                    print("    Status: valid")
            else:
                print("    Status: valid (no expiry)")
            print()
        else:
            # Check env var
            import os

            env_vars = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
            env_val = os.getenv(env_vars.get(provider, ""))
            if env_val:
                print(f"  {provider}: via ${env_vars[provider]} env var")
            else:
                print(f"  {provider}: not configured")
            print()


def _format_expiry(ts: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_config(config_arg: str | None) -> Path | None:
    """Resolve config file path. Auto-discovers agents.yml if not specified."""
    if config_arg:
        return Path(config_arg)
    # Auto-discover in current directory
    for name in ("agents.yml", "agents.yaml"):
        p = Path(name)
        if p.exists():
            return p
    return None


def _cmd_run(args: argparse.Namespace) -> None:
    """Run agents from config."""
    import os

    from agentino.core.runner import create_runner

    config_path = _resolve_config(args.config)
    if not config_path or not config_path.exists():
        print(
            "Error: config file not found. Provide a path or create agents.yml in the current directory."
        )
        sys.exit(1)

    # Auto-load .env from config directory (then parent dirs)
    for env_dir in [config_path.parent, config_path.parent.parent, Path.cwd()]:
        env_file = env_dir / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            break

    if args.project:
        project_dir = Path(args.project).resolve()
        if not project_dir.is_dir():
            print(f"Error: project directory not found: {project_dir}")
            sys.exit(1)
        os.environ["AGENTINO_PROJECT_DIR"] = str(project_dir)

    # Set config dir so tools (parallel_explore etc.) know where repos are
    os.environ["AGENTINO_CONFIG_DIR"] = str(config_path.parent.resolve())

    # Load tools from tools/ directory next to config if it exists
    tools = _discover_tools(config_path.parent)

    # json/jsonl modes write structured output to stdout — silence the
    # human-readable renderer so the two streams don't collide.
    machine_mode = getattr(args, "mode", "text") in ("json", "jsonl")
    verbose = (not args.quiet) and not machine_mode
    runner = create_runner(
        config_path,
        tools=tools,
        session_dir=args.session_dir,
        usage_file=args.usage_file,
        verbose=verbose,
        no_session=getattr(args, "no_session", False),
    )

    if args.gateway:
        from agentino.transport import build_gateway

        gateway = build_gateway(runner.config, session_dir=args.session_dir)
        names = ", ".join(gateway.channels[i].name for i in range(len(gateway.channels)))
        print(f"\n  Agentino gateway — {len(gateway.channels)} channel(s): {names}")
        print("  Press Ctrl+C to stop\n")
        gateway.run()
        return

    try:
        if args.serve:
            runner.serve(port=args.serve)
        elif args.message:
            import asyncio

            if machine_mode:
                import contextlib
                import io as _io

                from agentino.cli.json_emitter import JsonEmitter

                target = runner._resolve_agent(args.agent)
                # Construct the emitter BEFORE redirecting stdout so it
                # holds a reference to the real stdout for envelope output;
                # the redirect inside the run silences any incidental
                # prints from the staged-pipeline renderer / tool surface.
                emitter = JsonEmitter(mode=args.mode)
                prev_handler = target.on_event

                def _chain(event):
                    emitter.handle(event)
                    if prev_handler:
                        prev_handler(event)

                target.on_event = _chain
                noise = _io.StringIO()
                try:
                    with contextlib.redirect_stdout(noise):
                        reply = asyncio.run(runner.one_shot(args.message, agent_name=args.agent))
                finally:
                    target.on_event = prev_handler
                emitter.emit_envelope(reply, model=getattr(target, "model", "") or "")
            else:
                reply = asyncio.run(runner.one_shot(args.message, agent_name=args.agent))
                _print_final(reply)
            # --iterate: drop into REPL after one-shot to refine the result
            if getattr(args, "iterate", False) and not machine_mode:
                print("\n  Entering iteration mode — type follow-ups to improve the document.\n")
                runner.repl(agent_name=args.agent, session_id=args.session_id)
        else:
            runner.repl(agent_name=args.agent, session_id=args.session_id)
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
        sys.exit(1)


def _cmd_chat(args: argparse.Namespace) -> None:
    """Quick chat without config file."""
    from agentino.builtin_tools import BUILTIN_TOOLS
    from agentino.core.agent import Agent
    from agentino.core.session import Session
    from agentino.extras.usage import UsageTracker

    agent = Agent(
        model=args.model,  # None = auto-detect
        instructions=args.instructions,
        tools=BUILTIN_TOOLS,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    tracker = UsageTracker()
    tracker.bind(agent.model)
    agent.on_event = tracker.on_event

    session = Session("./sessions/chat.jsonl")

    print(f"\n  Agentino chat — {agent.model} via {agent._llm.provider}")
    print("   Ctrl+C to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            import asyncio

            reply = asyncio.run(agent.run(user_input, session=session))
            print(f"\nAssistant: {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{tracker.summary()}")
            print("Goodbye!")
            break


def _cmd_agents(args: argparse.Namespace) -> None:
    """List agents in a config file."""
    from agentino.core.runner import create_runner

    config_path = _resolve_config(args.config)
    if not config_path or not config_path.exists():
        print("Error: config file not found.")
        sys.exit(1)

    runner = create_runner(config_path)
    for info in runner.list_agents():
        tools = ", ".join(info["tools"]) if info["tools"] else "none"
        print(f"  {info['name']:15s} model={info['model']:20s} tools=[{tools}]")
        print(f"  {'':15s} {info['instructions']}")
        print()


def _print_final(text: str) -> None:
    """Print the final agent response, converting markdown to ANSI."""
    import re

    _DIM = "\033[2m"
    _BOLD = "\033[1m"
    _RESET = "\033[0m"

    def _md_to_ansi(line: str) -> str:
        # **bold** → ANSI bold
        line = re.sub(r"\*\*(.+?)\*\*", rf"{_BOLD}\1{_RESET}", line)
        # `code` → dim
        line = re.sub(r"`(.+?)`", rf"{_DIM}\1{_RESET}", line)
        # - bullet → •
        line = re.sub(r"^(\s*)- ", r"\1• ", line)
        # ### heading → bold
        line = re.sub(r"^#{1,4}\s+(.+)", rf"{_BOLD}\1{_RESET}", line)
        return line

    print(f"\n{_DIM}{'─' * 60}{_RESET}")
    for line in text.strip().split("\n"):
        print(f"  {_md_to_ansi(line)}")
    print(f"{_DIM}{'─' * 60}{_RESET}\n")


def _discover_tools(directory: Path) -> list:
    """Auto-discover @tool-decorated functions from tools/ directory."""
    from agentino.core.tool import Tool

    tools_dir = directory / "tools"
    if not tools_dir.is_dir():
        return []

    import importlib.util

    discovered: list[Tool] = []

    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, Tool):
                        discovered.append(attr)
        except Exception as e:
            print(f"Warning: failed to load {py_file}: {e}", file=sys.stderr)

    return discovered


if __name__ == "__main__":
    main()
