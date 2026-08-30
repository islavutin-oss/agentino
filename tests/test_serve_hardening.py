"""The HTTP endpoint runs an agent for whoever can reach it.

`serve()` has no authentication of its own. Binding every interface therefore
published an unauthenticated agent to the network — on a shared host, to
everyone on it. Loopback is the only safe default; exposing it is a decision
the caller makes explicitly, behind something that authenticates.
"""

from __future__ import annotations

import inspect

from agentino.core.runner import Runner


def test_the_http_server_binds_loopback_by_default():
    assert inspect.signature(Runner.serve).parameters["host"].default == "127.0.0.1"


def test_the_whatsapp_transport_binds_loopback_by_default():
    from agentino.transport.whatsapp import WhatsAppChannel

    assert inspect.signature(WhatsAppChannel.__init__).parameters["host"].default == "127.0.0.1"


def test_no_module_still_defaults_to_every_interface():
    """A new transport copied from an old one is how this comes back."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "agentino"
    offenders = []
    for f in root.rglob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            for name, default in zip(
                [a.arg for a in args.args][-len(args.defaults) :] if args.defaults else [],
                args.defaults,
            ):
                if (
                    name == "host"
                    and isinstance(default, ast.Constant)
                    and default.value == "0.0.0.0"
                ):
                    offenders.append(f"{f.relative_to(root)}:{node.lineno} {node.name}")
    assert not offenders, "these default to every interface: " + ", ".join(offenders)
