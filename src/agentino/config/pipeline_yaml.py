"""Config — pipeline and gateway building."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from agentino.core.agent import Agent
from agentino.pipeline.core import Pipeline, RouterPipeline, Step
from agentino.transport.gateway import GatewayConfig


def build_pipeline(
    cfg: dict[str, Any],
    agents: dict[str, Agent],
) -> Pipeline | RouterPipeline:
    """Build a Pipeline or RouterPipeline from config."""
    pipeline_type = cfg.get("type", "sequence")

    if pipeline_type == "router":
        router_name = cfg["router"]
        if router_name not in agents:
            raise ValueError(f"Router agent '{router_name}' not found")
        routes = {}
        for intent, agent_name in cfg.get("routes", {}).items():
            if agent_name not in agents:
                raise ValueError(f"Route agent '{agent_name}' not found")
            routes[intent] = agents[agent_name]
        return RouterPipeline(
            router=agents[router_name],
            routes=routes,
            default=cfg.get("default"),
        )

    # Sequence pipeline
    steps: list[Step] = []
    for step_cfg in cfg.get("steps", []):
        agent_name = step_cfg["agent"]
        if agent_name not in agents:
            raise ValueError(f"Step agent '{agent_name}' not found")
        condition = None
        condition_str = step_cfg.get("condition")
        if condition_str:
            condition = build_condition(condition_str)
        steps.append(
            Step(
                name=step_cfg.get("name", agent_name),
                agent=agents[agent_name],
                message=step_cfg.get("message", ""),
                condition=condition,
            )
        )
    return Pipeline(steps)


def build_condition(condition_str: str) -> Callable[[dict[str, str]], bool]:
    """Build a condition function from a simple expression string.

    Patterns: "always", "failure", "{step} contains 'text'"
    """
    if condition_str.strip().lower() == "always":
        return lambda ctx: True

    match = re.match(r"\{(\w+)\}\s+contains\s+['\"](.+?)['\"]", condition_str)
    if match:
        step_name, text = match.groups()
        return lambda ctx, s=step_name, t=text: t.lower() in ctx.get(s, "").lower()

    keyword = condition_str.strip().strip("'\"")
    return lambda ctx, k=keyword: any(k.lower() in v.lower() for v in ctx.values())


def build_gateway_config(cfg: dict[str, Any]) -> GatewayConfig:
    """Build GatewayConfig from the gateway section of agents.yml."""
    session_dir = (
        cfg.pop("session_dir", "./sessions")
        if isinstance(cfg.get("session_dir"), str)
        else "./sessions"
    )
    commands = cfg.pop("commands", "") if isinstance(cfg.get("commands"), str) else ""
    chat_history = int(cfg.pop("chat_history", 10)) if "chat_history" in cfg else 10

    channels: dict[str, list[dict[str, Any]]] = {}
    for key, value in cfg.items():
        if key in ("session_dir", "commands"):
            continue
        if isinstance(value, list):
            channels[key] = [dict(v) for v in value if isinstance(v, dict)]
        elif isinstance(value, dict):
            channels[key] = [dict(value)]

    return GatewayConfig(
        channels=channels, session_dir=session_dir, commands=commands, chat_history=chat_history
    )
