"""Pipeline — async multi-agent sequence and router patterns."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentino.core.agent import Agent


# ---------------------------------------------------------------------------
# Step definition
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """A single step in a pipeline."""

    name: str
    agent: Agent
    message: str | Callable[[dict[str, str]], str]
    condition: Callable[[dict[str, str]], bool] | None = None


# ---------------------------------------------------------------------------
# Pipeline — sequence of agents
# ---------------------------------------------------------------------------


class Pipeline:
    """Chain agents in sequence, with each step seeing previous results.

    Usage:
        pipeline = Pipeline([
            Step("check", checker, "Check CI status"),
            Step("report", reporter, lambda ctx: f"Write report: {ctx['check']}"),
            Step("notify", notifier, lambda ctx: f"Send: {ctx['report']}"),
        ])
        result = await pipeline.run()
    """

    def __init__(self, steps: list[Step | tuple]) -> None:
        self.steps: list[Step] = []
        for s in steps:
            if isinstance(s, Step):
                self.steps.append(s)
            elif isinstance(s, tuple):
                self.steps.append(self._tuple_to_step(s))
            else:
                raise TypeError(f"Expected Step or tuple, got {type(s)}")

    @staticmethod
    def _tuple_to_step(t: tuple) -> Step:
        if len(t) == 3:
            return Step(name=t[0], agent=t[1], message=t[2])
        elif len(t) == 4:
            return Step(name=t[0], agent=t[1], message=t[2], condition=t[3])
        raise ValueError(f"Step tuple must have 3 or 4 elements, got {len(t)}")

    async def run(self, initial_message: str | None = None) -> dict[str, str]:
        """Execute all steps in sequence. Returns {step_name: result}."""
        context: dict[str, str] = {}
        if initial_message:
            context["input"] = initial_message

        for step in self.steps:
            if step.condition and not step.condition(context):
                continue

            if callable(step.message):
                msg = step.message(context)
            else:
                msg = step.message
                if context:
                    last_key = list(context.keys())[-1]
                    msg = msg.replace("{previous}", context.get(last_key, ""))
                    for key, value in context.items():
                        msg = msg.replace(f"{{{key}}}", value)

            result = await step.agent.run(msg)
            context[step.name] = result

        return context

    @classmethod
    def router(
        cls,
        router: Agent,
        routes: dict[str, Agent],
        default: str | None = None,
        router_message: str | None = None,
    ) -> RouterPipeline:
        """Create a router pipeline that classifies intent and delegates."""
        return RouterPipeline(
            router=router, routes=routes, default=default, router_message=router_message
        )


# ---------------------------------------------------------------------------
# Router — intent classification → specialist delegation
# ---------------------------------------------------------------------------


class RouterPipeline:
    """Routes messages to specialist agents based on intent classification."""

    def __init__(
        self,
        router: Agent,
        routes: dict[str, Agent],
        default: str | None = None,
        router_message: str | None = None,
    ):
        self.router = router
        self.routes = routes
        self.default = default
        self.router_message = router_message

    async def run(self, message: str) -> str:
        """Classify intent and route to the appropriate agent."""
        prompt = self.router_message or message
        classification = await self.router.run(
            prompt if prompt == message else f"{prompt}\n\nUser message: {message}"
        )
        intent = classification.strip().lower()

        agent = self._resolve_route(intent)
        return await agent.run(message)

    def _resolve_route(self, intent: str) -> Agent:
        if intent in self.routes:
            return self.routes[intent]

        for key, agent in self.routes.items():
            if key in intent:
                return agent

        if self.default and self.default in self.routes:
            return self.routes[self.default]

        return next(iter(self.routes.values()))

    async def classify(self, message: str) -> str:
        """Just classify without running the specialist."""
        prompt = self.router_message or message
        result = await self.router.run(
            prompt if prompt == message else f"{prompt}\n\nUser message: {message}"
        )
        return result.strip().lower()


# ---------------------------------------------------------------------------
# Parallel pipeline — run multiple agents concurrently
# ---------------------------------------------------------------------------


class ParallelPipeline:
    """Run multiple agents in parallel on the same input.

    Returns a dict of {name: result} for each agent.
    Now truly parallel thanks to async.
    """

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents

    async def run(self, message: str) -> dict[str, str]:
        """Run all agents concurrently on the same message."""
        tasks = {
            name: asyncio.create_task(agent.run(message)) for name, agent in self.agents.items()
        }
        results: dict[str, str] = {}
        for name, task in tasks.items():
            results[name] = await task
        return results
