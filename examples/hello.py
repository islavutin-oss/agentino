"""The smallest useful agentino program.

Run it with a model endpoint configured:

    export AGENTINO_BASE_URL="https://api.openai.com/v1"
    export AGENTINO_API_KEY="sk-..."
    python examples/hello.py
"""

import asyncio

from agentino import Agent, tool


@tool
def greet(name: str) -> str:
    """Say hello to someone by name."""
    return f"Hello, {name}! Welcome to Agentino."


agent = Agent(
    instructions="You are a friendly greeter. Use the greet tool when asked to say hello.",
    tools=[greet],
)


async def main() -> None:
    print(await agent.run("Say hi to Alex"))


if __name__ == "__main__":
    # Agent.run is a coroutine. Calling it without awaiting returns the
    # coroutine object and never runs the agent — which is exactly what this
    # example used to print.
    asyncio.run(main())
