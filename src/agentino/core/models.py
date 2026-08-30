"""Model registry — per-model metadata: cost, context window, max output, reasoning.

Borrow #5 from pi (`packages/ai/src/models.ts`).

Why this exists:
- We hardcode `max_tokens=4096` (or 500/300/400) inline in service code today.
- Cost projection / billing requires knowing per-token prices, which today live
  nowhere structured.
- "Should I switch to a cheaper model?" decisions need the metadata in one place.

Use:
    from agentino.core.models import lookup_model, ModelInfo
    info = lookup_model("gpt-5.4-codex")
    print(info.context_window, info.cost.input_per_mtok)

Registering a custom model:
    from agentino.core.models import register_model, ModelInfo, ModelCost
    register_model(ModelInfo(
        id="my-model",
        provider="openai",
        context_window=128_000,
        max_output_tokens=8_000,
        reasoning=False,
        input_modalities=("text",),
        cost=ModelCost(0.5, 1.5, 0.1, 0.0),
    ))

Cost calculation:
    cost_usd = info.cost.estimate(usage)  # takes an agentino Usage object
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCost:
    """Token costs in USD per million tokens (consistent with Anthropic/OpenAI billing)."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0

    def estimate(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Return USD cost for the given token counts. Pure function — no I/O."""
        return (
            (prompt_tokens / 1_000_000) * self.input_per_mtok
            + (completion_tokens / 1_000_000) * self.output_per_mtok
            + (cache_read_tokens / 1_000_000) * self.cache_read_per_mtok
            + (cache_write_tokens / 1_000_000) * self.cache_write_per_mtok
        )


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata for one model. Frozen so registry entries are not silently mutated."""

    id: str
    provider: str  # "anthropic" | "openai" | "openai-codex" | "openrouter" | ...
    context_window: int
    max_output_tokens: int
    reasoning: bool = False  # True if model supports extended thinking / reasoning effort
    input_modalities: tuple[str, ...] = ("text",)
    cost: ModelCost = field(default_factory=ModelCost)
    aliases: tuple[str, ...] = ()  # alternate ids the user may type


# ----------------------------------------------------------------------
# Built-in registry — populated with the models we actually use today.
# Numbers source: vendor pricing pages as of 2026-05; verify before billing.
# ----------------------------------------------------------------------

_REGISTRY: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo) -> None:
    """Add or replace a model entry. Aliases also resolve to this entry."""
    _REGISTRY[info.id] = info
    for alias in info.aliases:
        _REGISTRY[alias] = info


def lookup_model(model_id: str) -> ModelInfo | None:
    """Resolve a model id (or alias) to its info. Returns None if unknown."""
    return _REGISTRY.get(model_id)


def all_models() -> Iterable[ModelInfo]:
    """Iterate registered models (deduped by id, aliases collapse to the same entry)."""
    seen: set[str] = set()
    for info in _REGISTRY.values():
        if info.id in seen:
            continue
        seen.add(info.id)
        yield info


# Anthropic Claude
register_model(
    ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        context_window=200_000,
        max_output_tokens=64_000,
        reasoning=True,
        input_modalities=("text", "image"),
        cost=ModelCost(
            input_per_mtok=3.0,
            output_per_mtok=15.0,
            cache_read_per_mtok=0.30,
            cache_write_per_mtok=3.75,
        ),
        aliases=("claude-sonnet-4-20250514",),
    )
)
register_model(
    ModelInfo(
        id="claude-haiku-4-5-20251001",
        provider="anthropic",
        context_window=200_000,
        max_output_tokens=8_192,
        reasoning=False,
        input_modalities=("text", "image"),
        cost=ModelCost(
            input_per_mtok=1.0,
            output_per_mtok=5.0,
            cache_read_per_mtok=0.10,
            cache_write_per_mtok=1.25,
        ),
        aliases=("claude-haiku-4-5",),
    )
)
register_model(
    ModelInfo(
        id="claude-opus-4-7",
        provider="anthropic",
        context_window=1_000_000,
        max_output_tokens=64_000,
        reasoning=True,
        input_modalities=("text", "image"),
        cost=ModelCost(
            input_per_mtok=15.0,
            output_per_mtok=75.0,
            cache_read_per_mtok=1.50,
            cache_write_per_mtok=18.75,
        ),
    )
)

# OpenAI / Codex (via Router)
register_model(
    ModelInfo(
        id="gpt-5.4-codex",
        provider="openai-codex",
        context_window=400_000,
        max_output_tokens=16_000,
        reasoning=True,
        input_modalities=("text", "image"),
        cost=ModelCost(),  # Subscription-included; per-token cost N/A
    )
)
register_model(
    ModelInfo(
        id="gpt-5.3-codex",
        provider="openai-codex",
        context_window=400_000,
        max_output_tokens=16_000,
        reasoning=True,
        input_modalities=("text", "image"),
        cost=ModelCost(),  # Subscription-included via ChatGPT Plus; per-token cost N/A
    )
)
register_model(
    ModelInfo(
        id="gpt-5.4",
        provider="openai-codex",
        context_window=400_000,
        max_output_tokens=16_000,
        reasoning=True,
        input_modalities=("text", "image"),
        cost=ModelCost(),
    )
)
register_model(
    ModelInfo(
        id="gpt-4o",
        provider="openai",
        context_window=128_000,
        max_output_tokens=16_384,
        reasoning=False,
        input_modalities=("text", "image"),
        cost=ModelCost(input_per_mtok=2.5, output_per_mtok=10.0),
    )
)
