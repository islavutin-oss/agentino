"""Tests for the model registry — Borrow #5."""

from __future__ import annotations

import pytest

from agentino.core.models import (
    ModelCost,
    ModelInfo,
    all_models,
    lookup_model,
    register_model,
)


class TestLookup:
    def test_known_model_resolves(self):
        info = lookup_model("claude-haiku-4-5-20251001")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.context_window == 200_000
        assert info.reasoning is False

    def test_alias_resolves_to_same_entry(self):
        a = lookup_model("claude-sonnet-4-5")
        b = lookup_model("claude-sonnet-4-20250514")
        assert a is b

    def test_unknown_returns_none(self):
        assert lookup_model("does-not-exist-xyz") is None


class TestCostEstimation:
    def test_zero_usage_zero_cost(self):
        cost = ModelCost(input_per_mtok=3.0, output_per_mtok=15.0)
        assert cost.estimate() == 0.0

    def test_basic_input_output_cost(self):
        cost = ModelCost(input_per_mtok=3.0, output_per_mtok=15.0)
        # 1M prompt + 500K completion → 3 + 7.5 = 10.5
        assert cost.estimate(prompt_tokens=1_000_000, completion_tokens=500_000) == pytest.approx(
            10.5
        )

    def test_cache_tokens_priced_separately(self):
        cost = ModelCost(input_per_mtok=3.0, cache_read_per_mtok=0.30, cache_write_per_mtok=3.75)
        # 1M cache_read = $0.30; 1M cache_write = $3.75; 0 fresh input
        out = cost.estimate(cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
        assert out == pytest.approx(4.05)

    def test_haiku_real_world_estimate(self):
        info = lookup_model("claude-haiku-4-5-20251001")
        # 100K input + 5K output → 0.10 + 0.025 = 0.125 USD
        usd = info.cost.estimate(prompt_tokens=100_000, completion_tokens=5_000)
        assert usd == pytest.approx(0.125, abs=1e-6)


class TestRegistration:
    def test_register_overwrites_existing(self):
        custom = ModelInfo(
            id="claude-haiku-4-5-20251001",  # collide on purpose
            provider="anthropic",
            context_window=999_999,
            max_output_tokens=1,
            cost=ModelCost(),
        )
        register_model(custom)
        try:
            assert lookup_model("claude-haiku-4-5-20251001").context_window == 999_999
        finally:
            # Restore real entry so test isolation holds for siblings
            from agentino.core.models import register_model as _reg

            _reg(
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

    def test_all_models_dedupe_aliases(self):
        ids = [m.id for m in all_models()]
        # Sanity: at least the headline models present, and no duplicate ids
        assert "claude-sonnet-4-5" in ids
        assert "gpt-5.3-codex" in ids
        assert len(ids) == len(set(ids))


class TestReasoningFlag:
    def test_haiku_not_reasoning(self):
        assert lookup_model("claude-haiku-4-5-20251001").reasoning is False

    def test_sonnet_reasoning(self):
        assert lookup_model("claude-sonnet-4-5").reasoning is True

    def test_codex_reasoning(self):
        assert lookup_model("gpt-5.3-codex").reasoning is True


class TestExportedFromPackage:
    def test_top_level_import(self):
        from agentino import ModelCost, ModelInfo, lookup_model

        assert lookup_model("gpt-4o") is not None
        assert ModelInfo is not None and ModelCost is not None
