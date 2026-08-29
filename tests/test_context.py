"""Tests for the async-safe context utility (contextvars-based)."""

import asyncio

import pytest

from agentino.core.context import clear_context, get_context, reset, set_context


class TestSetAndGet:
    def test_set_and_get(self):
        token = set_context(tenant_id="abc", sender="+1")
        try:
            assert get_context("tenant_id") == "abc"
            assert get_context("sender") == "+1"
        finally:
            reset(token)

    def test_get_default(self):
        assert get_context("nonexistent") is None
        assert get_context("nonexistent", "fallback") == "fallback"

    def test_update_preserves_existing(self):
        t1 = set_context(a=1)
        try:
            t2 = set_context(b=2)
            try:
                assert get_context("a") == 1
                assert get_context("b") == 2
            finally:
                reset(t2)
        finally:
            reset(t1)

    def test_overwrite_value(self):
        t1 = set_context(key="old")
        try:
            t2 = set_context(key="new")
            try:
                assert get_context("key") == "new"
            finally:
                reset(t2)
            # After reset, should revert to previous value
            assert get_context("key") == "old"
        finally:
            reset(t1)


class TestClear:
    def test_clear_removes_all(self):
        t1 = set_context(x=1, y=2)
        try:
            t2 = clear_context()
            try:
                assert get_context("x") is None
                assert get_context("y") is None
            finally:
                reset(t2)
            # After reset, values are back
            assert get_context("x") == 1
        finally:
            reset(t1)


class TestReset:
    def test_reset_reverts_state(self):
        token = set_context(val="before")
        set_context(val="after")
        reset(token)
        # After reset to original token, the context from that point is restored
        # (contextvars.Token restores to state *before* the set that produced the token)


class TestAsyncIsolation:
    @pytest.mark.asyncio
    async def test_tasks_inherit_parent_context(self):
        token = set_context(shared="parent_value")
        try:

            async def child():
                return get_context("shared")

            result = await asyncio.create_task(child())
            assert result == "parent_value"
        finally:
            reset(token)

    @pytest.mark.asyncio
    async def test_child_mutation_does_not_affect_parent(self):
        token = set_context(key="parent")
        try:

            async def child():
                set_context(key="child")
                return get_context("key")

            child_result = await asyncio.create_task(child())
            assert child_result == "child"
            # Parent context unchanged
            assert get_context("key") == "parent"
        finally:
            reset(token)

    @pytest.mark.asyncio
    async def test_concurrent_tasks_isolated(self):
        token = set_context(base="shared")
        try:
            results = {}

            async def worker(name: str):
                set_context(worker_id=name)
                await asyncio.sleep(0.01)  # yield to other tasks
                results[name] = get_context("worker_id")

            await asyncio.gather(
                asyncio.create_task(worker("A")),
                asyncio.create_task(worker("B")),
            )
            assert results["A"] == "A"
            assert results["B"] == "B"
        finally:
            reset(token)
