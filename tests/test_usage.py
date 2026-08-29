"""Tests for usage tracking."""

import json
import tempfile
from pathlib import Path

from agentino.core.message import Event, Usage
from agentino.extras.usage import UsageEntry, UsageTracker, _estimate_cost


class TestUsageEntry:
    def test_to_jsonl(self):
        entry = UsageEntry(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            timestamp=1234567890.0,
            session_id="sess-1",
            agent_name="maria",
        )
        d = entry.to_jsonl()
        assert d["model"] == "gpt-4o"
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["ts"] == 1234567890.0
        assert d["session_id"] == "sess-1"
        assert d["agent_name"] == "maria"

    def test_from_jsonl(self):
        data = {"model": "gpt-4o", "prompt_tokens": 200, "completion_tokens": 80, "ts": 1234.0}
        entry = UsageEntry.from_jsonl(data)
        assert entry.model == "gpt-4o"
        assert entry.total_tokens == 280

    def test_optional_fields_omitted(self):
        entry = UsageEntry(model="gpt-4o", prompt_tokens=10, completion_tokens=5, timestamp=0.0)
        d = entry.to_jsonl()
        assert "session_id" not in d
        assert "agent_name" not in d


class TestEstimateCost:
    def test_known_model(self):
        cost = _estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost is not None
        assert cost == 2.50 + 10.00

    def test_prefix_match(self):
        cost = _estimate_cost("gpt-4o-2024-08-06", 1_000_000, 0)
        assert cost is not None
        assert cost == 2.50

    def test_unknown_model(self):
        cost = _estimate_cost("my-custom-model", 1000, 500)
        assert cost is None

    def test_anthropic_pricing(self):
        cost = _estimate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        assert cost is not None
        assert cost == 3.00 + 15.00


class TestUsageTracker:
    def test_record_and_total(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", Usage(prompt_tokens=100, completion_tokens=50))
        tracker.record("gpt-4o", Usage(prompt_tokens=200, completion_tokens=100))
        assert tracker.total.prompt_tokens == 300
        assert tracker.total.completion_tokens == 150

    def test_by_model(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", Usage(prompt_tokens=100, completion_tokens=50))
        tracker.record("gpt-4o-mini", Usage(prompt_tokens=200, completion_tokens=80))
        by_model = tracker.by_model
        assert "gpt-4o" in by_model
        assert "gpt-4o-mini" in by_model
        assert by_model["gpt-4o"].total_tokens == 150

    def test_cost_estimate(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", Usage(prompt_tokens=1000, completion_tokens=500))
        cost = tracker.cost_estimate
        assert cost > 0

    def test_persist_to_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            tracker = UsageTracker(path=path)
            tracker.record("gpt-4o", Usage(prompt_tokens=100, completion_tokens=50))
            tracker.record("gpt-4o-mini", Usage(prompt_tokens=200, completion_tokens=80))

            # Read raw file
            with open(path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert data["model"] == "gpt-4o"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_from_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write(
                json.dumps(
                    {"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50, "ts": 1.0}
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {"model": "gpt-4o", "prompt_tokens": 200, "completion_tokens": 80, "ts": 2.0}
                )
                + "\n"
            )
            path = f.name

        try:
            tracker = UsageTracker(path=path)
            assert tracker.total.prompt_tokens == 300
            assert len(tracker.entries) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_on_event_callback(self):
        tracker = UsageTracker()
        tracker.bind("gpt-4o")
        event = Event(type="llm_response", usage=Usage(prompt_tokens=50, completion_tokens=20))
        tracker.on_event(event)
        assert tracker.total.total_tokens == 70
        assert tracker.entries[0].model == "gpt-4o"

    def test_ignores_non_usage_events(self):
        tracker = UsageTracker()
        tracker.on_event(Event(type="text", data="hello"))
        tracker.on_event(Event(type="tool_start", name="search"))
        assert len(tracker.entries) == 0

    def test_summary(self):
        tracker = UsageTracker()
        tracker.record("gpt-4o", Usage(prompt_tokens=1000, completion_tokens=500))
        summary = tracker.summary()
        assert "1,000 prompt" in summary
        assert "gpt-4o" in summary
        assert "$" in summary

    def test_session_and_agent_name(self):
        tracker = UsageTracker(session_id="sess-1", agent_name="maria")
        tracker.record("gpt-4o", Usage(prompt_tokens=10, completion_tokens=5))
        assert tracker.entries[0].session_id == "sess-1"
        assert tracker.entries[0].agent_name == "maria"

    def test_empty_tracker(self):
        tracker = UsageTracker()
        assert tracker.total.total_tokens == 0
        assert tracker.by_model == {}
        assert tracker.cost_estimate == 0.0
