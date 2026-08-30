"""Tests for the knowledge base — tokenization, TF-IDF, retrieval, SQLite persistence."""

import pytest

from agentino.extras.knowledge import (
    KnowledgeBase,
    KnowledgeEntry,
    _blob_to_vec,
    _cosine,
    _cosine_dense,
    _to_vector,
    _tokenize,
    _vec_to_blob,
)

# -------------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------------


class TestTokenize:
    def test_basic_latin(self):
        tokens = _tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_non_ascii_is_lowercased(self):
        tokens = _tokenize("Γειά Κόσμε")
        assert "γειά" in tokens
        assert "κόσμε" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = _tokenize("hello, world! foo-bar")
        assert "hello" in tokens
        assert "world" in tokens
        # foo and bar are separate tokens (hyphen is a separator)
        assert "foo" in tokens
        assert "bar" in tokens

    def test_numbers_preserved(self):
        tokens = _tokenize("version 3 costs 10")
        assert "3" in tokens
        assert "10" in tokens


# -------------------------------------------------------------------
# TF vectors & cosine similarity
# -------------------------------------------------------------------


class TestVectors:
    def test_to_vector_normalized(self):
        vec = _to_vector(["a", "b", "a"])
        assert abs(vec["a"] - 2 / 3) < 1e-6
        assert abs(vec["b"] - 1 / 3) < 1e-6

    def test_to_vector_empty(self):
        vec = _to_vector([])
        assert vec == {}

    def test_cosine_identical(self):
        v = {"a": 0.5, "b": 0.5}
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert abs(_cosine(a, b)) < 1e-6

    def test_cosine_empty(self):
        assert _cosine({}, {"a": 1.0}) == 0.0
        assert _cosine({"a": 1.0}, {}) == 0.0


class TestDenseVectors:
    def test_blob_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = _vec_to_blob(vec)
        restored = _blob_to_vec(blob)
        assert len(restored) == 5
        for a, b in zip(vec, restored):
            assert abs(a - b) < 1e-6

    def test_cosine_dense_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_dense(v, v) - 1.0) < 1e-6

    def test_cosine_dense_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_dense(a, b)) < 1e-6

    def test_cosine_dense_mismatched_length(self):
        assert _cosine_dense([1.0], [1.0, 2.0]) == 0.0

    def test_cosine_dense_empty(self):
        assert _cosine_dense([], []) == 0.0


# -------------------------------------------------------------------
# KnowledgeBase — indexing & retrieval (in-memory, no agent_name)
# -------------------------------------------------------------------


class TestIndexing:
    def test_index_single_file(self, tmp_path):
        md = tmp_path / "facts.md"
        md.write_text("## pricing.en\nOur plans start at $10/month.\n")
        kb = KnowledgeBase()
        count = kb.index_file(md)
        assert count == 1
        assert len(kb.entries) == 1
        assert kb.entries[0].topic == "pricing"
        assert kb.entries[0].language == "en"
        assert "$10/month" in kb.entries[0].text

    def test_index_multiple_sections(self, tmp_path):
        md = tmp_path / "facts.md"
        md.write_text(
            "## pricing.en\nPlans start at $10.\n\n"
            "## pricing.de\nPläne beginnen bei 10€.\n\n"
            "## hours.en\nWe are open 9-5.\n"
        )
        kb = KnowledgeBase()
        count = kb.index_file(md)
        assert count == 3
        topics = {e.topic for e in kb.entries}
        assert topics == {"pricing", "hours"}

    def test_index_empty_section_skipped(self, tmp_path):
        md = tmp_path / "facts.md"
        md.write_text("## empty.en\n\n## notempty.en\nSome content.\n")
        kb = KnowledgeBase()
        count = kb.index_file(md)
        assert count == 1
        assert kb.entries[0].topic == "notempty"

    def test_index_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("## topic_a.en\nFact A.\n")
        (tmp_path / "b.md").write_text("## topic_b.en\nFact B.\n")
        (tmp_path / "not_md.txt").write_text("ignored")
        kb = KnowledgeBase()
        total = kb.index_directory(tmp_path)
        assert total == 2

    def test_index_nonexistent_directory(self, tmp_path):
        kb = KnowledgeBase()
        assert kb.index_directory(tmp_path / "nope") == 0

    def test_no_matching_sections(self, tmp_path):
        md = tmp_path / "facts.md"
        md.write_text("# This is a regular markdown file\nNo topic sections here.\n")
        kb = KnowledgeBase()
        count = kb.index_file(md)
        assert count == 0


class TestRetrieval:
    @pytest.fixture()
    def kb(self, tmp_path):
        md = tmp_path / "facts.md"
        md.write_text(
            "## pricing.en\nOur plans start at $10/month for basic.\n\n"
            "## hours.en\nWe are open Monday to Friday 9am to 5pm.\n\n"
            "## pricing.de\nUnsere Pläne beginnen bei 10€ pro Monat.\n"
        )
        kb = KnowledgeBase(min_score=0.0)
        kb.index_file(md)
        return kb

    def test_retrieve_relevant(self, kb):
        results = kb.retrieve("how much does it cost", language="en")
        assert len(results) >= 1
        # pricing should be the top hit for a cost query
        topics = [e.topic for e in results]
        assert "pricing" in topics

    def test_retrieve_language_filter(self, kb):
        results = kb.retrieve("Preis", language="de")
        # Should get German pricing entry (boosted) + English entries
        langs = {e.language for e in results}
        assert "de" in langs

    def test_retrieve_empty_kb(self):
        kb = KnowledgeBase()
        assert kb.retrieve("anything") == []

    def test_retrieve_empty_query(self, kb):
        assert kb.retrieve("") == []

    def test_retrieve_deduplicates_by_topic(self, kb):
        results = kb.retrieve("plans pricing cost", language="en", top_k=10)
        topics = [e.topic for e in results]
        assert len(topics) == len(set(topics)), "Duplicate topics in results"

    def test_retrieve_top_k_limit(self, kb):
        results = kb.retrieve("open hours pricing", language="en", top_k=1)
        assert len(results) <= 1


# -------------------------------------------------------------------
# SQLite persistence
# -------------------------------------------------------------------


class TestSQLitePersistence:
    def test_index_and_reload_from_cache(self, tmp_path, monkeypatch):
        # Use tmp_path for the DB instead of ~/.agentino
        tmp_path / "db"
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        md_dir = tmp_path / "knowledge"
        md_dir.mkdir()
        (md_dir / "facts.md").write_text("## topic_one.en\nFact one text.\n")

        kb1 = KnowledgeBase(agent_name="test_agent")
        count1 = kb1.index_directory(md_dir)
        assert count1 == 1

        # Second load — should hit cache (same hash)
        kb2 = KnowledgeBase(agent_name="test_agent")
        count2 = kb2.index_directory(md_dir)
        assert count2 == 1  # loaded from cache
        assert len(kb2.entries) == 1
        assert kb2.entries[0].text == "Fact one text."

    def test_reindex_on_content_change(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        md_dir = tmp_path / "knowledge"
        md_dir.mkdir()
        md = md_dir / "facts.md"
        md.write_text("## topic.en\nVersion 1.\n")

        kb1 = KnowledgeBase(agent_name="test_reindex")
        kb1.index_directory(md_dir)
        assert kb1.entries[0].text == "Version 1."

        # Change the file
        md.write_text("## topic.en\nVersion 2.\n")

        kb2 = KnowledgeBase(agent_name="test_reindex")
        kb2.index_directory(md_dir)
        assert kb2.entries[0].text == "Version 2."

    def test_clear_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        KnowledgeBase(agent_name="test_clear")
        db_path = tmp_path / ".agentino" / "knowledge" / "test_clear.db"
        assert db_path.exists()

        KnowledgeBase.clear_cache("test_clear")
        assert not db_path.exists()


# -------------------------------------------------------------------
# Search tool generation
# -------------------------------------------------------------------


class TestSearchTool:
    def test_make_search_tool_returns_tool(self, tmp_path):
        md = tmp_path / "facts.md"
        md.write_text("## info.en\nSome useful information.\n")
        kb = KnowledgeBase()
        kb.index_file(md)
        t = kb.make_search_tool()
        assert t.name == "search_knowledge"
        assert "knowledge" in t.description.lower() or "factual" in t.description.lower()

    @pytest.mark.asyncio
    async def test_search_tool_callable(self, tmp_path):
        md = tmp_path / "facts.md"
        md.write_text("## cats.en\nImportant facts about cats and their behavior.\n")
        kb = KnowledgeBase(min_score=0.0)
        kb.index_file(md)
        t = kb.make_search_tool()
        result = await t.execute({"query": "cats"})
        assert "cats" in result

    @pytest.mark.asyncio
    async def test_search_tool_no_results(self):
        kb = KnowledgeBase()
        t = kb.make_search_tool()
        result = await t.execute({"query": "anything"})
        assert "no relevant" in result.lower()

    def test_custom_tool_description(self):
        kb = KnowledgeBase(tool_description="Search wine menu for pairing info.")
        t = kb.make_search_tool()
        assert t.description == "Search wine menu for pairing info."


# -------------------------------------------------------------------
# Format facts
# -------------------------------------------------------------------


class TestFormatFacts:
    def test_format_entries(self):
        entries = [
            KnowledgeEntry(id="a", topic="a", language="en", text="Fact A."),
            KnowledgeEntry(id="b", topic="b", language="en", text="Fact B."),
        ]
        kb = KnowledgeBase()
        text = kb.format_facts(entries)
        assert "Fact A." in text
        assert "Fact B." in text
        assert "RELEVANT FACTS" in text

    def test_format_empty(self):
        kb = KnowledgeBase()
        assert kb.format_facts([]) == ""
