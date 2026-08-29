"""Built-in knowledge base — auto-indexes markdown files from skills/*/knowledge/.

Hybrid retrieval: TF-IDF (keyword) + optional dense embeddings (semantic).
Persistent storage via SQLite — per-agent DB with hash-based re-indexing.
No required external dependencies — dense embeddings are opt-in via an
OpenAI-compatible embedding provider (e.g. local llama.cpp, OpenAI, any compatible API).

Usage (automatic via config):
    # In agents.yml, just add a skill with knowledge/ dir:
    skills: [qa]     # skills/qa/knowledge/*.md auto-indexed

    # Agentino registers a search_knowledge tool — LLM calls it when needed.
    # Zero token waste on requests that don't need facts.

    # Optional: configure dense embeddings for semantic search
    knowledge:
      embedding_base_url: http://localhost:8120/v1
      embedding_model: nomic-embed-text-v1.5

Persistence:
    # Per-agent SQLite DB at ~/.agentino/knowledge/{agent_name}.db
    # Hash-based re-indexing: only re-parses and re-embeds changed files.
    # Falls back to pure in-memory if agent_name is not provided.

Markdown format (topic sections):
    ## topic_name.language_code
    Factual text here...

    ## pricing.en
    Our plans start at $10/month...

    ## pricing.de
    Unsere Pläne beginnen bei 10€/Monat...
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeEntry:
    """A single fact entry from knowledge files."""

    id: str
    topic: str
    language: str
    text: str


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens (supports Latin, Cyrillic, Greek, CJK)."""
    return [
        t
        for t in re.split(
            r"[^0-9A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\+]+",
            text.lower(),
        )
        if t
    ]


def _to_vector(tokens: list[str]) -> dict[str, float]:
    """Convert token list to normalized TF vector."""
    vec: dict[str, float] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0.0) + 1.0
    n = float(len(tokens) or 1)
    for k in vec:
        vec[k] /= n
    return vec


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _cosine_dense(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# SQLite BLOB serialization for dense vectors
# ---------------------------------------------------------------------------


def _vec_to_blob(vec: list[float]) -> bytes:
    """Serialize a float list to a compact binary BLOB."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    """Deserialize a BLOB back to a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class KnowledgeBase:
    """Hybrid knowledge base with optional SQLite persistence.

    Retrieval modes:
    - TF-IDF only (default, no dependencies)
    - Hybrid: TF-IDF + dense embeddings via OpenAI-compatible API

    Persistence modes:
    - In-memory only (agent_name=None) — re-indexes on every startup
    - SQLite per-agent DB (agent_name set) — hash-based, skips unchanged files

    Parses `## topic.lang` sections and indexes them for retrieval.
    No domain-specific logic — the framework provides the mechanism,
    the app provides the domain knowledge via markdown files and tool descriptions.
    """

    # Retrieval tuning defaults (overridable via constructor or env vars)
    DEFAULT_LANGUAGE_BOOST = 0.03
    DEFAULT_MIN_SCORE = 0.10
    DEFAULT_DENSE_WEIGHT = 0.6  # blend: (1-w)*tfidf + w*dense
    DEFAULT_TOP_K = 3

    def __init__(
        self,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        tool_description: str | None = None,
        agent_name: str | None = None,
        # Retrieval tuning — YAML config → env var → default
        dense_weight: float | None = None,
        language_boost: float | None = None,
        min_score: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.entries: list[KnowledgeEntry] = []
        self.vectors: dict[str, dict[str, float]] = {}

        # Custom tool description (app-specific, not hardcoded)
        self._tool_description = tool_description

        # Retrieval tuning: constructor arg → env var → class default
        self.LANGUAGE_BOOST = (
            language_boost
            if language_boost is not None
            else float(os.environ.get("AGENTINO_LANGUAGE_BOOST", self.DEFAULT_LANGUAGE_BOOST))
        )
        self.MIN_SCORE = (
            min_score
            if min_score is not None
            else float(os.environ.get("AGENTINO_MIN_SCORE", self.DEFAULT_MIN_SCORE))
        )
        self.DENSE_WEIGHT = (
            dense_weight
            if dense_weight is not None
            else float(os.environ.get("AGENTINO_DENSE_WEIGHT", self.DEFAULT_DENSE_WEIGHT))
        )
        self.TOP_K = (
            top_k
            if top_k is not None
            else int(os.environ.get("AGENTINO_TOP_K", self.DEFAULT_TOP_K))
        )

        # Dense embeddings (optional)
        self._embedding_base_url = embedding_base_url or os.environ.get("AGENTINO_EMBEDDING_URL")
        self._embedding_model = embedding_model or os.environ.get("AGENTINO_EMBEDDING_MODEL", "")
        self._embedding_api_key = embedding_api_key or os.environ.get(
            "AGENTINO_EMBEDDING_KEY", "no-key"
        )
        self._dense_vectors: dict[str, list[float]] = {}
        self._embedding_client: Any = None
        self._query_embed_cache: dict[str, list[float]] = {}
        self._query_cache_max: int = 128

        # SQLite persistence (per-agent DB)
        self._agent_name = agent_name
        self._db: sqlite3.Connection | None = None
        if agent_name:
            self._init_db(agent_name)

    # ------------------------------------------------------------------
    # SQLite persistence layer
    # ------------------------------------------------------------------

    def _init_db(self, agent_name: str) -> None:
        """Initialize per-agent SQLite DB at ~/.agentino/knowledge/{agent_name}.db."""
        db_dir = Path.home() / ".agentino" / "knowledge"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / f"{agent_name}.db"
        self._db = sqlite3.connect(str(db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                language TEXT NOT NULL,
                text TEXT NOT NULL,
                source_file TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                tfidf_vector TEXT,
                dense_vector BLOB
            );
            CREATE TABLE IF NOT EXISTS file_hashes (
                source_file TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_source_file
                ON entries(source_file);
        """)
        self._db.commit()

    def _file_hash(self, content: str) -> str:
        """Compute a short hash of file content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_stored_hash(self, source_file: str) -> str | None:
        """Get the stored hash for a source file, or None if not indexed."""
        if not self._db:
            return None
        row = self._db.execute(
            "SELECT file_hash FROM file_hashes WHERE source_file = ?",
            (source_file,),
        ).fetchone()
        return row[0] if row else None

    def _load_entries_from_db(self, source_file: str) -> int:
        """Load cached entries from SQLite into memory. Returns count."""
        if not self._db:
            return 0
        rows = self._db.execute(
            "SELECT id, topic, language, text, tfidf_vector, dense_vector "
            "FROM entries WHERE source_file = ?",
            (source_file,),
        ).fetchall()
        count = 0
        for row in rows:
            entry_id, topic, lang, text, tfidf_json, dense_blob = row
            entry = KnowledgeEntry(id=entry_id, topic=topic, language=lang, text=text)
            self.entries.append(entry)
            if tfidf_json:
                self.vectors[entry_id] = json.loads(tfidf_json)
            if dense_blob:
                self._dense_vectors[entry_id] = _blob_to_vec(dense_blob)
            count += 1
        return count

    def _save_entries_to_db(self, source_file: str, file_hash: str, entry_ids: list[str]) -> None:
        """Persist entries for a source file to SQLite."""
        if not self._db:
            return
        # Remove old entries for this file
        self._db.execute("DELETE FROM entries WHERE source_file = ?", (source_file,))
        # Insert new entries
        for eid in entry_ids:
            entry = next((e for e in self.entries if e.id == eid), None)
            if not entry:
                continue
            tfidf_json = json.dumps(self.vectors.get(eid, {}))
            dense_blob = (
                _vec_to_blob(self._dense_vectors[eid]) if eid in self._dense_vectors else None
            )
            self._db.execute(
                "INSERT OR REPLACE INTO entries (id, topic, language, text, source_file, file_hash, tfidf_vector, dense_vector) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    eid,
                    entry.topic,
                    entry.language,
                    entry.text,
                    source_file,
                    file_hash,
                    tfidf_json,
                    dense_blob,
                ),
            )
        # Update file hash
        self._db.execute(
            "INSERT OR REPLACE INTO file_hashes (source_file, file_hash) VALUES (?, ?)",
            (source_file, file_hash),
        )
        self._db.commit()

    def _save_dense_vectors_to_db(self) -> None:
        """Persist dense vectors to SQLite for entries that have them."""
        if not self._db or not self._dense_vectors:
            return
        for entry_id, vec in self._dense_vectors.items():
            self._db.execute(
                "UPDATE entries SET dense_vector = ? WHERE id = ?",
                (_vec_to_blob(vec), entry_id),
            )
        self._db.commit()

    @staticmethod
    def clear_cache(agent_name: str) -> None:
        """Delete the SQLite DB for an agent (force full re-index on next startup)."""
        db_path = Path.home() / ".agentino" / "knowledge" / f"{agent_name}.db"
        if db_path.exists():
            db_path.unlink()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_file(self, path: Path) -> int:
        """Index a markdown file. Returns number of entries added.

        Parses `## topic_name.lang_code` sections. Language codes are any
        lowercase alphabetic string (not restricted to specific languages).

        With SQLite persistence: skips re-parsing if file hash is unchanged.
        """
        text = path.read_text(encoding="utf-8")
        file_hash = self._file_hash(text)
        source_file = str(path.resolve())

        # Check cache — skip if file unchanged
        if self._db:
            stored_hash = self._get_stored_hash(source_file)
            if stored_hash == file_hash:
                count = self._load_entries_from_db(source_file)
                return count

        # Parse markdown sections
        pattern = re.compile(
            r"^##\s+([a-z_]+)\.([a-z]{2,5})\s*$\n(.*?)(?=^##\s+[a-z_]+\.[a-z]{2,5}\s*$|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        new_entry_ids: list[str] = []
        count = 0
        for match in pattern.finditer(text):
            topic = match.group(1).strip()
            lang = match.group(2).strip()
            body = match.group(3).strip()
            if not body:
                continue

            # Extract #keywords: line — used for TF-IDF indexing but stripped
            # from the display text. This bridges vocabulary gaps between user
            # queries and entry text (e.g., "checkout" → "payment" entry).
            keywords_text = ""
            display_body = body
            kw_match = re.match(r"^#keywords?:\s*(.+)$", body, re.MULTILINE)
            if kw_match:
                keywords_text = kw_match.group(1)
                display_body = body[: kw_match.start()] + body[kw_match.end() :]
                display_body = display_body.strip()

            entry_id = f"{topic}_{lang}"
            entry = KnowledgeEntry(
                id=entry_id,
                topic=topic,
                language=lang,
                text=display_body,
            )
            self.entries.append(entry)
            # Index both the body text AND keywords for TF-IDF matching
            index_text = f"{display_body} {keywords_text}" if keywords_text else display_body
            self.vectors[entry_id] = _to_vector(_tokenize(index_text))
            new_entry_ids.append(entry_id)
            count += 1

        # Persist to SQLite (dense vectors saved later after embedding)
        if self._db and new_entry_ids:
            self._save_entries_to_db(source_file, file_hash, new_entry_ids)

        return count

    def index_directory(self, directory: Path) -> int:
        """Index all *.md files in a directory. Returns total entries.

        With SQLite: unchanged files load from cache, changed files re-index.
        Dense embeddings are only fetched for new/changed entries.
        """
        if not directory.is_dir():
            return 0

        # Track which entries need dense embeddings (new/changed only)
        entries_before = len(self.entries)
        total = 0
        for md_file in sorted(directory.glob("*.md")):
            total += self.index_file(md_file)
        new_entries = self.entries[entries_before:]

        # Build dense embeddings for new entries only
        if new_entries and self._embedding_base_url:
            # Only embed entries that don't already have dense vectors (loaded from cache)
            needs_embedding = [e for e in new_entries if e.id not in self._dense_vectors]
            if needs_embedding:
                self._build_dense_index_incremental(needs_embedding)
                # Persist new dense vectors
                if self._db:
                    self._save_dense_vectors_to_db()

        return total

    # ------------------------------------------------------------------
    # Dense embeddings
    # ------------------------------------------------------------------

    def _build_dense_index(self) -> None:
        """Build dense embedding vectors for all entries using configured provider."""
        self._build_dense_index_incremental(self.entries)

    def _build_dense_index_incremental(self, entries_to_embed: list[KnowledgeEntry]) -> None:
        """Build dense embeddings for a subset of entries."""
        try:
            import httpx
        except ImportError:
            return
        if not self._embedding_base_url or not entries_to_embed:
            return
        try:
            url = f"{self._embedding_base_url.rstrip('/')}/embeddings"
            headers = {"Authorization": f"Bearer {self._embedding_api_key}"}
            embedded_count = 0
            for entry in entries_to_embed:
                try:
                    resp = httpx.post(
                        url,
                        json={"model": self._embedding_model, "input": entry.text},
                        headers=headers,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("data", [])
                    if items:
                        self._dense_vectors[entry.id] = items[0]["embedding"]
                        embedded_count += 1
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).debug(f"Embedding failed for {entry.id}: {e}")
            if embedded_count < len(entries_to_embed):
                import logging

                logging.getLogger(__name__).warning(
                    f"Dense embeddings: {embedded_count}/{len(entries_to_embed)} entries embedded"
                )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Dense embedding indexing failed ({len(entries_to_embed)} entries): {e}"
            )

    def _embed_query(self, text: str) -> list[float] | None:
        """Get dense embedding for a query string (cached)."""
        if not self._embedding_base_url or not self._dense_vectors:
            return None
        # Check cache
        cached = self._query_embed_cache.get(text)
        if cached is not None:
            return cached
        try:
            import httpx

            resp = httpx.post(
                f"{self._embedding_base_url.rstrip('/')}/embeddings",
                json={"model": self._embedding_model, "input": [text]},
                headers={"Authorization": f"Bearer {self._embedding_api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            # Cache with eviction
            if len(self._query_embed_cache) >= self._query_cache_max:
                oldest = next(iter(self._query_embed_cache))
                del self._query_embed_cache[oldest]
            self._query_embed_cache[text] = embedding
            return embedding
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        message: str,
        language: str = "en",
        top_k: int = 3,
    ) -> list[KnowledgeEntry]:
        """Retrieve top-K relevant facts for a user message.

        Hybrid scoring: TF-IDF (keyword) + optional dense embeddings (semantic).
        Language preference applied on top. Deduplicates by topic.
        """
        if not self.entries:
            return []

        qvec = _to_vector(_tokenize(message))
        if not qvec:
            return []

        # Get dense query embedding if available
        query_emb = self._embed_query(message)
        use_dense = query_emb is not None and bool(self._dense_vectors)

        # Score all entries
        scored: list[tuple[float, KnowledgeEntry]] = []
        for entry in self.entries:
            if entry.language not in (language, "en"):
                continue

            # TF-IDF score
            tfidf_score = _cosine(qvec, self.vectors[entry.id])

            # Hybrid: blend TF-IDF + dense
            if use_dense and entry.id in self._dense_vectors:
                dense_score = _cosine_dense(query_emb, self._dense_vectors[entry.id])
                w = self.DENSE_WEIGHT
                score = (1 - w) * tfidf_score + w * dense_score
            else:
                score = tfidf_score

            # Language preference
            if entry.language == language:
                score += self.LANGUAGE_BOOST

            if score >= self.MIN_SCORE:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by topic, prefer entry in requested language
        # Build lookup: topic → entry in requested language (if exists)
        topic_lang_map: dict[str, KnowledgeEntry] = {}
        for entry in self.entries:
            if entry.language == language:
                topic_lang_map[entry.topic] = entry

        seen_topics: set[str] = set()
        result: list[KnowledgeEntry] = []
        for _, entry in scored:
            if entry.topic in seen_topics:
                continue
            # Prefer entry in requested language over highest-scoring language
            preferred = topic_lang_map.get(entry.topic, entry)
            result.append(preferred)
            seen_topics.add(entry.topic)
            if len(result) >= top_k:
                break

        if not result and scored:
            import logging

            top_scores = [f"{s:.3f}" for s, _ in scored[:3]]
            logging.getLogger(__name__).debug(
                f"Knowledge miss: top scores {top_scores} below threshold {self.MIN_SCORE}"
            )

        return result

    def format_facts(self, entries: list[KnowledgeEntry]) -> str:
        """Format retrieved entries for injection into agent context."""
        if not entries:
            return ""
        lines = [f"- {e.text}" for e in entries]
        return (
            "\nRELEVANT FACTS (retrieved from knowledge base):\n"
            + "\n".join(lines)
            + "\n\nFACT USAGE GUIDE:\n"
            "- Use factual lines from above verbatim.\n"
            "- Do not shorten, rewrite, or paraphrase.\n"
            "- You may add minimal connective text in the same language.\n"
        )

    def make_search_tool(self):
        """Create a search_knowledge Tool backed by this knowledge base.

        The LLM calls this tool when it needs factual information,
        instead of facts being injected into every request.
        """
        from agentino.core.tool import Tool

        kb = self  # capture reference

        # Use app-provided description or generic default
        tool_desc = kb._tool_description or (
            "Search the knowledge base for factual information. "
            "Call this when the user asks a question that requires factual knowledge."
        )

        async def search_knowledge(query: str) -> str:
            """Search the knowledge base for relevant facts."""
            from agentino.core import context as ctx

            language = ctx.get_context("language", "en")
            entries = kb.retrieve(query, language=language, top_k=kb.TOP_K)
            if not entries:
                return "No relevant facts found."
            lines = [e.text for e in entries]
            return (
                "FACTS (use verbatim — do not shorten, rewrite, or paraphrase):\n\n"
                + "\n\n".join(lines)
            )

        # Override the function docstring with app-provided description
        search_knowledge.__doc__ = tool_desc

        return Tool(
            name="search_knowledge",
            description=tool_desc,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or topic to search for",
                    },
                },
                "required": ["query"],
            },
            fn=search_knowledge,
        )
