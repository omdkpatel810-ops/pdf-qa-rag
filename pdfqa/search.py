"""Find the chunks that answer a question: by meaning, by keywords, or both.

- **Meaning** ("semantic"): the question's vector against every chunk's vector. Vectors have length 1, so
  one matrix product gives every cosine similarity. That's exact, not approximate, and takes milliseconds
  for tens of thousands of chunks, so an approximate index (HNSW and the like) isn't worth its complexity yet.
- **Keywords**: BM25 over SQLite's full-text index. It finds exact terms (course codes like COMP 1012, rare
  names, numbers) that a meaning-based model can blur.
- **Both** ("hybrid", the default): each method ranks its best chunks, and reciprocal rank fusion adds
  1 / (60 + rank) from each list. A chunk near the top of either list rises, and one on both rises most.
  Ranks are used rather than scores because BM25 and cosine scores aren't on the same scale.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from pdfqa import library
from pdfqa.embed import Embedder

MODES = ("hybrid", "semantic", "keyword")
RRF_K = 60  # the constant from Cormack, Clarke and Buettcher (2009); it damps the very top ranks
CANDIDATES = 50  # how deep each method's list goes before fusing
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Hit:
    chunk: library.FoundChunk
    score: float  # cosine for "semantic", BM25 for "keyword", fused rank score for "hybrid"
    ranks: dict[str, int] = field(default_factory=dict)  # this chunk's place (1 is best) in each method's list


class VectorIndex:
    """Every chunk's vector for one model, loaded once and searched in memory."""

    def __init__(self, conn: sqlite3.Connection, embedder: Embedder) -> None:
        self.embedder = embedder
        self.ids, self.matrix = library.load_vectors(conn, embedder.name, embedder.dimensions)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, question: str, limit: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        if not len(self.ids):
            return []
        scores = self.matrix @ self.embedder.query(question)
        if allowed is not None:
            scores = np.where(np.isin(self.ids, allowed), scores, -np.inf)
        limit = min(limit, len(scores))
        best = np.argpartition(-scores, limit - 1)[:limit]
        best = best[np.argsort(-scores[best], kind="stable")]
        return [(int(self.ids[i]), float(scores[i])) for i in best if np.isfinite(scores[i])]


def keyword_query(question: str) -> str | None:
    """An FTS5 query matching any of the question's words, each quoted so punctuation can't break it."""
    words = _WORD.findall(question)
    return " OR ".join(f'"{word}"' for word in words) if words else None


def search(
    conn: sqlite3.Connection,
    question: str,
    *,
    index: VectorIndex | None,
    mode: str = "hybrid",
    limit: int = 5,
    document_id: int | None = None,
) -> list[Hit]:
    """The `limit` best chunks for a question, best first. `index` is needed for "semantic" and "hybrid"."""
    if mode not in MODES:
        raise ValueError(f"Unknown search mode {mode!r}. Use one of: {', '.join(MODES)}.")
    if not question.strip():
        raise ValueError("Ask a question: the search text is empty.")
    if limit < 1:
        raise ValueError("Ask for at least one result.")
    allowed = _document_chunks(conn, document_id)
    depth = max(limit, CANDIDATES)

    lists: dict[str, list[tuple[int, float]]] = {}
    if mode in ("semantic", "hybrid"):
        if index is None or not len(index):
            raise ValueError("No chunks have vectors yet, so meaning search can't run. Make them with: pdfqa index")
        lists["semantic"] = index.search(question, depth, allowed)
    if mode in ("keyword", "hybrid"):
        match = keyword_query(question)
        lists["keyword"] = library.keyword_search(conn, match, depth, document_id) if match else []

    ranks = {name: {chunk_id: rank for rank, (chunk_id, _) in enumerate(found, 1)} for name, found in lists.items()}
    if mode == "hybrid":
        fused: dict[int, float] = {}
        for found in ranks.values():
            for chunk_id, rank in found.items():
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
        scored = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]
    else:
        scored = lists[mode][:limit]

    chunks = library.get_found_chunks(conn, (chunk_id for chunk_id, _ in scored))
    return [
        Hit(chunks[chunk_id], score, {name: found[chunk_id] for name, found in ranks.items() if chunk_id in found})
        for chunk_id, score in scored
    ]


def _document_chunks(conn: sqlite3.Connection, document_id: int | None) -> np.ndarray | None:
    if document_id is None:
        return None
    library.get_document(conn, document_id)  # NotFound if there's no such document
    rows = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
    return np.fromiter((row[0] for row in rows), dtype=np.int64)
