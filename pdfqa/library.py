"""The library: documents, their cleaned pages, their chunks and the chunks' vectors, in one SQLite file.

Version 2 of the layout adds what search needs: a vector per chunk (tagged with the model that made it) and
a full-text index of the chunks for keyword search. Triggers keep the index in step with the chunks, and a
library made by version 1 is upgraded the first time it's opened.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pdfqa.chunk import Chunk
from pdfqa.clean import Page

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE documents (
    id            INTEGER PRIMARY KEY,
    title         TEXT    NOT NULL CHECK (length(title) > 0),
    file_name     TEXT    NOT NULL,
    sha256        TEXT    NOT NULL UNIQUE CHECK (length(sha256) = 64),
    page_count    INTEGER NOT NULL CHECK (page_count >= 1),
    chunk_size    INTEGER NOT NULL CHECK (chunk_size > 0),
    chunk_overlap INTEGER NOT NULL CHECK (chunk_overlap >= 0 AND chunk_overlap < chunk_size),
    added_at      TEXT    NOT NULL
);

CREATE TABLE pages (
    document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    number      INTEGER NOT NULL CHECK (number >= 1),
    label       TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    PRIMARY KEY (document_id, number)
) WITHOUT ROWID;

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position    INTEGER NOT NULL CHECK (position >= 0),
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL CHECK (page_end >= page_start),
    word_count  INTEGER NOT NULL CHECK (word_count > 0),
    text        TEXT    NOT NULL,
    UNIQUE (document_id, position)
);

-- Finding the chunks on a page is a range check on (page_start, page_end).
CREATE INDEX chunks_by_page ON chunks (document_id, page_start, page_end);
"""

_SCHEMA_2 = """
CREATE TABLE vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
    model    TEXT    NOT NULL,
    vector   BLOB    NOT NULL  -- float32, little-endian
);
CREATE INDEX vectors_by_model ON vectors (model);

-- Keyword search: a full-text index over the chunks' text, ranked with BM25. "porter" matches word endings
-- (credits, credit), "unicode61" folds case and accents.
CREATE VIRTUAL TABLE chunk_words USING fts5 (
    text, content = 'chunks', content_rowid = 'id', tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER chunk_words_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_words (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunk_words_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_words (chunk_words, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER chunk_words_update AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunk_words (chunk_words, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunk_words (rowid, text) VALUES (new.id, new.text);
END;
INSERT INTO chunk_words (chunk_words) VALUES ('rebuild');  -- index chunks made before version 2
"""


class NotFound(Exception):
    pass


@dataclass(frozen=True)
class Document:
    id: int
    title: str
    file_name: str
    sha256: str
    page_count: int
    chunk_size: int
    chunk_overlap: int
    added_at: datetime
    chunk_count: int
    empty_pages: tuple[int, ...]  # pages with no text, usually scanned images


@dataclass(frozen=True)
class StoredChunk:
    id: int
    position: int
    page_start: int
    page_end: int
    word_count: int
    text: str


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the library, creating it (and its folder) the first time."""
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"{path} was made by a newer version of pdfqa. Upgrade pdfqa to open it.")
    try:
        if version == 0:
            conn.executescript(f"BEGIN; {_SCHEMA} {_SCHEMA_2} PRAGMA user_version = {SCHEMA_VERSION}; COMMIT;")
        elif version == 1:
            conn.executescript(f"BEGIN; {_SCHEMA_2} PRAGMA user_version = {SCHEMA_VERSION}; COMMIT;")
    except sqlite3.OperationalError as error:
        if conn.in_transaction:
            conn.rollback()
        conn.close()
        if "fts5" in str(error):
            raise RuntimeError("This Python's SQLite has no full-text search (FTS5), which pdfqa needs.") from error
        raise
    return conn


def add_document(
    conn: sqlite3.Connection,
    *,
    title: str,
    file_name: str,
    sha256: str,
    pages: Sequence[Page],
    chunks: Sequence[Chunk],
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    with conn:
        cursor = conn.execute(
            "INSERT INTO documents (title, file_name, sha256, page_count, chunk_size, chunk_overlap, added_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, file_name, sha256, len(pages), chunk_size, chunk_overlap, _now()),
        )
        document_id = int(cursor.lastrowid)  # type: ignore[arg-type]
        conn.executemany(
            "INSERT INTO pages (document_id, number, label, text) VALUES (?, ?, ?, ?)",
            [(document_id, p.number, p.label, p.text) for p in pages],
        )
        _insert_chunks(conn, document_id, chunks)
    return document_id


def replace_chunks(
    conn: sqlite3.Connection, document_id: int, chunks: Sequence[Chunk], chunk_size: int, chunk_overlap: int
) -> None:
    with conn:
        updated = conn.execute(
            "UPDATE documents SET chunk_size = ?, chunk_overlap = ? WHERE id = ?",
            (chunk_size, chunk_overlap, document_id),
        ).rowcount
        if not updated:
            raise NotFound(document_id)
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        _insert_chunks(conn, document_id, chunks)


def _insert_chunks(conn: sqlite3.Connection, document_id: int, chunks: Sequence[Chunk]) -> None:
    conn.executemany(
        "INSERT INTO chunks (document_id, position, page_start, page_end, word_count, text) VALUES (?, ?, ?, ?, ?, ?)",
        [(document_id, c.position, c.page_start, c.page_end, c.word_count, c.text) for c in chunks],
    )


def remove_document(conn: sqlite3.Connection, document_id: int) -> Document:
    document = get_document(conn, document_id)
    with conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return document


_DOCUMENT_QUERY = """
SELECT d.*,
       (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count,
       (SELECT group_concat(p.number) FROM pages p WHERE p.document_id = d.id AND p.text = '') AS empty_pages
FROM documents d
"""


def get_document(conn: sqlite3.Connection, document_id: int) -> Document:
    row = conn.execute(_DOCUMENT_QUERY + " WHERE d.id = ?", (document_id,)).fetchone()
    if row is None:
        raise NotFound(document_id)
    return _document(row)


def find_by_hash(conn: sqlite3.Connection, sha256: str) -> Document | None:
    row = conn.execute(_DOCUMENT_QUERY + " WHERE d.sha256 = ?", (sha256,)).fetchone()
    return _document(row) if row else None


def list_documents(conn: sqlite3.Connection) -> list[Document]:
    return [_document(row) for row in conn.execute(_DOCUMENT_QUERY + " ORDER BY d.id")]


def get_pages(conn: sqlite3.Connection, document_id: int) -> list[Page]:
    get_document(conn, document_id)
    rows = conn.execute("SELECT number, label, text FROM pages WHERE document_id = ? ORDER BY number", (document_id,))
    return [Page(number=r["number"], label=r["label"], text=r["text"]) for r in rows]


def get_chunks(conn: sqlite3.Connection, document_id: int, page: int | None = None) -> list[StoredChunk]:
    """All of a document's chunks in order, or only those that include text from `page`."""
    get_document(conn, document_id)
    sql = "SELECT id, position, page_start, page_end, word_count, text FROM chunks WHERE document_id = ?"
    params: list[int] = [document_id]
    if page is not None:
        sql += " AND page_start <= ? AND page_end >= ?"
        params += [page, page]
    rows = conn.execute(sql + " ORDER BY position", params)
    return [StoredChunk(**dict(row)) for row in rows]


# Vectors -----------------------------------------------------------------------------------------------


def chunks_without_vectors(
    conn: sqlite3.Connection, model: str, document_id: int | None = None
) -> list[tuple[int, str]]:
    """(id, text) of each chunk with no vector from `model` yet, in document order."""
    sql = (
        "SELECT c.id, c.text FROM chunks c LEFT JOIN vectors v ON v.chunk_id = c.id AND v.model = ?"
        " WHERE v.chunk_id IS NULL"
    )
    params: list[object] = [model]
    if document_id is not None:
        sql += " AND c.document_id = ?"
        params.append(document_id)
    return [(row[0], row[1]) for row in conn.execute(sql + " ORDER BY c.document_id, c.position", params)]


def store_vectors(conn: sqlite3.Connection, model: str, chunk_ids: Sequence[int], vectors: np.ndarray) -> None:
    rows = [(i, model, np.asarray(v, dtype="<f4").tobytes()) for i, v in zip(chunk_ids, vectors, strict=True)]
    with conn:
        conn.executemany("INSERT OR REPLACE INTO vectors (chunk_id, model, vector) VALUES (?, ?, ?)", rows)


def load_vectors(conn: sqlite3.Connection, model: str, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    """Every chunk's vector from `model`: their chunk ids, and a (chunks, dimensions) matrix."""
    rows = conn.execute("SELECT chunk_id, vector FROM vectors WHERE model = ? ORDER BY chunk_id", (model,)).fetchall()
    ids = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
    matrix = np.frombuffer(b"".join(row[1] for row in rows), dtype="<f4").reshape(len(rows), dimensions)
    return ids, matrix


def vector_counts(conn: sqlite3.Connection, model: str) -> dict[int, int]:
    """How many of each document's chunks have a vector from `model`."""
    rows = conn.execute(
        "SELECT c.document_id, count(*) FROM vectors v JOIN chunks c ON c.id = v.chunk_id"
        " WHERE v.model = ? GROUP BY c.document_id",
        (model,),
    )
    return {row[0]: row[1] for row in rows}


# Keyword search ----------------------------------------------------------------------------------------


def keyword_search(
    conn: sqlite3.Connection, match: str, limit: int, document_id: int | None = None
) -> list[tuple[int, float]]:
    """(chunk id, BM25 score) of the chunks matching an FTS5 query, best first. Higher scores are better."""
    sql = (
        "SELECT w.rowid, -bm25(chunk_words) AS score FROM chunk_words w JOIN chunks c ON c.id = w.rowid"
        " WHERE chunk_words MATCH ?"
    )
    params: list[object] = [match]
    if document_id is not None:
        sql += " AND c.document_id = ?"
        params.append(document_id)
    rows = conn.execute(sql + " ORDER BY score DESC, w.rowid LIMIT ?", [*params, limit])
    return [(row[0], row[1]) for row in rows]


@dataclass(frozen=True)
class FoundChunk:
    id: int
    document_id: int
    title: str
    position: int
    page_start: int
    page_end: int
    label_start: str  # the page numbers as printed, e.g. "iv"
    label_end: str
    text: str


def get_found_chunks(conn: sqlite3.Connection, chunk_ids: Iterable[int]) -> dict[int, FoundChunk]:
    """The chunks with these ids, with their document's title and printed page numbers."""
    ids = list(chunk_ids)
    if not ids:
        return {}
    marks = ", ".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT c.id, c.document_id, d.title, c.position, c.page_start, c.page_end, c.text,
               ps.label AS label_start, pe.label AS label_end
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        JOIN pages ps ON ps.document_id = c.document_id AND ps.number = c.page_start
        JOIN pages pe ON pe.document_id = c.document_id AND pe.number = c.page_end
        WHERE c.id IN ({marks})
        """,
        ids,
    )
    return {row["id"]: FoundChunk(**dict(row)) for row in rows}


def _document(row: sqlite3.Row) -> Document:
    empty = row["empty_pages"]
    return Document(
        id=row["id"],
        title=row["title"],
        file_name=row["file_name"],
        sha256=row["sha256"],
        page_count=row["page_count"],
        chunk_size=row["chunk_size"],
        chunk_overlap=row["chunk_overlap"],
        added_at=datetime.fromisoformat(row["added_at"]),
        chunk_count=row["chunk_count"],
        empty_pages=tuple(sorted(int(n) for n in empty.split(","))) if empty else (),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
