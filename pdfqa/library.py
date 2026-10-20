"""The library: documents, their cleaned pages and their chunks, in one SQLite file."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pdfqa.chunk import Chunk
from pdfqa.clean import Page

SCHEMA_VERSION = 1

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
    if version == 0:
        with conn:
            conn.executescript(_SCHEMA + f"PRAGMA user_version = {SCHEMA_VERSION};")
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
