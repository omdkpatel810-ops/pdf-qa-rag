from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from pdfqa import library
from pdfqa.chunk import Chunk
from pdfqa.clean import Page

SHA = "a" * 64


def pages(*texts: str) -> list[Page]:
    return [Page(number=n, label=str(n), text=text) for n, text in enumerate(texts, start=1)]


def chunks(*spans: tuple[int, int]) -> list[Chunk]:
    return [
        Chunk(position=i, text=f"chunk {i}", page_start=start, page_end=end, word_count=2)
        for i, (start, end) in enumerate(spans)
    ]


def add(conn, sha=SHA, title="Faculty of Science", page_texts=("One.", "", "Three."), spans=((1, 1), (1, 3))):
    return library.add_document(
        conn,
        title=title,
        file_name="science.pdf",
        sha256=sha,
        pages=pages(*page_texts),
        chunks=chunks(*spans),
        chunk_size=200,
        chunk_overlap=40,
    )


def test_creates_the_folder_and_schema(tmp_path):
    path = tmp_path / "nested" / "library.db"

    conn = library.connect(path)

    assert path.exists()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == library.SCHEMA_VERSION
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.close()


def test_reopening_keeps_the_data(tmp_path):
    conn = library.connect(tmp_path / "library.db")
    add(conn)
    conn.close()

    conn = library.connect(tmp_path / "library.db")
    assert [d.title for d in library.list_documents(conn)] == ["Faculty of Science"]
    conn.close()


def test_in_memory_library():
    conn = library.connect(":memory:")
    assert library.list_documents(conn) == []
    conn.close()


def test_refuses_a_library_from_a_newer_version(tmp_path):
    path = tmp_path / "library.db"
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {library.SCHEMA_VERSION + 1}")
    raw.close()

    with pytest.raises(RuntimeError, match="made by a newer version of pdfqa"):
        library.connect(path)


def test_add_and_read_back(conn):
    document_id = add(conn)

    doc = library.get_document(conn, document_id)
    assert (doc.title, doc.file_name, doc.sha256, doc.page_count) == ("Faculty of Science", "science.pdf", SHA, 3)
    assert (doc.chunk_size, doc.chunk_overlap, doc.chunk_count) == (200, 40, 2)
    assert doc.empty_pages == (2,)
    assert doc.added_at.tzinfo == timezone.utc
    assert abs((datetime.now(timezone.utc) - doc.added_at).total_seconds()) < 60

    assert library.get_pages(conn, document_id) == pages("One.", "", "Three.")
    assert [(c.position, c.page_start, c.page_end, c.text) for c in library.get_chunks(conn, document_id)] == [
        (0, 1, 1, "chunk 0"),
        (1, 1, 3, "chunk 1"),
    ]


def test_chunks_for_one_page_include_chunks_that_span_it(conn):
    document_id = add(conn, spans=((1, 1), (1, 3), (3, 3)))

    assert [c.position for c in library.get_chunks(conn, document_id, page=2)] == [1]
    assert [c.position for c in library.get_chunks(conn, document_id, page=3)] == [1, 2]


def test_find_by_hash(conn):
    document_id = add(conn)

    assert library.find_by_hash(conn, SHA).id == document_id
    assert library.find_by_hash(conn, "b" * 64) is None


def test_the_same_file_cannot_be_added_twice(conn):
    add(conn)

    with pytest.raises(sqlite3.IntegrityError):
        add(conn, title="Copy")
    assert len(library.list_documents(conn)) == 1


def test_list_in_the_order_added(conn):
    add(conn, sha="a" * 64, title="First", page_texts=("x",), spans=((1, 1),))
    add(conn, sha="b" * 64, title="Second", page_texts=("y",), spans=((1, 1),))

    assert [(d.title, d.empty_pages) for d in library.list_documents(conn)] == [("First", ()), ("Second", ())]


def test_replace_chunks(conn):
    document_id = add(conn)

    library.replace_chunks(conn, document_id, chunks((1, 3)), chunk_size=300, chunk_overlap=0)

    doc = library.get_document(conn, document_id)
    assert (doc.chunk_size, doc.chunk_overlap, doc.chunk_count) == (300, 0, 1)


def test_remove_deletes_pages_and_chunks_too(conn):
    document_id = add(conn)

    removed = library.remove_document(conn, document_id)

    assert removed.title == "Faculty of Science"
    assert conn.execute("SELECT count(*) FROM pages").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda conn: library.get_document(conn, 99),
        lambda conn: library.get_pages(conn, 99),
        lambda conn: library.get_chunks(conn, 99),
        lambda conn: library.remove_document(conn, 99),
        lambda conn: library.replace_chunks(conn, 99, chunks((1, 1)), 200, 40),
    ],
)
def test_missing_documents(conn, call):
    with pytest.raises(library.NotFound):
        call(conn)


def test_the_schema_rejects_impossible_chunks(conn):
    with pytest.raises(sqlite3.IntegrityError):
        add(conn, spans=((3, 1),))
    assert library.list_documents(conn) == []  # the whole document is rolled back
