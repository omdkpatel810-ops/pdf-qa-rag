"""Vectors, the keyword index and search, on a small library built directly (no PDFs needed)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from helpers import WordHashEmbedder

from pdfqa import library
from pdfqa.chunk import Chunk
from pdfqa.clean import Page
from pdfqa.ingest import index
from pdfqa.search import VectorIndex, keyword_query, search

SCIENCE = [
    "Students must complete 120 credit hours to graduate with a Bachelor of Science major.",
    "COMP 1012 teaches programming for scientists and engineers using Python.",
    "The Dean's Honour List recognizes students with a grade point average of 3.5 or higher.",
]
FURNITURE = [
    "Assembly takes about two hours for a large wardrobe.",
    "Every quote is free and comes within one business day.",
]


def add(conn, title: str, texts: list[str], sha: str, labels: tuple[str, ...] = ("i", "ii", "1")) -> int:
    pages = [Page(number=n, label=label, text="text") for n, label in enumerate(labels, start=1)]
    chunks = [
        Chunk(position=i, text=text, page_start=min(i + 1, 3), page_end=min(i + 2, 3), word_count=len(text.split()))
        for i, text in enumerate(texts)
    ]
    return library.add_document(
        conn,
        title=title,
        file_name=f"{title}.pdf",
        sha256=sha * 64,
        pages=pages,
        chunks=chunks,
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def embedder() -> WordHashEmbedder:
    return WordHashEmbedder()


@pytest.fixture
def shelf(conn, embedder):
    """Two documents, with vectors."""
    science, furniture = add(conn, "Faculty of Science", SCIENCE, "a"), add(conn, "Assembly", FURNITURE, "b")
    index(conn, embedder)
    return conn, science, furniture


def texts(hits) -> list[str]:
    return [hit.chunk.text for hit in hits]


# Vectors ------------------------------------------------------------------------------------------------


def test_index_makes_each_missing_vector_once(conn, embedder):
    science = add(conn, "Faculty of Science", SCIENCE, "a")
    seen = []

    made = index(conn, embedder, progress=lambda done, total: seen.append((done, total)))

    assert made.chunks == 3 and seen[-1] == (3, 3)
    assert library.vector_counts(conn, embedder.name) == {science: 3}
    assert index(conn, embedder).chunks == 0  # nothing left to do
    assert library.chunks_without_vectors(conn, "another/model") != []  # each model has its own vectors


def test_index_saves_as_it_goes(conn, embedder, monkeypatch):
    monkeypatch.setattr("pdfqa.ingest.EMBED_BATCH", 2)
    add(conn, "Faculty of Science", SCIENCE, "a")
    index(conn, embedder)
    assert embedder.calls == 2  # 3 chunks in batches of 2, each saved before the next


def test_index_one_document(conn, embedder):
    science, furniture = add(conn, "Faculty of Science", SCIENCE, "a"), add(conn, "Assembly", FURNITURE, "b")
    index(conn, embedder, furniture)
    assert library.vector_counts(conn, embedder.name) == {furniture: 2}
    assert [text for _, text in library.chunks_without_vectors(conn, embedder.name, science)] == SCIENCE


def test_vectors_round_trip_exactly(shelf, embedder):
    conn, *_ = shelf
    ids, matrix = library.load_vectors(conn, embedder.name, embedder.dimensions)
    assert matrix.dtype == np.float32 and matrix.shape == (5, 64)
    expected = embedder.passages(SCIENCE + FURNITURE)
    assert np.array_equal(matrix, expected)
    assert ids.tolist() == sorted(ids.tolist())


def test_an_empty_library_loads_no_vectors(conn, embedder):
    ids, matrix = library.load_vectors(conn, embedder.name, embedder.dimensions)
    assert ids.shape == (0,) and matrix.shape == (0, 64)


# Keeping the keyword index and vectors in step -------------------------------------------------------------


def test_removing_a_document_removes_its_words_and_vectors(shelf, embedder):
    conn, science, _ = shelf
    library.remove_document(conn, science)
    assert library.keyword_search(conn, '"credit"', 10) == []
    assert library.vector_counts(conn, embedder.name).keys() == {2}


def test_rechunking_updates_the_index(shelf, embedder):
    conn, science, _ = shelf
    new = [Chunk(position=0, text="Ice hockey is played at Max Bell Centre.", page_start=1, page_end=1, word_count=8)]
    library.replace_chunks(conn, science, new, 200, 40)
    assert library.keyword_search(conn, '"credit"', 10) == []
    assert len(library.keyword_search(conn, '"hockey"', 10)) == 1
    assert library.vector_counts(conn, embedder.name).get(science) is None  # old vectors went with old chunks


def test_editing_a_chunk_updates_the_index(shelf):
    conn, *_ = shelf
    with conn:
        conn.execute("UPDATE chunks SET text = 'Curling season starts in November.' WHERE position = 0")
    assert len(library.keyword_search(conn, '"curling"', 10)) == 2  # position 0 of both documents
    assert library.keyword_search(conn, '"wardrobe"', 10) == []


def test_upgrading_a_version_1_library(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(library._SCHEMA + "PRAGMA user_version = 1;")
    old.execute(
        "INSERT INTO documents VALUES (1, 'Old', 'old.pdf', ?, 1, 200, 40, '2026-10-19T00:00:00+00:00')", ("c" * 64,)
    )
    old.execute("INSERT INTO pages VALUES (1, 1, '1', 'text')")
    old.execute("INSERT INTO chunks VALUES (1, 1, 0, 1, 1, 4, 'Chunks from before search')")
    old.commit()
    old.close()

    conn = library.connect(path)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert [chunk_id for chunk_id, _ in library.keyword_search(conn, '"search"', 5)] == [1]
    assert library.chunks_without_vectors(conn, "any") == [(1, "Chunks from before search")]
    conn.close()


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        ("CREATE VIRTUAL TABLE t USING nofts5 (x);", "has no full-text search"),
        ("CREATE TABLE broken (;", "syntax error"),
    ],
)
def test_a_failed_upgrade_changes_nothing(tmp_path, monkeypatch, schema, error):
    monkeypatch.setattr(library, "_SCHEMA_2", schema)
    with pytest.raises((RuntimeError, sqlite3.OperationalError), match=error):
        library.connect(tmp_path / "library.db")
    check = sqlite3.connect(tmp_path / "library.db")
    assert check.execute("PRAGMA user_version").fetchone()[0] == 0
    assert check.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
    check.close()


# Search ----------------------------------------------------------------------------------------------------


def test_keywords_find_exact_terms(shelf):
    conn, *_ = shelf
    hits = search(conn, "What is COMP 1012?", index=None, mode="keyword", limit=2)
    assert texts(hits)[0] == SCIENCE[1]
    assert hits[0].ranks == {"keyword": 1} and hits[0].score > 0


def test_keywords_match_other_forms_of_a_word(shelf):
    conn, *_ = shelf
    assert texts(search(conn, "credits", index=None, mode="keyword", limit=1)) == [SCIENCE[0]]


def test_meaning_search(shelf, embedder):
    conn, *_ = shelf
    hits = search(conn, "Python programming", index=VectorIndex(conn, embedder), mode="semantic", limit=2)
    assert texts(hits)[0] == SCIENCE[1]
    assert hits[0].score == pytest.approx(float(embedder.query("Python programming") @ embedder.query(SCIENCE[1])))


def test_hybrid_fuses_both_rankings(shelf, embedder):
    conn, *_ = shelf
    hits = search(conn, "free quote", index=VectorIndex(conn, embedder), limit=5)
    assert texts(hits)[0] == FURNITURE[1]
    assert hits[0].ranks == {"semantic": 1, "keyword": 1}
    assert hits[0].score == pytest.approx(2 / 61)
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))


def test_results_carry_their_citation(shelf, embedder):
    conn, science, _ = shelf
    hit = search(conn, "Honour List", index=None, mode="keyword", limit=1)[0]
    assert (hit.chunk.document_id, hit.chunk.title, hit.chunk.position) == (science, "Faculty of Science", 2)
    assert (hit.chunk.page_start, hit.chunk.page_end, hit.chunk.label_start, hit.chunk.label_end) == (3, 3, "1", "1")


def test_search_one_document(shelf, embedder):
    conn, science, furniture = shelf
    for mode in ("keyword", "semantic", "hybrid"):
        hits = search(
            conn, "quote students", index=VectorIndex(conn, embedder), mode=mode, limit=10, document_id=furniture
        )
        assert {hit.chunk.document_id for hit in hits} == {furniture}


def test_asking_for_more_than_there_is(shelf, embedder):
    conn, *_ = shelf
    assert len(search(conn, "students", index=VectorIndex(conn, embedder), mode="semantic", limit=50)) == 5


def test_words_that_match_nothing(shelf):
    conn, *_ = shelf
    assert search(conn, "zeppelin", index=None, mode="keyword") == []
    assert search(conn, "?!", index=None, mode="keyword") == []


def test_keyword_queries_are_safe_from_search_syntax():
    assert keyword_query('NEAR(a b) "quoted" OR -x*') == '"NEAR" OR "a" OR "b" OR "quoted" OR "OR" OR "x"'
    assert keyword_query("...") is None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"mode": "fuzzy"}, "Unknown search mode"),
        ({"question": "   "}, "the search text is empty"),
        ({"limit": 0}, "at least one result"),
        ({"mode": "semantic"}, "No chunks have vectors yet"),
    ],
)
def test_requests_that_cant_be_searched(conn, options, message):
    add(conn, "Faculty of Science", SCIENCE, "a")
    arguments = {"question": "credits", "index": None, **options}
    with pytest.raises(ValueError, match=message):
        search(conn, **arguments)


def test_searching_a_missing_document(shelf):
    conn, *_ = shelf
    with pytest.raises(library.NotFound):
        search(conn, "credits", index=None, mode="keyword", document_id=99)


def test_an_empty_vector_index(conn, embedder):
    assert VectorIndex(conn, embedder).search("anything", 5) == []
    assert library.get_found_chunks(conn, []) == {}
