from __future__ import annotations

import pytest
from helpers import LOREM, words_to_lines

from pdfqa import library
from pdfqa.extract import PdfError
from pdfqa.ingest import Added, AlreadyAdded, add_pdf, rechunk


def long_pages(count: int, repeats: int = 4) -> list[list[str]]:
    """Pages of real-looking text; each starts and ends differently, like real pages do."""
    sentences = [s.strip() + "." for s in LOREM.split(".") if s.strip()]
    pages = []
    for n in range(count):
        rotated = sentences[n % len(sentences) :] + sentences[: n % len(sentences)]
        pages.append(words_to_lines(" ".join(rotated * repeats)))
    return pages


def test_adds_a_pdf_with_its_pages_and_chunks(conn, make_pdf):
    path = make_pdf("flex.pdf", long_pages(3), title="Flex Handbook", header="Flex Handbook", footer="{n}")

    result = add_pdf(conn, path, size=60, overlap=10)

    assert isinstance(result, Added)
    doc = result.document
    assert (doc.title, doc.file_name, doc.page_count, doc.chunk_size, doc.chunk_overlap) == (
        "Flex Handbook",
        "flex.pdf",
        3,
        60,
        10,
    )
    assert result.removed_lines == 6
    assert result.unreadable_pages == []
    assert result.seconds >= 0
    stored = library.get_chunks(conn, doc.id)
    assert len(stored) == doc.chunk_count > 3
    assert all(c.word_count <= 60 for c in stored)
    assert stored[0].text.startswith("Assembly takes about two hours")
    assert "Flex Handbook" not in " ".join(c.text for c in stored)


@pytest.mark.parametrize(
    ("given", "metadata", "expected"),
    [
        ("My Title", "Metadata Title", "My Title"),
        ("   ", "Metadata Title", "Metadata Title"),
        (None, None, "handbook"),
    ],
)
def test_title_comes_from_the_caller_then_the_pdf_then_the_file_name(conn, make_pdf, given, metadata, expected):
    path = make_pdf("handbook.pdf", title=metadata)

    assert add_pdf(conn, path, title=given).document.title == expected


def test_adding_the_same_file_again_returns_the_existing_document(conn, make_pdf, tmp_path):
    path = make_pdf()
    first = add_pdf(conn, path)
    copy = tmp_path / "renamed copy.pdf"
    copy.write_bytes(path.read_bytes())

    again = add_pdf(conn, copy)

    assert isinstance(again, AlreadyAdded)
    assert again.document.id == first.document.id
    assert len(library.list_documents(conn)) == 1


def test_a_scanned_pdf_is_refused(conn, make_pdf):
    path = make_pdf("scan.pdf", [[], []])

    with pytest.raises(PdfError, match="scan.pdf has no text to read. It's probably scanned images"):
        add_pdf(conn, path)
    assert library.list_documents(conn) == []


def test_pages_without_text_are_kept_and_reported(conn, make_pdf):
    path = make_pdf(pages=[["Typed page one."], [], ["Typed page three."]])

    doc = add_pdf(conn, path).document

    assert doc.page_count == 3
    assert doc.empty_pages == (2,)


def test_a_pdf_with_no_pages_is_refused(conn, make_pdf, monkeypatch):
    from pdfqa import ingest
    from pdfqa.extract import RawDocument

    monkeypatch.setattr(ingest, "read_pdf", lambda path, progress=None: RawDocument(None, "c" * 64, [], []))

    with pytest.raises(PdfError, match="has no pages"):
        add_pdf(conn, make_pdf())


def test_bad_settings_are_refused_before_reading(conn, tmp_path):
    with pytest.raises(ValueError, match="Overlap must be between"):
        add_pdf(conn, tmp_path / "missing.pdf", size=100, overlap=80)


def test_rechunk_uses_the_stored_pages(conn, make_pdf):
    path = make_pdf(pages=long_pages(2))
    doc = add_pdf(conn, path, size=200, overlap=40).document
    path.unlink()  # the PDF isn't needed any more

    smaller = rechunk(conn, doc.id, size=50, overlap=0)

    assert (smaller.chunk_size, smaller.chunk_overlap) == (50, 0)
    assert smaller.chunk_count > doc.chunk_count
    assert all(c.word_count <= 50 for c in library.get_chunks(conn, doc.id))


def test_rechunk_checks_its_settings(conn, make_pdf):
    doc = add_pdf(conn, make_pdf()).document

    with pytest.raises(ValueError, match="Chunk size must be between"):
        rechunk(conn, doc.id, size=5, overlap=0)
