"""Add a PDF to the library: read it, clean it, chunk it and store it."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from pdfqa import library
from pdfqa.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE, check_settings, chunk_pages
from pdfqa.clean import clean_pages
from pdfqa.extract import PdfError, Progress, read_pdf


@dataclass(frozen=True)
class Added:
    document: library.Document
    removed_lines: int
    unreadable_pages: list[int]
    seconds: float


@dataclass(frozen=True)
class AlreadyAdded:
    document: library.Document


def add_pdf(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    title: str | None = None,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    progress: Progress | None = None,
) -> Added | AlreadyAdded:
    check_settings(size, overlap)
    started = time.perf_counter()
    path = Path(path)
    raw = read_pdf(path, progress=progress)

    existing = library.find_by_hash(conn, raw.sha256)
    if existing:
        return AlreadyAdded(existing)
    if not raw.pages:
        raise PdfError(f"{path.name} has no pages.")

    cleaned = clean_pages(raw.pages)
    chunks = chunk_pages(cleaned.pages, size, overlap)
    if not chunks:
        raise PdfError(
            f"{path.name} has no text to read. It's probably scanned images; "
            "text recognition (OCR) isn't supported yet."
        )

    document_id = library.add_document(
        conn,
        title=(title or "").strip() or raw.title or path.stem,
        file_name=path.name,
        sha256=raw.sha256,
        pages=cleaned.pages,
        chunks=chunks,
        chunk_size=size,
        chunk_overlap=overlap,
    )
    return Added(
        document=library.get_document(conn, document_id),
        removed_lines=cleaned.removed_lines,
        unreadable_pages=raw.unreadable_pages,
        seconds=time.perf_counter() - started,
    )


def rechunk(conn: sqlite3.Connection, document_id: int, size: int, overlap: int) -> library.Document:
    """Split a document again with new settings, from its stored pages; the PDF isn't needed."""
    check_settings(size, overlap)
    pages = library.get_pages(conn, document_id)
    library.replace_chunks(conn, document_id, chunk_pages(pages, size, overlap), size, overlap)
    return library.get_document(conn, document_id)
