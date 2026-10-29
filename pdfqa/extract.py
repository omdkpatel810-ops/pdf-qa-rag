"""Read a PDF file into pages of raw text.

Uses pypdf. On the sample PDFs it was 1.7x faster than pdfminer.six and kept
table rows on one line; pypdfium2 was 14x faster still but broke the UofM
calendar into one letter per line. pypdf is the slow step for long documents,
so pages are read in parallel, one process per CPU core.
"""

from __future__ import annotations

import hashlib
import io
import logging
import multiprocessing
import os
import re
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

Progress = Callable[[int, int], None]

PARALLEL_FROM_PAGES = 48  # below this, starting processes costs more than it saves
_BATCH_PAGES = 16
_MAX_WORKERS = 8


def _quiet_pypdf() -> None:
    # pypdf logs a warning for every small defect in a file, which is noise for people using the CLI.
    logging.getLogger("pypdf").setLevel(logging.ERROR)


_quiet_pypdf()


class PdfError(Exception):
    """A PDF that can't be read. The message is written for people, not developers."""


@dataclass(frozen=True)
class RawPage:
    number: int  # 1-based position in the file
    label: str  # the page number printed in the PDF, like "iv" or "12"
    text: str  # as extracted, before cleaning


@dataclass(frozen=True)
class RawDocument:
    title: str | None
    sha256: str
    pages: list[RawPage]
    unreadable_pages: list[int]  # pages pypdf failed on; they are kept with no text


def read_pdf(path: str | Path, progress: Progress | None = None, workers: int | None = None) -> RawDocument:
    """Read every page's text, the printed page labels and the title from the PDF's metadata.

    `workers` is how many processes read pages at once; by default, one per CPU core.
    """
    path = Path(path)
    data = _read_bytes(path)
    _check_is_pdf(path, data)
    reader = _open(path, data)
    page_count = len(reader.pages)
    labels = _page_labels(reader, page_count)

    workers = min(workers or _available_cpus(), _MAX_WORKERS)
    texts: list[str | None] | None = None
    if workers > 1 and page_count >= PARALLEL_FROM_PAGES:
        try:
            texts = _extract_in_parallel(data, page_count, workers, progress)
        except (OSError, BrokenProcessPool):
            texts = None  # no subprocesses here (some sandboxes forbid them); read them one by one
    if texts is None:
        texts = []
        for index in range(page_count):
            texts.append(_page_text(reader, index))
            if progress:
                progress(index + 1, page_count)

    return RawDocument(
        title=_title(reader),
        sha256=hashlib.sha256(data).hexdigest(),
        pages=[RawPage(number=i + 1, label=labels[i], text=text or "") for i, text in enumerate(texts)],
        unreadable_pages=[i + 1 for i, text in enumerate(texts) if text is None],
    )


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise PdfError(f"No such file: {path}") from None
    except IsADirectoryError:
        raise PdfError(f"{path} is a folder, not a PDF.") from None
    except PermissionError:
        raise PdfError(f"Not allowed to read {path}.") from None


def _check_is_pdf(path: Path, data: bytes) -> None:
    head = data[:1024]
    if b"%PDF-" in head:
        return
    if not data:
        raise PdfError(f"{path.name} is empty.")
    if head.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise PdfError(f"{path.name} isn't a PDF, it's a web page. The download probably went wrong.")
    raise PdfError(f"{path.name} isn't a PDF.")


def _open(path: Path, data: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfError(f"{path.name} is password-protected. Remove the password and try again.")
        len(reader.pages)  # damaged files often fail only here
    except PdfError:
        raise
    except Exception as exc:  # pypdf raises many exception types for damaged files
        raise PdfError(f"{path.name} looks damaged and couldn't be opened ({exc}).") from exc
    return reader


def _page_text(reader: PdfReader, index: int) -> str | None:
    """The page's text, or None if pypdf fails on it: one bad page shouldn't lose the whole document."""
    try:
        return reader.pages[index].extract_text() or ""
    except Exception:
        return None


# Reading pages in parallel ----------------------------------------------------

_worker_reader: PdfReader | None = None


def _start_worker(data: bytes) -> None:
    """Runs once in each worker process: open the PDF there, so each batch only sends page numbers."""
    global _worker_reader
    _quiet_pypdf()
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        reader.decrypt("")
    _worker_reader = reader


def _extract_batch(start: int, stop: int) -> list[str | None]:
    assert _worker_reader is not None, "_start_worker runs first"
    return [_page_text(_worker_reader, index) for index in range(start, stop)]


def _extract_in_parallel(data: bytes, page_count: int, workers: int, progress: Progress | None) -> list[str | None]:
    texts: list[str | None] = [None] * page_count
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=_process_context(), initializer=_start_worker, initargs=(data,)
    ) as pool:
        batches = {
            pool.submit(_extract_batch, start, min(start + _BATCH_PAGES, page_count)): start
            for start in range(0, page_count, _BATCH_PAGES)
        }
        for finished in as_completed(batches):
            start = batches[finished]
            batch = finished.result()
            texts[start : start + len(batch)] = batch
            done += len(batch)
            if progress:
                progress(done, page_count)
    return texts


def _process_context():
    """Start workers from a clean process, never by forking this one.

    By the time a second PDF is read, the embedding model may be running threads in this process, and
    forking a process with threads can deadlock. A fork server (or, on Windows, a fresh interpreter) avoids it.
    """
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects container limits on Linux
    except AttributeError:  # macOS and Windows
        return os.cpu_count() or 1


# Labels and title -------------------------------------------------------------


def _page_labels(reader: PdfReader, page_count: int) -> list[str]:
    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = []
    if len(labels) != page_count:
        labels = [str(n) for n in range(1, page_count + 1)]
    return [label or str(n) for n, label in enumerate(labels, start=1)]


_WORD_PREFIX = re.compile(r"^Microsoft (Word|PowerPoint) - ", re.IGNORECASE)
_FILE_EXTENSION = re.compile(r"\.(docx?|pptx?|pdf|tex)$", re.IGNORECASE)


def _title(reader: PdfReader) -> str | None:
    """The title from the PDF's metadata, unless it's missing or junk like "untitled"."""
    try:
        raw = reader.metadata.title if reader.metadata else None
    except Exception:
        return None
    if not isinstance(raw, str):
        return None
    title = _FILE_EXTENSION.sub("", _WORD_PREFIX.sub("", " ".join(raw.split())))
    if not title or title.lower() in {"untitled", "title", "document"}:
        return None
    return title
