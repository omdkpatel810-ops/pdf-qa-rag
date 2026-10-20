"""Split a document's pages into overlapping chunks of whole sentences.

Each chunk holds up to `size` words. It starts with the last sentences of the
chunk before it, up to `overlap` words, so a fact that falls on a boundary still
appears whole in one chunk. Once a chunk is at least half full, it ends early at
a paragraph break rather than mixing two topics. Every chunk records the pages
it came from, which later become the answer's citations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pdfqa.clean import BULLET, Page
from pdfqa.sentences import split_sentences

DEFAULT_SIZE = 200
DEFAULT_OVERLAP = 40
MIN_SIZE = 20
MAX_SIZE = 2000
_PARAGRAPH_FILL = 0.5  # how full a chunk must be before it ends early at a paragraph break


@dataclass(frozen=True)
class Chunk:
    position: int  # 0-based order within the document
    text: str
    page_start: int
    page_end: int
    word_count: int


@dataclass(frozen=True)
class _Unit:
    """A sentence, or a piece of one that's too long to fit in a chunk."""

    words: tuple[str, ...]
    page: int
    starts_paragraph: bool


def check_settings(size: int, overlap: int) -> None:
    if not MIN_SIZE <= size <= MAX_SIZE:
        raise ValueError(f"Chunk size must be between {MIN_SIZE} and {MAX_SIZE} words.")
    if not 0 <= overlap <= size // 2:
        raise ValueError(f"Overlap must be between 0 and {size // 2} words (half the chunk size).")


def chunk_pages(pages: Sequence[Page], size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    check_settings(size, overlap)
    units = _units(pages, size, overlap)

    chunks: list[Chunk] = []
    current: list[_Unit] = []
    words = 0
    fresh = 0  # units in `current` that no earlier chunk has

    for unit in units:
        n = len(unit.words)
        paragraph_break = unit.starts_paragraph and words >= size * _PARAGRAPH_FILL
        if fresh and (words + n > size or paragraph_break):
            chunks.append(_make_chunk(len(chunks), current))
            current = _tail(current, overlap)
            words = sum(len(u.words) for u in current)
            if words + n > size:
                current, words = [], 0
            fresh = 0
        current.append(unit)
        words += n
        fresh += 1

    if fresh:
        chunks.append(_make_chunk(len(chunks), current))
    return chunks


def _units(pages: Sequence[Page], size: int, overlap: int) -> list[_Unit]:
    # Overlong sentences (often tables or lists without full stops) are cut into
    # overlap-sized pieces, so the overlap still works across them.
    piece = overlap or size
    units: list[_Unit] = []
    for page in pages:
        for p, paragraph in enumerate(page.text.split("\n\n")):
            for s, sentence in enumerate(split_sentences(paragraph)):
                words = tuple(sentence.split())
                pieces = [words] if len(words) <= size else [words[i : i + piece] for i in range(0, len(words), piece)]
                for k, part in enumerate(pieces):
                    # A page's first paragraph usually continues from the page before,
                    # unless it starts with a bullet.
                    starts = s == 0 and k == 0 and (p > 0 or bool(BULLET.match(paragraph)))
                    units.append(_Unit(words=part, page=page.number, starts_paragraph=starts))
    return units


def _tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """The last whole units, up to `overlap` words."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        total += len(unit.words)
        if total > overlap:
            break
        tail.append(unit)
    return tail[::-1]


def _make_chunk(position: int, units: list[_Unit]) -> Chunk:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i:
            parts.append("\n\n" if unit.starts_paragraph else " ")
        parts.append(" ".join(unit.words))
    return Chunk(
        position=position,
        text="".join(parts),
        page_start=min(u.page for u in units),
        page_end=max(u.page for u in units),
        word_count=sum(len(u.words) for u in units),
    )
