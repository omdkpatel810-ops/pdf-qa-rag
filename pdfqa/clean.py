"""Turn the raw text of each page into clean paragraphs.

PDFs store lines, not paragraphs, so the text needs repair before it's useful:

* ligatures like "ﬁ" and odd spaces are replaced with plain characters;
* headers and footers that repeat on most pages are removed, as is arXiv's
  sideways identifier stamp;
* lines are joined back into paragraphs, undoing end-of-line hyphenation
  ("representa-" + "tion") but keeping real compounds ("task-" + "specific");
* web addresses broken over several lines are put back together.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pdfqa.extract import RawPage

_REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
    "\ufeff": "",  # byte order mark
}
SOFT_HYPHEN = "\u00ad"
_WHITESPACE = re.compile(r"\s+")  # includes no-break, thin and other unicode spaces

_ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s+\[[\w.-]+\]\s+\d{1,2}\s+\w+\s+\d{4}$")
BULLET = re.compile(r"^(?:[•◦▪▫‣⁃●○■□–—*-]|\d{1,2}[.)])\s")
_ENDS_SENTENCE = re.compile(r"[.!?:][\"'”’)\]]*$")
_ENDS_WITH_PUNCTUATION = re.compile(r"[.,;:!?]$")
_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_URL_START = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
# Words are letters in any language, optionally joined by hyphens: "state-of-the-art".
_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_HYPHENATED_END = re.compile(r"([^\W\d_]+(?:-[^\W\d_]+)*)-$")
_LEADING_WORD = re.compile(r"^([^\W\d_]+)(-[^\W\d_])?")

# A line counts as a header or footer if it sits at the top or bottom of at least
# half the pages (and at least 3 of them), ignoring page numbers.
_EDGE_SHARE = 0.5
_EDGE_MIN_PAGES = 3
_EDGE_ROUNDS = 2  # a header can be two lines, e.g. a running title above a page number


@dataclass(frozen=True)
class Page:
    number: int
    label: str
    text: str  # paragraphs separated by a blank line; "" when the page has no text


@dataclass(frozen=True)
class CleanedDocument:
    pages: list[Page]
    removed_lines: int  # repeated headers and footers, and arXiv stamps


def clean_pages(raw_pages: Sequence[RawPage]) -> CleanedDocument:
    lines = [[normalize_line(line) for line in page.text.splitlines()] for page in raw_pages]
    lines, stamps = remove_stamps(lines)
    lines, repeated = remove_repeated_edges(lines)
    vocabulary = build_vocabulary(lines)
    pages = [
        Page(number=raw.number, label=raw.label, text=join_lines(page_lines, vocabulary))
        for raw, page_lines in zip(raw_pages, lines, strict=True)
    ]
    return CleanedDocument(pages=pages, removed_lines=stamps + repeated)


def normalize_line(line: str) -> str:
    """Plain characters and single spaces. Soft hyphens survive here so joining can use them."""
    line = unicodedata.normalize("NFC", line)
    for old, new in _REPLACEMENTS.items():
        line = line.replace(old, new)
    return _WHITESPACE.sub(" ", line).strip()


# Headers, footers and stamps ------------------------------------------------


def remove_stamps(pages: list[list[str]]) -> tuple[list[list[str]], int]:
    """Drop arXiv's identifier stamp, which pypdf reads into the middle of page 1's text."""
    kept = [[line for line in lines if not _ARXIV_STAMP.match(line)] for lines in pages]
    return kept, sum(map(len, pages)) - sum(map(len, kept))


def edge_key(line: str) -> str:
    """What's left of a line once page numbers are ignored, so "Page 3" matches "Page 4"."""
    key = re.sub(r"\d+", "#", line.lower())
    return key.strip(" #.-–—|·•:")


def remove_repeated_edges(pages: list[list[str]]) -> tuple[list[list[str]], int]:
    pages = [list(lines) for lines in pages]
    pages_with_text = sum(1 for lines in pages if any(lines))
    if pages_with_text < _EDGE_MIN_PAGES:
        return pages, 0
    needed = max(_EDGE_MIN_PAGES, math.ceil(pages_with_text * _EDGE_SHARE))

    removed = 0
    for from_top in (True, False):
        for _ in range(_EDGE_ROUNDS):
            edges = [_edge_index(lines, from_top) for lines in pages]
            counts = Counter(edge_key(lines[i]) for lines, i in zip(pages, edges, strict=True) if i is not None)
            repeated = {key for key, count in counts.items() if count >= needed}
            if not repeated:
                break
            for lines, i in zip(pages, edges, strict=True):
                if i is not None and edge_key(lines[i]) in repeated:
                    del lines[i]
                    removed += 1
    return pages, removed


def _edge_index(lines: list[str], from_top: bool) -> int | None:
    """The first (or last) line with text, unless it's the only one: a header never stands alone."""
    if sum(1 for line in lines if line) < 2:
        return None
    order = range(len(lines)) if from_top else range(len(lines) - 1, -1, -1)
    return next(i for i in order if lines[i])


# Joining lines --------------------------------------------------------------


def build_vocabulary(pages: list[list[str]]) -> Counter[str]:
    """How often each word and hyphenated compound appears whole in the document.

    The pieces of a word split across two lines ("representa", "tion") are left
    out, so they never count as words.
    """
    vocabulary: Counter[str] = Counter()
    after_split = False
    for lines in pages:
        for line in lines:
            words = _WORD.findall(line)
            if after_split:
                words = words[1:]
            split = line.endswith(("-", SOFT_HYPHEN))
            if split:
                words = words[:-1]
            vocabulary.update(word.lower() for word in words)
            after_split = split
    return vocabulary


def join_lines(lines: Sequence[str], vocabulary: Counter[str]) -> str:
    """Join a page's lines into paragraphs separated by blank lines."""
    lengths = [len(line) for line in lines if line]
    if not lengths:
        return ""
    typical = statistics.median(lengths)

    paragraphs: list[str] = []
    current = ""
    for i, line in enumerate(lines):
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        current = join_two(current, line, vocabulary) if current else line
        following = lines[i + 1] if i + 1 < len(lines) else ""
        if following and _paragraph_ends(lines, i, typical):
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraph.replace(SOFT_HYPHEN, "") for paragraph in paragraphs)


def join_two(text: str, line: str, vocabulary: Counter[str]) -> str:
    """Append the next line to the text so far, repairing hyphenation and broken web addresses."""
    if _ends_inside_url(text, line):
        return text + line
    if text.endswith(SOFT_HYPHEN):
        return text[:-1] + line
    end = _HYPHENATED_END.search(text)
    start = _LEADING_WORD.match(line)
    if end and start:
        if keeps_hyphen(end.group(1), start.group(1), bool(start.group(2)), vocabulary):
            return text + line
        return text[:-1] + line
    if end and line[0].isdigit():
        return text + line  # "COVID-" + "19"
    return f"{text} {line}"


def keeps_hyphen(head: str, tail: str, tail_is_compound: bool, vocabulary: Counter[str]) -> bool:
    """Whether a hyphen at the end of a line belongs to the word ("task-specific") or only splits it.

    The document is its own dictionary: whichever spelling it uses elsewhere wins.
    When it uses neither, a hyphen is kept inside a longer compound ("left-to-" + "right")
    or between two words the document uses on their own ("sequence-" + "aligned").
    """
    if tail[0].isupper() and not (head.isupper() and tail.isupper()):
        return True  # "non-" + "English", but "INTRO-" + "DUCTION" is checked like any other word
    hyphenated = vocabulary[f"{head}-{tail}".lower()]
    joined = vocabulary[f"{head}{tail}".lower()]
    if hyphenated != joined:
        return hyphenated > joined
    if "-" in head or tail_is_compound:
        return True
    head, tail = head.lower(), tail.lower()
    return len(head) >= 3 and len(tail) >= 3 and vocabulary[head] > 0 and vocabulary[tail] > 0


def _ends_inside_url(text: str, line: str) -> bool:
    last = text.rsplit(" ", 1)[-1].lstrip("(<[\"'")
    if not (_URL_START.match(last) and last[-1] in "/-._=?&#"):
        return False
    first = line.split(" ", 1)[0]
    return any(char in first for char in "/.-_)?=&#")


def _paragraph_ends(lines: Sequence[str], i: int, typical: float) -> bool:
    line, following = lines[i], lines[i + 1]
    if BULLET.match(following):
        return True
    if _ENDS_SENTENCE.search(line) and len(line) < 0.8 * typical:
        return True
    return _is_heading(lines, i, typical) or _is_heading(lines, i + 1, typical)


def _is_heading(lines: Sequence[str], i: int, typical: float) -> bool:
    """A short line without end punctuation, starting a section, like "3.1 Pre-training BERT"."""
    line = lines[i]
    following = lines[i + 1] if i + 1 < len(lines) else ""
    if not line or not following or not following[0].isupper():
        return False
    if len(line) >= 0.7 * typical or len(line.split()) > 10 or _ENDS_WITH_PUNCTUATION.search(line):
        return False
    return line[0].isupper() or bool(_SECTION_NUMBER.match(line))
