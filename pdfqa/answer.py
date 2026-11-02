"""Answer a question from the library: find the passages, have a model write from them, cite the pages.

The model sees the question and the five best passages from search, numbered [1] to [5], each with its
document and page. It's told to use only those passages, to end each sentence with the number of the
passage it came from, and to say "The documents don't say." when they don't. Small models follow that
loosely, so the reply is tidied afterwards:

- "Source [2]", "[Source 2]", "[2, 3]" and "[2-3]" all become [2] or [2][3].
- Numbers that don't match a passage are dropped, never shown as a citation to nothing.
- A "The documents don't say." tacked on after a real, cited answer is removed.
- An answer counts as "not found" when every sentence of it says the documents don't cover the question.
- Maths written as LaTeX, which small models do even when told not to, becomes plain text the way the PDFs
  write it: `$\\frac{1}{\\sqrt{d_k}}$` becomes 1/√dk. Only real maths changes: "$5 and $10" stays.

Each [n] maps to a passage, and each passage to its document and printed page numbers, so the citations
are checked links rather than text the model made up.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass

from pdfqa.generate import Writer
from pdfqa.search import Hit, VectorIndex, search

NOT_FOUND = "The documents don't say."
SYSTEM = "You answer questions about documents, using only the numbered sources the user gives you, and you cite them."
INSTRUCTIONS = (
    "Answer the question above in two to four sentences, using only facts from the sources. "
    "After each sentence, put the number of the source it came from in square brackets, for example [2]. "
    "Write plain text: no Markdown, no LaTeX. "
    f"If none of the sources answer the question, say only this sentence instead: {NOT_FOUND}"
)

_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_SOURCE_WORD = re.compile(r"\b[Ss]ources? \[(\d+)\]|\[[Ss]ources? (\d+)\]")
_NUMBER_LIST = re.compile(r"\[(\d+(?:\s*(?:,|and|-|–)\s*\d+)+)\]")
_CITATION = re.compile(r"\[(\d+)\]")
_SAYS_NOTHING = re.compile(
    r"(documents?|sources?|passages?|text)\s+(do not|don't|does not|doesn't)\s+"
    r"(say|mention|contain|provide|include|state|answer|specify|give|list|cover|discuss|describe)"
    r"|\bno (information|mention|details?)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MATH_SPAN = re.compile(r"\$\$?(.+?)\$\$?|\\\((.+?)\\\)", re.DOTALL)
_TEXT_COMMAND = re.compile(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]*)\}")
_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SYMBOL = re.compile(r"\\(cdot|times|approx|leq|geq|le|ge)\b")
_SYMBOLS = {"cdot": "·", "times": "×", "approx": "≈", "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥"}
_SUBSCRIPT = re.compile(r"_\{?([A-Za-z0-9]+)\}?")


@dataclass(frozen=True)
class Answer:
    question: str
    text: str  # with citations like [2]
    sources: list[Hit]  # the passages the model was given; [n] is sources[n - 1]
    cited: list[int]  # the source numbers the text cites, in the order they first appear
    found: bool  # False when the documents don't answer the question
    model: str
    seconds: float


@dataclass(frozen=True)
class Sources:
    """First event: the passages found, before the model starts writing."""

    hits: list[Hit]


@dataclass(frozen=True)
class Text:
    """The next piece of the answer as the model writes it."""

    text: str


@dataclass(frozen=True)
class Done:
    """Last event: the finished, tidied answer."""

    answer: Answer


Event = Sources | Text | Done


def prompt(question: str, hits: list[Hit]) -> str:
    """The message the model gets: the question, the numbered passages, and the question again with the rules."""
    passages = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        pages = (
            f"page {chunk.label_start}"
            if chunk.page_start == chunk.page_end
            else f"pages {chunk.label_start}–{chunk.label_end}"
        )
        passages.append(f"Source [{number}] ({chunk.title}, {pages}):\n{chunk.text}")
    return (
        f"Question: {question}\n\nSources:\n\n" + "\n\n".join(passages) + f"\n\nQuestion: {question}\n\n{INSTRUCTIONS}"
    )


def ask(
    conn: sqlite3.Connection,
    question: str,
    *,
    index: VectorIndex | None,
    writer: Writer,
    limit: int = 5,
    mode: str = "hybrid",
    document_id: int | None = None,
    max_tokens: int = 400,
) -> Iterator[Event]:
    """Search, then stream the answer: Sources, then Text pieces, then Done."""
    started = time.perf_counter()
    question = question.strip()
    hits = search(conn, question, index=index, mode=mode, limit=limit, document_id=document_id)
    yield Sources(hits)
    if not hits:
        yield Done(Answer(question, NOT_FOUND, [], [], False, writer.name, time.perf_counter() - started))
        return
    pieces = []
    for piece in writer.stream(SYSTEM, prompt(question, hits), max_tokens):
        pieces.append(piece)
        yield Text(piece)
    text, cited, found = tidy("".join(pieces), len(hits))
    yield Done(Answer(question, text, hits, cited, found, writer.name, time.perf_counter() - started))


def tidy(text: str, sources: int) -> tuple[str, list[int], bool]:
    """Clean up a reply: normalise citations, drop ones to missing sources. Returns (text, cited, found)."""
    text = _THINKING.sub("", text)
    text = _MATH_SPAN.sub(_plain_span, text)
    text = _commands(text)
    text = _SOURCE_WORD.sub(lambda m: f"[{m.group(1) or m.group(2)}]", text)
    text = _NUMBER_LIST.sub(lambda m: "".join(f"[{n}]" for n in _numbers(m.group(1))), text)
    text = _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= sources else "", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)  # "fact [9]." without the [9] leaves "fact ."
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    cited = list(dict.fromkeys(int(n) for n in _CITATION.findall(text)))
    if cited and text.endswith(NOT_FOUND) and text != NOT_FOUND:
        text = text[: -len(NOT_FOUND)].rstrip()
    if not text:
        return NOT_FOUND, cited, False
    # Not found when every sentence says the documents don't cover it, cited or not ("The sources don't
    # give the cost [2]."). One real sentence is enough to count as an answer.
    found = not all(_SAYS_NOTHING.search(sentence) for sentence in _SENTENCE_END.split(text))
    return text, cited, found


def _plain_span(match: re.Match[str]) -> str:
    """A $...$ or \\(...\\) span as plain text, if it's maths; money like "$5 and $10" is left alone."""
    expression = match.group(1) or match.group(2)
    if not re.search(r"[\\^_]", expression):
        return match.group(0)
    expression = _SUBSCRIPT.sub(r"\1", _commands(expression))
    return expression.replace("{", "").replace("}", "").strip()


def _commands(text: str) -> str:
    """LaTeX commands as plain text. Safe outside maths too: it only touches backslash commands."""
    text = _TEXT_COMMAND.sub(r"\1", text)
    text = _SQRT.sub(lambda m: "√" + _SUBSCRIPT.sub(r"\1", m.group(1)), text)
    text = _FRAC.sub(lambda m: f"{m.group(1)}/{m.group(2)}", text)
    return _SYMBOL.sub(lambda m: _SYMBOLS[m.group(1)], text)


def _numbers(listed: str) -> list[int]:
    """ "2, 3" → [2, 3]; "2-4" → [2, 3, 4]; "2 and 5" → [2, 5]."""
    numbers: list[int] = []
    for part in re.split(r"\s*(?:,|and)\s*", listed):
        if re.fullmatch(r"\d+\s*[-–]\s*\d+", part):
            first, last = (int(n) for n in re.split(r"\s*[-–]\s*", part))
            numbers.extend(range(first, last + 1) if last - first < 10 else [first, last])
        else:
            numbers.append(int(part))
    return numbers
