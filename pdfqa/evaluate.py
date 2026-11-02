"""Measure how well pdfqa answers, on a test set of questions with known answers.

A question in the test set (a JSON file) says three things:

- `question`: what to ask.
- `answer`: patterns (regular expressions, case ignored) that a right answer must all match, such as
  `["mask", "next sentence"]` for BERT's two pre-training tasks. `null` marks a question the documents
  don't answer, where the right reply is "The documents don't say."
- `evidence`: patterns a passage must all match to hold the answer, such as `["3\\.5 days"]`.

Whether a passage holds the answer is decided by its text, not its page, so the same test set can compare
chunk sizes: a passage counts if it contains the evidence however the pages were split.

Two things are measured:

- **Search**: where the first passage holding the answer ranks. Recall@k is the share of questions with one
  in the top k; MRR (mean reciprocal rank) averages 1/rank, so first place scores 1, second 0.5, and so on.
- **Answers**: right (all answer patterns match, or "not found" for unanswerable questions); *supported*
  (one of the cited passages holds the answer); and *citation precision* (the share of all citations that
  point at a passage holding the answer).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from pdfqa.answer import Answer, Done, ask
from pdfqa.generate import Writer
from pdfqa.search import VectorIndex, search

DEPTH = 10  # how far down the results to look for the answer; MRR counts ranks up to here


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: tuple[str, ...]  # empty when the documents don't answer it
    evidence: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.answer)

    def holds_answer(self, text: str) -> bool:
        """Whether a passage contains the answer."""
        return self.answerable and all(re.search(p, text, re.IGNORECASE) for p in self.evidence)

    def is_right(self, answer: Answer) -> bool:
        if not self.answerable:
            return not answer.found
        return answer.found and all(re.search(p, answer.text, re.IGNORECASE) for p in self.answer)


def load_questions(path: str | Path) -> list[Question]:
    """Read and check a test set. Raises ValueError naming the problem."""
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Can't read the test set {path}: {error}") from error
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} should hold a list of questions.")
    questions, seen = [], set()
    for number, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {number} in {path} isn't an object with a question.")
        name = str(item.get("id") or f"question {number}")
        if not str(item.get("question") or "").strip():
            raise ValueError(f"{name} in {path} has no question.")
        if name in seen:
            raise ValueError(f"Two questions in {path} have the id {name!r}.")
        seen.add(name)
        answer, evidence = tuple(item.get("answer") or ()), tuple(item.get("evidence") or ())
        if answer and not evidence:
            raise ValueError(f"{name} in {path} has an answer but no evidence to find it by.")
        for pattern in answer + evidence:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"{name} in {path} has a broken pattern {pattern!r}: {error}.") from error
        questions.append(Question(name, item["question"].strip(), answer, evidence))
    return questions


# Search


@dataclass(frozen=True)
class Found:
    question: Question
    rank: int | None  # where the first passage holding the answer came, 1 is best; None if not in the top DEPTH


def find(
    conn: sqlite3.Connection, questions: Iterable[Question], *, index: VectorIndex | None, mode: str
) -> list[Found]:
    """Search for each answerable question and note where the answer first appears."""
    results = []
    for question in questions:
        if not question.answerable:
            continue
        hits = search(conn, question.question, index=index, mode=mode, limit=DEPTH)
        rank = next((n for n, hit in enumerate(hits, start=1) if question.holds_answer(hit.chunk.text)), None)
        results.append(Found(question, rank))
    return results


def recall(found: list[Found], k: int) -> float:
    return sum(1 for f in found if f.rank is not None and f.rank <= k) / len(found) if found else 0.0


def mrr(found: list[Found]) -> float:
    return sum(1 / f.rank for f in found if f.rank) / len(found) if found else 0.0


# Answers


@dataclass(frozen=True)
class Scored:
    question: Question
    answer: Answer
    right: bool
    retrieved: bool  # one of the passages the model was given holds the answer (answerable questions only)
    supported: bool  # one of the cited passages holds the answer (answerable questions only)
    citations: int
    good_citations: int  # citations pointing at a passage that holds the answer


def score(question: Question, answer: Answer) -> Scored:
    cited = [answer.sources[n - 1].chunk.text for n in answer.cited]
    good = sum(question.holds_answer(text) for text in cited)
    return Scored(
        question=question,
        answer=answer,
        right=question.is_right(answer),
        retrieved=any(question.holds_answer(hit.chunk.text) for hit in answer.sources),
        supported=good > 0,
        citations=len(cited),
        good_citations=good,
    )


def answer_all(
    conn: sqlite3.Connection,
    questions: Iterable[Question],
    *,
    index: VectorIndex | None,
    writer: Writer,
    limit: int = 5,
    progress: Callable[[Scored], None] | None = None,
) -> list[Scored]:
    """Ask every question and score the answers."""
    results = []
    for question in questions:
        events = ask(conn, question.question, index=index, writer=writer, limit=limit)
        scored = score(question, next(e.answer for e in events if isinstance(e, Done)))
        results.append(scored)
        if progress:
            progress(scored)
    return results


@dataclass(frozen=True)
class Summary:
    answerable: int
    right: int  # answerable questions answered rightly
    abstained: int  # answerable questions answered "the documents don't say"
    retrieved: int  # answerable questions whose answer was among the passages given to the model
    supported: int  # right answers with a citation to a passage holding the answer
    unanswerable: int
    declined: int  # unanswerable questions answered "the documents don't say" (right)
    citations: int
    good_citations: int
    seconds: float  # median time per answer

    @property
    def citation_precision(self) -> float:
        return self.good_citations / self.citations if self.citations else 0.0


def summarize(results: list[Scored]) -> Summary:
    answerable = [r for r in results if r.question.answerable]
    unanswerable = [r for r in results if not r.question.answerable]
    times = sorted(r.answer.seconds for r in results)
    middle = len(times) // 2
    median = (times[middle] if len(times) % 2 else (times[middle - 1] + times[middle]) / 2) if times else 0.0
    return Summary(
        answerable=len(answerable),
        right=sum(r.right for r in answerable),
        abstained=sum(not r.answer.found for r in answerable),
        retrieved=sum(r.retrieved for r in answerable),
        supported=sum(r.right and r.supported for r in answerable),
        unanswerable=len(unanswerable),
        declined=sum(r.right for r in unanswerable),
        citations=sum(r.citations for r in answerable),
        good_citations=sum(r.good_citations for r in answerable),
        seconds=median,
    )
