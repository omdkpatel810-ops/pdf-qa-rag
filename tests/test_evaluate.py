"""The test-set format, search metrics and answer scoring, on a small library with stand-in models."""

from __future__ import annotations

import json

import pytest
from helpers import ScriptedWriter, WordHashEmbedder, shelf_library

from pdfqa import evaluate
from pdfqa.answer import Answer
from pdfqa.evaluate import Found, Question, answer_all, find, load_questions, mrr, recall, score, summarize
from pdfqa.search import VectorIndex

QUESTIONS = [
    {
        "id": "wardrobe",
        "question": "How long does wardrobe assembly take?",
        "answer": ["two hours|2 hours"],
        "evidence": ["two hours"],
    },
    {
        "id": "comp",
        "question": "Which course teaches Python?",
        "answer": ["COMP 1012"],
        "evidence": ["COMP 1012", "Python"],
    },
    {"id": "parking", "question": "What does parking cost?", "answer": None, "evidence": []},
]


@pytest.fixture
def embedder():
    return WordHashEmbedder()


@pytest.fixture
def shelf(conn, embedder):
    shelf_library(conn, embedder)
    return conn


@pytest.fixture
def questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(QUESTIONS))
    return load_questions(path)


def test_loading_a_test_set(questions):
    assert [q.id for q in questions] == ["wardrobe", "comp", "parking"]
    assert questions[1] == Question("comp", "Which course teaches Python?", ("COMP 1012",), ("COMP 1012", "Python"))
    assert [q.answerable for q in questions] == [True, True, False]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "Can't read the test set"),
        ("{}", "should hold a list of questions"),
        ("[]", "should hold a list of questions"),
        ('["just text"]', "Question 1 in .* isn't an object"),
        ('[{"id": "a"}]', "a in .* has no question"),
        ('[{"question": "q"}, {"question": "q2", "id": "question 1"}]', "Two questions in .* have the id 'question 1'"),
        ('[{"question": "q", "answer": ["x"]}]', "question 1 in .* has an answer but no evidence"),
        ('[{"question": "q", "answer": ["("], "evidence": ["x"]}]', r"question 1 in .* has a broken pattern '\('"),
    ],
)
def test_bad_test_sets(tmp_path, content, message):
    path = tmp_path / "questions.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_questions(path)


def test_a_missing_test_set(tmp_path):
    with pytest.raises(ValueError, match="Can't read the test set"):
        load_questions(tmp_path / "missing.json")


def answer(text: str, found: bool = True, cited=(), sources=()) -> Answer:
    return Answer("q", text, list(sources), list(cited), found, "test", 1.0)


def test_right_and_wrong_answers(questions):
    wardrobe, comp, parking = questions
    assert wardrobe.is_right(answer("It takes about Two Hours [1]."))  # case doesn't matter
    assert not wardrobe.is_right(answer("Three hours."))
    assert not wardrobe.is_right(answer("Not two hours, the documents don't say.", found=False))
    assert comp.holds_answer("COMP 1012 teaches Python.") and not comp.holds_answer("COMP 1012 teaches C.")
    assert parking.is_right(answer("The documents don't say.", found=False))
    assert not parking.is_right(answer("Parking costs $5."))
    assert not parking.holds_answer("anything")


def test_where_search_finds_each_answer(shelf, embedder, questions):
    found = find(shelf, questions, index=VectorIndex(shelf, embedder), mode="hybrid")
    assert [(f.question.id, f.rank) for f in found] == [("wardrobe", 1), ("comp", 1)]  # unanswerable ones skipped

    nothing = find(shelf, [Question("x", "zebra", ("x",), ("never written",))], index=None, mode="keyword")
    assert nothing[0].rank is None


def test_recall_and_mrr():
    q = Question("q", "q", ("a",), ("a",))
    found = [Found(q, 1), Found(q, 2), Found(q, 4), Found(q, None)]
    assert [recall(found, k) for k in (1, 3, 5)] == [0.25, 0.5, 0.75]
    assert mrr(found) == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert recall([], 5) == mrr([]) == 0.0


def test_answering_and_scoring_every_question(shelf, embedder, questions):
    writer = ScriptedWriter("Assembly takes two hours [1][2].")
    seen = []
    results = answer_all(
        shelf, questions, index=VectorIndex(shelf, embedder), writer=writer, limit=2, progress=seen.append
    )

    assert seen == results and [r.question.id for r in results] == ["wardrobe", "comp", "parking"]
    wardrobe, comp, parking = results
    assert (wardrobe.right, wardrobe.retrieved, wardrobe.supported, wardrobe.citations, wardrobe.good_citations) == (
        True,
        True,
        True,
        2,
        1,
    )
    # The scripted reply is about wardrobes, so it's wrong for COMP 1012, though one passage it cites holds that answer.
    assert (comp.right, comp.retrieved, comp.supported, comp.good_citations) == (False, True, True, 1)
    assert not parking.right  # it answered instead of saying the documents don't say

    total = summarize(results)
    assert (total.answerable, total.right, total.abstained, total.retrieved, total.supported) == (2, 1, 0, 2, 1)
    assert (total.unanswerable, total.declined, total.citations, total.good_citations) == (1, 0, 4, 2)
    assert total.citation_precision == 0.5
    assert total.seconds >= 0


def test_scoring_a_refusal(questions):
    scored = score(questions[2], answer("The documents don't say.", found=False))
    assert scored.right and not scored.supported and scored.citations == 0


def test_summaries_of_odd_and_even_counts(questions):
    def timed(seconds):
        return evaluate.Scored(questions[2], Answer("q", "x", [], [], False, "t", seconds), True, False, False, 0, 0)

    assert summarize([timed(1), timed(5), timed(3)]).seconds == 3
    assert summarize([timed(1), timed(5)]).seconds == 3
    empty = summarize([])
    assert (empty.seconds, empty.citation_precision) == (0.0, 0.0)
