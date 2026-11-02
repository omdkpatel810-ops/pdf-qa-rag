"""The real models: bge-small-en-v1.5 (133 MB) and Qwen3-1.7B (2.1 GB). They're downloads, so these run
only when asked:

    PDFQA_REAL_MODEL=1 python -m pytest tests/test_model.py

GitHub Actions runs them on every push, with the models cached between runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pdfqa.answer import SYSTEM, prompt, tidy
from pdfqa.embed import BgeSmall
from pdfqa.generate import LocalModel
from pdfqa.library import FoundChunk
from pdfqa.search import Hit

pytestmark = pytest.mark.skipif(
    not os.environ.get("PDFQA_REAL_MODEL"), reason="set PDFQA_REAL_MODEL=1 to download the model"
)

PASSAGES = [
    "Students must complete 120 credit hours to graduate with a Bachelor of Science major.",
    "COMP 1012 teaches programming for scientists and engineers using Python.",
    "Assembly takes about two hours for a large wardrobe.",
    "BERT is pre-trained with a masked language model and next sentence prediction.",
]
# The first numbers of the first passage's vector, from fastembed (Qdrant's independent implementation).
REFERENCE = [0.06086, 0.04955, -0.0307, -0.05463, 0.01702, -0.03839]


@pytest.fixture(scope="module")
def model() -> BgeSmall:
    return BgeSmall()


def test_matches_an_independent_implementation(model):
    vectors = model.passages(PASSAGES)
    assert vectors.shape == (4, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert np.allclose(vectors[0, :6], REFERENCE, atol=2e-4)


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("How many credits do I need for a science degree?", 0),
        ("Which course teaches coding?", 1),  # no words in common with the answer
        ("How long does it take to put a closet together?", 2),
        ("How was the language model trained?", 3),
    ],
)
def test_questions_find_the_passage_that_answers_them(model, question, answer):
    scores = model.passages(PASSAGES) @ model.query(question)
    assert int(np.argmax(scores)) == answer


# Writing answers


@pytest.fixture(scope="module")
def writer() -> LocalModel:
    return LocalModel()


def answer(writer: LocalModel, question: str) -> tuple[str, list[int], bool]:
    hits = [
        Hit(FoundChunk(i, 1, "notes", i, i + 1, i + 1, str(i + 1), str(i + 1), text), 1.0)
        for i, text in enumerate(PASSAGES)
    ]
    return tidy("".join(writer.stream(SYSTEM, prompt(question, hits), 150)), len(hits))


def test_answers_from_the_sources_and_cites_them(writer):
    text, cited, found = answer(writer, "What are the two tasks BERT is pre-trained on?")
    assert found and 4 in cited, text
    assert "masked" in text.lower() and "next sentence" in text.lower(), text


def test_says_when_the_sources_dont_answer(writer):
    text, cited, found = answer(writer, "How much does a parking pass cost on campus?")
    assert not found and not cited, text
