"""The real bge-small-en-v1.5 model. It's a 133 MB download, so these run only when asked:

    PDFQA_REAL_MODEL=1 python -m pytest tests/test_model.py

GitHub Actions runs them on every push, with the model cached between runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pdfqa.embed import BgeSmall

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
