"""Answering: the prompt, the stream of events, and tidying the model's citations."""

from __future__ import annotations

import pytest
from helpers import ScriptedWriter, WordHashEmbedder, shelf_library

from pdfqa.answer import INSTRUCTIONS, NOT_FOUND, SYSTEM, Done, Sources, Text, ask, prompt, tidy
from pdfqa.search import VectorIndex, search


@pytest.fixture
def embedder() -> WordHashEmbedder:
    return WordHashEmbedder()


@pytest.fixture
def shelf(conn, embedder):
    shelf_library(conn, embedder)
    return conn


def test_the_prompt_numbers_each_passage_with_its_document_and_page(shelf, embedder):
    hits = search(shelf, "How long does wardrobe assembly take?", index=VectorIndex(shelf, embedder), limit=2)
    text = prompt("How long does wardrobe assembly take?", hits)

    assert text.startswith("Question: How long does wardrobe assembly take?\n\nSources:\n\n")
    assert "Source [1] (Assembly, page i):\nAssembly takes about two hours for a large wardrobe." in text
    assert "Source [2] (" in text and "Source [3]" not in text
    assert text.endswith(f"Question: How long does wardrobe assembly take?\n\n{INSTRUCTIONS}")


def test_a_passage_across_pages_names_both(shelf, embedder):
    hit = search(shelf, "wardrobe", index=None, mode="keyword", limit=1)[0]
    spread = type(hit)(hit.chunk.__class__(**{**hit.chunk.__dict__, "page_end": 2, "label_end": "1"}), hit.score)
    assert "(Assembly, pages i–1)" in prompt("q", [spread])


def test_asking_streams_sources_then_text_then_the_answer(shelf, embedder):
    writer = ScriptedWriter("Assembly takes ", "about two hours ", "Source [1].")
    events = list(
        ask(shelf, "How long does wardrobe assembly take?", index=VectorIndex(shelf, embedder), writer=writer)
    )

    assert isinstance(events[0], Sources) and events[0].hits[0].chunk.title == "Assembly"
    assert [e.text for e in events[1:-1] if isinstance(e, Text)] == [
        "Assembly takes ",
        "about two hours ",
        "Source [1].",
    ]
    answer = events[-1].answer
    assert isinstance(events[-1], Done)
    assert answer.text == "Assembly takes about two hours [1]."
    assert answer.cited == [1] and answer.found
    assert answer.model == "test/scripted" and answer.seconds >= 0
    assert answer.sources == events[0].hits

    system, user, max_tokens = writer.prompts[0]
    assert system == SYSTEM and user.startswith("Question: How long") and max_tokens == 400


def test_asking_with_options(shelf, embedder):
    writer = ScriptedWriter()
    events = list(
        ask(shelf, "  credit hours  ", index=None, writer=writer, mode="keyword", limit=1, document_id=1, max_tokens=50)
    )
    assert events[-1].answer.question == "credit hours"
    assert [hit.chunk.title for hit in events[0].hits] == ["Faculty of Science"]
    assert writer.prompts[0][2] == 50


def test_nothing_found_means_no_model_call(shelf):
    writer = ScriptedWriter()
    events = list(ask(shelf, "zebra", index=None, writer=writer, mode="keyword"))
    assert events[0].hits == []
    answer = events[-1].answer
    assert (answer.text, answer.found, answer.cited) == (NOT_FOUND, False, [])
    assert writer.prompts == []


@pytest.mark.parametrize(
    ("reply", "text", "cited", "found"),
    [
        ("Two hours [1]. Free quotes [2][3].", "Two hours [1]. Free quotes [2][3].", [1, 2, 3], True),
        ("Two hours, as Source [2] says.", "Two hours, as [2] says.", [2], True),
        ("Two hours (sources [2]).", "Two hours ([2]).", [2], True),
        ("Two hours [Source 3].", "Two hours [3].", [3], True),
        ("Two hours [1, 3].", "Two hours [1][3].", [1, 3], True),
        ("Two hours [2 and 4].", "Two hours [2][4].", [2, 4], True),
        ("Two hours [2-4].", "Two hours [2][3][4].", [2, 3, 4], True),
        (
            "Two hours [1–30].",
            "Two hours [1].",
            [1],
            True,
        ),  # a range too wide to list keeps its ends, then 30 is dropped
        ("Two hours [9]. Free quotes [2].", "Two hours. Free quotes [2].", [2], True),  # there's no source 9
        ("Two hours [2]. Then [1] again [2].", "Two hours [2]. Then [1] again [2].", [2, 1], True),
        (f"Two hours [1].\n\n{NOT_FOUND}", "Two hours [1].", [1], True),
        (NOT_FOUND, NOT_FOUND, [], False),
        ("The documents do not mention parking.", "The documents do not mention parking.", [], False),
        ("The sources don't say what parking costs.", "The sources don't say what parking costs.", [], False),
        ("<think>hmm</think>\nTwo hours [1].", "Two hours [1].", [1], True),
        ("  ", NOT_FOUND, [], False),
        ("Two hours, roughly.", "Two hours, roughly.", [], True),  # an answer without citations is still an answer
    ],
)
def test_tidying_replies(reply, text, cited, found):
    assert tidy(reply, sources=5) == (text, cited, found)
