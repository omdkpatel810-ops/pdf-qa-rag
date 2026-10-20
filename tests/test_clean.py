from __future__ import annotations

from collections import Counter

import pytest

from pdfqa.clean import (
    SOFT_HYPHEN,
    build_vocabulary,
    clean_pages,
    edge_key,
    join_lines,
    join_two,
    normalize_line,
    remove_repeated_edges,
    remove_stamps,
)
from pdfqa.extract import RawPage, read_pdf


def raw(*pages: str) -> list[RawPage]:
    return [RawPage(number=n, label=str(n), text=text) for n, text in enumerate(pages, start=1)]


# Characters -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ﬁne-tuning with ﬂoating oﬀsets", "fine-tuning with floating offsets"),
        ("Faculty of Science\xa0\xa0\xa0\xa0 3", "Faculty of Science 3"),
        ("  tabs\tand   spaces  ", "tabs and spaces"),
        ("zero​width﻿", "zerowidth"),
        ("pre‐train and non‑stop", "pre-train and non-stop"),
        ("café", "café"),
    ],
)
def test_normalize_line(line, expected):
    assert normalize_line(line) == expected


def test_normalize_keeps_a_soft_hyphen_for_joining():
    assert normalize_line("represen­") == "represen" + SOFT_HYPHEN


# Headers, footers and stamps ------------------------------------------------


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("Faculty of Science 3", "faculty of science"),
        ("4 Faculty of Science", "faculty of science"),
        ("Page 3 of 12", "page # of"),
        ("- 7 -", ""),
        ("12", ""),
    ],
)
def test_edge_key_ignores_page_numbers(line, key):
    assert edge_key(line) == key


BODY = [
    "Wardrobes take two hours.",
    "Desks take one hour.",
    "Sheds take a full day.",
    "Swing sets vary.",
    "Beds are quick.",
]


def test_removes_headers_and_footers_that_repeat_on_most_pages():
    # Odd and even pages put the page number on different sides, as printed books do.
    pages = [
        [f"Faculty of Science {n}" if n % 2 else f"{n} Faculty of Science", BODY[n], f"Page {n} of 4"]
        for n in range(1, 5)
    ]

    cleaned, removed = remove_repeated_edges(pages)

    assert cleaned == [[BODY[n]] for n in range(1, 5)]
    assert removed == 8


def test_removes_two_line_headers():
    pages = [["Course Calendar", "Chapter 2", BODY[n], str(n)] for n in range(1, 4)]

    cleaned, removed = remove_repeated_edges(pages)

    assert cleaned == [[BODY[n]] for n in range(1, 4)]
    assert removed == 9


def test_keeps_edge_lines_that_repeat_on_fewer_than_half_the_pages():
    pages = [["Introduction", "Welcome."], ["Introduction", "Thanks."]] + [
        [BODY[n], f"Footnote {letter}."] for n, letter in zip(range(1, 5), "abcd", strict=True)
    ]

    cleaned, removed = remove_repeated_edges(pages)

    assert removed == 0
    assert cleaned == pages


def test_needs_at_least_three_pages_with_text_to_spot_a_header():
    pages = [["Draft", "One."], ["Draft", "Two."], [], []]

    assert remove_repeated_edges(pages) == (pages, 0)


def test_blank_edge_lines_are_skipped_when_looking_for_headers():
    pages = [["", "Header", BODY[n], ""] for n in range(1, 4)]

    cleaned, removed = remove_repeated_edges(pages)

    assert removed == 3
    assert [[line for line in page if line] for page in cleaned] == [[BODY[n]] for n in range(1, 4)]


def test_never_removes_the_only_line_on_a_page():
    # Pages that are just "Question 1", "Question 2"... look like headers but are all there is.
    pages = [[f"Question {n}"] for n in range(1, 6)]

    assert remove_repeated_edges(pages) == (pages, 0)


def test_removes_the_arxiv_stamp():
    pages = [["the masked", "arXiv:1810.04805v2 [cs.CL] 24 May 2019", "word"], ["arXiv:1810.04805 is a paper id."]]

    cleaned, removed = remove_stamps(pages)

    assert cleaned == [["the masked", "word"], ["arXiv:1810.04805 is a paper id."]]
    assert removed == 1


# Hyphenation ----------------------------------------------------------------


def test_vocabulary_counts_whole_words_but_not_the_pieces_of_split_words():
    vocabulary = build_vocabulary([["a new representa-", "tion of task-specific text", "Plain line."]])

    assert vocabulary["task-specific"] == 1
    assert vocabulary["text"] == 1
    assert vocabulary["plain"] == 1
    assert vocabulary["representa"] == 0
    assert vocabulary["tion"] == 0


@pytest.mark.parametrize(
    ("text", "line", "vocabulary", "expected"),
    [
        # A word split at the line end, which the document never hyphenates.
        ("a new representa-", "tion model", {}, "a new representation model"),
        # The document writes "task-specific" elsewhere, so the hyphen stays.
        ("without task-", "specific changes", {"task-specific": 2}, "without task-specific changes"),
        # The document writes "downstream" elsewhere, so the hyphen goes.
        ("to down-", "stream tasks", {"downstream": 3, "down-stream": 1}, "to downstream tasks"),
        # Neither spelling elsewhere: a longer compound keeps its hyphen.
        ("a left-to-", "right model", {}, "a left-to-right model"),
        ("an English-", "to-German model", {}, "an English-to-German model"),
        # Neither spelling elsewhere: two words the document uses on their own keep it.
        ("instead of sequence-", "aligned RNNs", {"sequence": 5, "aligned": 1}, "instead of sequence-aligned RNNs"),
        # ...unless one is too short to be a word on its own.
        ("it was in-", "spired by", {"in": 9, "spired": 1}, "it was inspired by"),
        # A capital after the hyphen means a real compound.
        ("non-", "English text", {}, "non-English text"),
        # All capitals are judged like any other word.
        ("INTRO-", "DUCTION", {}, "INTRODUCTION"),
        ("COVID-", "19 cases", {}, "COVID-19 cases"),
        # A soft hyphen always means the word was split.
        ("represen" + SOFT_HYPHEN, "tation", {}, "representation"),
        # A dash between words isn't a hyphen.
        ("one -", "two", {}, "one - two"),
        ("plain", "line", {}, "plain line"),
    ],
)
def test_join_two(text, line, vocabulary, expected):
    assert join_two(text, line, Counter(vocabulary)) == expected


# Web addresses ----------------------------------------------------------------


def test_puts_a_web_address_back_together():
    lines = [
        "Applied Mathematics with",
        "Economics Option, B.Sc. Major",
        "(https://catalog.umanitoba.ca/",
        "undergraduate-studies/science/",
        "mathematics/applied-mathematics-",
        "bsc-major-economics-option/) is a program.",
    ]

    assert join_lines(lines, Counter()) == (
        "Applied Mathematics with Economics Option, B.Sc. Major "
        "(https://catalog.umanitoba.ca/undergraduate-studies/science/mathematics/"
        "applied-mathematics-bsc-major-economics-option/) is a program."
    )


def test_a_web_address_at_the_end_of_a_line_is_not_glued_to_the_next_word():
    assert join_two("See https://example.com/", "for details", Counter()) == "See https://example.com/ for details"


# Paragraphs -------------------------------------------------------------------


def test_joins_lines_into_paragraphs():
    lines = [
        "Language model pre-training has been shown to be",
        "effective for improving many natural language tasks.",
        "These include sentence-level tasks such as natural",
        "language inference and paraphrasing.",
        "There are two existing strategies for applying the",
        "models to downstream tasks, which we compare.",
    ]

    assert join_lines(lines, Counter()) == (
        "Language model pre-training has been shown to be effective for improving many natural language tasks. "
        "These include sentence-level tasks such as natural language inference and paraphrasing.\n\n"
        "There are two existing strategies for applying the models to downstream tasks, which we compare."
    )


def test_headings_get_their_own_paragraph():
    lines = [
        "and this finishes the previous section of the paper.",
        "2 Related Work",
        "There is a long history of pre-training general language",
        "representations, and we briefly review the most widely",
    ]

    assert join_lines(lines, Counter()).split("\n\n") == [
        "and this finishes the previous section of the paper.",
        "2 Related Work",
        "There is a long history of pre-training general language representations, "
        "and we briefly review the most widely",
    ]


def test_bullets_start_new_paragraphs():
    lines = [
        "Applicants need all of the following courses:",
        "• English 40S",
        "• Mathematics 40S, Pre-Calculus or Applied, which counts",
        "as one course",
        "1. A science 40S course",
    ]

    assert join_lines(lines, Counter()).split("\n\n") == [
        "Applicants need all of the following courses:",
        "• English 40S",
        "• Mathematics 40S, Pre-Calculus or Applied, which counts as one course",
        "1. A science 40S course",
    ]


def test_blank_lines_end_paragraphs():
    assert join_lines(["First part", "", "", "second part"], Counter()) == "First part\n\nsecond part"


def test_a_page_with_no_lines_is_empty():
    assert join_lines(["", ""], Counter()) == ""


# The whole document -----------------------------------------------------------


def test_clean_pages_uses_the_whole_document_as_its_dictionary():
    pages = raw(
        "Running Head\nBERT uses task-specific heads and a new representation.\n1",
        "Running Head\nIt needs no task-\nspecific architecture; the representa-\ntion is shared.\n2",
        "Running Head\nThe end.\n3",
    )

    cleaned = clean_pages(pages)

    assert [p.text for p in cleaned.pages] == [
        "BERT uses task-specific heads and a new representation.",
        "It needs no task-specific architecture; the representation is shared.",
        "The end.",
    ]
    assert cleaned.removed_lines == 6
    assert [(p.number, p.label) for p in cleaned.pages] == [(1, "1"), (2, "2"), (3, "3")]


def test_cleans_a_real_pdf_end_to_end(make_pdf):
    pages = [
        [
            "Flex assembles furniture all over Winnipeg. Every job gets a free",
            "quote, and we bring our own tools. Large wardrobes take about two",
            "hours, and we offer evening slots.",
        ],
        ["Customers book online and pay after the job is done."],
        ["Questions? Visit https://example.com/flex/", "booking-help/ any time."],
    ]
    path = make_pdf(pages=pages, header="Flex Furniture Assembly", footer="Page {n}")

    cleaned = clean_pages(read_pdf(path).pages)

    assert [p.text for p in cleaned.pages] == [
        "Flex assembles furniture all over Winnipeg. Every job gets a free quote, and we bring our own tools. "
        "Large wardrobes take about two hours, and we offer evening slots.",
        "Customers book online and pay after the job is done.",
        "Questions? Visit https://example.com/flex/booking-help/ any time.",
    ]
    assert cleaned.removed_lines == 6
