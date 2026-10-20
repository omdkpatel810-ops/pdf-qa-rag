from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdfqa.chunk import MAX_SIZE, MIN_SIZE, check_settings, chunk_pages
from pdfqa.clean import Page


def sentence(first: int, words: int) -> str:
    """`words` numbered words ending in a full stop, like "W1 W2 W3."."""
    return " ".join(f"W{n}" for n in range(first, first + words)) + "."


def page(number: int, *paragraphs: str) -> Page:
    return Page(number=number, label=str(number), text="\n\n".join(paragraphs))


def test_packs_whole_sentences_up_to_the_size_and_repeats_the_last_one():
    text = " ".join(sentence(n, 10) for n in range(1, 60, 10))  # six 10-word sentences

    chunks = chunk_pages([page(1, text)], size=30, overlap=10)

    assert [c.text.split()[0] for c in chunks] == ["W1", "W21", "W41"]
    assert [c.word_count for c in chunks] == [30, 30, 20]
    assert chunks[1].text.startswith(sentence(21, 10))  # the overlap: chunk 1's last sentence


def test_no_overlap():
    text = " ".join(sentence(n, 10) for n in range(1, 40, 10))

    chunks = chunk_pages([page(1, text)], size=20, overlap=0)

    assert [c.text for c in chunks] == [
        f"{sentence(1, 10)} {sentence(11, 10)}",
        f"{sentence(21, 10)} {sentence(31, 10)}",
    ]


def test_drops_the_overlap_when_it_would_not_leave_room_for_the_next_sentence():
    chunks = chunk_pages([page(1, f"{sentence(1, 10)} {sentence(11, 15)}")], size=20, overlap=10)

    assert [c.text for c in chunks] == [sentence(1, 10), sentence(11, 15)]


def test_records_the_pages_each_chunk_came_from():
    pages = [page(1, sentence(1, 15)), page(2, sentence(16, 15)), page(3, sentence(31, 15))]

    chunks = chunk_pages(pages, size=30, overlap=0)

    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 2), (3, 3)]


def test_ends_a_half_full_chunk_at_a_paragraph_break():
    text = page(1, sentence(1, 12), sentence(13, 12), sentence(25, 12))

    chunks = chunk_pages([text], size=40, overlap=0)

    # 12 words isn't half of 40, so the second paragraph joins the first. At 24 words the
    # chunk is half full, so the third paragraph starts a new chunk even though it would fit.
    assert [c.text for c in chunks] == [f"{sentence(1, 12)}\n\n{sentence(13, 12)}", sentence(25, 12)]


def test_keeps_short_paragraphs_together():
    text = page(1, sentence(1, 5), sentence(6, 5), sentence(11, 5))

    [chunk] = chunk_pages([text], size=40, overlap=0)

    assert chunk.text == "\n\n".join([sentence(1, 5), sentence(6, 5), sentence(11, 5)])


def test_a_page_usually_continues_the_paragraph_from_the_page_before():
    pages = [page(1, sentence(1, 12)), page(2, sentence(13, 12))]

    [chunk] = chunk_pages(pages, size=40, overlap=0)

    assert chunk.text == f"{sentence(1, 12)} {sentence(13, 12)}"


def test_a_bullet_at_the_top_of_a_page_starts_a_new_paragraph():
    pages = [page(1, "Bring these:", "• English 40S"), page(2, "• Mathematics 40S")]

    [chunk] = chunk_pages(pages, size=40, overlap=0)

    assert chunk.text == "Bring these:\n\n• English 40S\n\n• Mathematics 40S"


def test_cuts_a_sentence_longer_than_a_chunk_into_overlap_sized_pieces():
    table = " ".join(f"T{n}" for n in range(1, 101))  # 100 words without a full stop

    chunks = chunk_pages([page(1, table)], size=40, overlap=10)

    assert [c.word_count for c in chunks] == [40, 40, 40]
    assert chunks[1].text.split()[0] == "T31"  # starts with the last 10 words of chunk 1
    assert chunks[-1].text.split()[-1] == "T100"


def test_pages_without_text_make_no_chunks():
    assert chunk_pages([page(1, ""), page(2, "")]) == []


@pytest.mark.parametrize(
    ("size", "overlap", "message"),
    [
        (MIN_SIZE - 1, 0, f"Chunk size must be between {MIN_SIZE} and {MAX_SIZE} words."),
        (MAX_SIZE + 1, 0, f"Chunk size must be between {MIN_SIZE} and {MAX_SIZE} words."),
        (100, 51, "Overlap must be between 0 and 50 words (half the chunk size)."),
        (100, -1, "Overlap must be between 0 and 50 words (half the chunk size)."),
    ],
)
def test_rejects_bad_settings(size, overlap, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        check_settings(size, overlap)


# Properties that must hold for any document ---------------------------------


@st.composite
def documents(draw):
    """Pages of paragraphs of sentences, every word unique so it can be traced to its chunk."""
    size = draw(st.integers(MIN_SIZE, 120))
    overlap = draw(st.integers(0, size // 2))
    sentence_lengths = st.lists(st.one_of(st.integers(1, 30), st.integers(1, 3 * size)), min_size=1, max_size=6)
    layout = draw(st.lists(st.lists(sentence_lengths, max_size=4), min_size=1, max_size=5))

    pages, word_pages, next_word = [], {}, 1
    for number, paragraphs in enumerate(layout, start=1):
        texts = []
        for lengths in paragraphs:
            sentences = []
            for length in lengths:
                sentences.append(sentence(next_word, length))
                for n in range(next_word, next_word + length):
                    word_pages[f"W{n}"] = number
                next_word += length
            texts.append(" ".join(sentences))
        pages.append(page(number, *texts))
    return pages, word_pages, size, overlap


@settings(max_examples=300, deadline=None)
@given(documents())
def test_every_word_lands_in_order_within_the_limits(document):
    pages, word_pages, size, overlap = document
    all_words = list(word_pages)  # in document order
    index = {word: i for i, word in enumerate(all_words)}

    chunks = chunk_pages(pages, size=size, overlap=overlap)

    if not all_words:
        assert chunks == []
        return
    assert [c.position for c in chunks] == list(range(len(chunks)))
    previous_end = 0
    for i, chunk in enumerate(chunks):
        words = [w.rstrip(".") for w in chunk.text.split()]
        start, end = index[words[0]], index[words[-1]] + 1
        assert words == all_words[start:end], "a chunk is a contiguous run of the document"
        assert 1 <= chunk.word_count == len(words) <= size
        assert (chunk.page_start, chunk.page_end) == (word_pages[words[0]], word_pages[words[-1]])
        if i == 0:
            assert start == 0
        else:
            assert start <= previous_end, "no words are skipped between chunks"
            assert previous_end - start <= overlap, "chunks overlap by at most `overlap` words"
            assert end > previous_end, "every chunk adds something new"
        previous_end = end
    assert previous_end == len(all_words)
