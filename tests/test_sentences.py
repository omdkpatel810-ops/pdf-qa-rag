from __future__ import annotations

import pytest

from pdfqa.sentences import split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two! Three? Four", ["One.", "Two!", "Three?", "Four"]),
        ("It works, e.g. on tables. Next one.", ["It works, e.g. on tables.", "Next one."]),
        ("See Fig. 3 for details. Then stop.", ["See Fig. 3 for details.", "Then stop."]),
        ("As shown by Peters et al. (2018), it helps. Yes.", ["As shown by Peters et al. (2018), it helps.", "Yes."]),
        ("Written by J. Smith in Winnipeg. Done.", ["Written by J. Smith in Winnipeg.", "Done."]),
        ("A B.Sc. Major takes four years. Honours too.", ["A B.Sc. Major takes four years.", "Honours too."]),
        ("Made in the U.S. Every part.", ["Made in the U.S. Every part."]),
        ("The score was 80.5 on the test. It improved.", ["The score was 80.5 on the test.", "It improved."]),
        ("It ended in 2018. A new one began.", ["It ended in 2018.", "A new one began."]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
        ("Use the grid (p. 1). Everyone does.", ["Use the grid (p. 1).", "Everyone does."]),
        ("Lower case follows. so no split here.", ["Lower case follows. so no split here."]),
        ("Ends with a list. • First item", ["Ends with a list.", "• First item"]),
        ("Wait... What happened?", ["Wait...", "What happened?"]),
        ("", []),
        ("   ", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected
