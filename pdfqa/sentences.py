"""Split text into sentences, without breaking after "e.g.", "Fig." or "et al."."""

from __future__ import annotations

import re

# Words that end with a period but don't end a sentence (compared lowercase, without the period).
ABBREVIATIONS = frozenset(
    """
    al approx ca cf ch co corp dept dr e.g eq eqs est etc fig figs i.e inc jr ltd mr mrs ms
    p pp prof sec sr st vol vols vs
    jan feb mar apr jun jul aug sep sept oct nov dec
    """.split()
)

# Sentence-ending punctuation, any closing quotes or brackets, then whitespace.
_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*\s+")
_STARTS_SENTENCE = re.compile(r"[\"'“‘(\[]?[A-Z0-9•]")
_DOTTED = re.compile(r"^(?:[A-Za-z]{1,3}\.)+$")  # U.S., B.Sc., Ph.D.


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        end = boundary.end()
        if not _STARTS_SENTENCE.match(text, end):
            continue
        if boundary.group().startswith(".") and _is_abbreviation(text[start : boundary.start() + 1]):
            continue
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


def _is_abbreviation(text: str) -> bool:
    """Whether the word just before the period is an abbreviation or someone's initial."""
    word = text.rsplit(None, 1)[-1].lstrip("([\"'“‘") if text.strip() else ""
    if len(word) == 2 and word[0].isalpha():  # an initial, like the "J." in "J. Smith"
        return True
    if _DOTTED.match(word) and word.count(".") > 1:
        return True
    return word[:-1].lower() in ABBREVIATIONS
