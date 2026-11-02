"""Time answers from the local model: python scripts/answer_speed.py [--db PATH]

Loads both models once, as `pdfqa serve` would, then asks a few questions about the sample PDFs and reports,
for each, how long until the first word appears (searching plus reading the prompt) and how fast the rest
is written.
"""

from __future__ import annotations

import argparse
import statistics
import time

from pdfqa import library
from pdfqa.answer import Done, Text, ask
from pdfqa.cli import default_db_path
from pdfqa.embed import BgeSmall
from pdfqa.generate import LocalModel
from pdfqa.search import VectorIndex

QUESTIONS = [
    "What are the two tasks BERT is pre-trained on?",
    "Why divide the dot product by the square root of the dimension?",
    "How many credit hours does a B.Sc. Major need?",
    "What does a parking pass cost on campus?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(default_db_path()))
    args = parser.parse_args()

    conn = library.connect(args.db)
    writer = LocalModel()
    started = time.perf_counter()
    writer._load()
    print(f"Loading the model: {time.perf_counter() - started:.1f} s")
    index = VectorIndex(conn, BgeSmall())
    tokenizer = writer._tokenizer

    firsts, rates = [], []
    for question in QUESTIONS:
        started = time.perf_counter()
        first = None
        for event in ask(conn, question, index=index, writer=writer):
            if isinstance(event, Text) and first is None:
                first = time.perf_counter() - started
            elif isinstance(event, Done):
                answer = event.answer
        tokens = len(tokenizer.encode(answer.text, add_special_tokens=False).ids)
        writing = answer.seconds - (first or answer.seconds)
        rate = tokens / writing if writing else 0
        firsts.append(first or answer.seconds)
        rates.append(rate)
        print(f"{answer.seconds:5.1f} s  first word {first or 0:4.1f} s  {tokens:3} tokens at {rate:.1f}/s  {question}")
        print(f"         {answer.text}")
    print(f"Median: first word {statistics.median(firsts):.1f} s, {statistics.median(rates):.1f} tokens a second")
    conn.close()


if __name__ == "__main__":
    main()
