"""Time searches on a library: python scripts/search_speed.py [--db PATH]

Loads the model and the vectors once, as a long-running app would, then times each mode on a few
questions (median of five runs each). The first question also reports how long loading took.
"""

from __future__ import annotations

import argparse
import statistics
import time

from pdfqa import library
from pdfqa.cli import default_db_path
from pdfqa.embed import BgeSmall
from pdfqa.search import MODES, VectorIndex, search

QUESTIONS = [
    "How many credit hours do I need to graduate?",
    "Which course teaches programming for science students?",
    "Can I take a course pass/fail?",
    "What happens if my grade point average is too low?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(default_db_path()))
    args = parser.parse_args()

    conn = library.connect(args.db)
    model = BgeSmall()
    started = time.perf_counter()
    model.query("warm up")  # loads the model
    loaded_model = time.perf_counter() - started
    started = time.perf_counter()
    index = VectorIndex(conn, model)
    loaded_vectors = time.perf_counter() - started
    print(f"{len(index):,} chunks. Loading: model {loaded_model:.2f} s, vectors {loaded_vectors * 1000:.0f} ms.")

    for mode in MODES:
        times = []
        for question in QUESTIONS:
            for _ in range(5):
                started = time.perf_counter()
                search(conn, question, index=index, mode=mode, limit=5)
                times.append(time.perf_counter() - started)
        print(f"{mode:>8}: median {statistics.median(times) * 1000:.1f} ms a question")
    conn.close()


if __name__ == "__main__":
    main()
