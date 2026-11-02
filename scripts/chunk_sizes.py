"""Compare chunk sizes on a test set: python scripts/chunk_sizes.py QUESTIONS.json [--db PATH]

Copies the library, splits every document again at each size (overlap 20% of the size), makes new
vectors, and measures search on the copy. The library itself isn't changed. Because a passage counts when
its text holds the answer, the same test set works for every size.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from pdfqa import library
from pdfqa.cli import default_db_path
from pdfqa.embed import BgeSmall
from pdfqa.evaluate import find, load_questions, mrr, recall
from pdfqa.ingest import index, rechunk
from pdfqa.search import MODES, VectorIndex

SIZES = [(100, 20), (200, 40), (300, 60)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", type=Path)
    parser.add_argument("--db", default=str(default_db_path()))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    embedder = BgeSmall()
    print(f"{'Size':>5} {'Chunks':>7}  {'Mode':<9}{'Top 1':>7}{'Top 3':>7}{'Top 5':>7}{'MRR':>7}")
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "library.db"
        shutil.copy(args.db, copy)
        conn = library.connect(copy)
        for size, overlap in SIZES:
            started = time.perf_counter()
            for doc in library.list_documents(conn):
                rechunk(conn, doc.id, size, overlap)
            index(conn, embedder)
            chunks = sum(doc.chunk_count for doc in library.list_documents(conn))
            vectors = VectorIndex(conn, embedder)
            for mode in MODES:
                found = find(conn, questions, index=vectors, mode=mode)
                recalls = "".join(f"{recall(found, k):>7.0%}" for k in (1, 3, 5))
                print(f"{size:>5} {chunks:>7}  {mode:<9}{recalls}{mrr(found):>7.2f}", flush=True)
            print(f"      (re-split and re-embedded in {time.perf_counter() - started:.0f} s)", flush=True)
        conn.close()


if __name__ == "__main__":
    main()
