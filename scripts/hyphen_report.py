"""List every word a PDF hyphenates at the end of a line, and whether cleaning keeps the hyphen.

    python scripts/hyphen_report.py samples/bert.pdf samples/attention.pdf

This is how the README's hyphenation numbers were checked: run it, then read the lists.
"""

from __future__ import annotations

import sys

from pdfqa import clean
from pdfqa.extract import read_pdf


def decisions(path: str) -> tuple[list[str], list[str]]:
    raw = read_pdf(path)
    lines = [[clean.normalize_line(line) for line in page.text.splitlines()] for page in raw.pages]
    lines, _ = clean.remove_stamps(lines)
    lines, _ = clean.remove_repeated_edges(lines)
    vocabulary = clean.build_vocabulary(lines)

    kept, joined = [], []
    for page in lines:
        for line, following in zip(page, page[1:], strict=False):
            end = clean._HYPHENATED_END.search(line) if line else None
            start = clean._LEADING_WORD.match(following) if following else None
            if not end or not start or clean._ends_inside_url(line, following):
                continue
            keep = clean.keeps_hyphen(end.group(1), start.group(1), bool(start.group(2)), vocabulary)
            (kept if keep else joined).append(f"{end.group(1)}-{start.group(1)}")
    return kept, joined


def main(paths: list[str]) -> None:
    for path in paths:
        kept, joined = decisions(path)
        print(f"{path}: {len(kept) + len(joined)} line-end hyphens, {len(kept)} kept, {len(joined)} joined")
        print("  kept:  ", ", ".join(kept))
        print("  joined:", ", ".join(joined))


if __name__ == "__main__":
    main(sys.argv[1:])
