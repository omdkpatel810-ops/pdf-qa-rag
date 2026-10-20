from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from helpers import build_pdf

from pdfqa import library

MakePdf = Callable[..., Path]


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    def make(name: str = "doc.pdf", pages: Sequence[Sequence[str]] | None = None, **options) -> Path:
        return build_pdf(tmp_path / name, pages if pages is not None else [["Hello there."]], **options)

    return make


@pytest.fixture
def conn(tmp_path: Path):
    connection = library.connect(tmp_path / "library.db")
    yield connection
    connection.close()
