from __future__ import annotations

import io
from pathlib import Path

import pytest
from helpers import LOREM, WordHashEmbedder, words_to_lines

from pdfqa import __version__
from pdfqa.cli import default_db_path, main
from pdfqa.embed import EmbeddingError


class Terminal(io.StringIO):
    """An output stream that says it's a terminal, so progress is shown."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def embedder() -> WordHashEmbedder:
    return WordHashEmbedder()


@pytest.fixture
def run(tmp_path: Path, embedder):
    db = tmp_path / "library.db"

    def run(*args: str, err: io.StringIO | None = None) -> tuple[int, str, str]:
        out, err = io.StringIO(), err or io.StringIO()
        code = main(["--db", str(db), *args], out=out, err=err, embedder=embedder)
        return code, out.getvalue(), err.getvalue()

    return run


def text_pages(count: int) -> list[list[str]]:
    sentences = [s.strip() + "." for s in LOREM.split(".") if s.strip()]
    return [words_to_lines(" ".join(sentences[n % 4 :] + sentences[: n % 4]) * 3) for n in range(count)]


def test_default_library_location(monkeypatch, tmp_path):
    monkeypatch.delenv("PDFQA_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_db_path() == tmp_path / ".pdfqa" / "library.db"

    monkeypatch.setenv("PDFQA_DB", "~/elsewhere.db")
    assert default_db_path() == Path("~/elsewhere.db").expanduser()


def test_uses_pdfqa_db_when_no_db_option_is_given(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PDFQA_DB", str(tmp_path / "from-env.db"))

    assert main(["list"]) == 0
    assert (tmp_path / "from-env.db").exists()
    assert "The library is empty" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"pdfqa {__version__}"


def test_add_then_list_show_and_chunks(run, make_pdf):
    path = make_pdf(
        "handbook.pdf", text_pages(3) + [[]], title="Flex Handbook", header="Flex Handbook", footer="Page {n}"
    )

    code, out, err = run("add", str(path), "--size", "60", "--overlap", "10")
    assert (code, err) == (0, "")
    assert out.startswith('Added #1 "Flex Handbook" (handbook.pdf): 4 pages, ')
    assert "Removed 6 header or footer lines." in out
    assert "1 page with no text, probably scanned: 4." in out

    code, out, _ = run("list")
    assert code == 0
    header, row = out.splitlines()
    assert header.split() == ["#", "Title", "Pages", "Chunks", "Added"]
    assert row.split()[:5] == ["1", "Flex", "Handbook", "4", row.split()[4]]

    code, out, _ = run("show", "1")
    assert code == 0
    assert "#1 Flex Handbook" in out
    assert "Pages    4, 1 page with no text: 4" in out
    assert "of up to 60 words with a 10-word overlap" in out

    code, out, _ = run("show", "1", "--page", "2")
    assert code == 0
    assert out.startswith("Page 2 of 4\n\n")
    assert "Flex Handbook" not in out

    code, out, _ = run("show", "1", "--page", "4")
    assert "No text on this page. It's probably a scanned image." in out

    code, out, _ = run("chunks", "1", "--limit", "2")
    assert code == 0
    assert out.startswith("── chunk 1 · p. 1 · ")
    assert out.count("── chunk") == 2
    assert "more chunks. Use --limit N or --all to see more." in out

    code, out, _ = run("chunks", "1", "--all")
    assert "more chunks" not in out

    code, out, _ = run("chunks", "1", "--page", "3", "--all")
    assert "── chunk" in out
    assert all(("p. 3" in line or "pp. 2–3" in line) for line in out.splitlines() if line.startswith("── chunk"))


def test_chunk_headings_use_printed_page_numbers(run, make_pdf):
    path = make_pdf(pages=text_pages(3), roman_front_pages=1)
    run("add", str(path), "--size", "200", "--overlap", "0")

    _, out, _ = run("chunks", "1", "--all")

    assert "· p. i ·" in out or "· pp. i–1 ·" in out

    _, out, _ = run("show", "1", "--page", "1")
    assert out.startswith('Page 1 of 3 (printed as "i")')


def test_adding_several_files_continues_past_a_bad_one(run, make_pdf, tmp_path):
    good = make_pdf("good.pdf", text_pages(1))
    bad = tmp_path / "bad.pdf"
    bad.write_text("<html>not found</html>")

    code, out, err = run("add", str(bad), str(good))

    assert code == 1
    assert "error: bad.pdf isn't a PDF, it's a web page." in err
    assert 'Added #1 "good"' in out


def test_adding_the_same_file_twice(run, make_pdf):
    path = make_pdf("guide.pdf")
    run("add", str(path))

    code, out, _ = run("add", str(path))

    assert code == 0
    assert out.strip() == (
        'guide.pdf is already in the library as #1 "guide". To split it differently, use: pdfqa rechunk 1 --size N'
    )


def test_title_only_works_for_one_file(run, make_pdf):
    one, two = make_pdf("one.pdf"), make_pdf("two.pdf", [["Other text."]])

    code, _, err = run("add", str(one), str(two), "--title", "Both")

    assert code == 1
    assert err.strip() == "error: --title can only be used when adding one file."


def test_shows_progress_for_long_pdfs_in_a_terminal(run, make_pdf):
    path = make_pdf(pages=[[f"Page {n} has its own words here."] for n in range(1, 21)])
    terminal = Terminal()

    code, _, err = run("add", str(path), err=terminal)

    assert code == 0
    assert "\rReading doc.pdf: 1/20 pages" in err
    assert err.endswith("\rReading doc.pdf: 20/20 pages\n")


def test_no_progress_for_short_pdfs(run, make_pdf):
    terminal = Terminal()

    run("add", str(make_pdf()), err=terminal)

    assert terminal.getvalue() == ""


def test_reports_unreadable_pages(run, make_pdf, monkeypatch):
    from pypdf import PageObject

    original = PageObject.extract_text
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad font")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky)

    _, out, _ = run("add", str(make_pdf(pages=[["One."], ["Two."]])))

    assert "Couldn't read 1 page: 1." in out


def test_rechunk_and_remove(run, make_pdf):
    run("add", str(make_pdf(pages=text_pages(2))))
    _, out, _ = run("show", "1")
    assert "Pages    2, all with text" in out

    code, out, _ = run("rechunk", "1", "--size", "40", "--overlap", "0")
    assert code == 0
    assert out.startswith("Split #1 again: ")
    assert "of up to 40 words with a 0-word overlap (was 2)." in out

    code, out, _ = run("remove", "1")
    assert (code, out.strip()) == (0, 'Removed #1 "doc".')

    _, out, _ = run("list")
    assert "The library is empty" in out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["show", "9"], "error: No document #9. See what's in the library with: pdfqa list"),
        (["chunks", "9"], "error: No document #9."),
        (["remove", "9"], "error: No document #9."),
        (["rechunk", "9"], "error: No document #9."),
        (["show", "1", "--page", "7"], "error: #1 has pages 1 to 1."),
        (["chunks", "1", "--page", "0"], "error: #1 has pages 1 to 1."),
        (["chunks", "1", "--limit", "0"], "error: --limit must be at least 1."),
        (["rechunk", "1", "--size", "10"], "error: Chunk size must be between 20 and 2000 words."),
    ],
)
def test_errors(run, make_pdf, args, message):
    run("add", str(make_pdf()))

    code, out, err = run(*args)

    assert code == 1
    assert out == ""
    assert err.startswith(message)


def test_add_errors_are_reported(run, tmp_path):
    code, _, err = run("add", str(tmp_path / "missing.pdf"))

    assert code == 1
    assert err.strip() == f"error: No such file: {tmp_path / 'missing.pdf'}"


def test_cannot_open_the_library(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    err = io.StringIO()

    code = main(["--db", str(blocker / "library.db"), "list"], out=io.StringIO(), err=err)

    assert code == 1
    assert err.getvalue().startswith("error: Can't open the library:")


def test_long_titles_are_shortened_in_the_list(run, make_pdf):
    run("add", str(make_pdf()), "--title", "A very long title " * 5)

    _, out, _ = run("list")

    assert "…" in out.splitlines()[1]


def test_lists_many_page_numbers_briefly(run, make_pdf):
    run("add", str(make_pdf(pages=[["Typed."]] + [[]] * 14)))

    _, out, _ = run("show", "1")

    assert "14 pages with no text: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 and 2 more" in out


# Search vectors and search ------------------------------------------------------------------------------

COURSE_PAGES = [
    ["Students must complete 120 credit hours to graduate with a Bachelor of Science major."],
    ["COMP 1012 teaches programming for scientists and engineers using Python."],
    ["Furniture assembly takes about two hours for a large wardrobe in most homes. Every quote is free."],
]


class Offline:
    """An embedder whose model can't be downloaded."""

    dimensions = 64

    def __init__(self, name: str = "test/offline") -> None:
        self.name = name

    def passages(self, texts, progress=None):
        raise EmbeddingError("Couldn't download the model (OSError).")

    def query(self, text):
        raise EmbeddingError("Couldn't download the model (OSError).")


def test_adding_makes_search_vectors(run, make_pdf, embedder):
    code, out, _ = run("add", str(make_pdf("courses.pdf", COURSE_PAGES)), "--size", "20", "--overlap", "0")
    assert code == 0
    assert out.splitlines()[-1].startswith("  Made 3 search vectors in ")

    _, out, _ = run("show", "1")
    assert f"  Vectors  3 of 3, from {embedder.name}\n" in out

    _, out, _ = run("index")
    assert out.strip() == 'Every chunk already has a search vector. Ask something with: pdfqa search "your question"'


def test_search_shows_the_best_chunks_with_their_pages(run, make_pdf):
    run("add", str(make_pdf("courses.pdf", COURSE_PAGES, title="Faculty of Science")), "--size", "20", "--overlap", "0")

    code, out, _ = run("search", "Which", "course", "teaches", "Python?", "-k", "2")

    assert code == 0
    first = out.split("\n\n")[0].splitlines()
    assert first[0].startswith('── 1 · #1 "Faculty of Science" p. 2 · meaning #1 · keywords #1 ──')
    assert first[1] == "COMP 1012 teaches programming for scientists and engineers using Python."
    assert out.count("── ") == 2


def test_search_modes_documents_and_length(run, make_pdf):
    run("add", str(make_pdf("courses.pdf", COURSE_PAGES)), "--size", "20", "--overlap", "0")
    run("add", str(make_pdf("long.pdf", [[("wardrobe " * 12).strip()] * 8])), "--size", "100", "--overlap", "0")

    _, out, _ = run("search", "wardrobe", "--mode", "keyword", "--doc", "2")
    assert "keywords #1" in out and "meaning" not in out and '#1 "courses"' not in out
    assert out.splitlines()[1].endswith(" …") and len(out.splitlines()[1].split()) == 61

    _, out, _ = run("search", "wardrobe", "--mode", "keyword", "--doc", "2", "--full")
    assert len(out.splitlines()[1].split()) == 96

    _, out, _ = run("search", "wardrobe", "--mode", "semantic", "-k", "1")
    assert "meaning #1" in out and "keywords" not in out


def test_search_with_no_match(run, make_pdf):
    run("add", str(make_pdf("courses.pdf", COURSE_PAGES)))
    code, out, _ = run("search", "zeppelin", "--mode", "keyword")
    assert (code, out.strip()) == (0, "Nothing matched. Try other words, or search by meaning with --mode semantic.")


def test_search_needs_vectors_for_meaning(tmp_path, make_pdf):
    db = str(tmp_path / "library.db")
    main(
        ["--db", db, "add", str(make_pdf("courses.pdf", COURSE_PAGES))],
        out=io.StringIO(),
        err=io.StringIO(),
        embedder=Offline(),
    )
    err = io.StringIO()
    code = main(
        ["--db", db, "search", "credits", "--mode", "semantic"], out=io.StringIO(), err=err, embedder=WordHashEmbedder()
    )
    assert code == 1 and "No chunks have vectors yet" in err.getvalue()


def test_adding_works_offline_and_index_catches_up_later(tmp_path, make_pdf):
    db = str(tmp_path / "library.db")
    out, err = io.StringIO(), io.StringIO()

    code = main(["--db", db, "add", str(make_pdf("courses.pdf", COURSE_PAGES))], out=out, err=err, embedder=Offline())
    assert code == 0
    assert "  No search vectors yet. Couldn't download the model (OSError). Then run: pdfqa index" in out.getvalue()

    out, terminal = io.StringIO(), Terminal()
    assert main(["--db", db, "index"], out=out, err=terminal, embedder=WordHashEmbedder()) == 0
    assert out.getvalue().startswith("Made 1 search vector in ")
    assert "chunks a second" in out.getvalue()

    # The vectors are there, but the model that turns the question into one isn't.
    err = io.StringIO()
    assert main(["--db", db, "search", "credits"], out=out, err=err, embedder=Offline(WordHashEmbedder.name)) == 1
    assert err.getvalue().strip().endswith("Or search by words alone with --mode keyword.")


def test_index_shows_progress_on_a_terminal(tmp_path, make_pdf):
    db = str(tmp_path / "library.db")
    path = make_pdf("long.pdf", text_pages(6))
    main(
        ["--db", db, "add", str(path), "--size", "20", "--overlap", "0"],
        out=io.StringIO(),
        err=io.StringIO(),
        embedder=Offline(),
    )

    terminal = Terminal()
    main(["--db", db, "index"], out=io.StringIO(), err=terminal, embedder=WordHashEmbedder())
    assert terminal.getvalue().startswith("\rMaking search vectors: ")
    assert terminal.getvalue().endswith(" chunks\n")


def test_rechunking_remakes_the_vectors(run, make_pdf, embedder):
    run("add", str(make_pdf("courses.pdf", COURSE_PAGES)), "--size", "20", "--overlap", "0")
    code, out, _ = run("rechunk", "1", "--size", "200", "--overlap", "0")
    assert code == 0 and out.splitlines()[-1].startswith("  Made 1 search vector in ")
    _, out, _ = run("show", "1")
    assert "Vectors  1 of 1" in out


def test_show_points_out_missing_vectors(tmp_path, make_pdf):
    db = str(tmp_path / "library.db")
    main(
        ["--db", db, "add", str(make_pdf("courses.pdf", COURSE_PAGES))],
        out=io.StringIO(),
        err=io.StringIO(),
        embedder=Offline(),
    )
    out = io.StringIO()
    main(["--db", db, "show", "1"], out=out, err=io.StringIO(), embedder=WordHashEmbedder())
    assert "Vectors  0 of 1, from test/word-hash. Make the rest with: pdfqa index" in out.getvalue()
