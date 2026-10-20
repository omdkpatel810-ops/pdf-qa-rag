from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pypdf import PageObject

from pdfqa.extract import PARALLEL_FROM_PAGES, PdfError, read_pdf


def test_reads_each_page_with_its_number_and_label(make_pdf):
    path = make_pdf(pages=[["First page."], ["Second page."], ["Third page."]])

    doc = read_pdf(path)

    assert [(p.number, p.label) for p in doc.pages] == [(1, "1"), (2, "2"), (3, "3")]
    assert [p.text.strip() for p in doc.pages] == ["First page.", "Second page.", "Third page."]
    assert doc.unreadable_pages == []


def test_keeps_the_printed_page_numbers(make_pdf):
    path = make_pdf(pages=[["Contents"], ["Preface"], ["Chapter one"], ["Chapter two"]], roman_front_pages=2)

    assert [p.label for p in read_pdf(path).pages] == ["i", "ii", "1", "2"]


def test_fingerprints_the_file(make_pdf):
    path = make_pdf()

    assert read_pdf(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("metadata_title", "expected"),
    [
        ("Faculty of Science", "Faculty of Science"),
        ("  Lots   of   space ", "Lots of space"),
        ("Microsoft Word - Course Outline.docx", "Course Outline"),
        ("untitled", None),
        ("", None),
        (None, None),
    ],
)
def test_title_comes_from_the_metadata_unless_it_is_junk(make_pdf, metadata_title, expected):
    assert read_pdf(make_pdf(title=metadata_title)).title == expected


def test_reports_progress_page_by_page(make_pdf):
    calls = []

    read_pdf(make_pdf(pages=[["a"], ["b"], ["c"]]), progress=lambda done, total: calls.append((done, total)))

    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_a_page_with_no_text_is_kept_empty(make_pdf):
    doc = read_pdf(make_pdf(pages=[["Typed page."], []]))

    assert doc.pages[1].text == ""
    assert doc.unreadable_pages == []


def test_a_page_that_fails_is_kept_empty_and_reported(make_pdf, monkeypatch):
    original = PageObject.extract_text
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("bad font")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky)

    doc = read_pdf(make_pdf(pages=[["One."], ["Two."], ["Three."]]))

    assert [p.text.strip() for p in doc.pages] == ["One.", "", "Three."]
    assert doc.unreadable_pages == [2]


def test_opens_a_pdf_with_only_an_owner_password(make_pdf):
    path = make_pdf(pages=[["Secret-free text."]], user_password="", owner_password="owner")

    assert read_pdf(path).pages[0].text.strip() == "Secret-free text."


def test_password_protected_pdf(make_pdf):
    path = make_pdf(user_password="hunter22")

    with pytest.raises(PdfError, match=r"doc\.pdf is password-protected\. Remove the password"):
        read_pdf(path)


def test_missing_file(tmp_path: Path):
    with pytest.raises(PdfError, match="No such file"):
        read_pdf(tmp_path / "nope.pdf")


def test_folder_instead_of_file(tmp_path: Path):
    with pytest.raises(PdfError, match="is a folder, not a PDF"):
        read_pdf(tmp_path)


def test_unreadable_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "locked.pdf"
    path.write_bytes(b"%PDF-1.4")

    def deny(self):
        raise PermissionError

    monkeypatch.setattr(Path, "read_bytes", deny)
    with pytest.raises(PdfError, match="Not allowed to read"):
        read_pdf(path)


def test_empty_file(tmp_path: Path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(PdfError, match="empty.pdf is empty"):
        read_pdf(path)


def test_a_web_page_saved_as_pdf(tmp_path: Path):
    path = tmp_path / "catalog.pdf"
    path.write_text("\n\n<!doctype html>\n<html><title>Page Not Found</title></html>")

    with pytest.raises(PdfError, match="isn't a PDF, it's a web page. The download probably went wrong"):
        read_pdf(path)


def test_some_other_file(tmp_path: Path):
    path = tmp_path / "notes.pdf"
    path.write_text("just some notes")

    with pytest.raises(PdfError, match=r"^notes\.pdf isn't a PDF\.$"):
        read_pdf(path)


def test_damaged_pdf(make_pdf):
    path = make_pdf()
    path.write_bytes(path.read_bytes()[:300])

    with pytest.raises(PdfError, match=r"doc\.pdf looks damaged and couldn't be opened"):
        read_pdf(path)


def test_broken_page_labels_fall_back_to_page_numbers(make_pdf, monkeypatch):
    import pypdf

    def broken(self):
        raise KeyError("/Nums")

    monkeypatch.setattr(pypdf.PdfReader, "page_labels", property(broken))

    assert [p.label for p in read_pdf(make_pdf(pages=[["a"], ["b"]])).pages] == ["1", "2"]


def test_unreadable_metadata_means_no_title(make_pdf, monkeypatch):
    import pypdf

    def broken(self):
        raise ValueError("bad info dictionary")

    monkeypatch.setattr(pypdf.PdfReader, "metadata", property(broken))

    assert read_pdf(make_pdf(title="Something")).title is None


# Reading pages in parallel ----------------------------------------------------


def many_pages(count: int) -> list[list[str]]:
    return [[f"This is page {n}.", f"It talks about topic number {n * 7}."] for n in range(1, count + 1)]


def test_reads_long_pdfs_in_parallel_with_the_same_result(make_pdf):
    path = make_pdf(pages=many_pages(PARALLEL_FROM_PAGES + 2))
    progress = []

    parallel = read_pdf(path, workers=2, progress=lambda done, total: progress.append((done, total)))

    assert parallel == read_pdf(path, workers=1)
    assert progress[-1] == (PARALLEL_FROM_PAGES + 2, PARALLEL_FROM_PAGES + 2)
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)


def test_falls_back_to_one_process_when_subprocesses_are_not_allowed(make_pdf, monkeypatch):
    from pdfqa import extract

    def refuse(*args, **kwargs):
        raise OSError("no subprocesses in this sandbox")

    monkeypatch.setattr(extract, "ProcessPoolExecutor", refuse)
    path = make_pdf(pages=many_pages(PARALLEL_FROM_PAGES))

    doc = read_pdf(path, workers=4)

    assert [p.text.strip().splitlines()[0] for p in doc.pages][:2] == ["This is page 1.", "This is page 2."]


def test_a_worker_opens_the_pdf_once_and_reads_batches(make_pdf):
    from pdfqa import extract

    path = make_pdf(pages=many_pages(3), user_password="", owner_password="owner")

    extract._start_worker(path.read_bytes())
    batch = extract._extract_batch(1, 3)

    assert [text.splitlines()[0] for text in batch] == ["This is page 2.", "This is page 3."]


def test_counts_available_cpus(monkeypatch):
    from pdfqa import extract

    assert extract._available_cpus() >= 1
    monkeypatch.delattr(extract.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(extract.os, "cpu_count", lambda: None)
    assert extract._available_cpus() == 1
