"""Command line: pdfqa add | list | show | chunks | rechunk | remove | index | search | ask | serve."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from pdfqa import __version__, library
from pdfqa.answer import Done, Sources, Text, ask
from pdfqa.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE
from pdfqa.embed import BgeSmall, Embedder, EmbeddingError
from pdfqa.extract import PdfError
from pdfqa.generate import DOWNLOAD_SIZE, GenerationError, LocalModel, Writer, make_writer
from pdfqa.ingest import Added, add_pdf, index, rechunk
from pdfqa.search import MODES, Hit, VectorIndex, search

_TITLE_WIDTH = 44
_RULE_WIDTH = 72
_FOUND_BY = {"semantic": "meaning", "keyword": "keywords"}


def default_db_path() -> Path:
    """Use $PDFQA_DB if it's set, otherwise ~/.pdfqa/library.db."""
    override = os.environ.get("PDFQA_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".pdfqa" / "library.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfqa",
        description="Ask questions about your PDFs and get answers with page citations.",
    )
    parser.add_argument("--version", action="version", version=f"pdfqa {__version__}")
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="library file to use (default: $PDFQA_DB or ~/.pdfqa/library.db)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    add = commands.add_parser("add", help="add PDFs to the library")
    add.add_argument("files", nargs="+", type=Path, metavar="PDF")
    add.add_argument("--title", help="title to use instead of the one in the PDF (one file only)")
    _chunk_options(add)

    commands.add_parser("list", help="list the documents in the library")

    show = commands.add_parser("show", help="details of a document, or the text of one page")
    show.add_argument("document", type=int, metavar="ID")
    show.add_argument("--page", type=int, help="print the cleaned text of this page (1 is the first)")

    chunks = commands.add_parser("chunks", help="print a document's chunks")
    chunks.add_argument("document", type=int, metavar="ID")
    chunks.add_argument("--page", type=int, help="only chunks with text from this page")
    chunks.add_argument("--limit", type=int, default=10, help="how many to print (default: 10)")
    chunks.add_argument("--all", action="store_true", help="print every chunk")

    again = commands.add_parser("rechunk", help="split a document again with a different size or overlap")
    again.add_argument("document", type=int, metavar="ID")
    _chunk_options(again)

    remove = commands.add_parser("remove", help="remove a document from the library")
    remove.add_argument("document", type=int, metavar="ID")

    commands.add_parser("index", help="make search vectors for any chunks that don't have them yet")

    find = commands.add_parser("search", help="find the chunks that best answer a question")
    find.add_argument("question", nargs="+", help="the question (quotes are optional)")
    find.add_argument(
        "--mode",
        choices=MODES,
        default="hybrid",
        help="by meaning (semantic), by words (keyword), or both combined (hybrid, the default)",
    )
    find.add_argument("-k", "--limit", type=int, default=5, help="how many chunks to show (default: 5)")
    find.add_argument("--doc", type=int, metavar="ID", help="only search this document")
    find.add_argument("--full", action="store_true", help="print whole chunks instead of their first 60 words")

    question = commands.add_parser("ask", help="answer a question from your PDFs, citing the pages")
    question.add_argument("question", nargs="+", help="the question (quotes are optional)")
    _answer_options(question)
    question.add_argument("--doc", type=int, metavar="ID", help="only use this document")

    serve = commands.add_parser("serve", help="open the web page for asking questions and adding PDFs")
    serve.add_argument("--port", type=int, default=8000, help="port on this computer (default: 8000)")
    serve.add_argument("--no-browser", action="store_true", help="don't open the page in a browser")
    _answer_options(serve)
    return parser


def _answer_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=os.environ.get("PDFQA_MODEL", "local"),
        help="local (Qwen3-1.7B on this computer, the default), claude (needs your ANTHROPIC_API_KEY), "
        "or a Claude model id. Set PDFQA_MODEL to change the default.",
    )
    parser.add_argument("-k", "--limit", type=int, default=5, help="how many passages the model reads (default: 5)")


def _chunk_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE, help=f"most words in a chunk (default: {DEFAULT_SIZE})"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"words repeated from the previous chunk (default: {DEFAULT_OVERLAP})",
    )


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
    embedder: Embedder | None = None,
    writer: Writer | None = None,
) -> int:
    """Run a command. `embedder` and `writer` let tests use small stand-ins instead of downloading models."""
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    args.embedder = embedder or BgeSmall()
    args.writer = writer
    if args.command in ("ask", "serve") and writer is None:
        try:
            args.writer = make_writer(args.model)
        except ValueError as exc:
            print(f"error: {exc}", file=err)
            return 1
    try:
        conn = library.connect(args.db or default_db_path())
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        print(f"error: Can't open the library: {exc}", file=err)
        return 1
    try:
        return _COMMANDS[args.command](conn, args, out, err)
    except library.NotFound as exc:
        print(f"error: No document #{exc.args[0]}. See what's in the library with: pdfqa list", file=err)
        return 1
    except (ValueError, EmbeddingError, GenerationError) as exc:
        print(f"error: {exc}", file=err)
        return 1
    finally:
        conn.close()


def _add(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.title and len(args.files) > 1:
        raise ValueError("--title can only be used when adding one file.")
    failed = 0
    for path in args.files:
        try:
            result = add_pdf(
                conn, path, title=args.title, size=args.size, overlap=args.overlap, progress=_progress(path, err)
            )
        except PdfError as exc:
            print(f"error: {exc}", file=err)
            failed += 1
            continue
        if isinstance(result, Added):
            _print_added(result, out)
            _make_vectors(conn, args.embedder, result.document.id, out, err)
        else:
            doc = result.document
            print(
                f'{path.name} is already in the library as #{doc.id} "{doc.title}". '
                f"To split it differently, use: pdfqa rechunk {doc.id} --size N",
                file=out,
            )
    return 1 if failed else 0


def _progress(path: Path, err: TextIO):
    if not err.isatty():
        return None

    def report(done: int, total: int) -> None:
        if total >= 20:
            end = "\n" if done == total else ""
            print(f"\rReading {path.name}: {done}/{total} pages", end=end, file=err, flush=True)

    return report


def _make_vectors(conn: sqlite3.Connection, embedder: Embedder, document_id: int, out: TextIO, err: TextIO) -> None:
    """Make a document's search vectors. The document stays added even if that fails."""
    try:
        made = index(conn, embedder, document_id, _counter("Making search vectors", "chunks", err))
    except EmbeddingError as exc:
        print(f"  No search vectors yet. {exc} Then run: pdfqa index", file=out)
        return
    print(f"  Made {_plural(made.chunks, 'search vector')} in {made.seconds:.1f} s.", file=out)


def _counter(label: str, unit: str, err: TextIO):
    if not err.isatty():
        return None

    def report(done: int, total: int) -> None:
        if total >= 20:
            end = "\n" if done == total else ""
            print(f"\r{label}: {done}/{total} {unit}", end=end, file=err, flush=True)

    return report


def _print_added(result: Added, out: TextIO) -> None:
    doc = result.document
    print(
        f'Added #{doc.id} "{doc.title}" ({doc.file_name}): {_plural(doc.page_count, "page")}, '
        f"{_plural(doc.chunk_count, 'chunk')} in {result.seconds:.1f} s.",
        file=out,
    )
    if result.removed_lines:
        print(f"  Removed {_plural(result.removed_lines, 'header or footer line')}.", file=out)
    if doc.empty_pages:
        print(
            f"  {_plural(len(doc.empty_pages), 'page')} with no text, probably scanned: {_numbers(doc.empty_pages)}.",
            file=out,
        )
    if result.unreadable_pages:
        unreadable = result.unreadable_pages
        print(f"  Couldn't read {_plural(len(unreadable), 'page')}: {_numbers(unreadable)}.", file=out)


def _list(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    documents = library.list_documents(conn)
    if not documents:
        print("The library is empty. Add a PDF with: pdfqa add FILE.pdf", file=out)
        return 0
    id_width = max(2, len(str(documents[-1].id)) + 1)
    print(f"{'#':>{id_width}}  {'Title':<{_TITLE_WIDTH}}  {'Pages':>5}  {'Chunks':>6}  Added", file=out)
    for doc in documents:
        print(
            f"{doc.id:>{id_width}}  {_shorten(doc.title, _TITLE_WIDTH):<{_TITLE_WIDTH}}  "
            f"{doc.page_count:>5}  {doc.chunk_count:>6}  {doc.added_at:%Y-%m-%d}",
            file=out,
        )
    return 0


def _show(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    doc = library.get_document(conn, args.document)
    if args.page is not None:
        return _show_page(conn, doc, args.page, out)

    counts = [c.word_count for c in library.get_chunks(conn, doc.id)]
    if doc.empty_pages:
        pages = f"{doc.page_count}, {_plural(len(doc.empty_pages), 'page')} with no text: {_numbers(doc.empty_pages)}"
    else:
        pages = f"{doc.page_count}, all with text"
    print(f"#{doc.id} {doc.title}", file=out)
    print(f"  File     {doc.file_name}", file=out)
    print(f"  Pages    {pages}", file=out)
    print(
        f"  Chunks   {doc.chunk_count} of up to {doc.chunk_size} words with a {doc.chunk_overlap}-word overlap "
        f"(average {round(sum(counts) / len(counts))}, shortest {min(counts)})",
        file=out,
    )
    vectors = library.vector_counts(conn, args.embedder.name).get(doc.id, 0)
    missing = "" if vectors == doc.chunk_count else ". Make the rest with: pdfqa index"
    print(f"  Vectors  {vectors} of {doc.chunk_count}, from {args.embedder.name}{missing}", file=out)
    print(f"  Added    {doc.added_at:%Y-%m-%d %H:%M} UTC", file=out)
    return 0


def _show_page(conn: sqlite3.Connection, doc: library.Document, number: int, out: TextIO) -> int:
    if not 1 <= number <= doc.page_count:
        raise ValueError(f"#{doc.id} has pages 1 to {doc.page_count}.")
    page = library.get_pages(conn, doc.id)[number - 1]
    printed = f' (printed as "{page.label}")' if page.label != str(page.number) else ""
    print(f"Page {page.number} of {doc.page_count}{printed}", file=out)
    print(file=out)
    print(page.text or "(No text on this page. It's probably a scanned image.)", file=out)
    return 0


def _chunks(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    doc = library.get_document(conn, args.document)
    if args.page is not None and not 1 <= args.page <= doc.page_count:
        raise ValueError(f"#{doc.id} has pages 1 to {doc.page_count}.")
    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    labels = {p.number: p.label for p in library.get_pages(conn, doc.id)}
    chunks = library.get_chunks(conn, doc.id, page=args.page)
    shown = chunks if args.all else chunks[: args.limit]
    for chunk in shown:
        pages = _page_ref(chunk.page_start, chunk.page_end, labels)
        heading = f"── chunk {chunk.position + 1} · {pages} · {chunk.word_count} words "
        print(heading + "─" * max(3, _RULE_WIDTH - len(heading)), file=out)
        print(chunk.text, file=out)
        print(file=out)
    hidden = len(chunks) - len(shown)
    if hidden:
        print(f"… {_plural(hidden, 'more chunk')}. Use --limit N or --all to see more.", file=out)
    return 0


def _rechunk(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    before = library.get_document(conn, args.document)
    doc = rechunk(conn, args.document, args.size, args.overlap)
    print(
        f"Split #{doc.id} again: {_plural(doc.chunk_count, 'chunk')} of up to {doc.chunk_size} words "
        f"with a {doc.chunk_overlap}-word overlap (was {before.chunk_count}).",
        file=out,
    )
    _make_vectors(conn, args.embedder, doc.id, out, err)
    return 0


def _remove(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    doc = library.remove_document(conn, args.document)
    print(f'Removed #{doc.id} "{doc.title}".', file=out)
    return 0


def _index(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    made = index(conn, args.embedder, progress=_counter("Making search vectors", "chunks", err))
    if not made.chunks:
        print('Every chunk already has a search vector. Ask something with: pdfqa search "your question"', file=out)
    else:
        rate = made.chunks / made.seconds if made.seconds else 0
        print(
            f"Made {_plural(made.chunks, 'search vector')} in {made.seconds:.1f} s ({rate:.0f} chunks a second).",
            file=out,
        )
    return 0


def _search(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    question = " ".join(args.question)
    try:
        vectors = VectorIndex(conn, args.embedder) if args.mode != "keyword" else None
        hits = search(conn, question, index=vectors, mode=args.mode, limit=args.limit, document_id=args.doc)
    except EmbeddingError as exc:
        raise EmbeddingError(f"{exc} Or search by words alone with --mode keyword.") from exc
    if not hits:
        print("Nothing matched. Try other words, or search by meaning with --mode semantic.", file=out)
        return 0
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        pages = (
            f"p. {chunk.label_start}"
            if chunk.page_start == chunk.page_end
            else f"pp. {chunk.label_start}–{chunk.label_end}"
        )
        found_by = " · ".join(f"{_FOUND_BY[name]} #{rank}" for name, rank in hit.ranks.items())
        heading = f'── {number} · #{chunk.document_id} "{_shorten(chunk.title, 32)}" {pages} · {found_by} '
        print(heading + "─" * max(3, _RULE_WIDTH - len(heading)), file=out)
        words = chunk.text.split()
        print(chunk.text if args.full or len(words) <= 60 else " ".join(words[:60]) + " …", file=out)
        print(file=out)
    return 0


def _ask(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    question = " ".join(args.question)
    if not library.list_documents(conn):
        raise ValueError("The library is empty, so there's nothing to answer from. Add a PDF with: pdfqa add FILE.pdf")
    live = out.isatty()  # in a terminal, show the answer as it's written; otherwise print the tidied one
    if isinstance(args.writer, LocalModel) and not args.writer.is_downloaded():
        print(f"Downloading the answer model, Qwen3-1.7B ({DOWNLOAD_SIZE}, only this once)…", file=err, flush=True)
    vectors = VectorIndex(conn, args.embedder)
    status = _Status(err)
    for event in ask(conn, question, index=vectors, writer=args.writer, limit=args.limit, document_id=args.doc):
        if isinstance(event, Sources):
            status.show(f"Reading {_plural(len(event.hits), 'passage')}…")
        elif isinstance(event, Text):
            status.clear()
            if live:
                print(event.text, end="", file=out, flush=True)
        elif isinstance(event, Done):
            status.clear()
            _print_answer(event.answer, live, out)
    return 0


def _print_answer(answer, live: bool, out: TextIO) -> None:
    if live:
        print(file=out)
    else:
        print(answer.text, file=out)
    print(file=out)
    if answer.found and answer.cited:
        print("Sources", file=out)
        for number in answer.cited:
            print(f"  [{number}] {_source_line(answer.sources[number - 1])}", file=out)
    elif answer.sources:
        print("Closest passages, if you want to look yourself:", file=out)
        for hit in answer.sources[:3]:
            print(f"  {_source_line(hit)}", file=out)
    print(file=out)
    print(f"{answer.model} · {answer.seconds:.1f} s", file=out)


def _source_line(hit: Hit) -> str:
    chunk = hit.chunk
    pages = (
        f"p. {chunk.label_start}"
        if chunk.page_start == chunk.page_end
        else f"pp. {chunk.label_start}–{chunk.label_end}"
    )
    words = chunk.text.split()
    quote = " ".join(words[:18]) + (" …" if len(words) > 18 else "")
    return f'"{_shorten(chunk.title, 32)}" {pages}: {quote}'


class _Status:
    """A one-line note on stderr while waiting (terminals only), erased when output starts."""

    def __init__(self, err: TextIO) -> None:
        self.err, self.width = err, 0

    def show(self, text: str) -> None:
        if self.err.isatty():
            print(f"\r{text}", end="", file=self.err, flush=True)
            self.width = len(text)

    def clear(self) -> None:
        if self.width:
            print("\r" + " " * self.width + "\r", end="", file=self.err, flush=True)
            self.width = 0


def _serve(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from pdfqa.web import serve

    conn.close()  # the server opens a connection per request
    return serve(
        args.db or default_db_path(),
        args.embedder,
        args.writer,
        port=args.port,
        browser=not args.no_browser,
        out=out,
        limit=args.limit,
    )


_COMMANDS = {
    "add": _add,
    "list": _list,
    "show": _show,
    "chunks": _chunks,
    "rechunk": _rechunk,
    "remove": _remove,
    "index": _index,
    "search": _search,
    "ask": _ask,
    "serve": _serve,
}


def _page_ref(start: int, end: int, labels: dict[int, str]) -> str:
    first, last = labels.get(start, str(start)), labels.get(end, str(end))
    return f"p. {first}" if start == end else f"pp. {first}–{last}"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _numbers(numbers: Sequence[int], limit: int = 12) -> str:
    shown = ", ".join(str(n) for n in numbers[:limit])
    return shown + (f" and {len(numbers) - limit} more" if len(numbers) > limit else "")


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
