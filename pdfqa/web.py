"""The web page: `pdfqa serve` runs a small server on this computer and opens it in a browser.

It uses only Python's standard library (http.server). Answers and progress stream back as one JSON object
per line, so the page can show the answer as it's written.

    GET    /                      the page
    GET    /api/documents         the library
    POST   /api/documents?name=F  add a PDF (the request body); streams progress, then the document
    DELETE /api/documents/ID      remove one
    POST   /api/ask               {"question": ..., "doc": ID or null}; streams sources, text, then the answer
    GET    /api/pages/ID/N        one page's cleaned text, for showing a citation in context

It listens on 127.0.0.1 only. Two checks stop other websites open in the same browser from using it:
requests must name this server as their Host (which defeats DNS rebinding), and requests that change
anything must carry an X-Pdfqa header, which a page from another site can't add without the server's
permission (a CORS preflight it never grants).
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import parse_qs, urlparse

from pdfqa import library
from pdfqa.answer import Done, Sources, Text, ask
from pdfqa.embed import Embedder, EmbeddingError
from pdfqa.extract import PdfError
from pdfqa.generate import GenerationError, Writer
from pdfqa.ingest import Added, add_pdf, index
from pdfqa.search import Hit, VectorIndex

MAX_UPLOAD = 300 * 1024 * 1024  # 300 MB; the 1,239-page UofM calendar is 9 MB
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}


class App:
    """What the request handlers share: the library's path, the models, and one lock for model work."""

    def __init__(self, db_path: str | Path, embedder: Embedder, writer: Writer, limit: int = 5) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.writer = writer
        self.limit = limit
        self.busy = threading.Lock()  # one answer or one PDF at a time: they'd only slow each other down
        self._vectors: VectorIndex | None = None

    def connect(self) -> sqlite3.Connection:
        return library.connect(self.db_path)

    def vectors(self, conn: sqlite3.Connection) -> VectorIndex:
        """The vectors, loaded once and reloaded after the library changes. Call while holding `busy`."""
        if self._vectors is None:
            self._vectors = VectorIndex(conn, self.embedder)
        return self._vectors

    def changed(self) -> None:
        self._vectors = None


class Handler(BaseHTTPRequestHandler):
    server: Server
    protocol_version = "HTTP/1.0"  # each response ends by closing the connection, which ends a stream

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        pass  # keep the terminal quiet

    # Routing

    def do_GET(self) -> None:  # noqa: N802 - the base class's name
        if not self._local():
            return
        path = urlparse(self.path).path
        if path == "/":
            page = resources.files("pdfqa").joinpath("static/index.html").read_bytes()
            self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
        elif path == "/api/documents":
            with self._library() as conn:
                self._json(
                    {"documents": [_document(d) for d in library.list_documents(conn)], "model": self.app.writer.name}
                )
        elif match := re.fullmatch(r"/api/pages/(\d+)/(\d+)", path):
            self._page(int(match[1]), int(match[2]))
        else:
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if not (self._local() and self._from_page()):
            return
        url = urlparse(self.path)
        if url.path == "/api/documents":
            name = parse_qs(url.query).get("name", ["document.pdf"])[0]
            self._add(name)
        elif url.path == "/api/ask":
            self._ask()
        else:
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:  # noqa: N802
        if not (self._local() and self._from_page()):
            return
        match = re.fullmatch(r"/api/documents/(\d+)", urlparse(self.path).path)
        if not match:
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        with self.app.busy, self._library() as conn:
            try:
                doc = library.remove_document(conn, int(match[1]))
            except library.NotFound:
                self._json({"error": "That document isn't in the library."}, HTTPStatus.NOT_FOUND)
                return
            self.app.changed()
        self._json({"removed": _document(doc)})

    # Actions

    def _page(self, document_id: int, number: int) -> None:
        with self._library() as conn:
            try:
                doc = library.get_document(conn, document_id)
            except library.NotFound:
                self._json({"error": "That document isn't in the library."}, HTTPStatus.NOT_FOUND)
                return
            if not 1 <= number <= doc.page_count:
                self._json({"error": f"It has pages 1 to {doc.page_count}."}, HTTPStatus.NOT_FOUND)
                return
            page = library.get_pages(conn, document_id)[number - 1]
        self._json({"document": _document(doc), "number": page.number, "label": page.label, "text": page.text})

    def _add(self, name: str) -> None:
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_UPLOAD:
            self._json({"error": "Send a PDF of up to 300 MB."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        name = Path(name.replace("\\", "/")).name or "document.pdf"  # a name, never a path
        with tempfile.TemporaryDirectory(prefix="pdfqa-") as folder:
            path = Path(folder) / name
            with path.open("wb") as file:
                try:
                    _copy(self.rfile, file, size)
                except ValueError as error:
                    self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                    return
            self._start_stream()
            with self.app.busy, self._library() as conn:
                try:
                    result = add_pdf(
                        conn, path, progress=lambda done, total: self._line(stage="reading", done=done, total=total)
                    )
                    if isinstance(result, Added):
                        self.app.changed()
                        index(
                            conn,
                            self.app.embedder,
                            result.document.id,
                            lambda done, total: self._line(stage="vectors", done=done, total=total),
                        )
                        self._line(added=_document(library.get_document(conn, result.document.id)))
                    else:
                        self._line(already=_document(result.document))
                except (PdfError, ValueError, EmbeddingError) as error:
                    self._line(error=str(error))

    def _ask(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            question = str(body.get("question") or "").strip()
            document_id = body.get("doc")
            document_id = int(document_id) if document_id is not None else None
        except (ValueError, TypeError):
            self._json({"error": 'Send JSON like {"question": "..."}.'}, HTTPStatus.BAD_REQUEST)
            return
        if not question:
            self._json({"error": "Ask a question: it's empty."}, HTTPStatus.BAD_REQUEST)
            return
        self._start_stream()
        with self.app.busy, self._library() as conn:
            try:
                if not library.list_documents(conn):
                    raise ValueError("The library is empty. Add a PDF first.")
                for event in ask(
                    conn,
                    question,
                    index=self.app.vectors(conn),
                    writer=self.app.writer,
                    limit=self.app.limit,
                    document_id=document_id,
                ):
                    if isinstance(event, Sources):
                        self._line(sources=[_source(hit) for hit in event.hits])
                    elif isinstance(event, Text):
                        self._line(text=event.text)
                    elif isinstance(event, Done):
                        answer = event.answer
                        self._line(
                            answer={
                                "text": answer.text,
                                "cited": answer.cited,
                                "found": answer.found,
                                "model": answer.model,
                                "seconds": round(answer.seconds, 1),
                            }
                        )
            except library.NotFound:
                self._line(error="That document isn't in the library any more.")
            except (ValueError, EmbeddingError, GenerationError) as error:
                self._line(error=str(error))

    # Checks

    def _local(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in _LOCAL_HOSTS:
            self._json({"error": "Open this page at http://127.0.0.1 or http://localhost."}, HTTPStatus.FORBIDDEN)
            return False
        return True

    def _from_page(self) -> bool:
        if self.headers.get("X-Pdfqa") != "1":
            self._json(
                {"error": "Requests that change the library must come from the pdfqa page."}, HTTPStatus.FORBIDDEN
            )
            return False
        return True

    # Responses

    @property
    def app(self) -> App:
        return self.server.app

    def _library(self) -> _Connection:
        return _Connection(self.app)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _start_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _line(self, **data: Any) -> None:
        self.wfile.write(json.dumps(data).encode() + b"\n")
        self.wfile.flush()


class _Connection:
    """`with` a library connection that's always closed."""

    def __init__(self, app: App) -> None:
        self.conn = app.connect()

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, *exc: object) -> None:
        self.conn.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, app: App, port: int = 8000) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.app = app

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


def serve(
    db_path: str | Path,
    embedder: Embedder,
    writer: Writer,
    *,
    port: int = 8000,
    browser: bool = True,
    out: TextIO,
    limit: int = 5,
) -> int:
    try:
        server = Server(App(db_path, embedder, writer, limit), port)
    except OSError as error:
        raise ValueError(
            f"Can't use port {port} ({error.strerror}). Try another one: pdfqa serve --port 8001"
        ) from error
    print(f"PDF Q&A is running at {server.url}  (answers by {writer.name})", file=out)
    print("Press Ctrl+C to stop.", file=out, flush=True)
    if browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=out)
    finally:
        server.server_close()
    return 0


def _copy(source: Any, target: Any, size: int) -> None:
    remaining = size
    while remaining:
        block = source.read(min(remaining, 1 << 20))
        if not block:
            raise ValueError("The upload stopped before the end of the file.")
        target.write(block)
        remaining -= len(block)


def _document(doc: library.Document) -> dict[str, Any]:
    return {"id": doc.id, "title": doc.title, "file": doc.file_name, "pages": doc.page_count, "chunks": doc.chunk_count}


def _source(hit: Hit) -> dict[str, Any]:
    chunk = asdict(hit.chunk)
    chunk["ranks"] = hit.ranks
    return chunk
