"""The web server, run for real on a free port, with stand-ins for both models."""

from __future__ import annotations

import http.client
import io
import json
import socket
import threading

import pytest
from helpers import LOREM, ScriptedWriter, WordHashEmbedder, build_pdf, shelf_library, words_to_lines

from pdfqa import library, web
from pdfqa.generate import GenerationError
from pdfqa.web import App, Server, serve

PAGE_HEADERS = {"X-Pdfqa": "1"}


class Client:
    def __init__(self, server: Server) -> None:
        self.port = server.server_address[1]

    def request(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        headers = dict(headers or {})
        if host:
            conn.putrequest(method, path, skip_host=True)
            conn.putheader("Host", host)
            for name, value in headers.items():
                conn.putheader(name, value)
            conn.putheader("Content-Length", str(len(body or b"")))
            conn.endheaders(body)
        else:
            conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, response.getheader("Content-Type"), data

    def json(self, method, path, body=None, headers=None, host=None):
        status, _, data = self.request(method, path, body, headers, host)
        return status, json.loads(data)

    def lines(self, method, path, body=None, headers=PAGE_HEADERS):
        status, kind, data = self.request(method, path, body, headers)
        assert (status, kind) == (200, "application/x-ndjson")
        return [json.loads(line) for line in data.splitlines()]


@pytest.fixture
def embedder():
    return WordHashEmbedder()


@pytest.fixture
def writer():
    return ScriptedWriter("Assembly takes ", "about two hours [1].")


@pytest.fixture
def db(tmp_path, embedder):
    path = tmp_path / "library.db"
    conn = library.connect(path)
    shelf_library(conn, embedder)
    conn.close()
    return path


@pytest.fixture
def client(db, embedder, writer):
    server = Server(App(db, embedder, writer, limit=2), port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield Client(server)
    server.shutdown()
    server.server_close()


def test_the_page(client):
    status, kind, body = client.request("GET", "/")
    assert (status, kind) == (200, "text/html; charset=utf-8")
    assert b"<title>PDF Q&amp;A</title>" in body


def test_the_library(client):
    status, data = client.json("GET", "/api/documents")
    assert status == 200
    assert data["model"] == "test/scripted"
    assert [(d["id"], d["title"], d["pages"], d["chunks"]) for d in data["documents"]] == [
        (1, "Faculty of Science", 2, 2),
        (2, "Assembly", 2, 2),
    ]


def test_asking(client, writer):
    lines = client.lines("POST", "/api/ask", {"question": "How long does wardrobe assembly take?"})
    sources, *texts, answer = lines
    assert len(sources["sources"]) == 2  # the server's limit
    first = sources["sources"][0]
    assert (first["title"], first["label_start"], first["text"]) == (
        "Assembly",
        "i",
        "Assembly takes about two hours for a large wardrobe.",
    )
    assert set(first["ranks"]) == {"semantic", "keyword"}
    assert [t["text"] for t in texts] == ["Assembly takes ", "about two hours [1]."]
    assert answer["answer"]["text"] == "Assembly takes about two hours [1]."
    assert (answer["answer"]["cited"], answer["answer"]["found"], answer["answer"]["model"]) == (
        [1],
        True,
        "test/scripted",
    )


def test_asking_about_one_document(client):
    lines = client.lines("POST", "/api/ask", {"question": "hours", "doc": 1})
    assert {s["title"] for s in lines[0]["sources"]} == {"Faculty of Science"}

    assert client.lines("POST", "/api/ask", {"question": "hours", "doc": 99}) == [
        {"error": "That document isn't in the library any more."}
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"question": "  "}, "Ask a question: it's empty."),
        (b"{not json", 'Send JSON like {"question": "..."}.'),
        ({"question": "q", "doc": "x"}, 'Send JSON like {"question": "..."}.'),
    ],
)
def test_bad_questions(client, body, message):
    assert client.json("POST", "/api/ask", body, PAGE_HEADERS) == (400, {"error": message})


def test_model_errors_reach_the_page(client, writer):
    writer.error = GenerationError("Couldn't download the model.")
    lines = client.lines("POST", "/api/ask", {"question": "hours"})
    assert lines[-1] == {"error": "Couldn't download the model."}


def test_asking_an_empty_library(client):
    for document_id in (1, 2):
        assert client.json("DELETE", f"/api/documents/{document_id}", headers=PAGE_HEADERS)[0] == 200
    assert client.lines("POST", "/api/ask", {"question": "hours"}) == [
        {"error": "The library is empty. Add a PDF first."}
    ]


def test_adding_a_pdf_streams_progress(client, tmp_path):
    lines = words_to_lines(LOREM.replace("wardrobe", "bookcase") * 3)
    pdf = build_pdf(tmp_path / "upload.pdf", [lines] * 2, title="Bookcase guide").read_bytes()

    result = client.lines("POST", "/api/documents?name=../../evil/My%20guide.pdf", pdf)
    stages = {line.get("stage") for line in result[:-1]}
    assert stages == {"reading", "vectors"}
    added = result[-1]["added"]
    assert (added["id"], added["title"], added["file"], added["pages"]) == (3, "Bookcase guide", "My guide.pdf", 2)

    assert client.lines("POST", "/api/documents?name=again.pdf", pdf)[-1]["already"]["id"] == 3

    # The new document is searchable straight away: the vectors were reloaded.
    answer = client.lines("POST", "/api/ask", {"question": "bookcase", "doc": 3})
    assert answer[0]["sources"][0]["title"] == "Bookcase guide"


def test_adding_something_that_isnt_a_pdf(client):
    result = client.lines("POST", "/api/documents?name=notes.pdf", b"<html>hello</html>")
    assert result == [{"error": "notes.pdf isn't a PDF, it's a web page. The download probably went wrong."}]


def test_adding_nothing(client):
    assert client.json("POST", "/api/documents", b"", PAGE_HEADERS) == (413, {"error": "Send a PDF of up to 300 MB."})


def test_an_upload_that_stops_early(client):
    with socket.create_connection(("127.0.0.1", client.port), timeout=10) as sock:
        sock.sendall(
            b"POST /api/documents?name=a.pdf HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Pdfqa: 1\r\nContent-Length: 100\r\n\r\n%PDF-1.7"
        )
        sock.shutdown(socket.SHUT_WR)
        reply = sock.makefile("rb").read()
    assert b"400 Bad Request" in reply
    assert b"The upload stopped before the end of the file." in reply


def test_removing(client):
    status, data = client.json("DELETE", "/api/documents/2", headers=PAGE_HEADERS)
    assert (status, data["removed"]["title"]) == (200, "Assembly")
    assert client.json("DELETE", "/api/documents/2", headers=PAGE_HEADERS) == (
        404,
        {"error": "That document isn't in the library."},
    )
    assert client.json("DELETE", "/api/somewhere", headers=PAGE_HEADERS)[0] == 404


def test_a_page_for_a_citation(client):
    status, data = client.json("GET", "/api/pages/2/1")
    assert status == 200
    assert (data["number"], data["label"], data["document"]["title"]) == (1, "i", "Assembly")
    assert data["text"].startswith("Assembly takes about two hours")
    assert client.json("GET", "/api/pages/2/3") == (404, {"error": "It has pages 1 to 2."})
    assert client.json("GET", "/api/pages/9/1") == (404, {"error": "That document isn't in the library."})


def test_unknown_addresses(client):
    assert client.json("GET", "/nowhere")[0] == 404
    assert client.json("POST", "/api/nowhere", {}, PAGE_HEADERS)[0] == 404


def test_other_hosts_are_refused(client):
    # A website that points its own name at 127.0.0.1 (DNS rebinding) still sends its name as the Host.
    status, data = client.json("GET", "/api/documents", host="evil.example:8000")
    assert (status, data) == (403, {"error": "Open this page at http://127.0.0.1 or http://localhost."})
    assert client.json("GET", "/api/documents", host="localhost:8000")[0] == 200


@pytest.mark.parametrize(
    ("method", "path"), [("POST", "/api/ask"), ("POST", "/api/documents"), ("DELETE", "/api/documents/1")]
)
def test_changes_need_the_page_header(client, method, path):
    status, data = client.json(method, path, {"question": "hours"})
    assert (status, data) == (403, {"error": "Requests that change the library must come from the pdfqa page."})


def test_serve_prints_the_address_and_stops_on_ctrl_c(monkeypatch, db, embedder, writer):
    opened = []
    monkeypatch.setattr(web.webbrowser, "open", opened.append)

    def interrupted(self, poll_interval=0.5):
        raise KeyboardInterrupt

    monkeypatch.setattr(Server, "serve_forever", interrupted)
    out = io.StringIO()
    assert serve(db, embedder, writer, port=0, out=out) == 0
    url = out.getvalue().split()[5]
    assert (
        out.getvalue()
        == f"PDF Q&A is running at {url}  (answers by test/scripted)\nPress Ctrl+C to stop.\n\nStopped.\n"
    )
    assert opened == [url] and url.startswith("http://127.0.0.1:")


def test_serve_on_a_busy_port(db, embedder, writer):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(ValueError, match=f"Can't use port {port}.*pdfqa serve --port 8001"):
            serve(db, embedder, writer, port=port, browser=False, out=io.StringIO())
