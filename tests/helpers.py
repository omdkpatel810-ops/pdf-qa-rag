"""Helpers for building test PDFs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfWriter
from pypdf.constants import PageLabelStyle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

LOREM = (
    "Assembly takes about two hours for a large wardrobe. "
    "Our team brings every tool needed for the job. "
    "Customers can book an evening or a weekend slot. "
    "Every quote is free and comes within one business day. "
)


def build_pdf(
    path: Path,
    pages: Sequence[Sequence[str]],
    *,
    title: str | None = None,
    header: str | None = None,
    footer: str | None = None,
    roman_front_pages: int = 0,
    user_password: str | None = None,
    owner_password: str | None = None,
) -> Path:
    """A PDF with exactly these lines on each page. `{n}` in a header or footer becomes the page number.

    An empty list of lines makes a page with no text at all, like a scanned image.
    """
    pdf = canvas.Canvas(str(path), pagesize=letter)
    if title is not None:
        pdf.setTitle(title)
    for number, lines in enumerate(pages, start=1):
        if lines and header:
            pdf.drawString(72, 750, header.format(n=number))
        y = 720
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 14
        if lines and footer:
            pdf.drawString(72, 40, footer.format(n=number))
        if not lines:
            pdf.rect(100, 100, 300, 300, fill=1)  # an "image" with no text
        pdf.showPage()
    pdf.save()

    if roman_front_pages or user_password is not None:
        writer = PdfWriter(clone_from=str(path))
        if roman_front_pages:
            writer.set_page_label(0, roman_front_pages - 1, style=PageLabelStyle.LOWERCASE_ROMAN, start=1)
            writer.set_page_label(roman_front_pages, len(pages) - 1, style=PageLabelStyle.DECIMAL, start=1)
        if user_password is not None:
            writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm="AES-256")
        with path.open("wb") as handle:
            writer.write(handle)
    return path


def words_to_lines(text: str, width: int = 80) -> list[str]:
    """Wrap text into lines of at most `width` characters, like a PDF would."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
