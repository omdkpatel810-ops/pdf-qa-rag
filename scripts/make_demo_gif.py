"""Record docs/demo.gif: a question typed into the web page, the answer as it's written, then a citation
opened on its page.

It drives a real browser against a running `pdfqa serve`, so the GIF shows real answers at their real
pace, sped up 4x (the label in the corner says so). Needs Playwright (`pip install playwright` and
`playwright install chromium`):

    pdfqa serve --no-browser &
    python scripts/make_demo_gif.py [--url http://127.0.0.1:8000/]
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
QUESTION = "Why are dot products scaled by the square root of the dimension?"
WIDTH, HEIGHT = 1000, 640
SPEED = 4  # frames are taken every second and shown for a quarter of one
FRAME_SECONDS = 1.0


def label(frame: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 16, box[3] - box[1] + 12
    draw.rounded_rectangle((WIDTH - w - 12, HEIGHT - h - 12, WIDTH - 12, HEIGHT - 12), radius=8, fill=(29, 27, 22))
    draw.text((WIDTH - w - 4, HEIGHT - h - 7), text, font=font, fill=(255, 253, 248))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000/")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    args = parser.parse_args()

    frames: list[tuple[Image.Image, int]] = []  # (picture, milliseconds on screen)

    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": WIDTH, "height": HEIGHT}, color_scheme="light")

        def snap(ms: int, note: str | None = None) -> None:
            frame = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
            frames.append((label(frame, note) if note else frame, ms))

        page.goto(args.url)
        page.wait_for_selector(".doc")
        snap(1200)
        page.click("#question")
        for start in range(0, len(QUESTION), 6):
            page.type("#question", QUESTION[start : start + 6])
            snap(90)
        snap(700)
        page.click("#go")
        while not page.evaluate("document.querySelector('.meta')?.textContent.length > 0"):
            time.sleep(FRAME_SECONDS)
            snap(int(1000 * FRAME_SECONDS / SPEED), f"{SPEED}x speed")
        snap(3500)
        page.click(".src.cited button:has-text('Show on the page')")
        page.wait_for_selector("dialog[open] mark")
        time.sleep(0.6)
        snap(4500)

    pictures = [picture.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for picture, _ in frames]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pictures[0].save(
        args.out, save_all=True, append_images=pictures[1:], duration=[ms for _, ms in frames], loop=0, optimize=True
    )
    print(
        f"{len(frames)} frames, {sum(ms for _, ms in frames) / 1000:.0f} s, {args.out.stat().st_size / 1e6:.1f} MB: {args.out}"
    )


if __name__ == "__main__":
    main()
