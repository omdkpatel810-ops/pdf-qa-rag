"""Download the sample PDFs used in the README into samples/ (which git ignores).

python scripts/get_samples.py           # the three small ones (about 3 MB)
python scripts/get_samples.py --large   # also the 1,239-page UofM calendar (21 MB)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SAMPLES = {
    "bert.pdf": "https://arxiv.org/pdf/1810.04805v2",
    "attention.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "uofm-science.pdf": "https://catalog.umanitoba.ca/undergraduate-studies/science/science.pdf",
}
LARGE = {
    "uofm-undergrad-2023-24.pdf": "https://catalog.umanitoba.ca/pdf/23-24%20Undergraduate%20Studies.pdf",
}
FOLDER = Path(__file__).resolve().parent.parent / "samples"


def download(name: str, url: str) -> None:
    target = FOLDER / name
    if target.exists():
        print(f"{name}: already downloaded")
        return
    request = urllib.request.Request(url, headers={"User-Agent": "pdf-qa-rag sample downloader"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    if not data.startswith(b"%PDF-"):
        raise SystemExit(f"{name}: {url} didn't return a PDF; the link may have moved.")
    target.write_bytes(data)
    print(f"{name}: {len(data) / 1_000_000:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--large", action="store_true", help="also download the full UofM calendar")
    args = parser.parse_args()
    FOLDER.mkdir(exist_ok=True)
    for name, url in {**SAMPLES, **(LARGE if args.large else {})}.items():
        download(name, url)


if __name__ == "__main__":
    sys.exit(main())
