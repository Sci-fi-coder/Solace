"""Extract page-preserving text from the curated local PDF knowledge sources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TypedDict

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / "data" / "kb" / "sources.yaml"
WORK_DIR = ROOT / "data" / "kb_work"
DEFAULT_RAW_DIR = Path(r"C:\Users\DELL\kb_raw")


class SourceRecord(TypedDict):
    """Fields used by the local source registry."""

    id: str
    local_filename: str


class PageText(TypedDict):
    """Extracted text for one 1-based PDF page."""

    page_number: int
    text: str


def load_sources(path: Path = SOURCES_PATH) -> list[SourceRecord]:
    """Load source ids and exact local filenames from the source registry."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError(f"Invalid source registry: {path}")
    sources: list[SourceRecord] = []
    for row in payload["sources"]:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid source record in {path}")
        source_id = row.get("id")
        filename = row.get("local_filename")
        if not isinstance(source_id, str) or not isinstance(filename, str):
            raise ValueError(f"Source record requires id and local_filename in {path}")
        sources.append({"id": source_id, "local_filename": filename})
    return sources


def raw_directory() -> Path:
    """Return KB_RAW_DIR or the documented default without reading dotenv files."""
    configured = os.environ.get("KB_RAW_DIR")
    return Path(configured) if configured else DEFAULT_RAW_DIR


def inventory(raw_dir: Path, sources: list[SourceRecord]) -> tuple[dict[str, Path], list[str]]:
    """Print all files, match exact mapped filenames, and report unmatched or missing files."""
    expected = {source["local_filename"]: source["id"] for source in sources}
    found: dict[str, Path] = {}
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"KB_RAW_DIR is not a directory: {raw_dir}")

    print(f"KB_RAW_DIR: {raw_dir}")
    files = sorted(path for path in raw_dir.iterdir() if path.is_file())
    if not files:
        print("Files found: none")
    for path in files:
        page_count = "n/a"
        if path.suffix.casefold() == ".pdf":
            import fitz

            try:
                with fitz.open(path) as document:
                    page_count = str(len(document))
            except (fitz.FileDataError, RuntimeError) as exc:
                page_count = f"unreadable ({type(exc).__name__})"
        source_id = expected.get(path.name)
        if source_id is not None:
            found[source_id] = path
            match = f"matched: {source_id}"
        else:
            match = "unmatched: ignored"
        print(f"{path.name}\t{path.stat().st_size} bytes\t{page_count} pages\t{match}")

    missing = [
        source["id"] for source in sources if source["local_filename"] not in
        {path.name for path in files}
    ]
    for source_id in missing:
        filename = next(item["local_filename"] for item in sources if item["id"] == source_id)
        print(f"Missing source: {source_id} ({filename})")
    return found, missing


def extract_pdf(source_id: str, pdf_path: Path, output_dir: Path = WORK_DIR) -> list[int]:
    """Extract each PDF page into one JSON record and return pages with no text."""
    import fitz

    pages: list[PageText] = []
    empty_pages: list[int] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            if not text.strip():
                empty_pages.append(page_number)
            pages.append({"page_number": page_number, "text": text})

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_id}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    print(f"Extracted {len(pages)} pages: {output_path}")
    if empty_pages:
        print(f"Pages with no extractable text ({source_id}): {empty_pages}")
    else:
        print(f"Pages with no extractable text ({source_id}): none")
    return empty_pages


def main() -> int:
    """Inventory mapped PDFs and extract text for each exactly matched source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Extract only this source id.")
    args = parser.parse_args()

    sources = load_sources()
    if args.source and args.source not in {source["id"] for source in sources}:
        parser.error(f"unknown source id: {args.source}")
    found, _ = inventory(raw_directory(), sources)
    selected = [args.source] if args.source else [source["id"] for source in sources]
    extracted = 0
    for source_id in selected:
        pdf_path = found.get(source_id)
        if pdf_path is None:
            continue
        extract_pdf(source_id, pdf_path)
        extracted += 1
    if extracted == 0:
        print("No exact-mapped PDFs were available to extract.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
