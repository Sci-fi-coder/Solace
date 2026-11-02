"""Shared deny-list and line-level sanitizer for RAG passage ingestion.

This module is imported by both ``scripts/kb_build.py`` and tests without
pulling in all of ``scripts/kb_draft_cards.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.cards import banned_content_reasons

# ── Deny-list ────────────────────────────────────────────────────────────────

_PAGE_RANGE_RE = re.compile(r"(\d+)-(\d+)")


def _parse_page_set(spec: str) -> frozenset[int]:
    """Parse a page-range spec like '21-22,36-40,139-148' into a set of ints."""
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        m = _PAGE_RANGE_RE.fullmatch(part)
        if m:
            pages.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            pages.add(int(part))
    return frozenset(pages)


def build_deny_set(deny_sections: list[dict]) -> frozenset[int]:
    """Return the union of all denied page numbers from a sources.yaml deny_sections list.

    Args:
        deny_sections: The ``deny_sections`` list from a sources.yaml source entry.
            Each item has a ``pages`` key (string) and a ``name`` key.
    """
    denied: set[int] = set()
    for section in deny_sections:
        pages_spec = str(section.get("pages", ""))
        denied.update(_parse_page_set(pages_spec))
    return frozenset(denied)


# ── Line-level sanitizer ──────────────────────────────────────────────────────

_SOURCE_LINE_EXCLUSION_REASONS = frozenset(
    {
        "phone number",
        "emergency or helpline number",
        "URL",
        "dosage",
        "medication name or reference",
        "helpline name or reference",
    }
)

_MIN_CHARS = 200


@dataclass
class PageText:
    """A single page of text from a knowledge-base source."""

    page_number: int
    text: str


def sanitize_page(page: PageText) -> tuple[PageText | None, str | None]:
    """Remove lines containing contact/medication details; reject very short pages.

    Args:
        page: The page to sanitize.

    Returns:
        A tuple of (sanitized PageText or None, rejection reason or None).
        If the page is rejected, the first element is None.
    """
    clean_lines: list[str] = []
    for line in page.text.splitlines():
        reasons = banned_content_reasons(line)
        if _SOURCE_LINE_EXCLUSION_REASONS.intersection(reasons):
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines).strip()
    if len(text) < _MIN_CHARS:
        return None, f"too short after sanitisation ({len(text)} characters; minimum {_MIN_CHARS})"
    return PageText(page_number=page.page_number, text=text), None


# ── Word-based chunker ────────────────────────────────────────────────────────

_CHUNK_WORDS = 200
_OVERLAP_WORDS = 40


def chunk_text(text: str, chunk_words: int = _CHUNK_WORDS, overlap_words: int = _OVERLAP_WORDS) -> list[str]:
    """Split ``text`` into overlapping word-based chunks.

    Args:
        text: Source text to split.
        chunk_words: Target word count per chunk.
        overlap_words: Number of words to overlap between consecutive chunks.

    Returns:
        A list of text chunks (at least one, even for very short text).
    """
    words = text.split()
    if not words:
        return []
    step = max(chunk_words - overlap_words, 1)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


__all__ = [
    "PageText",
    "build_deny_set",
    "chunk_text",
    "sanitize_page",
]
