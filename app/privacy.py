"""Best-effort PII scrubbing applied before any text leaves the machine for an LLM provider."""

from __future__ import annotations

import re
from collections.abc import Callable

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "[email]"),
    (re.compile(r"(?i)\b(?:https?://|www\.)\S+"), "[link]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[id]"),  # PAN-style identifiers
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[id]"),  # 12-digit national-id style numbers
    (re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?){2,4}\d{2,4}(?!\w)"), "[number]"),
)


def scrub_pii(text: str) -> str:
    """Replace emails, links, id-like numbers and phone-like numbers with placeholders."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def message_filter(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Gateway pre-send filter: scrub the content of every message."""
    return [{**m, "content": scrub_pii(m.get("content", ""))} for m in messages]


def install_privacy_filter(gateway: object) -> None:
    """Register the scrubber on a gateway that supports add_filter()."""
    add: Callable[..., None] | None = getattr(gateway, "add_filter", None)
    if add is not None:
        add(message_filter)
