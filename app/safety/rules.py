"""Reviewable, rule-based safety screening patterns."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import yaml

from app.contracts import SafetyLevel

logger = logging.getLogger(__name__)
_PATTERN_FILES = ("patterns_en.yaml", "patterns_hinglish.yaml")
_TYPO_REPLACEMENTS = {
    "suicde": "suicide",
    "suicidal": "suicidal",
    "selfharm": "self harm",
    "k1ll": "kill",
    "h4rm": "harm",
}
_LEET_TRANSLATION = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s"})
_NEGATION = re.compile(
    r"\b(?:not|never|don't|do not|won't|will not|wouldn't|would not)"
    r"(?:\s+(?:going to|want to|plan to|intend to))?\s*$",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"[.!?;\n]")
_REPEATED_LETTER = re.compile(r"([a-z])\1{2,}")


class PatternEntry(TypedDict):
    """A stable rule code and its human-editable regular expression."""

    code: str
    expression: str


def normalize_text(text: str) -> str:
    """Normalize casing, common obfuscations, repeated letters, and punctuation."""
    normalized = text.lower().translate(_LEET_TRANSLATION)
    for typo, correction in _TYPO_REPLACEMENTS.items():
        normalized = re.sub(rf"\b{re.escape(typo)}\b", correction, normalized)
    normalized = _REPEATED_LETTER.sub(r"\1", normalized)
    normalized = re.sub(r"[^\w\s']", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@lru_cache(maxsize=1)
def _load_patterns() -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    """Load and compile safety patterns from the reviewable YAML data files."""
    data_directory = Path(__file__).resolve().parents[2] / "data" / "safety"
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {
        "crisis": [],
        "elevated": [],
    }
    for filename in _PATTERN_FILES:
        path = data_directory / filename
        with path.open(encoding="utf-8") as handle:
            groups = yaml.safe_load(handle)
        if not isinstance(groups, dict):
            raise ValueError(f"Safety patterns in {filename} must be a mapping.")
        for level in ("crisis", "elevated"):
            categories = groups.get(level, {})
            if not isinstance(categories, dict):
                raise ValueError(f"Safety pattern level {level!r} in {filename} must be a mapping.")
            for category, expressions in categories.items():
                if not isinstance(expressions, list):
                    raise ValueError(f"Safety pattern category {category!r} must be a list.")
                for expression in expressions:
                    if not isinstance(expression, str):
                        raise ValueError(f"Safety patterns in {filename} must be strings.")
                    compiled[level].append(
                        (
                            f"{filename.removesuffix('.yaml')}:{category}",
                            re.compile(rf"\b(?:{expression})\b"),
                        )
                    )
    return compiled


def _is_negated(text: str, match_start: int) -> bool:
    """Ignore local negative intent, without suppressing a later positive clause."""
    sentence_start = max(
        (boundary.end() for boundary in _SENTENCE_BOUNDARY.finditer(text, 0, match_start)),
        default=0,
    )
    prefix = text[sentence_start:match_start]
    return _NEGATION.search(prefix) is not None


def classify_rules(message: str) -> tuple[SafetyLevel, list[str]]:
    """Return the highest matching rule level and stable reason codes."""
    normalized = normalize_text(message)
    patterns = _load_patterns()
    crisis_reasons = [
        code
        for code, pattern in patterns["crisis"]
        if any(not _is_negated(normalized, match.start()) for match in pattern.finditer(normalized))
    ]
    if crisis_reasons:
        return SafetyLevel.crisis, list(dict.fromkeys(crisis_reasons))
    elevated_reasons = [
        code
        for code, pattern in patterns["elevated"]
        if any(not _is_negated(normalized, match.start()) for match in pattern.finditer(normalized))
    ]
    if elevated_reasons:
        return SafetyLevel.elevated, list(dict.fromkeys(elevated_reasons))
    return SafetyLevel.none, []
