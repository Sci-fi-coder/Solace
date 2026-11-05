"""Deterministic theme/emotion matching for optional historical analogies."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.contracts import Phase, SafetyLevel, UserNeed
from app.rolemodels.library import RoleModelLibrary
from app.rolemodels.models import Person, RoleModelCard

_PERSON_PATTERNS: dict[Person, re.Pattern[str]] = {
    Person.gandhi: re.compile(r"(?i)\bgandhi\b"),
    Person.vivekananda: re.compile(r"(?i)\b(?:swami\s+)?vivekananda\b"),
    Person.marcus_aurelius: re.compile(r"(?i)\b(?:marcus\s+aurelius|marcus|aurelius)\b"),
    Person.kalam: re.compile(
        r"(?i)\b(?:a\.?\s*p\.?\s*j\.?\s*)?(?:abdul\s+)?kalam\b"
    ),
}
_EMOTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "anger": re.compile(r"(?i)\b(?:ang(?:er|ry)|furious|irritat\w*|frustrat\w*)\b"),
    "anxiety_fear": re.compile(
        r"(?i)\b(?:anxious|anxiety|afraid|fear|fearful|nervous|worried|worry|panic|"
        r"overwhelmed|stress(?:ed)?)\b"
    ),
    "neutral": re.compile(r"(?i)\b(?:numb|flat|neutral|uncertain|unsure)\b"),
    "positive": re.compile(r"(?i)\b(?:hopeful|relieved|grateful|glad|better)\b"),
    "sadness": re.compile(r"(?i)\b(?:sad|low|lonely|grief|hopeless|empty|down)\b"),
    "shame_guilt": re.compile(r"(?i)\b(?:ashamed|guilty|shame|guilt|worthless|failure)\b"),
}
_TAG_SYNONYMS: dict[str, re.Pattern[str]] = {
    "humiliation": re.compile(r"(?i)\bhumiliat(?:ion|ed|ing)\b"),
    "self_doubt": re.compile(r"(?i)\b(?:self[- ]doubt|doubt myself|doubt my)\b"),
    "uncertainty": re.compile(r"(?i)\b(?:uncertain(?:ty)?|unsure)\b"),
    "failure": re.compile(r"(?i)\b(?:fail(?:ed|ing)?|failure)\b"),
}


def requested_person(text: str) -> Person | None:
    """Return a uniquely named supported person, if the user explicitly mentions one."""
    matches = [person for person, pattern in _PERSON_PATTERNS.items() if pattern.search(text)]
    return matches[0] if len(matches) == 1 else None


def _tag_pattern(tag: str) -> re.Pattern[str]:
    words = r"[\s_-]+".join(re.escape(part) for part in tag.split("_"))
    return re.compile(rf"(?i)(?<!\w){words}(?!\w)")


def _tag_matches(tag: str, text: str) -> bool:
    """Match a supplied tag or its narrow, deterministic inflections."""
    pattern = _TAG_SYNONYMS.get(tag, _tag_pattern(tag))
    return bool(pattern.search(text))


def select_role_model(
    library: RoleModelLibrary,
    text: str,
    *,
    emotion_labels: Sequence[str] = (),
) -> RoleModelCard | None:
    """Return a unique clear theme/emotion match or an explicitly requested person card."""
    explicit = requested_person(text)
    candidates = library.for_person(explicit) if explicit is not None else library.all()
    if not candidates:
        return None

    text_emotions = {
        label for label, pattern in _EMOTION_PATTERNS.items() if pattern.search(text)
    }
    text_emotions.update(label for label in emotion_labels if label in _EMOTION_PATTERNS)
    scored: list[tuple[int, int, RoleModelCard]] = []
    for card in candidates:
        theme_hits = sum(_tag_matches(tag, text) for tag in card.tags)
        emotion_hits = len(set(card.emotions) & text_emotions)
        score = 2 * theme_hits + emotion_hits
        if score:
            scored.append((score, theme_hits, card))

    if scored:
        scored.sort(key=lambda result: (-result[0], -result[1], result[2].id))
        best_score, best_themes, best_card = scored[0]
        if explicit is not None:
            return best_card
        tied = [
            item
            for item in scored
            if item[0] == best_score and item[1] == best_themes
        ]
        return best_card if len(tied) == 1 else None

    if explicit is not None:
        return min(candidates, key=lambda card: card.id)
    return None


def role_model_allowed(
    *,
    safety_level: SafetyLevel,
    crisis_seen: bool,
    phase: Phase,
    move: str,
    need: UserNeed,
    explicit_request: bool,
    use_turns: Sequence[int],
    current_turn: int,
) -> bool:
    """Apply safety, listening, phase, and per-session frequency restrictions."""
    if safety_level != SafetyLevel.none or crisis_seen:
        return False
    if need == UserNeed.vent:
        return False
    if move in {
        "reflect",
        "summarise",
        "follow_up",
        "close",
    }:
        return False
    if move == "ask_clarifying" and not explicit_request:
        return False
    if not explicit_request and phase not in (Phase.understanding, Phase.guidance):
        return False
    if len(use_turns) >= 2:
        return False
    return not (use_turns and not explicit_request and current_turn - use_turns[-1] < 5)


__all__ = ["requested_person", "role_model_allowed", "select_role_model"]
