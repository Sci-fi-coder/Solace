"""Deterministic checks for historical-card use in generated replies."""

from __future__ import annotations

import re

from app.rolemodels.models import Person, RoleModelCard

_PERSON_PATTERNS: dict[Person, re.Pattern[str]] = {
    Person.gandhi: re.compile(r"(?i)\bgandhi\b"),
    Person.vivekananda: re.compile(r"(?i)\b(?:swami\s+)?vivekananda\b"),
    Person.marcus_aurelius: re.compile(r"(?i)\b(?:marcus\s+aurelius|marcus|aurelius)\b"),
    Person.kalam: re.compile(r"(?i)\b(?:a\.?\s*p\.?\s*j\.?\s*)?(?:abdul\s+)?kalam\b"),
}
_QUOTED_TEXT = re.compile(r'"[^"\n]+"|“[^”\n]+”|(?<!\w)\'[^\'\n]{3,}\'')
_DIGIT_NUMBER = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2}|[0-9]+)\b")
_NUMBER_WORD = re.compile(r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b")
_NUMBER_CONTEXT = re.compile(
    r"(?i)\b(?:years?|months?|decades?|centur(?:y|ies)|times?|campaigns?|"
    r"launches?|people|groups?|followers?|members?|books?|leaders?)\b"
)
_IDENTITY_CLAIM = re.compile(
    r"(?i)\b(?:the same situation|identical situation|exactly like|went through the same|"
    r"the same thing happened to)\b"
)
_ANALOGY_FRAME = re.compile(r"(?i)\b(?:analogy|parallel|reminds? me of|resonates?)\b")
_FACTUAL_CLAIM = re.compile(
    r"(?i)\b(?:was|were|did|led|wrote|said|founded|fought|won|served|worked|lived|"
    r"travelled|traveled|became|created|invented|studied|published|died|born|met|"
    r"married|imprisoned|removed|arrived|launched|held|organized|organised)\b"
)
_NAMED_PERSON_CLAIM = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:was|were|led|wrote|said|founded|fought|won|served|"
    r"worked|lived|became|created|invented|studied|published|died|born|met|married|"
    r"imprisoned|removed|arrived|launched|held|organized|organised)\b"
)
_CAPITALIZED_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")
_COMMON_SENTENCE_STARTS = frozenset(
    {
        "A",
        "An",
        "And",
        "As",
        "At",
        "But",
        "For",
        "From",
        "He",
        "Her",
        "Here",
        "I",
        "If",
        "In",
        "It",
        "Let",
        "Maybe",
        "One",
        "Perhaps",
        "She",
        "Solace",
        "Sometimes",
        "That",
        "The",
        "There",
        "They",
        "This",
        "We",
        "What",
        "When",
        "While",
        "You",
    }
)
_WORD_RE = re.compile(r"[a-z]{4,}")
_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "because",
        "been",
        "being",
        "could",
        "from",
        "have",
        "into",
        "more",
        "only",
        "should",
        "their",
        "there",
        "these",
        "this",
        "through",
        "those",
        "very",
        "what",
        "when",
        "which",
        "while",
        "with",
        "your",
    }
)


def _historical_sentences(reply: str) -> list[str]:
    """Return sentences that name any supported historical person."""
    sentences = re.split(r"(?<=[.!?])\s+", reply)
    return [
        sentence
        for sentence in sentences
        if any(pattern.search(sentence) for pattern in _PERSON_PATTERNS.values())
    ]


def _supported_terms(card: RoleModelCard) -> set[str]:
    source = " ".join((card.title, card.summary, card.lesson, card.framing))
    return {word for word in _WORD_RE.findall(source.lower()) if word not in _STOP_WORDS}


def _supported_numbers(card: RoleModelCard) -> set[str]:
    """Return digits explicitly present in the selected card's narrative."""
    source = " ".join((card.title, card.summary, card.lesson, card.framing))
    return set(_DIGIT_NUMBER.findall(source))


def _unsupported_capitalized_names(reply: str, card: RoleModelCard) -> bool:
    """Detect likely unprovided proper names while allowing card-mentioned places."""
    narrative = " ".join((card.title, card.summary, card.lesson, card.framing))
    allowed = set(_CAPITALIZED_WORD.findall(narrative))
    allowed.update(
        {
            Person.gandhi: {"Gandhi", "Mahatma"},
            Person.vivekananda: {"Swami", "Vivekananda"},
            Person.marcus_aurelius: {"Marcus", "Aurelius"},
            Person.kalam: {"A", "P", "J", "Abdul", "Kalam"},
        }[card.person]
    )
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        for index, match in enumerate(_CAPITALIZED_WORD.finditer(sentence)):
            word = match.group()
            if word in allowed or any(
                pattern.fullmatch(word) for pattern in _PERSON_PATTERNS.values()
            ):
                continue
            if index == 0 and word in _COMMON_SENTENCE_STARTS:
                continue
            return True
    return False


def check_role_model_reply(
    reply: str,
    *,
    selected_card: RoleModelCard | None,
    used_card_id: str | None,
) -> list[str]:
    """Return issues that make historical provenance unsafe or unsupported."""
    sentences = _historical_sentences(reply)
    issues: list[str] = []
    if used_card_id is not None and (
        selected_card is None or used_card_id != selected_card.id
    ):
        issues.append("claims use of a historical card that was not selected")
    if sentences and used_card_id is None:
        issues.append("mentions a historical person without declaring the selected card as used")
    if not sentences:
        if used_card_id is not None:
            issues.append("declares historical-card use without mentioning its person")
        if selected_card is None:
            for match in _NAMED_PERSON_CLAIM.finditer(reply):
                if match.group(1) not in _COMMON_SENTENCE_STARTS:
                    issues.append("introduces a named historical figure without selected context")
                    break
        return issues
    if selected_card is None:
        issues.append("mentions a historical person without selected historical context")
        return issues

    for person, pattern in _PERSON_PATTERNS.items():
        if person != selected_card.person and pattern.search(reply):
            issues.append("introduces a historical person outside the selected card")
            break
    if _QUOTED_TEXT.search(reply):
        issues.append("uses quotation marks in historical context")
    if _unsupported_capitalized_names(reply, selected_card):
        issues.append("introduces a named person or place outside the selected card")
    if _IDENTITY_CLAIM.search(reply):
        issues.append("claims the user's situation is the same as the historical person's")

    if used_card_id == selected_card.id and not _ANALOGY_FRAME.search(reply):
        issues.append("historical example must be clearly framed as an analogy")

    supported = _supported_terms(selected_card)
    supported_numbers = _supported_numbers(selected_card)
    for sentence in sentences:
        digit_numbers = _DIGIT_NUMBER.findall(sentence)
        spelled_numbers = _NUMBER_WORD.findall(sentence)
        if digit_numbers and not set(digit_numbers) <= supported_numbers:
            issues.append("introduces an unsupported historical date or number")
            break
        if spelled_numbers and _NUMBER_CONTEXT.search(sentence):
            number_words = {word.lower() for word in spelled_numbers}
            if not number_words <= supported:
                issues.append("introduces an unsupported historical date or number")
                break
        terms = {word for word in _WORD_RE.findall(sentence.lower()) if word not in _STOP_WORDS}
        if _FACTUAL_CLAIM.search(sentence) and len(terms & supported) < 3:
            issues.append("historical claim is not grounded in the selected card")
            break
    return list(dict.fromkeys(issues))


__all__ = ["check_role_model_reply"]
