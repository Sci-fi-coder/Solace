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
_ANALOGY_FRAME = re.compile(
    r"(?i)\b(?:"
    r"(?:analogy|parallel|comparison)\b[^.!?]{0,80}\b(?:for|to|with)\b[^.!?]{0,60}"
    r"(?:your\b|you\b|this\b|that\b|what\b|the\b[^.!?]{0,30}\b"
    r"(?:situation|experience|concern|conflict|challenge|team|boundary|support|need)\b|"
    r"[a-z]+ing\b[^.!?]{0,50}\b"
    r"(?:you|your|this|that|someone|people|situation|experience|concern|conflict|"
    r"challenge|team|boundary|support|need)\b)"
    r"|(?:your\b|you\b|what\b|the\b[^.!?]{0,30}\b"
    r"(?:situation|experience|concern|conflict|challenge|team|boundary|support|need)\b)"
    r"[^.!?]{0,60}\b(?:analogy|parallel|comparison)\b"
    r"|as an (?:analogy|parallel|comparison)\b[^.!?]{0,40}\b"
    r"(?:you\b|your\b|what\b|(?:this|that)\b[^.!?]{0,50}\b"
    r"(?:you|your|what|situation|experience|concern|conflict|challenge|team|boundary|"
    r"support|need)\b)"
    r"|(?:compared\s+(?:to|with)|reminds?\s+me\s+of|resonates?\s+with)\s+"
    r"(?:your\b|you\b|this\b|that\b|what\b|the\b[^.!?]{0,30}\b"
    r"(?:situation|experience|concern|conflict|challenge|team|boundary|support|need)\b)"
    r")"
)
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
_SENTENCE_BOUNDARY = re.compile(r"(?<=[!?])\s+|(?<=[a-z0-9])\.\s+")
_SENTENCE_INITIAL_WORD = re.compile(r"^\s*[\"'“‘([{]*\s*([A-Za-z][A-Za-z'-]*)")
_GRAMMATICAL_SENTENCE_STARTS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "he",
        "her",
        "i",
        "if",
        "in",
        "it",
        "she",
        "so",
        "that",
        "the",
        "there",
        "they",
        "this",
        "we",
        "what",
        "when",
        "while",
        "you",
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
_UNVERIFIED_STATUS = re.compile(r"(?i)\b(?:unverified|unreviewed)\b")


def _historical_sentences(reply: str) -> list[str]:
    """Return sentences that name any supported historical person."""
    sentences = _SENTENCE_BOUNDARY.split(reply)
    return [
        sentence
        for sentence in sentences
        if any(pattern.search(sentence) for pattern in _PERSON_PATTERNS.values())
    ]


def has_historical_reference(reply: str) -> bool:
    """Return whether a reply names a supported figure or makes a named-person claim."""
    if _historical_sentences(reply):
        return True
    return any(
        _NAMED_PERSON_CLAIM.search(sentence)
        for sentence in _SENTENCE_BOUNDARY.split(reply)
    )


def _supported_terms(card: RoleModelCard) -> set[str]:
    source = " ".join((card.title, card.summary, card.lesson, card.framing))
    return {word for word in _WORD_RE.findall(source.lower()) if word not in _STOP_WORDS}


def _supported_numbers(card: RoleModelCard) -> set[str]:
    """Return digits explicitly present in the selected card's narrative."""
    source = " ".join((card.title, card.summary, card.lesson, card.framing))
    return set(_DIGIT_NUMBER.findall(source))


def _is_grammatical_sentence_start(sentence: str, word_start: int) -> bool:
    initial = _SENTENCE_INITIAL_WORD.match(sentence)
    return (
        initial is not None
        and initial.start(1) == word_start
        and initial.group(1).lower() in _GRAMMATICAL_SENTENCE_STARTS
    )


def _unsupported_capitalized_names(reply: str, card: RoleModelCard) -> list[str]:
    """Return unsupported capitalized tokens, excluding ordinary sentence openings."""
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
    unsupported: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(reply):
        initial = _SENTENCE_INITIAL_WORD.match(sentence)
        for match in _CAPITALIZED_WORD.finditer(sentence):
            word = match.group()
            if word in allowed or any(
                pattern.fullmatch(word) for pattern in _PERSON_PATTERNS.values()
            ):
                continue
            if initial is not None and match.start() == initial.start(1):
                continue
            unsupported.append(word)
        for match in _NAMED_PERSON_CLAIM.finditer(sentence):
            name = match.group(1)
            if _is_grammatical_sentence_start(sentence, match.start(1)):
                continue
            if name not in allowed and not any(
                pattern.fullmatch(name) for pattern in _PERSON_PATTERNS.values()
            ):
                unsupported.append(name)
    return list(dict.fromkeys(unsupported))


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
        if selected_card is None and any(
            not _is_grammatical_sentence_start(sentence, match.start(1))
            for sentence in _SENTENCE_BOUNDARY.split(reply)
            for match in _NAMED_PERSON_CLAIM.finditer(sentence)
        ):
            issues.append("introduces a named historical figure without selected context")
        return issues
    if selected_card is None:
        issues.append("mentions a historical person without selected historical context")
        return issues

    if not selected_card.reviewed:
        if not _UNVERIFIED_STATUS.search(" ".join(sentences)):
            issues.append("unreviewed historical context must be identified as unverified")
        if any(_FACTUAL_CLAIM.search(sentence) for sentence in sentences):
            issues.append("unreviewed historical details must not be presented as established facts")

    for person, pattern in _PERSON_PATTERNS.items():
        if person != selected_card.person and pattern.search(reply):
            issues.append("introduces a historical person outside the selected card")
            break
    if _QUOTED_TEXT.search(reply):
        issues.append("uses quotation marks in historical context")
    unsupported_names = _unsupported_capitalized_names(reply, selected_card)
    if unsupported_names:
        issues.append(f"unsupported capitalized token(s): {', '.join(unsupported_names)}")
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
