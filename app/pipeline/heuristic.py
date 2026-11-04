"""Rule-based planner used when the planning model is unavailable."""

from __future__ import annotations

import re

from app.contracts import (
    EmotionReading,
    Phase,
    Role,
    SafetyLevel,
    UserNeed,
)
from app.pipeline.guardrails import SessionRuntime
from app.pipeline.types import EmotionGuess, PlannerOutput

_EMOTION_PATTERNS = {
    "anxiety_fear": (
        r"stress(?:ed)?",
        r"anxiety",
        r"anxious",
        r"worr(?:y|ied|ies|ying)",
        r"panic",
        r"nervous",
        r"scared",
        r"afraid",
        r"overwhelmed",
        r"pressure",
        r"tense",
        r"dread",
        r"exams?",
        r"deadlines?",
        r"racing",
    ),
    "sadness": (
        r"sad(?:ness)?",
        r"low",
        r"empty",
        r"down",
        r"exhausted",
        r"hopeless",
        r"lonely",
        r"alone",
        r"cry(?:ing|ies)?",
        r"numb",
        r"drained",
        r"tired",
    ),
    "anger": (
        r"angry",
        r"anger",
        r"furious",
        r"frustrat(?:ion|ed)",
        r"irritat(?:ed|ion)",
        r"unfair",
        r"resentful",
        r"fed\s+up",
    ),
    "shame_guilt": (
        r"fail(?:ed|ure)?",
        r"guilt(?:y)?",
        r"ashamed",
        r"shame",
        r"disappointed",
        r"useless",
        r"worthless",
        r"burden",
        r"not\s+good\s+enough",
        r"let\s+someone\s+down",
    ),
    "positive": (
        r"thank(?:s|ful)?",
        r"better",
        r"helped",
        r"helps",
        r"grateful",
        r"relieved",
        r"glad",
    ),
}

_TRIGGER_PATTERNS = (
    ("exams", re.compile(r"\bexams?\b|\btests?\b", re.IGNORECASE)),
    ("deadlines", re.compile(r"\bdeadlines?\b|\bdue dates?\b", re.IGNORECASE)),
    (
        "family expectations",
        re.compile(
            r"\b(?:family|parents?|mum|mom|dad|relatives?)\b|"
            r"\bexpectations?\b|\bexpects?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "work or career",
        re.compile(
            r"\b(?:work(?:ing|ed|place)?|jobs?|career|colleagues?|manager|boss)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "relationships",
        re.compile(
            r"\b(?:relationships?|partner|boyfriend|girlfriend|breakup|dating|friendships?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "money",
        re.compile(r"\b(?:money|financial|finances|debt|bills?|rent|afford)\b", re.IGNORECASE),
    ),
)

_IMPACT_PATTERNS = (
    (
        "sleep difficulties",
        re.compile(
            r"\b(?:can't|cannot|couldn't|hard|trouble|difficulty|difficult)\s+"
            r"(?:to\s+)?sleep(?:ing)?\b|"
            r"\bnot sleeping(?:\s+(?:well|properly))?\b|"
            r"\bsleeping\s+(?:poorly|badly|less)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "appetite changes",
        re.compile(
            r"\b(?:appetite|eating)\b.{0,24}\b(?:chang\w*|decreas\w*|increas\w*|"
            r"less|more|lost|loss|little|low|high|poor)\b|"
            r"\b(?:lost|loss of|no)\s+appetite\b|\bcan't eat\b",
            re.IGNORECASE,
        ),
    ),
    (
        "difficulty concentrating",
        re.compile(
            r"\b(?:can't|cannot|couldn't|hard|trouble|difficulty|difficult)\s+"
            r"(?:to\s+)?(?:focus|concentrate|concentrating)\b|"
            r"\b(?:can't|cannot|couldn't)\s+concentrate\b",
            re.IGNORECASE,
        ),
    ),
    (
        "low energy or motivation",
        re.compile(
            r"\b(?:low|no|little)\s+(?:energy|motivation)\b|"
            r"\b(?:unmotivated|exhausted|drained|tired)\b",
            re.IGNORECASE,
        ),
    ),
)

_DURATION_PATTERN = re.compile(
    r"\b(?:for\s+)?(?:a\s+few|a\s+couple|couple|several|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"\d+)\s+(?:days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_SINCE_PATTERN = re.compile(r"\bsince\s+[^,.!?;\n]{1,60}", re.IGNORECASE)
_NEED_PATTERNS = (
    (
        UserNeed.vent,
        re.compile(r"\bjust listen\b|\bjust need to talk\b|\bneed to vent\b", re.IGNORECASE),
    ),
    (
        UserNeed.act,
        re.compile(
            r"\bwhat can i do\b|\bpractical\b|\bsomething to try\b|\btips\b|\badvice\b",
            re.IGNORECASE,
        ),
    ),
    (
        UserNeed.understand,
        re.compile(r"\bwhy am i\b|\bunderstand\b|\bexplain\b", re.IGNORECASE),
    ),
)
_CLOSING_PATTERN = re.compile(
    r"\b(?:done for today|that's all|that is all|bye|goodbye|good night|"
    r"i['’]m done|talk later)\b",
    re.IGNORECASE,
)
_ACK_PATTERN = re.compile(
    r"^(?:yes|yeah|yep|sure|ok|okay)(?:[\s,]+(?:let['’]s try it|lets try it|"
    r"that helped(?: a bit)?|it helped(?: a bit)?))?[.! ]*$|"
    r"^(?:thanks?(?:\s+you)?|thank you)(?:[,\s]+(?:that|it) helped(?: a bit)?)?[.! ]*$|"
    r"^(?:that helped(?: a bit)?|it helped(?: a bit)?)[.! ]*$",
    re.IGNORECASE,
)


def _emotion_counts(text: str) -> dict[str, int]:
    """Count whole-word emotion-pattern hits in a message."""
    counts: dict[str, int] = {}
    for label, expressions in _EMOTION_PATTERNS.items():
        counts[label] = sum(
            len(re.findall(rf"(?<!\w)(?:{expression})(?!\w)", text, re.IGNORECASE))
            for expression in expressions
        )
    return counts


def _emotion_guesses(
    text: str, rt: SessionRuntime
) -> tuple[list[EmotionGuess], bool]:
    """Return normalized top emotion guesses and whether the message is positive-only."""
    counts = _emotion_counts(text)
    matched = {label: count for label, count in counts.items() if count}
    positive_only = not (set(matched) - {"positive"})
    if rt.state is not None and rt.state.hypotheses and positive_only:
        old = [
            EmotionGuess(label=item.label, probability=item.probability)
            for item in rt.state.hypotheses
        ]
        return old[:3], True

    total = sum(matched.values())
    if total == 0:
        return [EmotionGuess(label="neutral", probability=1.0)], False
    return [
        EmotionGuess(label=label, probability=round(count / total, 3))
        for label, count in sorted(matched.items(), key=lambda item: (-item[1], item[0]))[:3]
    ], False


def _trigger(text: str) -> str | None:
    """Return the earliest deterministic trigger match."""
    matches = [
        (match.start(), label)
        for label, pattern in _TRIGGER_PATTERNS
        if (match := pattern.search(text)) is not None
    ]
    return min(matches)[1] if matches else None


def _impact(text: str) -> str | None:
    """Return the detected functional-impact categories."""
    matched = [label for label, pattern in _IMPACT_PATTERNS if pattern.search(text)]
    return "; ".join(matched) or None


def _duration(text: str) -> str | None:
    """Extract a short duration phrase from a message."""
    match = _DURATION_PATTERN.search(text) or _SINCE_PATTERN.search(text)
    if match is None:
        return None
    value = " ".join(match.group(0).split())
    return re.sub(r"^for\s+", "", value, flags=re.IGNORECASE)


def _need(text: str) -> UserNeed:
    """Detect an explicit conversational need."""
    return next(
        (need for need, pattern in _NEED_PATTERNS if pattern.search(text)),
        UserNeed.unknown,
    )


def _question(
    trigger: str | None, duration: str | None, impact: str | None
) -> str | None:
    """Choose one question for the first missing piece of context."""
    if duration is None:
        return "How long has this been going on?"
    if impact is None:
        return "How has this been affecting your days, like sleep, studies or mood?"
    if trigger is None:
        return "What do you think set this off?"
    return None


def _phase(rt: SessionRuntime, need: UserNeed, *, context_complete: bool) -> Phase:
    """Keep the conversation moving forward without resetting its current phase."""
    if rt.phase == Phase.checkin:
        return Phase.checkin if rt.turn_count == 0 else Phase.exploration
    if (
        context_complete
        and rt.phase == Phase.exploration
        and need == UserNeed.understand
    ):
        return Phase.understanding
    if (
        context_complete
        and rt.phase in (Phase.exploration, Phase.understanding)
        and need == UserNeed.act
    ):
        return Phase.guidance
    return rt.phase


def heuristic_plan(
    rt: SessionRuntime,
    text: str,
    emotion: EmotionReading | None = None,
) -> PlannerOutput:
    """Build a conservative planning result without calling a model."""
    del emotion  # The classifier is a weak hint; fallback emotion inference remains deterministic.
    emotions, preserved_emotions = _emotion_guesses(text, rt)
    trigger = _trigger(text) or (rt.state.trigger if rt.state else None)
    duration = _duration(text) or (rt.state.duration if rt.state else None)
    impact = _impact(text) or (rt.state.functional_impact if rt.state else None)
    detected_need = _need(text)
    effective_need = detected_need
    if effective_need == UserNeed.unknown and rt.state is not None:
        effective_need = rt.state.need
    closing = _CLOSING_PATTERN.search(text) is not None
    question = _question(trigger, duration, impact)
    phase = (
        Phase.closing
        if closing
        else _phase(rt, effective_need, context_complete=question is None)
    )

    if closing:
        move = "close"
    elif (
        rt.last_move == "offer_technique"
        and rt.techniques_offered
        and _ACK_PATTERN.fullmatch(text.strip())
    ):
        move = "follow_up"
    elif effective_need == UserNeed.vent or (
        rt.last_move in ("offer_technique", "psychoeducation", "follow_up")
        and detected_need != UserNeed.act
    ):
        move = "reflect"
    elif question is not None and phase in (Phase.checkin, Phase.exploration):
        move = "ask_clarifying"
    elif phase == Phase.guidance and effective_need == UserNeed.act:
        move = "offer_technique"
    elif effective_need == UserNeed.understand and phase == Phase.understanding:
        move = "psychoeducation"
    else:
        move = "reflect"

    if phase == Phase.guidance and effective_need == UserNeed.act:
        speaker = Role.mentor
    elif phase == Phase.guidance and effective_need == UserNeed.understand:
        speaker = Role.teacher
    else:
        speaker = rt.last_speaker or Role.counsellor

    retrieve = move in ("offer_technique", "psychoeducation")
    query_parts = [
        item.label.replace("_", " ")
        for item in emotions[:2]
        if item.label not in ("neutral", "positive")
    ]
    query_parts.extend(
        value
        for value in (trigger, impact, effective_need.value)
        if value and value not in ("unknown", "vent")
    )
    retrieve_query = " ".join(query_parts) if retrieve else None
    open_questions = [
        item
        for item, missing in (
            ("duration", duration is None),
            ("functional impact", impact is None),
            ("trigger", trigger is None),
        )
        if missing
    ][:3]
    intensity = rt.state.intensity if preserved_emotions and rt.state is not None else 0.5

    return PlannerOutput(
        safety_level=SafetyLevel.none,
        emotions=emotions,
        intensity=intensity,
        trigger=trigger,
        duration=duration,
        functional_impact=impact,
        need=detected_need,
        confidence=0.7,
        open_questions=open_questions,
        phase=phase,
        speaker=speaker,
        reason="Rule-based planning is being used because the planning model was unavailable.",
        move=move,
        clarifying_question=question,
        retrieve=retrieve,
        retrieve_query=retrieve_query,
        wants_new_help=detected_need == UserNeed.act,
    )


__all__ = ["heuristic_plan"]
