"""Local, deterministic reply checks (no LLM call) and safe fallback replies."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.contracts import CriticVerdict, Role
from app.pipeline.roles import ROLES

MAX_WORDS = 140

_DIAGNOSIS = re.compile(
    r"(?i)\byou(?:'ve| have| may have| might have| probably have| seem to have| likely have)\s+"
    r"(?:an? )?(?:depress\w*|anxiety disorder|ptsd|bipolar|adhd|ocd|panic disorder|disorder|"
    r"mental illness|clinical \w+)\b|\bdiagnos\w*\b|\byou(?:'re| are) (?:clinically )?(?:depressed|bipolar)\b"
)
_MEDICATION = re.compile(
    r"(?i)\b(?:medicat\w*|medicine\w*|pills?|antidepress\w*|dosage|prescri\w*|supplements?|\d+\s?mg)\b"
)
_CONTACT = re.compile(
    r"(?i)(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+\.\w+|\b(?:helpline|hotline|lifeline)\b|"
    r"(?<!\w)\+?\d[\d\s().-]{7,}\d"
)
_MINIMISING = re.compile(
    r"(?i)\b(?:just relax|calm down|cheer up|don'?t worry|do not worry|it could be worse|snap out of it|"
    r"everything happens for a reason|look on the bright side|at least you|stop overthinking)\b"
)
_LIVED_OR_CREDENTIAL = re.compile(
    r"(?i)\bas (?:your|a) (?:licensed |professional |qualified )?(?:therapist|doctor|psychologist|psychiatrist)\b|"
    r"\bI(?:'m| am) (?:a )?(?:human|real person|licensed|certified)\b|"
    r"\bwhen I (?:was|went|had|felt|lost|struggled)\b|\bin my (?:own )?(?:experience|life)\b|\bI (?:also )?went through\b"
)
_SAFETY_WORDS = re.compile(r"(?i)\b(?:suicid\w*|kill (?:yourself|myself)|self[- ]harm\w*|end (?:your|my) life)\b")
_DIRECTIVE = re.compile(r"(?i)\byou (?:should|must|have to|need to|ought to)\b")
_LIST_LINE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-•*])\s+")

FALLBACKS: dict[str, str] = {
    "reflect": (
        "That sounds really hard, and I'm glad you said it out loud. "
        "I'm here and listening. What feels heaviest right now?"
    ),
    "summarise": (
        "Let me check I'm following you. Does the picture I've shared of what you're going "
        "through feel about right?"
    ),
    "follow_up": (
        "I'm glad that was useful. Would you like to keep talking about it, or wrap up for today?"
    ),
    "close": (
        "Thank you for talking with me today. Be gentle with yourself, and come back whenever "
        "you'd like to talk again."
    ),
}


def fallback_reply(move: str, question: str | None = None) -> str:
    """A safe, generic reply used when generation or checking fails twice."""
    if move == "ask_clarifying" and question:
        return f"Thank you for telling me that. {question}"
    if move in ("offer_technique", "psychoeducation", "ask_clarifying"):
        return FALLBACKS["reflect"]
    return FALLBACKS.get(move, FALLBACKS["reflect"])


def check_reply(
    reply: str,
    *,
    role: Role,
    move: str,
    retrieved_ids: Sequence[str],
    used_card_ids: Sequence[str],
    previous_reply: str | None = None,
) -> tuple[CriticVerdict, list[str]]:
    """Run every rule; return the contract verdict and a flat list of issues."""
    issues: list[str] = []
    no_diagnosis = True
    grounded = True
    role_fidelity = True
    text = reply.strip()
    words = len(text.split())

    if words < 5:
        issues.append("reply is too short")
    if words > MAX_WORDS:
        issues.append(f"reply is too long ({words} words; limit {MAX_WORDS})")
    if _DIAGNOSIS.search(text):
        no_diagnosis = False
        issues.append("sounds like a diagnosis; never diagnose or name a disorder for the user")
    if _MEDICATION.search(text):
        no_diagnosis = False
        issues.append("mentions medication or prescriptions")
    if _CONTACT.search(text):
        issues.append("contains a phone number, link or helpline reference")
    if _MINIMISING.search(text):
        issues.append("uses a minimising phrase")
    if _SAFETY_WORDS.search(text):
        issues.append("uses self-harm or suicide wording; safety text is handled separately")
    if _LIVED_OR_CREDENTIAL.search(text):
        role_fidelity = False
        issues.append("claims lived experience or professional credentials")

    if previous_reply:
        opening = re.findall(r"[a-z']+", text.lower())[:3]
        if len(opening) == 3 and opening == re.findall(r"[a-z']+", previous_reply.lower())[:3]:
            issues.append("starts the same way as the previous reply; vary the opening")

    questions = text.count("?")
    if questions > 1:
        issues.append("asks more than one question")
    if move == "ask_clarifying" and questions < 1:
        issues.append("should ask the narrowing question")

    policy = ROLES[role]
    list_lines = len(_LIST_LINE.findall(text))
    advice = move in ("offer_technique", "psychoeducation")
    allowed_steps = policy.max_numbered_steps if advice else 2
    if list_lines > allowed_steps:
        role_fidelity = False
        issues.append(f"uses a list of {list_lines} items; keep it conversational for this move")
    if any(re.search(rf"(?i)\b{re.escape(t)}\b", text) for t in policy.banned_terms):
        role_fidelity = False
        issues.append(f"{policy.name} should not use clinical vocabulary")
    if role == Role.counsellor and _DIRECTIVE.search(text):
        role_fidelity = False
        issues.append("counsellor should not give directives like 'you should'")

    unknown = [c for c in used_card_ids if c not in set(retrieved_ids)]
    if unknown:
        grounded = False
        issues.append("cites cards that were not retrieved")
    if advice and not used_card_ids:
        grounded = False
        issues.append("technique or explanation is not tied to a knowledge card")

    verdict = CriticVerdict(
        passed=not issues,
        issues=issues,
        grounded=grounded,
        role_fidelity=role_fidelity,
        no_diagnosis=no_diagnosis,
    )
    return verdict, issues
