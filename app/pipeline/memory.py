"""Cross-session memory: profile updates, deterministic session summaries, greetings."""

from __future__ import annotations

import re
from typing import Any

from app.contracts import Role, UserProfile
from app.pipeline.guardrails import SessionRuntime
from app.pipeline.types import ProfileUpdate

MAX_THEMES = 8
_FRIENDLY = {"anxiety_fear": "anxiety", "shame_guilt": "guilt or shame", "others": "mixed feelings"}


def merge_profile(
    profile: UserProfile | None,
    *,
    user_id: str,
    display_name: str | None,
    update: ProfileUpdate,
) -> UserProfile:
    """Merge small planner suggestions into the profile; themes are short tags only."""
    base = profile or UserProfile(user_id=user_id, display_name=display_name)
    themes = list(base.recurring_themes)
    for theme in update.themes:
        tag = " ".join(theme.lower().split()[:3])
        if tag and tag not in themes:
            themes.append(tag)
    fields: dict[str, Any] = {"recurring_themes": themes[-MAX_THEMES:]}
    if update.preferred_role is not None:
        fields["preferred_role"] = update.preferred_role
    if update.values_note:
        fields["values_notes"] = update.values_note[:300]
    if display_name and not base.display_name:
        fields["display_name"] = display_name
    return base.model_copy(update=fields)


def session_summary(rt: SessionRuntime) -> dict[str, Any]:
    """Deterministic, non-verbatim summary stored for the next session."""
    top = None
    if rt.state and rt.state.hypotheses:
        top = max(rt.state.hypotheses, key=lambda h: h.probability).label
    return {
        "main_feeling": _FRIENDLY.get(top or "", top),
        "trigger": rt.state.trigger if rt.state else None,
        "themes": rt.themes[-3:],
        "techniques": rt.techniques_offered[-3:],
        "phase_reached": rt.phase.value,
        "turns": rt.turn_count,
    }


def resume_note(profile: UserProfile | None) -> str | None:
    """One gentle sentence about last time, or None."""
    if profile is None:
        return None
    last = profile.communication_prefs.get("last_session")
    if not isinstance(last, dict):
        return None
    topic = last.get("trigger") or (last.get("themes") or [None])[0] or last.get("main_feeling")
    if not topic:
        return None
    return f"Last time we spoke about {topic}."


def greeting(display_name: str | None, note: str | None) -> str:
    """Opening message; always discloses that this is an AI companion."""
    name = f" {display_name}" if display_name else ""
    lines = [
        f"Hi{name}. I'm Solace, an AI companion here to listen and help you think things through. "
        "I'm not a therapist and I can't help in emergencies."
    ]
    if note:
        lines.append(f"{note} How has that been since then?")
    else:
        lines.append("What's on your mind today?")
    return " ".join(lines)


_ROLE_WORDS = {
    Role.mentor: r"mentor|rohan",
    Role.counsellor: r"counsell?or|anika",
    Role.teacher: r"teacher|meera",
    Role.peer: r"peer|kabir|friend",
}
_PREFERENCE_CUE = re.compile(r"(?i)\b(prefer|rather|switch|talk to|speak to|bring in|go back to|want|like)\b")


def explicit_role_preference(text: str) -> Role | None:
    """A preferred role is stored only when the person names a persona AND states a preference."""
    if not _PREFERENCE_CUE.search(text):
        return None
    for role, words in _ROLE_WORDS.items():
        if re.search(rf"(?i)\b(?:{words})\b", text):
            return role
    return None
