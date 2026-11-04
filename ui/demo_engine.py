"""Scripted local engine for developing the Streamlit interface without provider keys."""

from __future__ import annotations

from app.contracts import (
    EmotionalState,
    EmotionHypothesis,
    Phase,
    Role,
    SafetyLevel,
    UserNeed,
)
from app.pipeline.types import Citation, SessionInfo, TurnOutput


class DemoEngine:
    """Provide deterministic session methods and scripted demo turns."""

    def __init__(self) -> None:
        """Initialize in-memory session state."""
        self._user_id: str | None = None
        self._session_id: str | None = None
        self._turn_count = 0
        self._session_count = 0

    def start_session(self, user_id: str, display_name: str | None = None) -> SessionInfo:
        """Start an in-memory demo session."""
        self._user_id = user_id
        self._session_count += 1
        self._session_id = f"demo-session-{self._session_count}"
        self._turn_count = 0
        name = display_name.strip() if display_name else ""
        greeting = f"Hi {name}, I'm Solace. What would feel helpful to talk about today?"
        if not name:
            greeting = "Hi, I'm Solace. What would feel helpful to talk about today?"
        return SessionInfo(
            session_id=self._session_id,
            user_id=user_id,
            greeting=greeting,
        )

    def handle_turn(
        self,
        session_id: str,
        text: str,
        *,
        enable_role_models: bool = False,
    ) -> TurnOutput:
        """Return the next scripted turn for a demo session."""
        del enable_role_models
        if session_id != self._session_id:
            raise ValueError("Unknown demo session.")
        if not text.strip():
            raise ValueError("Message cannot be empty.")

        self._turn_count += 1
        position = (self._turn_count - 1) % 6
        roles = (Role.counsellor, Role.mentor, Role.teacher, Role.peer)
        role = roles[position] if position < len(roles) else Role.counsellor
        safety = SafetyLevel.elevated if position == 4 else SafetyLevel.none
        crisis = position == 5
        citations = (
            [
                Citation(
                    card_id="demo-grounding",
                    title="A small grounding pause",
                    sources=["Synthetic demo source, page 1"],
                )
            ]
            if position == 1
            else []
        )
        phase = (
            Phase.guidance if position in (1, 2) else Phase.exploration
        )
        return TurnOutput(
            reply=(
                "This is a fixed crisis-safety demonstration, not a live safety assessment."
                if crisis
                else (
                    "Let's pause and notice one thing around you, if that feels okay."
                    if position == 1
                    else "Thanks for sharing that. What part feels most present right now?"
                )
            ),
            speaker=None if crisis else role,
            speaker_name=None if crisis else role.value.title(),
            handoff=(
                "A handoff to Mentor: let's look together at one small option."
                if position == 1
                else None
            ),
            phase=Phase.closing if crisis else phase,
            safety_level=SafetyLevel.crisis if crisis else safety,
            safety_action="crisis_override" if crisis else "proceed",
            state=EmotionalState(
                hypotheses=[
                    EmotionHypothesis(label="anxiety_fear", probability=0.62),
                    EmotionHypothesis(label="others", probability=0.24),
                ],
                intensity=0.55,
                trigger="A synthetic demo situation",
                duration="Recently",
                functional_impact="Not specified",
                need=UserNeed.unknown,
                confidence=0.68,
            ),
            state_summary="You have described a synthetic situation that feels important.",
            explanation=(
                ""
                if crisis
                else "This demonstration turn is scripted locally and does not contact a provider."
            ),
            citations=citations,
            move="reflect",
            degraded=position == 2,
            provider_used="demo",
            latency_ms=0.1,
            turn_id=f"demo-turn-{self._turn_count}",
            debug_steps=["demo: scripted response", "safety: synthetic state"],
        )

    def end_session(self, session_id: str) -> dict[str, object]:
        """End the in-memory session and return a small scripted summary."""
        if session_id != self._session_id:
            raise ValueError("Unknown demo session.")
        return {
            "session_id": session_id,
            "turns": self._turn_count,
            "summary": "A local scripted demo conversation.",
        }

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Clear matching in-memory demo session data."""
        if user_id != self._user_id:
            return {"deleted": 0}
        self._user_id = None
        self._session_id = None
        self._turn_count = 0
        return {"deleted": 1}
