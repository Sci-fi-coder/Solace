"""Pipeline data shapes. TurnOutput is the interface the UI and evaluation code consume."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts import EmotionalState, Phase, Role, SafetyLevel, UserNeed

_LABEL_ALIASES = {
    "anxiety": "anxiety_fear", "anxious": "anxiety_fear", "fear": "anxiety_fear", "worry": "anxiety_fear",
    "stress": "anxiety_fear", "nervous": "anxiety_fear", "shame": "shame_guilt", "guilt": "shame_guilt",
    "embarrassment": "shame_guilt", "sad": "sadness", "grief": "sadness", "lonely": "sadness",
    "loneliness": "sadness", "hopeless": "sadness", "angry": "anger", "frustration": "anger",
    "frustrated": "anger", "irritation": "anger", "happy": "positive", "joy": "positive", "relief": "positive",
    "mixed": "others", "confusion": "others", "confused": "others",
}
_VALID_LABELS = {"anger", "anxiety_fear", "neutral", "others", "positive", "sadness", "shame_guilt"}


def _fraction(value: Any) -> Any:
    """Accept 0-1 or 0-100 numbers from a language model."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 1 < value <= 100:
        return value / 100
    return value


def _clean_label(label: Any) -> str | None:
    key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    key = _LABEL_ALIASES.get(key, key)
    return key if key in _VALID_LABELS else None

Move = Literal[
    "ask_clarifying",  # one narrowing question, no advice
    "reflect",  # mirror feelings, no advice, may end with an open invitation
    "offer_technique",  # suggest one card-backed technique
    "psychoeducation",  # explain what is happening, card-backed
    "summarise",  # "what I understand so far" and check it
    "close",  # wrap up the session
]


class EmotionGuess(BaseModel):
    """One hypothesis in the planner's reading of the user's feeling."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(description="anger, anxiety_fear, neutral, others, positive, sadness, shame_guilt")
    probability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _tolerate_model_output(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data["probability"] = _fraction(data.get("probability", 0.0))
        return data


class ProfileUpdate(BaseModel):
    """Small, non-sensitive facts the planner thinks are worth remembering."""

    model_config = ConfigDict(frozen=True)

    preferred_role: Role | None = None
    themes: list[str] = Field(default_factory=list, description="Short topic tags, max 3 words each.")
    values_note: str | None = None


class PlannerOutput(BaseModel):
    """Structured result of the single planner call (Explorer + Conductor in one)."""

    model_config = ConfigDict(frozen=True)

    safety_level: SafetyLevel = SafetyLevel.none
    emotions: list[EmotionGuess] = Field(default_factory=list)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    trigger: str | None = None
    duration: str | None = None
    functional_impact: str | None = None
    need: UserNeed = UserNeed.unknown
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    open_questions: list[str] = Field(default_factory=list)
    phase: Phase = Phase.exploration
    speaker: Role = Role.counsellor
    reason: str = ""
    move: Move = "reflect"
    clarifying_question: str | None = None
    retrieve: bool = False
    retrieve_query: str | None = None
    profile_update: ProfileUpdate = Field(default_factory=ProfileUpdate)

    @model_validator(mode="before")
    @classmethod
    def _tolerate_model_output(cls, data: Any) -> Any:
        """Language models drift: wrong case, nulls, percentages, unknown labels. Normalise, don't fail."""
        if not isinstance(data, dict):
            return data
        data = {k: v for k, v in data.items() if v is not None}
        for key in ("safety_level", "need", "phase", "speaker", "move"):
            if isinstance(data.get(key), str):
                data[key] = data[key].strip().lower()
        for key in ("intensity", "confidence"):
            if key in data:
                data[key] = _fraction(data[key])
        if not any(key in data for key in ("move", "phase", "speaker", "emotions")):
            raise ValueError("planner output has none of the decision fields")
        raw = data.get("emotions")
        if isinstance(raw, list):
            cleaned: dict[str, float] = {}
            for item in raw:
                if isinstance(item, BaseModel):
                    item = item.model_dump()  # noqa: PLW2901
                if isinstance(item, dict):
                    label = _clean_label(item.get("label"))
                    if label:
                        prob = _fraction(item.get("probability", 0.0))
                        cleaned[label] = max(cleaned.get(label, 0.0), float(prob or 0.0))
            data["emotions"] = [{"label": k, "probability": min(v, 1.0)} for k, v in cleaned.items()]
        if isinstance(data.get("open_questions"), str):
            data["open_questions"] = [data["open_questions"]]
        if isinstance(data.get("profile_update"), dict):
            data["profile_update"] = {k: v for k, v in data["profile_update"].items() if v is not None}
        return data


class ReplyOutput(BaseModel):
    """Structured result of the role reply call."""

    model_config = ConfigDict(frozen=True)

    reply: str
    explanation: str = ""
    used_card_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _tolerate_model_output(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if v is not None}
            if isinstance(data.get("used_card_ids"), str):
                data["used_card_ids"] = [data["used_card_ids"]]
        return data


class Citation(BaseModel):
    """A knowledge card (and its source pages) that shaped a reply."""

    model_config = ConfigDict(frozen=True)

    card_id: str
    title: str
    sources: list[str] = Field(default_factory=list, description="e.g. 'dwm_stress p40, p41'")


class SessionInfo(BaseModel):
    """Returned when a session starts."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    user_id: str
    greeting: str
    resume_note: str | None = None


class TurnOutput(BaseModel):
    """Everything the UI needs to show one assistant turn and its explanation panels."""

    model_config = ConfigDict(frozen=True)

    reply: str
    speaker: Role | None = Field(description="None for fixed safety replies.")
    speaker_name: str | None = None
    handoff: str | None = Field(default=None, description="Shown when the speaker changes.")
    phase: Phase
    safety_level: SafetyLevel
    safety_action: str
    state: EmotionalState | None = None
    state_summary: str = Field(default="", description="'What I understand so far' panel text.")
    explanation: str = Field(default="", description="'Why I said this' panel text.")
    citations: list[Citation] = Field(default_factory=list)
    move: str | None = None
    degraded: bool = False
    provider_used: str = ""
    latency_ms: float = 0.0
    turn_id: str = ""
