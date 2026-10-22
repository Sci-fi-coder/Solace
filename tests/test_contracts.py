"""Tests for app/contracts — validation of Pydantic v2 models and enums."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from app.contracts import (
    ConductorDecision,
    CriticVerdict,
    EmotionHypothesis,
    EmotionReading,
    EmotionalState,
    Message,
    Phase,
    RetrievedItem,
    Role,
    SafetyLevel,
    SafetyResult,
    StepRecord,
    TaskKind,
    TurnTrace,
    UserNeed,
    UserProfile,
)


# ── Enum sanity ───────────────────────────────────────────────────────────────


def test_role_values() -> None:
    """All Role enum members have the correct string values."""
    assert Role.counsellor == "counsellor"
    assert Role.mentor == "mentor"
    assert Role.teacher == "teacher"
    assert Role.peer == "peer"


def test_phase_values() -> None:
    """Phase enum has all five session phases."""
    phases = {p.value for p in Phase}
    assert phases == {"checkin", "exploration", "understanding", "guidance", "closing"}


def test_safety_level_ordering() -> None:
    """SafetyLevel has none, elevated, and crisis."""
    assert SafetyLevel.none == "none"
    assert SafetyLevel.elevated == "elevated"
    assert SafetyLevel.crisis == "crisis"


def test_task_kind_values() -> None:
    """TaskKind has all four task types."""
    kinds = {t.value for t in TaskKind}
    assert kinds == {"converse", "classify", "judge", "extract"}


# ── SafetyResult ──────────────────────────────────────────────────────────────


def test_safety_result_valid() -> None:
    """SafetyResult accepts valid data."""
    sr = SafetyResult(
        level=SafetyLevel.none,
        reasons=[],
        source="rule",
        action="continue",
    )
    assert sr.level == SafetyLevel.none
    assert sr.source == "rule"


def test_safety_result_elevated() -> None:
    """SafetyResult stores reasons list."""
    sr = SafetyResult(
        level=SafetyLevel.elevated,
        reasons=["self-harm keyword detected"],
        source="rule",
        action="warn",
    )
    assert len(sr.reasons) == 1


def test_safety_result_missing_required_field() -> None:
    """SafetyResult raises ValidationError when required fields are missing."""
    with pytest.raises(ValidationError):
        SafetyResult(level=SafetyLevel.none)  # missing source and action


# ── EmotionReading ────────────────────────────────────────────────────────────


def test_emotion_reading_valid() -> None:
    """EmotionReading accepts valid data."""
    er = EmotionReading(
        label="sadness",
        score=0.82,
        top_k=[("sadness", 0.82), ("anxiety", 0.12)],
        source_model="distilroberta-base",
    )
    assert er.label == "sadness"
    assert er.score == pytest.approx(0.82)


def test_emotion_reading_score_bounds() -> None:
    """EmotionReading rejects scores outside [0, 1]."""
    with pytest.raises(ValidationError):
        EmotionReading(label="joy", score=1.5, source_model="test")
    with pytest.raises(ValidationError):
        EmotionReading(label="joy", score=-0.1, source_model="test")


# ── EmotionalState ────────────────────────────────────────────────────────────


def test_emotional_state_valid() -> None:
    """EmotionalState accepts valid data with optional fields absent."""
    state = EmotionalState(
        hypotheses=[EmotionHypothesis(label="sadness", probability=0.9)],
        intensity=0.7,
        need=UserNeed.vent,
        confidence=0.85,
    )
    assert state.trigger is None
    assert state.need == UserNeed.vent
    assert len(state.hypotheses) == 1


def test_emotional_state_intensity_bounds() -> None:
    """EmotionalState rejects intensity outside [0, 1]."""
    with pytest.raises(ValidationError):
        EmotionalState(intensity=1.1, need=UserNeed.unknown, confidence=0.5)


# ── UserProfile ───────────────────────────────────────────────────────────────


def test_user_profile_defaults() -> None:
    """UserProfile has sensible defaults for optional fields."""
    profile = UserProfile(user_id="u-001")
    assert profile.display_name is None
    assert profile.preferred_role is None
    assert profile.recurring_themes == []
    assert isinstance(profile.created_at, datetime)


def test_user_profile_with_role() -> None:
    """UserProfile accepts a preferred_role."""
    profile = UserProfile(user_id="u-002", preferred_role=Role.mentor)
    assert profile.preferred_role == Role.mentor


# ── Message ───────────────────────────────────────────────────────────────────


def test_message_user() -> None:
    """Message with role='user' is valid."""
    m = Message(role="user", content="Hello")
    assert m.speaker_role is None


def test_message_assistant_with_speaker() -> None:
    """Message with role='assistant' and speaker_role is valid."""
    m = Message(role="assistant", content="How are you?", speaker_role=Role.counsellor)
    assert m.speaker_role == Role.counsellor


# ── ConductorDecision ─────────────────────────────────────────────────────────


def test_conductor_decision_retrieve_false() -> None:
    """ConductorDecision with retrieve=False needs no retrieve_query."""
    d = ConductorDecision(
        phase=Phase.exploration,
        speaker=Role.counsellor,
        reason="User is venting; listen first.",
        retrieve=False,
    )
    assert d.retrieve_query is None


def test_conductor_decision_retrieve_true() -> None:
    """ConductorDecision with retrieve=True carries a query string."""
    d = ConductorDecision(
        phase=Phase.guidance,
        speaker=Role.teacher,
        reason="User wants psychoeducation on anxiety.",
        retrieve=True,
        retrieve_query="anxiety management techniques",
    )
    assert d.retrieve_query == "anxiety management techniques"


# ── CriticVerdict ─────────────────────────────────────────────────────────────


def test_critic_verdict_passed() -> None:
    """CriticVerdict with all checks True is valid."""
    v = CriticVerdict(
        passed=True,
        grounded=True,
        role_fidelity=True,
        no_diagnosis=True,
    )
    assert v.issues == []


def test_critic_verdict_failed() -> None:
    """CriticVerdict captures issues list."""
    v = CriticVerdict(
        passed=False,
        issues=["Reply contains clinical diagnosis"],
        grounded=True,
        role_fidelity=True,
        no_diagnosis=False,
    )
    assert not v.passed
    assert len(v.issues) == 1


# ── TurnTrace ─────────────────────────────────────────────────────────────────


def _make_safety() -> SafetyResult:
    return SafetyResult(level=SafetyLevel.none, source="rule", action="continue")


def test_turn_trace_minimal() -> None:
    """TurnTrace can be created with only required fields."""
    trace = TurnTrace(
        turn_id="t-001",
        session_id="s-001",
        user_message="I feel sad",
        safety=_make_safety(),
        final_reply="I hear you.",
        explanation="User venting; counsellor responds with empathy.",
        provider_used="mock/mock-model",
    )
    assert trace.turn_id == "t-001"
    assert trace.emotion is None
    assert trace.retrieved == []
    assert trace.steps == []


def test_turn_trace_with_steps() -> None:
    """TurnTrace stores StepRecords correctly."""
    trace = TurnTrace(
        turn_id="t-002",
        session_id="s-001",
        user_message="synthetic",
        safety=_make_safety(),
        final_reply="OK",
        explanation="test",
        provider_used="mock/mock-model",
        steps=[
            StepRecord(name="safety", summary="level=none", latency_ms=2.5),
            StepRecord(name="emotion", summary="label=neutral", latency_ms=10.0),
        ],
    )
    assert len(trace.steps) == 2
    assert trace.steps[0].name == "safety"


def test_turn_trace_serialisation() -> None:
    """TurnTrace round-trips through model_dump / model_validate."""
    trace = TurnTrace(
        turn_id="t-003",
        session_id="s-001",
        user_message="synthetic",
        safety=_make_safety(),
        final_reply="OK",
        explanation="test",
        provider_used="mock/mock-model",
    )
    data = trace.model_dump(mode="json")
    restored = TurnTrace.model_validate(data)
    assert restored.turn_id == trace.turn_id
