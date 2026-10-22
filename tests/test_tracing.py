"""Tests for app/tracing/recorder — TraceRecorder and TurnTrace assembly."""

from __future__ import annotations

import time
import uuid

import pytest

from app.contracts import SafetyLevel, SafetyResult, StepRecord, TurnTrace
from app.tracing.recorder import TraceRecorder


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safety() -> SafetyResult:
    return SafetyResult(level=SafetyLevel.none, source="rule", action="continue")


def _recorder() -> TraceRecorder:
    return TraceRecorder(
        turn_id=f"t-{uuid.uuid4().hex[:8]}",
        session_id=f"s-{uuid.uuid4().hex[:8]}",
        user_message="Synthetic test message",
    )


# ── Basic build ───────────────────────────────────────────────────────────────


def test_recorder_builds_trace() -> None:
    """TraceRecorder.build() returns a valid TurnTrace."""
    rec = _recorder()
    trace = rec.build(
        safety=_safety(),
        final_reply="I hear you.",
        explanation="User venting; empathy response.",
        provider_used="mock/mock-model",
    )
    assert isinstance(trace, TurnTrace)
    assert trace.turn_id == rec.turn_id
    assert trace.session_id == rec.session_id
    assert trace.final_reply == "I hear you."
    assert trace.steps == []


def test_recorder_preserves_user_message() -> None:
    """The user message is stored in the trace (not sanitised at this layer)."""
    rec = _recorder()
    trace = rec.build(
        safety=_safety(),
        final_reply="ok",
        explanation="test",
        provider_used="mock/mock",
    )
    assert trace.user_message == "Synthetic test message"


# ── Manual step recording ──────────────────────────────────────────────────────


def test_recorder_record_step() -> None:
    """rec.record() adds a StepRecord with the given values."""
    rec = _recorder()
    rec.record("safety", "level=none", latency_ms=3.5)
    rec.record("emotion", "label=neutral", latency_ms=12.0)

    trace = rec.build(
        safety=_safety(),
        final_reply="ok",
        explanation="test",
        provider_used="mock/mock",
    )
    assert len(trace.steps) == 2
    assert trace.steps[0].name == "safety"
    assert trace.steps[0].summary == "level=none"
    assert trace.steps[0].latency_ms == pytest.approx(3.5)
    assert trace.steps[1].name == "emotion"


# ── Context-manager step recording ────────────────────────────────────────────


def test_recorder_step_context_manager_records_timing() -> None:
    """step() context manager records a step with non-zero latency."""
    rec = _recorder()
    with rec.step("safety"):
        time.sleep(0.01)

    trace = rec.build(
        safety=_safety(),
        final_reply="ok",
        explanation="test",
        provider_used="mock/mock",
    )
    assert len(trace.steps) == 1
    assert trace.steps[0].name == "safety"
    assert trace.steps[0].latency_ms >= 10.0  # at least 10ms


def test_recorder_multiple_steps_preserved_in_order() -> None:
    """Multiple step() calls produce steps in insertion order."""
    rec = _recorder()
    with rec.step("safety"):
        pass
    with rec.step("emotion"):
        pass
    with rec.step("conductor"):
        pass

    trace = rec.build(
        safety=_safety(),
        final_reply="ok",
        explanation="test",
        provider_used="mock/mock",
    )
    names = [s.name for s in trace.steps]
    assert names == ["safety", "emotion", "conductor"]


# ── Optional fields ───────────────────────────────────────────────────────────


def test_recorder_optional_fields_default_to_none() -> None:
    """TurnTrace optional fields are None by default."""
    rec = _recorder()
    trace = rec.build(
        safety=_safety(),
        final_reply="ok",
        explanation="test",
        provider_used="mock/mock",
    )
    assert trace.emotion is None
    assert trace.state_before is None
    assert trace.state_after is None
    assert trace.decision is None
    assert trace.retrieved == []
    assert trace.draft is None
    assert trace.critic is None


# ── Context manager protocol ──────────────────────────────────────────────────


def test_recorder_used_as_context_manager() -> None:
    """TraceRecorder can be used as a context manager (with rec:)."""
    with TraceRecorder(
        turn_id="t-cm-001",
        session_id="s-cm-001",
        user_message="context manager test",
    ) as rec:
        rec.record("safety", "none", 1.0)
        trace = rec.build(
            safety=_safety(),
            final_reply="ok",
            explanation="cm test",
            provider_used="mock/mock",
        )

    assert trace.turn_id == "t-cm-001"
    assert len(trace.steps) == 1


# ── Serialisation ─────────────────────────────────────────────────────────────


def test_trace_serialises_to_json_compatible_dict() -> None:
    """TurnTrace.model_dump(mode='json') produces a JSON-serialisable dict."""
    import json
    rec = _recorder()
    rec.record("safety", "none", 2.0)
    trace = rec.build(
        safety=_safety(),
        final_reply="ok",
        explanation="test",
        provider_used="mock/mock",
    )
    data = trace.model_dump(mode="json")
    # Should not raise
    serialised = json.dumps(data)
    assert '"turn_id"' in serialised


def test_trace_round_trips_through_model_validate() -> None:
    """TurnTrace survives a model_dump → model_validate round-trip."""
    rec = _recorder()
    trace = rec.build(
        safety=_safety(),
        final_reply="round trip",
        explanation="test",
        provider_used="mock/mock",
    )
    data = trace.model_dump(mode="json")
    restored = TurnTrace.model_validate(data)
    assert restored.turn_id == trace.turn_id
    assert restored.final_reply == "round trip"
