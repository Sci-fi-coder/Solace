"""Tests for the deterministic planner fallback."""

from __future__ import annotations

import pytest

from app.contracts import EmotionalState, EmotionHypothesis, Phase, Role, UserNeed
from app.pipeline.guardrails import SessionRuntime
from app.pipeline.heuristic import heuristic_plan
from app.pipeline.types import EmotionGuess


def _runtime(
    *,
    state: EmotionalState | None = None,
    phase: Phase = Phase.checkin,
    turn_count: int = 0,
    last_move: str | None = None,
) -> SessionRuntime:
    """Build a minimal runtime for planner tests."""
    return SessionRuntime(
        session_id="test-session",
        user_id="test-user",
        state=state,
        phase=phase,
        turn_count=turn_count,
        last_move=last_move,
    )


def test_emotion_hits_are_normalized_and_limited_to_the_top_three() -> None:
    """Regex hit counts become normalized probabilities ordered by frequency."""
    plan = heuristic_plan(
        _runtime(),
        "I feel stressed and worried, sad and lonely, angry, and guilty.",
    )

    assert [guess.label for guess in plan.emotions] == [
        "anxiety_fear",
        "sadness",
        "anger",
    ]
    assert sum(guess.probability for guess in plan.emotions) == pytest.approx(5 / 6, abs=0.001)


def test_no_emotion_matches_produces_neutral() -> None:
    """A message without a configured emotion term receives a neutral hypothesis."""
    plan = heuristic_plan(_runtime(), "The window is open.")

    assert plan.emotions == [EmotionGuess(label="neutral", probability=1.0)]


@pytest.mark.parametrize(
    ("text", "trigger"),
    [
        ("My final exams are close.", "exams"),
        ("I have too many deadlines.", "deadlines"),
        ("My parents' expectations feel heavy.", "family expectations"),
        ("Work has been difficult lately.", "work or career"),
        ("My relationship has changed.", "relationships"),
        ("I am worried about money.", "money"),
    ],
)
def test_trigger_categories_are_detected(text: str, trigger: str) -> None:
    """Each supported trigger family maps to a stable trigger label."""
    assert heuristic_plan(_runtime(), text).trigger == trigger


def test_previous_trigger_is_preserved_without_a_new_match() -> None:
    """An unrelated message does not erase a known trigger."""
    state = EmotionalState(
        hypotheses=[EmotionHypothesis(label="anxiety_fear", probability=1.0)],
        intensity=0.6,
        trigger="exams",
        confidence=0.4,
    )

    assert heuristic_plan(_runtime(state=state), "I feel worried.").trigger == "exams"


def test_duration_and_functional_impact_are_extracted() -> None:
    """Duration and all stated daily impacts are extracted from one message."""
    plan = heuristic_plan(
        _runtime(),
        "For two weeks I can't sleep properly, my appetite has changed, "
        "I can't concentrate, and I have no energy.",
    )

    assert plan.duration == "two weeks"
    assert plan.functional_impact == (
        "sleep difficulties; appetite changes; difficulty concentrating; "
        "low energy or motivation"
    )


def test_since_duration_and_previous_impact_are_preserved() -> None:
    """A since-phrase is recognized while an unstated impact remains in state."""
    state = EmotionalState(
        hypotheses=[EmotionHypothesis(label="sadness", probability=1.0)],
        intensity=0.6,
        functional_impact="sleep difficulties",
        confidence=0.4,
    )

    plan = heuristic_plan(_runtime(state=state), "I've felt this way since exams began.")

    assert plan.duration == "since exams began"
    assert plan.functional_impact == "sleep difficulties"


def test_a_few_days_is_detected_and_missing_context_uses_one_question() -> None:
    """Duration language is recognized and the first missing fact gets one question."""
    plan = heuristic_plan(_runtime(), "I've felt low for a few days.")
    assert plan.duration == "a few days"
    assert plan.move == "ask_clarifying"
    assert plan.clarifying_question == "How has this been affecting your days, like sleep, studies or mood?"


@pytest.mark.parametrize(
    ("text", "need"),
    [
        ("Please just listen.", UserNeed.vent),
        ("What can I do about it?", UserNeed.act),
        ("I need something practical to try.", UserNeed.act),
        ("Why am I reacting this way?", UserNeed.understand),
        ("Can you explain this?", UserNeed.understand),
        ("I have a lot on my mind.", UserNeed.unknown),
    ],
)
def test_user_need_is_inferred_from_explicit_language(text: str, need: UserNeed) -> None:
    """Explicit requests map to venting, action, understanding, or unknown."""
    assert heuristic_plan(_runtime(), text).need == need


@pytest.mark.parametrize(
    "text",
    [
        "I'm done for today.",
        "That's all.",
        "Good night.",
        "Talk later.",
        "Goodbye.",
    ],
)
def test_closing_language_selects_close_and_closing_phase(text: str) -> None:
    """Recognized endings immediately close the session."""
    plan = heuristic_plan(_runtime(phase=Phase.guidance, turn_count=5), text)

    assert plan.move == "close"
    assert plan.phase == Phase.closing


def test_gratitude_preserves_an_existing_emotional_picture() -> None:
    """Positive-only feedback keeps the prior emotion and intensity."""
    state = EmotionalState(
        hypotheses=[
            EmotionHypothesis(label="anxiety_fear", probability=0.8),
            EmotionHypothesis(label="sadness", probability=0.2),
        ],
        intensity=0.7,
        trigger="exams",
        duration="two weeks",
        functional_impact="sleep difficulties",
        need=UserNeed.understand,
        confidence=0.7,
    )

    plan = heuristic_plan(_runtime(state=state), "Thanks, that helped a bit.")

    assert [guess.label for guess in plan.emotions] == ["anxiety_fear", "sadness"]
    assert plan.intensity == state.intensity
    assert plan.trigger == state.trigger
    assert plan.duration == state.duration
    assert plan.functional_impact == state.functional_impact


def test_guidance_selects_mentor_and_retrieves_for_action() -> None:
    """Action in guidance asks for a technique using state-derived retrieval terms."""
    state = EmotionalState(
        hypotheses=[EmotionHypothesis(label="anxiety_fear", probability=1.0)],
        intensity=0.5,
        trigger="exams",
        duration="two weeks",
        functional_impact="sleep difficulties",
        confidence=0.7,
    )
    runtime = _runtime(state=state, phase=Phase.understanding, turn_count=5)

    plan = heuristic_plan(runtime, "Can you give me something practical to try?")

    assert plan.phase == Phase.guidance
    assert plan.speaker == Role.mentor
    assert plan.move == "offer_technique"
    assert plan.retrieve
    assert all(term in (plan.retrieve_query or "") for term in ("anxiety fear", "exams", "sleep"))


def test_simple_acknowledgment_after_a_technique_selects_follow_up() -> None:
    """A simple yes or thanks follows up on the technique already offered."""
    runtime = _runtime(phase=Phase.guidance, turn_count=6, last_move="offer_technique")
    runtime.techniques_offered = ["slow-breathing-for-stress"]

    plan = heuristic_plan(runtime, "yes, let's try it")

    assert plan.move == "follow_up"
    assert not plan.retrieve


def test_only_advice_moves_request_retrieval() -> None:
    """Listening and clarifying plans do not request knowledge retrieval."""
    ask = heuristic_plan(_runtime(), "I've been stressed about exams.")
    reflect = heuristic_plan(
        _runtime(phase=Phase.exploration, turn_count=2),
        "I just need to vent.",
    )

    assert ask.move == "ask_clarifying" and not ask.retrieve
    assert reflect.move == "reflect" and not reflect.retrieve


def test_missing_context_prevents_an_early_understanding_phase() -> None:
    """An explanation request still gets a narrowing question before advice."""
    plan = heuristic_plan(
        _runtime(phase=Phase.exploration, turn_count=1),
        "I want to understand why I'm feeling anxious.",
    )

    assert plan.phase == Phase.exploration
    assert plan.move == "ask_clarifying"
    assert not plan.retrieve
