"""Deterministic guardrails around the planner: phase machine, ask-before-advise, handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.contracts import (
    EmotionalState,
    EmotionHypothesis,
    Message,
    Phase,
    Role,
    SafetyLevel,
    UserNeed,
)
from app.pipeline.types import Move, PlannerOutput

MAX_CLARIFYING = 4  # narrowing questions per session before we must summarise
CONFIDENT = 0.7  # state confidence at which we stop asking
MIN_TURNS_BETWEEN_HANDOFFS = 2
MAX_SESSION_TURNS = 14

_PHASE_ORDER = [Phase.checkin, Phase.exploration, Phase.understanding, Phase.guidance, Phase.closing]
_FRIENDLY = {
    "anxiety_fear": "anxiety or fear",
    "shame_guilt": "shame or guilt",
    "others": "mixed or unclear feelings",
    "neutral": "feeling flat or neutral",
    "positive": "something positive",
}
_GENERIC_QUESTIONS = [
    "What has been on your mind the most lately?",
    "When did you first notice this feeling?",
    "How has this been affecting your days?",
    "What would be most helpful to talk about right now?",
]


@dataclass
class SessionRuntime:
    """Mutable per-session state kept in memory by the engine."""

    session_id: str
    user_id: str
    state: EmotionalState | None = None
    phase: Phase = Phase.checkin
    turn_count: int = 0
    questions_asked: int = 0
    last_speaker: Role | None = None
    turns_since_handoff: int = 99
    last_move: str | None = None
    history: list[Message] = field(default_factory=list)
    gate_state: dict[str, object] = field(default_factory=dict)
    techniques_offered: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    preferred_role: Role | None = None
    resume_note: str | None = None
    degraded_turns: int = 0


@dataclass(frozen=True)
class Decision:
    """The plan after guardrails have been applied."""

    phase: Phase
    speaker: Role
    move: Move
    question: str | None
    retrieve: bool
    retrieve_query: str | None
    notes: tuple[str, ...]


def _idx(phase: Phase) -> int:
    return _PHASE_ORDER.index(phase)


def next_phase(prev: Phase, proposed: Phase, plan: PlannerOutput, rt: SessionRuntime) -> Phase:
    """Allow only sensible forward moves; closing is always reachable."""
    if proposed == Phase.closing:
        return Phase.closing
    if proposed == prev:
        return prev
    if _idx(proposed) < _idx(prev):
        return Phase.exploration if proposed == Phase.exploration else prev
    if _idx(proposed) == _idx(prev) + 1:
        return proposed
    # a jump of two or more phases is only allowed for a user who wants a plan and is understood
    if (
        prev == Phase.exploration
        and proposed == Phase.guidance
        and plan.need == UserNeed.act
        and plan.confidence >= 0.6
    ):
        return Phase.guidance
    return _PHASE_ORDER[_idx(prev) + 1]


def _fallback_question(plan: PlannerOutput, rt: SessionRuntime) -> str:
    if plan.clarifying_question:
        return plan.clarifying_question
    if plan.open_questions:
        return plan.open_questions[0]
    return _GENERIC_QUESTIONS[min(rt.questions_asked, len(_GENERIC_QUESTIONS) - 1)]


def build_query(plan: PlannerOutput, state: EmotionalState | None) -> str:
    """Retrieval query from the state model, not from the raw user message."""
    if plan.retrieve_query:
        return plan.retrieve_query
    parts: list[str] = []
    if state and state.hypotheses:
        parts.append(max(state.hypotheses, key=lambda h: h.probability).label.replace("_", " "))
    if state and state.trigger:
        parts.append(state.trigger)
    parts.append({UserNeed.act: "practical coping steps", UserNeed.understand: "explain"}.get(
        state.need if state else UserNeed.unknown, "coping technique"
    ))
    return " ".join(parts)


def enforce_plan(
    plan: PlannerOutput,
    rt: SessionRuntime,
    *,
    safety_level: SafetyLevel,
    state: EmotionalState,
) -> Decision:
    """Apply the rules that make Solace a structured session instead of a free chat."""
    notes: list[str] = []
    phase = next_phase(rt.phase, plan.phase, plan, rt)
    if rt.turn_count == 0:
        phase = Phase.checkin if phase == Phase.checkin else Phase.exploration
    if rt.phase == Phase.exploration and (
        state.confidence >= 0.6 or rt.questions_asked >= 3
    ) and phase == Phase.exploration:
        phase = Phase.understanding
        notes.append("moved to understanding: state is clear enough")
    move: Move = plan.move
    question = plan.clarifying_question

    if rt.turn_count + 1 >= MAX_SESSION_TURNS and move != "close":
        move = "close"
        phase = Phase.closing
        notes.append("session turn limit reached")
    if move == "close":
        phase = Phase.closing

    can_ask = (
        phase in (Phase.checkin, Phase.exploration)
        and rt.questions_asked < MAX_CLARIFYING
        and state.confidence < CONFIDENT
    )
    if move == "ask_clarifying":
        if can_ask:
            question = _fallback_question(plan, rt)
        else:
            move = "summarise" if _idx(phase) <= _idx(Phase.understanding) else "reflect"
            notes.append("question budget used or state clear: switched to " + move)

    advice_moves = ("offer_technique", "psychoeducation")
    if move in advice_moves:
        allowed = _idx(phase) >= _idx(Phase.understanding) or (
            state.need == UserNeed.act and state.confidence >= 0.6
        )
        if state.need == UserNeed.vent:
            move = "reflect"
            notes.append("user wants to be heard: listening first")
        elif not allowed:
            if can_ask:
                move, question = "ask_clarifying", _fallback_question(plan, rt)
            else:
                move = "summarise"
            notes.append("ask before advising: " + move)

    elevated = safety_level == SafetyLevel.elevated
    if elevated:
        if move in advice_moves:
            move = "reflect"
        speaker = Role.counsellor
        notes.append("elevated safety concern: counsellor with a gentle check-in")
    else:
        speaker = plan.speaker
        if (
            rt.last_speaker is not None
            and speaker != rt.last_speaker
            and rt.turns_since_handoff < MIN_TURNS_BETWEEN_HANDOFFS
        ):
            speaker = rt.last_speaker
            notes.append("handoff throttled to keep the conversation steady")

    retrieve = move in advice_moves and not elevated
    query = build_query(plan, state) if retrieve else None
    return Decision(
        phase=phase,
        speaker=speaker,
        move=move,
        question=question if move == "ask_clarifying" else None,
        retrieve=retrieve,
        retrieve_query=query,
        notes=tuple(notes),
    )


def state_from_plan(plan: PlannerOutput, previous: EmotionalState | None) -> EmotionalState:
    """Merge the planner's reading into the running state model (facts persist across turns)."""
    guesses = [g for g in plan.emotions if g.probability > 0]
    total = sum(g.probability for g in guesses)
    if guesses and total > 0:
        hyps = [
            EmotionHypothesis(label=g.label, probability=round(g.probability / total, 3))
            for g in sorted(guesses, key=lambda g: g.probability, reverse=True)[:4]
        ]
    elif previous is not None:
        hyps = list(previous.hypotheses)
    else:
        hyps = []
    prev = previous
    return EmotionalState(
        hypotheses=hyps,
        intensity=plan.intensity,
        trigger=plan.trigger or (prev.trigger if prev else None),
        duration=plan.duration or (prev.duration if prev else None),
        functional_impact=plan.functional_impact or (prev.functional_impact if prev else None),
        need=plan.need if plan.need != UserNeed.unknown else (prev.need if prev else UserNeed.unknown),
        confidence=plan.confidence,
        open_questions=plan.open_questions[:3],
    )


def render_state_summary(state: EmotionalState | None) -> str:
    """Plain-language 'what I understand so far' text, built without another LLM call."""
    if state is None:
        return "I'm just getting to know what's going on. Tell me what's on your mind."
    lines = ["Here is what I understand so far:"]
    if state.hypotheses:
        feelings = ", ".join(
            f"{_FRIENDLY.get(h.label, h.label)} ({round(h.probability * 100)}%)"
            for h in state.hypotheses[:3]
        )
        lines.append(f"- Main feelings: {feelings}")
    strength = "mild" if state.intensity < 0.34 else "moderate" if state.intensity < 0.67 else "strong"
    lines.append(f"- How strong it feels: {strength}")
    if state.trigger:
        lines.append(f"- What set it off: {state.trigger}")
    if state.duration:
        lines.append(f"- For how long: {state.duration}")
    if state.functional_impact:
        lines.append(f"- Effect on your days: {state.functional_impact}")
    wants = {
        UserNeed.vent: "to be heard",
        UserNeed.understand: "to understand what is happening",
        UserNeed.act: "a practical next step",
    }.get(state.need)
    if wants:
        lines.append(f"- What you seem to want right now: {wants}")
    if state.confidence >= CONFIDENT:
        lines.append("- I feel fairly sure about this picture. Tell me if I've got anything wrong.")
    else:
        lines.append("- I'm still working this out, so correct me if something is off.")
    if state.open_questions:
        lines.append(f"- Still unsure about: {state.open_questions[0]}")
    return "\n".join(lines)
