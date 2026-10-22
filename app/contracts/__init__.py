"""Shared Pydantic v2 contracts for Solace.

This module is the single source of truth for data shapes shared across all
layers of the system.  It has **zero** internal imports — only pydantic,
stdlib enums, and datetime.  Any layer may safely import from here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ── Enumerations ──────────────────────────────────────────────────────────────


class Role(str, Enum):
    """The counselling persona an agent embodies for a given turn."""

    counsellor = "counsellor"  # Person-centred (Rogers)
    mentor = "mentor"          # Solution-focused / GROW
    teacher = "teacher"        # CBT psychoeducation
    peer = "peer"              # Lived-experience peer support


class Phase(str, Enum):
    """Structured session phase.  Transitions are decided by the Conductor."""

    checkin = "checkin"
    exploration = "exploration"
    understanding = "understanding"
    guidance = "guidance"
    closing = "closing"


class SafetyLevel(str, Enum):
    """Severity of a safety concern detected by the Sentinel."""

    none = "none"
    elevated = "elevated"
    crisis = "crisis"


class UserNeed(str, Enum):
    """High-level intent inferred for the current turn."""

    vent = "vent"          # User primarily wants to be heard
    understand = "understand"  # User wants insight / psychoeducation
    act = "act"            # User wants a plan / next steps
    unknown = "unknown"    # Not yet determined


class TaskKind(str, Enum):
    """LLM task type — used to select the right provider fallback chain."""

    converse = "converse"    # Free-form dialogue turn
    classify = "classify"    # Label assignment (emotion, intent)
    judge = "judge"          # Evaluation / critic role
    extract = "extract"      # Structured JSON extraction


# ── Safety ────────────────────────────────────────────────────────────────────


class SafetyResult(BaseModel):
    """Output of the Safety Sentinel (runs before any LLM call)."""

    model_config = ConfigDict(frozen=True)

    level: SafetyLevel
    reasons: list[str] = Field(default_factory=list, description="Human-readable flags triggered.")
    source: str = Field(description="'rule' for the rule-based sentinel; 'classifier' if ML.")
    action: str = Field(description="What the system should do, e.g. 'continue', 'warn', 'escalate'.")


# ── Emotion ───────────────────────────────────────────────────────────────────


class EmotionReading(BaseModel):
    """Raw output from the emotion classifier for a single utterance."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(description="Top emotion label.")
    score: float = Field(ge=0.0, le=1.0, description="Confidence for top label.")
    top_k: list[tuple[str, float]] = Field(
        default_factory=list, description="Top-k (label, score) pairs."
    )
    source_model: str = Field(description="Model name / identifier used for classification.")


class EmotionHypothesis(BaseModel):
    """A single hypothesis in the Explorer's emotional state model."""

    model_config = ConfigDict(frozen=True)

    label: str
    probability: float = Field(ge=0.0, le=1.0)


class EmotionalState(BaseModel):
    """The Explorer's maintained model of the user's emotional state.

    Updated after each turn; persisted in the session trace.
    """

    model_config = ConfigDict(frozen=True)

    hypotheses: list[EmotionHypothesis] = Field(
        default_factory=list, description="Distribution over emotion labels."
    )
    intensity: float = Field(ge=0.0, le=1.0, description="Estimated arousal / distress intensity.")
    trigger: str | None = Field(default=None, description="Identified precipitating event, if any.")
    duration: str | None = Field(default=None, description="How long the feeling has persisted.")
    functional_impact: str | None = Field(
        default=None, description="How the emotion affects daily functioning."
    )
    need: UserNeed = Field(default=UserNeed.unknown)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this state model.")
    open_questions: list[str] = Field(
        default_factory=list, description="Questions the Explorer still wants to ask."
    )


# ── User Profile ──────────────────────────────────────────────────────────────


class UserProfile(BaseModel):
    """Persistent profile for a user, maintained across sessions."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    display_name: str | None = None
    preferred_role: Role | None = None
    recurring_themes: list[str] = Field(default_factory=list)
    communication_prefs: dict[str, Any] = Field(
        default_factory=dict, description="Free-form key/value communication preferences."
    )
    values_notes: str | None = Field(
        default=None, description="Free text capturing user's stated values and priorities."
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Messaging ─────────────────────────────────────────────────────────────────


class Message(BaseModel):
    """A single message in the conversation history."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(description="'user', 'assistant', or 'system'.")
    content: str
    speaker_role: Role | None = Field(
        default=None, description="Which agent role produced this assistant message."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── RAG ───────────────────────────────────────────────────────────────────────


class RetrievedItem(BaseModel):
    """A single document chunk returned by the RAG retriever."""

    model_config = ConfigDict(frozen=True)

    collection: str = Field(description="ChromaDB collection name.")
    doc_id: str
    source: str = Field(description="Original source path or URL.")
    text: str
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity score.")


# ── Conductor ─────────────────────────────────────────────────────────────────


class ConductorDecision(BaseModel):
    """The Conductor's orchestration decision for the current turn."""

    model_config = ConfigDict(frozen=True)

    phase: Phase
    speaker: Role
    reason: str = Field(description="One-sentence rationale for the speaker/phase choice.")
    retrieve: bool = Field(description="Whether to invoke the RAG retriever.")
    retrieve_query: str | None = Field(
        default=None, description="Query string for RAG if retrieve=True."
    )


# ── Critic ────────────────────────────────────────────────────────────────────


class CriticVerdict(BaseModel):
    """The Critic's quality gate assessment of a drafted reply."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    issues: list[str] = Field(default_factory=list, description="Specific issues found.")
    grounded: bool = Field(description="Reply is grounded in retrieved docs or session context.")
    role_fidelity: bool = Field(description="Reply matches the assigned role's style/paradigm.")
    no_diagnosis: bool = Field(description="Reply makes no clinical diagnosis.")


# ── Tracing ───────────────────────────────────────────────────────────────────


class StepRecord(BaseModel):
    """Timing record for a single step in the per-turn pipeline."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Step name, e.g. 'safety', 'emotion', 'conductor'.")
    summary: str = Field(description="One-line summary of the step outcome.")
    latency_ms: float


class TurnTrace(BaseModel):
    """Full audit trace for a single conversation turn.

    Persisted to SQLite; drives the explanation panel and offline evaluation.
    """

    model_config = ConfigDict(frozen=True)

    turn_id: str
    session_id: str
    user_message: str
    safety: SafetyResult
    emotion: EmotionReading | None = None
    state_before: EmotionalState | None = None
    state_after: EmotionalState | None = None
    decision: ConductorDecision | None = None
    retrieved: list[RetrievedItem] = Field(default_factory=list)
    draft: str | None = None
    critic: CriticVerdict | None = None
    final_reply: str
    explanation: str = Field(description="Human-readable explanation of why this reply was given.")
    steps: list[StepRecord] = Field(default_factory=list)
    provider_used: str = Field(description="Provider+model that produced the final reply.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
