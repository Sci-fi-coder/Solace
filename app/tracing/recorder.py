"""TraceRecorder — context manager for assembling TurnTrace objects.

Usage::

    with TraceRecorder(turn_id="t-1", session_id="s-1", user_message="...") as rec:
        with rec.step("safety"):
            result = sentinel.check(user_message)
            rec.record("safety", f"level={result.level}", ...)

    trace: TurnTrace = rec.build(
        safety=..., final_reply="...", explanation="...", provider_used="gemini/..."
    )
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from app.contracts import (
    ConductorDecision,
    CriticVerdict,
    EmotionReading,
    EmotionalState,
    RetrievedItem,
    SafetyResult,
    StepRecord,
    TurnTrace,
)


class TraceRecorder:
    """Collects StepRecords during a turn and assembles a TurnTrace at the end.

    Designed to be used as a context manager, but can also be used manually
    (call ``open()`` and ``close()`` instead of ``with``).
    """

    def __init__(
        self,
        *,
        turn_id: str,
        session_id: str,
        user_message: str,
    ) -> None:
        """
        Args:
            turn_id: Unique identifier for this turn.
            session_id: The parent session.
            user_message: The raw user utterance (stored in trace; not logged).
        """
        self.turn_id = turn_id
        self.session_id = session_id
        self.user_message = user_message
        self._steps: list[StepRecord] = []
        self._step_start: float | None = None
        self._current_step: str | None = None

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # No automatic cleanup needed; caller must call build()

    # ── Step recording ────────────────────────────────────────────────────────

    @contextmanager
    def step(self, name: str) -> Generator["TraceRecorder", None, None]:
        """Context manager that automatically records a step with timing.

        Usage::

            with rec.step("safety") as rec:
                result = do_safety_check()
                rec.finish_step(f"level={result.level}")
        """
        t0 = time.perf_counter()
        self._current_step = name
        try:
            yield self
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            # If finish_step wasn't called, record a placeholder summary.
            if self._steps and self._steps[-1].name == name:
                pass  # already recorded by finish_step
            else:
                self._steps.append(
                    StepRecord(name=name, summary="completed", latency_ms=latency_ms)
                )

    def record(self, name: str, summary: str, latency_ms: float) -> None:
        """Manually record a step with an explicit latency measurement.

        Args:
            name: Step name.
            summary: One-line description of the outcome.
            latency_ms: Wall-clock duration in milliseconds.
        """
        self._steps.append(StepRecord(name=name, summary=summary, latency_ms=latency_ms))

    def finish_step(self, summary: str) -> None:
        """Finish the currently active step() context with a custom summary.

        Must be called inside a ``with rec.step(name):`` block.
        """
        # We don't have the start time here, so we just append with 0 latency;
        # the step() context manager will not double-append because we track the name.
        if self._current_step is not None:
            name = self._current_step
            # Remove any auto-appended placeholder for this step
            self._steps = [s for s in self._steps if s.name != name]
            self._steps.append(StepRecord(name=name, summary=summary, latency_ms=0.0))

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        *,
        safety: SafetyResult,
        final_reply: str,
        explanation: str,
        provider_used: str,
        emotion: EmotionReading | None = None,
        state_before: EmotionalState | None = None,
        state_after: EmotionalState | None = None,
        decision: ConductorDecision | None = None,
        retrieved: list[RetrievedItem] | None = None,
        draft: str | None = None,
        critic: CriticVerdict | None = None,
    ) -> TurnTrace:
        """Assemble and return a TurnTrace from collected data.

        Args:
            safety: Safety Sentinel result (required).
            final_reply: The reply actually sent to the user.
            explanation: Human-readable explanation of why this reply was given.
            provider_used: "<provider>/<model>" string.
            emotion: Emotion classifier output (optional in Layer 0).
            state_before: Emotional state before this turn.
            state_after: Emotional state after this turn.
            decision: Conductor's orchestration decision.
            retrieved: RAG results used.
            draft: Initial draft before Critic gate.
            critic: Critic's verdict.
        """
        return TurnTrace(
            turn_id=self.turn_id,
            session_id=self.session_id,
            user_message=self.user_message,
            safety=safety,
            emotion=emotion,
            state_before=state_before,
            state_after=state_after,
            decision=decision,
            retrieved=retrieved or [],
            draft=draft,
            critic=critic,
            final_reply=final_reply,
            explanation=explanation,
            steps=list(self._steps),
            provider_used=provider_used,
            created_at=datetime.now(timezone.utc),
        )
