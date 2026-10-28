"""Rule-first safety gate with optional classifier escalation and sticky sessions."""

from __future__ import annotations

import logging
import time
from collections.abc import MutableMapping

from pydantic import ValidationError

from app.contracts import Message, SafetyLevel, SafetyResult
from app.llm.gateway import GatewayExhaustedError, StructuredOutputError
from app.safety.classifier import ClassifierResult, SafetyClassifier
from app.safety.responses import action_for
from app.safety.rules import classify_rules
from app.tracing.recorder import TraceRecorder

logger = logging.getLogger(__name__)
_LEVEL_RANK = {
    SafetyLevel.none: 0,
    SafetyLevel.elevated: 1,
    SafetyLevel.crisis: 2,
}


class SafetyGate:
    """Apply rule screening first, then let an LLM classifier escalate only."""

    def __init__(
        self,
        classifier: SafetyClassifier | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """Create a gate; an omitted classifier runs rules-only and records degraded status."""
        self._classifier = classifier
        self._trace_recorder = trace_recorder

    def check(
        self,
        message: str,
        history: list[Message],
        session_state: MutableMapping[str, object],
    ) -> SafetyResult:
        """Screen a turn, preserve crisis escalation, and record its trace step."""
        started = time.perf_counter()
        try:
            rule_level, rule_reasons = classify_rules(message)
            classifier_result: ClassifierResult | None = None
            degraded = self._classifier is None
            if self._classifier is not None:
                try:
                    classifier_result = self._classifier.classify(message, history)
                except (
                    GatewayExhaustedError,
                    StructuredOutputError,
                    ValidationError,
                ) as exc:
                    degraded = True
                    logger.warning(
                        "Safety classifier unavailable; using rules only (%s).",
                        type(exc).__name__,
                    )

            classifier_level = (
                classifier_result.level if classifier_result is not None else SafetyLevel.none
            )
            level = max((rule_level, classifier_level), key=_LEVEL_RANK.__getitem__)
            reasons = list(rule_reasons)
            if classifier_result is not None:
                reasons.extend(
                    reason
                    for reason in classifier_result.reasons
                    if reason not in reasons
                )

            if session_state.get("crisis_seen") is True and level != SafetyLevel.crisis:
                level = max((level, SafetyLevel.elevated), key=_LEVEL_RANK.__getitem__)
                reasons.append("sticky_crisis_escalation")
            if level == SafetyLevel.crisis:
                session_state["crisis_seen"] = True

            source = (
                "rule+classifier"
                if classifier_result is not None and rule_level != SafetyLevel.none
                else "classifier"
                if classifier_result is not None and rule_level == SafetyLevel.none
                else "rule"
            )
            result = SafetyResult(
                level=level,
                reasons=reasons,
                source=source,
                action=action_for(level),
                degraded=degraded,
                classifier_provider=classifier_result.provider if classifier_result else None,
                classifier_model=classifier_result.model if classifier_result else None,
                classifier_attempts=(
                    classifier_result.attempts if classifier_result else []
                ),
            )
            return result
        finally:
            if self._trace_recorder is not None:
                elapsed_ms = (time.perf_counter() - started) * 1000
                summary = "safety check failed"
                if "result" in locals():
                    summary = (
                        f"level={result.level.value}, action={result.action}, "
                        f"degraded={result.degraded}"
                    )
                self._trace_recorder.record("safety", summary, elapsed_ms)
