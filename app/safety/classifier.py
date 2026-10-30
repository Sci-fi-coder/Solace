"""LLM-backed safety level classification using the existing gateway."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.contracts import Message, SafetyLevel, TaskKind
from app.llm.gateway import LLMGateway


class SafetyClassification(BaseModel):
    """Validated structured output from the safety classifier."""

    model_config = ConfigDict(frozen=True)

    level: SafetyLevel
    reasons: list[str] = Field(default_factory=list)


class ClassifierResult(BaseModel):
    """Classifier output and key-free gateway diagnostics."""

    model_config = ConfigDict(frozen=True)

    level: SafetyLevel
    reasons: list[str] = Field(default_factory=list)
    provider: str
    model: str
    attempts: list[dict[str, str]] = Field(default_factory=list)


class SafetyClassifier(Protocol):
    """Interface implemented by the gateway classifier and deterministic test fakes."""

    def classify(self, message: str, history: Sequence[Message]) -> ClassifierResult:
        """Classify the message with up to six previous user turns as context."""


class GatewaySafetyClassifier:
    """Use TaskKind.classify with validated JSON and a zero-temperature request."""

    def __init__(self, gateway: LLMGateway) -> None:
        """Create a classifier backed by an already configured gateway."""
        self._gateway = gateway

    def classify(self, message: str, history: Sequence[Message]) -> ClassifierResult:
        """Classify a message and preserve provider fallback diagnostics."""
        prior_user_turns = [item.content for item in history if item.role == "user"][-6:]
        context = "\n".join(f"- {turn}" for turn in prior_user_turns) or "(none)"
        result = self._gateway.generate(
            TaskKind.classify,
            [
                {
                    "role": "system",
                    "content": (
                        "Classify safety risk only. Return JSON with level (none, elevated, crisis) "
                        "and concise general reasons. Never quote the user. Rules: crisis means "
                        "explicit imminent intent/plan to suicide or harm someone; elevated means "
                        "passive ideation, hopelessness, inability to continue, or self-harm urges "
                        "without explicit intent. Do not treat idioms or clearly negated intent as "
                        "risk. Do not diagnose."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Previous user turns (up to six):\n{context}\n\n"
                        f"Current user turn:\n{message}"
                    ),
                },
            ],
            json_schema=SafetyClassification,
            temperature=0,
            max_tokens=160,
        )
        parsed = result.parsed
        if not isinstance(parsed, SafetyClassification):
            parsed = SafetyClassification.model_validate(parsed)
        return ClassifierResult(
            level=parsed.level,
            reasons=parsed.reasons,
            provider=result.provider,
            model=result.model,
            attempts=[dict(attempt) for attempt in result.attempts],
        )


__all__ = [
    "ClassifierResult",
    "GatewaySafetyClassifier",
    "SafetyClassification",
    "SafetyClassifier",
]
