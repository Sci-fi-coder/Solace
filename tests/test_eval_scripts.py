"""Tests for synthetic evaluation datasets and disk-backed LLM result caching."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from app.contracts import Message, SafetyLevel
from app.safety.classifier import ClassifierResult
from scripts.eval_safety import CachedClassifier


class CountingClassifier:
    """Count deterministic classifier calls for cache behavior tests."""

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, message: str, history: Sequence[Message]) -> ClassifierResult:
        """Return a fixed crisis classification."""
        self.calls += 1
        return ClassifierResult(
            level=SafetyLevel.crisis,
            reasons=["synthetic"],
            provider="mock",
            model="synthetic-model",
        )


def test_evaluation_sets_meet_minimum_synthetic_sample_counts() -> None:
    """Checked-in synthetic evaluation sets meet the requested minimum sizes."""
    root = Path(__file__).parent / "data"
    safety_rows = (root / "safety_eval.jsonl").read_text(encoding="utf-8").splitlines()
    emotion_rows = (root / "emotion_eval.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(safety_rows) >= 80
    assert len(emotion_rows) >= 40
    assert all(isinstance(json.loads(row), dict) for row in safety_rows + emotion_rows)


def test_llm_evaluation_cache_uses_message_hash_and_model_without_storing_text(
    tmp_path: Path,
) -> None:
    """Repeated inputs are cache hits and cached results contain no raw message."""
    cache_path = tmp_path / "safety-cache.json"
    delegate = CountingClassifier()
    classifier = CachedClassifier(delegate, cache_path, "model-a")
    message = "Synthetic private text for cache key only."

    first = classifier.classify(message, [])
    second = classifier.classify(message, [])

    assert first == second
    assert delegate.calls == 1
    assert message not in cache_path.read_text(encoding="utf-8")

    other_model = CachedClassifier(delegate, cache_path, "model-b")
    other_model.classify(message, [])
    assert delegate.calls == 2
