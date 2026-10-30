"""Evaluate safety rules, optionally comparing them with the gateway classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import create_engine

from app.config import settings
from app.contracts import Message, SafetyLevel, TaskKind
from app.llm.gateway import LLMGateway
from app.safety.classifier import (
    ClassifierResult,
    GatewaySafetyClassifier,
    SafetyClassifier,
)
from app.safety.sentinel import SafetyGate
from app.storage.models import create_all

LEVELS = [SafetyLevel.none, SafetyLevel.elevated, SafetyLevel.crisis]
DATA_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "safety_eval.jsonl"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "safety_eval.json"
logger = logging.getLogger(__name__)


class CachedClassifier:
    """Cache structured classifier results without storing message content."""

    def __init__(
        self,
        delegate: SafetyClassifier,
        cache_path: Path,
        model_key: str,
    ) -> None:
        self._delegate = delegate
        self._cache_path = cache_path
        self._model_key = model_key
        self._cache: dict[str, object] = {}
        self.live_calls = 0
        self.cache_hits = 0
        self.providers: dict[str, int] = {}
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Safety cache must contain an object: {cache_path}")
            self._cache = payload

    def classify(self, message: str, history: Sequence[Message]) -> ClassifierResult:
        """Return a cached classification or make one gateway call and persist it."""
        inputs = json.dumps(
            {
                "model": self._model_key,
                "message": message,
                "history": [item.content for item in history if item.role == "user"][-6:],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(inputs.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return ClassifierResult.model_validate(cached)
        result = self._delegate.classify(message, history)
        self.live_calls += 1
        self.providers[result.provider] = self.providers.get(result.provider, 0) + 1
        self._cache[key] = result.model_dump(mode="json")
        self._write_cache()
        return result

    def _write_cache(self) -> None:
        """Write cache data atomically after each successful model response."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, self._cache_path)


def _read_samples(path: Path, limit: int | None) -> list[dict[str, str]]:
    """Read labelled synthetic examples from JSON Lines."""
    samples: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object on line {line_number}.")
            text = payload.get("text")
            expected = payload.get("expected_level")
            session_id = payload.get("session_id", f"sample-{line_number}")
            if not isinstance(text, str) or not isinstance(expected, str):
                raise ValueError(f"Missing text or expected_level on line {line_number}.")
            SafetyLevel(expected)
            samples.append(
                {"text": text, "expected_level": expected, "session_id": str(session_id)}
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def _predict(
    samples: list[dict[str, str]],
    classifier: SafetyClassifier | None,
) -> list[SafetyLevel]:
    """Run safety checks in dataset order with isolated state per synthetic session."""
    gate = SafetyGate(classifier=classifier)
    states: dict[str, dict[str, object]] = {}
    histories: dict[str, list[Message]] = {}
    predictions: list[SafetyLevel] = []
    for sample in samples:
        session_id = sample["session_id"]
        session_state = states.setdefault(session_id, {})
        history = histories.setdefault(session_id, [])
        result = gate.check(sample["text"], history, session_state)
        predictions.append(result.level)
        history.append(Message(role="user", content=sample["text"]))
    return predictions


def _print_metrics(label: str, samples: list[dict[str, str]], predicted: list[SafetyLevel]) -> None:
    """Print crisis metrics and a labelled confusion matrix."""
    actual = [SafetyLevel(sample["expected_level"]) for sample in samples]
    true_crisis = sum(expected == SafetyLevel.crisis for expected in actual)
    predicted_crisis = sum(level == SafetyLevel.crisis for level in predicted)
    correctly_detected = sum(
        expected == got == SafetyLevel.crisis
        for expected, got in zip(actual, predicted, strict=True)
    )
    none_total = sum(expected == SafetyLevel.none for expected in actual)
    false_positives = sum(
        expected == SafetyLevel.none and got != SafetyLevel.none
        for expected, got in zip(actual, predicted, strict=True)
    )
    precision = correctly_detected / predicted_crisis if predicted_crisis else 0.0
    recall = correctly_detected / true_crisis if true_crisis else 0.0
    false_positive_rate = false_positives / none_total if none_total else 0.0
    print(f"\n{label} ({len(samples)} examples)")
    print(f"Crisis recall: {recall:.3f} ({correctly_detected}/{true_crisis})")
    print(f"Crisis precision: {precision:.3f} ({correctly_detected}/{predicted_crisis})")
    print(f"False-positive rate on none: {false_positive_rate:.3f} "
          f"({false_positives}/{none_total})")
    print("Confusion matrix (rows=expected, columns=predicted):")
    print("expected\\predicted " + " ".join(f"{level.value:>9}" for level in LEVELS))
    for expected in LEVELS:
        counts = [
            sum(
                actual_item == expected and predicted_item == predicted_level
                for actual_item, predicted_item in zip(actual, predicted, strict=True)
            )
            for predicted_level in LEVELS
        ]
        print(f"{expected.value:>18} " + " ".join(f"{count:>9}" for count in counts))


def _print_classifier_differences(
    samples: list[dict[str, str]],
    rules_predictions: list[SafetyLevel],
    combined_predictions: list[SafetyLevel],
) -> None:
    """Report cases corrected by the classifier and new none-class false alarms."""
    caught = [
        sample["text"]
        for sample, rules_level, combined_level in zip(
            samples,
            rules_predictions,
            combined_predictions,
            strict=True,
        )
        if rules_level != SafetyLevel(sample["expected_level"])
        and combined_level == SafetyLevel(sample["expected_level"])
    ]
    false_alarms = []
    for sample, rules_level, combined_level in zip(
        samples,
        rules_predictions,
        combined_predictions,
        strict=True,
    ):
        expected = SafetyLevel(sample["expected_level"])
        if rules_level == expected and LEVELS.index(combined_level) > LEVELS.index(expected):
            false_alarms.append(
                f"{sample['text']} (expected {expected.value}, got {combined_level.value})"
            )
    print("\nLLM caught rules misses:")
    print("\n".join(f"- {text}" for text in caught) if caught else "- none")
    print("\nNew classifier false alarms:")
    print("\n".join(f"- {text}" for text in false_alarms) if false_alarms else "- none")


def main() -> None:
    """Run rules-only evaluation and the opt-in cached LLM comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true", help="Opt in to gateway API calls.")
    parser.add_argument("--limit", type=int, help="Maximum number of examples to evaluate.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least one.")
    limit = args.limit if args.limit is not None else (20 if args.with_llm else None)
    samples = _read_samples(DATA_PATH, limit)
    if not samples:
        raise ValueError(f"No safety evaluation examples found in {DATA_PATH}.")
    rules_predictions = _predict(samples, classifier=None)
    _print_metrics("Rules only", samples, rules_predictions)

    if args.with_llm:
        engine = create_engine("sqlite:///:memory:")
        create_all(engine)
        gateway = LLMGateway.from_settings(engine=engine)
        model_key = "|".join(settings.get_fallback_chain(TaskKind.classify.value))
        classifier = CachedClassifier(
            GatewaySafetyClassifier(gateway),
            args.cache,
            model_key,
        )
        combined_predictions = _predict(samples, classifier=classifier)
        _print_metrics("Rules + LLM", samples, combined_predictions)
        _print_classifier_differences(samples, rules_predictions, combined_predictions)
        print(
            f"\nLive classifier calls: {classifier.live_calls}; "
            f"cache hits: {classifier.cache_hits}; providers: {classifier.providers}"
        )
        engine.dispose()
    else:
        print("\nRules + LLM: not run (pass --with-llm to opt in; default is rules-only).")


if __name__ == "__main__":
    main()
