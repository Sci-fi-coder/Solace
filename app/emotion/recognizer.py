"""GoEmotions inference with configurable coarse labels and lazy model loading."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from app.contracts import EmotionReading
from app.tracing.recorder import TraceRecorder

logger = logging.getLogger(__name__)
DEFAULT_MODEL = "SamLowe/roberta-base-go_emotions"
DEFAULT_LABEL_MAP = {
    "admiration": "positive",
    "amusement": "positive",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "positive",
    "caring": "positive",
    "confusion": "others",
    "curiosity": "others",
    "desire": "positive",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "shame_guilt",
    "excitement": "positive",
    "fear": "anxiety_fear",
    "gratitude": "positive",
    "grief": "sadness",
    "joy": "positive",
    "love": "positive",
    "nervousness": "anxiety_fear",
    "neutral": "neutral",
    "optimism": "positive",
    "pride": "positive",
    "realization": "others",
    "relief": "positive",
    "remorse": "shame_guilt",
    "sadness": "sadness",
    "surprise": "others",
}


class EmotionClassifier(Protocol):
    """Predict fine-grained emotion probabilities without prescribing an implementation."""

    def predict(self, message: str) -> Sequence[tuple[str, float]]:
        """Return (fine label, score) pairs ordered by descending score."""


class HuggingFaceEmotionClassifier:
    """Adapter that imports and constructs transformers only when first used."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._pipeline: Callable[..., object] | None = None

    def predict(self, message: str) -> Sequence[tuple[str, float]]:
        """Run CPU inference and normalize the Transformers pipeline output."""
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,
                device=-1,
            )
        output = self._pipeline(message, truncation=True)
        return _parse_pipeline_output(output)


def _parse_pipeline_output(output: object) -> list[tuple[str, float]]:
    """Validate the dynamic output returned by the Transformers pipeline."""
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    if not isinstance(output, list):
        raise TypeError("Transformers pipeline returned an unexpected output shape.")
    predictions: list[tuple[str, float]] = []
    for item in output:
        if not isinstance(item, dict):
            raise TypeError("Transformers pipeline output entries must be mappings.")
        label = item.get("label")
        score = item.get("score")
        if not isinstance(label, str) or not isinstance(score, int | float):
            raise TypeError("Transformers pipeline output requires string labels and numeric scores.")
        predictions.append((label, float(score)))
    if not predictions:
        raise ValueError("Transformers pipeline returned no emotion predictions.")
    return sorted(predictions, key=lambda prediction: prediction[1], reverse=True)


class EmotionRecognizer:
    """Return fine and coarse labels from GoEmotions without loading at import time."""

    def __init__(
        self,
        classifier: EmotionClassifier | None = None,
        *,
        model_name: str = DEFAULT_MODEL,
        label_map: Mapping[str, str] | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """Configure an injectable classifier, model, label map, and optional trace recorder."""
        self._classifier = (
            classifier if classifier is not None else HuggingFaceEmotionClassifier(model_name)
        )
        self._model_name = model_name
        self._label_map = dict(label_map or DEFAULT_LABEL_MAP)
        self._trace_recorder = trace_recorder

    def read(self, message: str) -> EmotionReading:
        """Classify a message into a fine label and aggregated coarse emotion probabilities."""
        started = time.perf_counter()
        try:
            predictions = self._classifier.predict(message)
            if not predictions:
                raise ValueError("Emotion classifier returned no predictions.")
            fine_label, _ = predictions[0]
            coarse_scores: defaultdict[str, float] = defaultdict(float)
            for label, score in predictions:
                coarse_scores[self._label_map.get(label, "others")] += score
            coarse_label = max(coarse_scores, key=coarse_scores.__getitem__)
            reading = EmotionReading(
                label=coarse_label,
                score=min(1.0, max(0.0, coarse_scores[coarse_label])),
                top_k=sorted(
                    coarse_scores.items(), key=lambda pair: pair[1], reverse=True
                ),
                source_model=self._model_name,
                fine_label=fine_label,
                coarse_label=coarse_label,
            )
            return reading
        finally:
            if self._trace_recorder is not None:
                elapsed_ms = (time.perf_counter() - started) * 1000
                summary = "emotion classification failed"
                if "reading" in locals():
                    summary = (
                        f"label={reading.label}, fine_label={reading.fine_label}, "
                        f"source_model={reading.source_model}"
                    )
                self._trace_recorder.record("emotion", summary, elapsed_ms)


__all__ = [
    "DEFAULT_LABEL_MAP",
    "DEFAULT_MODEL",
    "EmotionClassifier",
    "EmotionRecognizer",
    "HuggingFaceEmotionClassifier",
]
