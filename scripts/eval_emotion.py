"""Evaluate coarse emotion labels with the local GoEmotions model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.emotion.recognizer import DEFAULT_MODEL, EmotionRecognizer

DATA_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "emotion_eval.jsonl"


def main() -> None:
    """Run local model inference on the labelled synthetic emotion set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least one.")
    rows: list[tuple[str, str]] = []
    with DATA_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object on line {line_number}.")
            text = payload.get("text")
            expected = payload.get("expected_label")
            if not isinstance(text, str) or not isinstance(expected, str):
                raise ValueError(f"Missing text or expected_label on line {line_number}.")
            rows.append((text, expected))
            if args.limit is not None and len(rows) >= args.limit:
                break
    if not rows:
        raise ValueError(f"No emotion evaluation examples found in {DATA_PATH}.")

    recognizer = EmotionRecognizer(model_name=args.model)
    actual = [expected for _, expected in rows]
    predicted = [recognizer.read(text).coarse_label or "others" for text, _ in rows]
    labels = sorted(set(actual) | set(predicted))
    correct = sum(
        expected == got for expected, got in zip(actual, predicted, strict=True)
    )
    print(f"Model: {args.model}")
    print(f"Examples: {len(rows)}")
    print(f"Accuracy: {correct / len(rows):.3f} ({correct}/{len(rows)})")
    print("Confusion matrix (rows=expected, columns=predicted):")
    print("expected\\predicted " + " ".join(f"{label:>14}" for label in labels))
    matrix: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for expected, got in zip(actual, predicted, strict=True):
        matrix[expected][got] += 1
    for expected in labels:
        print(
            f"{expected:>18} "
            + " ".join(f"{matrix[expected][got]:>14}" for got in labels)
        )


if __name__ == "__main__":
    main()
