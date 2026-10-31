"""Validate drafted knowledge-card YAML and report collection coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.rag.cards import COARSE_EMOTION_LABELS, Card, banned_content_reasons

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "data" / "kb" / "cards"
WORK_DIR = ROOT / "data" / "kb_work"


def _prose(card: Card) -> str:
    """Return the card's user-facing prose fields."""
    return "\n".join(
        [
            card.title,
            card.summary,
            *card.when_to_use,
            *card.when_not_to_use,
            *card.steps,
            card.notes,
        ]
    )


def _validate_citations(card: Card) -> list[str]:
    """Ensure every cited source page exists in the extracted PDF-page index."""
    errors: list[str] = []
    for citation in card.sources:
        extraction_path = WORK_DIR / f"{citation.source_id}.json"
        if not extraction_path.exists():
            errors.append(f"missing extracted source {citation.source_id}")
            continue
        pages = json.loads(extraction_path.read_text(encoding="utf-8"))
        if not isinstance(pages, list):
            errors.append(f"invalid page index for {citation.source_id}")
            continue
        page_numbers = {
            row.get("page_number")
            for row in pages
            if isinstance(row, dict) and isinstance(row.get("page_number"), int)
        }
        for page_number in citation.pages:
            if page_number not in page_numbers:
                errors.append(
                    f"{citation.source_id} PDF page {page_number} does not exist"
                )
    return errors


def validate_cards(cards_dir: Path = CARDS_DIR) -> int:
    """Validate schemas, ids, banned text, citations, and print coverage reports."""
    errors: list[str] = []
    cards: list[Card] = []
    id_paths: dict[str, Path] = {}
    files = sorted(cards_dir.glob("*.yaml")) if cards_dir.exists() else []
    if not files:
        errors.append(f"no card YAML files found in {cards_dir}")

    for path in files:
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
            card = Card.model_validate(payload)
        except (yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
            errors.append(f"{path.name}: schema error ({type(exc).__name__})")
            continue
        if path.stem != card.id:
            errors.append(f"{path.name}: filename does not match card id {card.id}")
        previous = id_paths.get(card.id)
        if previous is not None:
            errors.append(f"duplicate id {card.id}: {previous.name} and {path.name}")
        else:
            id_paths[card.id] = path
        for reason in banned_content_reasons(_prose(card)):
            errors.append(f"{path.name}: banned content pattern ({reason})")
        errors.extend(f"{path.name}: {reason}" for reason in _validate_citations(card))
        cards.append(card)

    emotion_counts = Counter(
        emotion for card in cards for emotion in card.emotions
    )
    category_counts = Counter(card.category for card in cards)
    print(f"Cards: {len(cards)}")
    print("Counts by category:")
    for category in (
        "technique",
        "psychoeducation",
        "communication_skill",
        "when_to_seek_help",
    ):
        print(f"  {category}: {category_counts[category]}")
    print("Emotion coverage (label: card count; target >= 2):")
    for label in sorted(COARSE_EMOTION_LABELS):
        count = emotion_counts[label]
        print(f"  {label}: {count}")
        if count < 2:
            print(f"    WARNING: {label} appears on fewer than two cards.")
    if category_counts["when_to_seek_help"] != 1:
        print(
            "WARNING: expected exactly one when_to_seek_help card; "
            f"found {category_counts['when_to_seek_help']}."
        )
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("Validation passed.")
    return 0


def main() -> int:
    """Run knowledge-card validation and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-dir", type=Path, default=CARDS_DIR)
    args = parser.parse_args()
    return validate_cards(args.cards_dir)


if __name__ == "__main__":
    raise SystemExit(main())
