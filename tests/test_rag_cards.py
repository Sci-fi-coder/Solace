"""Tests for the Pydantic knowledge-card contract and policy checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.cards import COARSE_EMOTION_LABELS, Card, banned_content_reasons


def valid_card_payload() -> dict[str, object]:
    """Return a minimal valid card payload for schema tests."""
    return {
        "id": "grounding-under-stress",
        "title": "Grounding Under Stress",
        "category": "technique",
        "summary": (
            "Notice a few neutral details around you to steady your attention. "
            "This can make it easier to choose a small next action."
        ),
        "when_to_use": ["When stress makes it hard to focus."],
        "when_not_to_use": ["When noticing the surroundings feels uncomfortable."],
        "steps": ["Name one thing you see.", "Notice one sound."],
        "emotions": ["anxiety_fear"],
        "role_affinity": ["counsellor", "mentor"],
        "sources": [{"source_id": "dwm_stress", "pages": [4, 5]}],
    }


def test_card_schema_uses_coarse_emotion_labels_and_unreviewed_default() -> None:
    """The card accepts recognizer labels and defaults to human review pending."""
    card = Card.model_validate(valid_card_payload())

    assert card.reviewed is False
    assert set(card.emotions) <= COARSE_EMOTION_LABELS
    assert card.sources[0].pages == [4, 5]


def test_card_schema_rejects_bad_emotion_role_summary_and_step_count() -> None:
    """The schema rejects unsupported labels, roles, summaries, and oversized plans."""
    payload = valid_card_payload()
    payload["emotions"] = ["calm"]
    with pytest.raises(ValidationError):
        Card.model_validate(payload)

    payload = valid_card_payload()
    payload["role_affinity"] = ["doctor"]
    with pytest.raises(ValidationError):
        Card.model_validate(payload)

    payload = valid_card_payload()
    payload["summary"] = "Only one sentence."
    with pytest.raises(ValidationError):
        Card.model_validate(payload)

    payload = valid_card_payload()
    payload["steps"] = ["step"] * 7
    with pytest.raises(ValidationError):
        Card.model_validate(payload)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Call +1 800 555 1234 for details.", "phone number"),
        ("Read https://example.org/help for more.", "URL"),
        ("Use 5 mg each morning.", "dosage"),
        ("Ask whether sertraline is appropriate.", "medication name or reference"),
        ("Call the Lifeline when distressed.", "helpline name or reference"),
        ("This meets the DSM-5 criteria.", "diagnostic criteria"),
        ("Ways to end my life should be listed.", "self-harm or suicide method detail"),
    ],
)
def test_banned_content_patterns(text: str, reason: str) -> None:
    """Banned contact, medication, diagnostic, and method details are detected."""
    assert reason in banned_content_reasons(text)


def test_unlisted_emotion_is_rejected() -> None:
    """The imported emotion-label set remains the schema's source of truth."""
    payload = valid_card_payload()
    payload["emotions"] = ["positive", "not-a-recognizer-label"]

    with pytest.raises(ValidationError, match="unsupported coarse emotion labels"):
        Card.model_validate(payload)
