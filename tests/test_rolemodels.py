"""Offline tests for historical role-model cards, selection, and critic rules."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import Phase, SafetyLevel, UserNeed
from app.rolemodels.critic import check_role_model_reply
from app.rolemodels.library import RoleModelLibrary, default_role_models_dir
from app.rolemodels.models import Person, RoleModelCard, RoleModelSource
from app.rolemodels.selector import role_model_allowed, select_role_model

CARD_DIR = Path(__file__).parents[1] / "data" / "role_models"
PEOPLE = {
    Person.gandhi: 6,
    Person.vivekananda: 6,
    Person.marcus_aurelius: 6,
    Person.kalam: 6,
}


@pytest.fixture()
def draft_library() -> RoleModelLibrary:
    """Load unreviewed cards explicitly for validation and offline tests."""
    return RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)


def test_all_24_drafts_validate_and_have_six_cards_per_person(
    draft_library: RoleModelLibrary,
) -> None:
    """Every supplied YAML card validates and the collection is balanced."""
    cards = draft_library.all()

    assert default_role_models_dir() == CARD_DIR
    assert len(cards) == 24
    assert Counter(card.person for card in cards) == Counter(PEOPLE)
    assert all(not card.reviewed for card in cards)
    assert all(card.verification.status == "needs_verification" for card in cards)
    assert all(card.sources and all(source.locator.strip() for source in card.sources) for card in cards)


def test_unreviewed_cards_are_excluded_by_default() -> None:
    """Normal library loading validates drafts but exposes only reviewed cards."""
    assert len(RoleModelLibrary.load(CARD_DIR)) == 0


def test_unreviewed_cards_require_the_explicit_validation_option() -> None:
    """The testing/validation option makes the supplied drafts available."""
    assert len(RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)) == 24


def test_schema_rejects_unknown_people_tags_urls_and_quote_fields(
    draft_library: RoleModelLibrary,
) -> None:
    """Card validation rejects unsupported people, tags, summary URLs, and quote fields."""
    base = draft_library.get("gandhi-pietermaritzburg")
    assert base is not None
    payload = base.model_dump(mode="json")

    for update, expected in (
        ({"person": "unsupported"}, "person"),
        ({"tags": ["invented-tag"]}, "unsupported role-model tags"),
        ({"summary": "Read https://example.org for more."}, "URLs"),
        ({"quotation": "invented"}, "quotation"),
    ):
        with pytest.raises(ValidationError, match=expected):
            RoleModelCard.model_validate({**payload, **update})


def test_schema_requires_a_source_and_locator(
    draft_library: RoleModelLibrary,
) -> None:
    """A card without bibliographic locator information cannot validate."""
    base = draft_library.get("gandhi-pietermaritzburg")
    assert base is not None
    payload = base.model_dump(mode="json")

    with pytest.raises(ValidationError, match="at least 1"):
        RoleModelCard.model_validate({**payload, "sources": []})
    with pytest.raises(ValidationError, match="at least 5 characters"):
        RoleModelSource(title="A source", locator=" ")


def test_loader_reports_the_exact_invalid_card_file(tmp_path: Path) -> None:
    """Library failures identify the YAML file and preserve the validation cause."""
    invalid = tmp_path / "invalid-card.yaml"
    invalid.write_text("id: invalid\nperson: unsupported\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid-card.yaml"):
        RoleModelLibrary.load(tmp_path, include_unreviewed=True)


def test_reviewed_defaults_to_false_and_verified_status_is_required(
    draft_library: RoleModelLibrary,
) -> None:
    """Cards remain inactive unless explicitly reviewed after source verification."""
    card = draft_library.get("gandhi-pietermaritzburg")
    assert card is not None
    payload = card.model_dump(mode="json")
    payload.pop("reviewed")

    assert RoleModelCard.model_validate(payload).reviewed is False
    with pytest.raises(ValidationError, match="verification.status"):
        RoleModelCard.model_validate(
            {**payload, "reviewed": True, "verification": {"status": "needs_verification"}}
        )


def test_selector_matches_theme_and_emotion_and_returns_no_unmatched_card(
    draft_library: RoleModelLibrary,
) -> None:
    """A distinctive theme/emotion combination selects a card deterministically."""
    selected = select_role_model(
        draft_library,
        "I feel angry and humiliated by an unfair experience.",
    )

    assert selected is not None
    assert selected.id == "gandhi-pietermaritzburg"
    assert select_role_model(draft_library, "I would like to order a sandwich.") is None


def test_selector_supports_explicit_person_requests(
    draft_library: RoleModelLibrary,
) -> None:
    """A direct person request selects only a card belonging to that person."""
    selected = select_role_model(draft_library, "Could you tell me about Marcus Aurelius?")

    assert selected is not None
    assert selected.person == Person.marcus_aurelius


def test_role_model_gate_enforces_safety_listening_phase_and_frequency() -> None:
    """Automatic inspiration is suppressed by safety, listening, phase, and use limits."""
    common = {
        "safety_level": SafetyLevel.none,
        "crisis_seen": False,
        "phase": Phase.guidance,
        "move": "offer_technique",
        "need": UserNeed.act,
        "explicit_request": False,
        "use_turns": [],
        "current_turn": 1,
    }

    assert role_model_allowed(**common)
    assert not role_model_allowed(**{**common, "safety_level": SafetyLevel.elevated})
    assert not role_model_allowed(**{**common, "crisis_seen": True})
    assert not role_model_allowed(**{**common, "need": UserNeed.vent})
    assert not role_model_allowed(**{**common, "move": "ask_clarifying"})
    assert not role_model_allowed(**{**common, "phase": Phase.exploration})
    assert not role_model_allowed(**{**common, "use_turns": [1], "current_turn": 5})
    assert role_model_allowed(**{**common, "use_turns": [1], "current_turn": 6})
    assert not role_model_allowed(**{**common, "use_turns": [1, 6], "current_turn": 7})


def test_explicit_person_request_bypasses_phase_and_interval_only() -> None:
    """Explicit requests bypass ordinary phase/spacing rules, never safety or hard cap."""
    common = {
        "safety_level": SafetyLevel.none,
        "crisis_seen": False,
        "phase": Phase.checkin,
        "move": "offer_technique",
        "need": UserNeed.act,
        "explicit_request": True,
        "use_turns": [1],
        "current_turn": 2,
    }

    assert role_model_allowed(**common)
    assert not role_model_allowed(**{**common, "safety_level": SafetyLevel.elevated})
    assert not role_model_allowed(**{**common, "use_turns": [1, 2]})


def test_critic_accepts_grounded_analogy_and_rejects_unsupported_claims(
    draft_library: RoleModelLibrary,
) -> None:
    """The critic blocks unsupported names, dates, quotations, and equivalence claims."""
    card = draft_library.get("gandhi-pietermaritzburg")
    assert card is not None
    valid = (
        "Gandhi was removed from a first-class railway compartment despite holding a ticket. "
        "That may offer an analogy for responding constructively to an unjust experience."
    )
    assert check_role_model_reply(valid, selected_card=card, used_card_id=card.id) == []

    invalid_replies = (
        "Gandhi was removed in 1947. That is an analogy for your response.",
        'Gandhi said "keep going". This analogy could be useful.',
        "Gandhi invented the telephone. This is an analogy.",
        "Gandhi's situation is exactly like yours.",
        "Churchill led a campaign. This may be an analogy.",
        "Vivekananda led a campaign. This may be an analogy.",
    )
    for reply_text in invalid_replies:
        assert check_role_model_reply(
            reply_text, selected_card=card, used_card_id=card.id
        )


def test_critic_rejects_historical_mentions_without_used_card_provenance(
    draft_library: RoleModelLibrary,
) -> None:
    """Mentioning a historical figure without the selected-card id is not provenance."""
    card = draft_library.get("gandhi-pietermaritzburg")
    assert card is not None

    assert check_role_model_reply(
        "Gandhi's story may be an analogy.", selected_card=card, used_card_id=None
    )
    assert check_role_model_reply(
        "Gandhi's story may be an analogy.", selected_card=None, used_card_id=None
    )
    assert check_role_model_reply(
        "Churchill led a campaign.", selected_card=None, used_card_id=None
    )
