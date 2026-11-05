"""Offline tests for historical role-model cards, selection, and critic rules."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import Phase, SafetyLevel, UserNeed
from app.rolemodels.critic import _unsupported_capitalized_names, check_role_model_reply
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
    """All 24 cards validate and preserve their individual verification states."""
    cards = draft_library.all()

    assert default_role_models_dir() == CARD_DIR
    assert len(cards) == 24
    assert Counter(card.person for card in cards) == Counter(PEOPLE)

    reviewed_cards = [card for card in cards if card.reviewed]
    unreviewed_cards = [card for card in cards if not card.reviewed]

    assert all(
        card.verification.status == "verified"
        for card in reviewed_cards
    )
    assert all(
        card.verification.status == "needs_verification"
        for card in unreviewed_cards
    )
    assert all(
        card.sources and all(source.locator.strip() for source in card.sources)
        for card in cards
    )

def test_unreviewed_cards_are_excluded_by_default() -> None:
    """Normal library loading exposes only cards whose metadata marks them reviewed."""
    reviewed_count = sum(
        card.reviewed
        for card in RoleModelLibrary.load(CARD_DIR, include_unreviewed=True).all()
    )

    assert len(RoleModelLibrary.load(CARD_DIR)) == reviewed_count


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


def test_explicit_person_request_bypasses_phase_spacing_and_clarifying_move_only() -> None:
    """Explicit requests bypass phase/spacing and clarification, never listening or safety."""
    common = {
        "safety_level": SafetyLevel.none,
        "crisis_seen": False,
        "phase": Phase.exploration,
        "move": "ask_clarifying",
        "need": UserNeed.act,
        "explicit_request": True,
        "use_turns": [1],
        "current_turn": 2,
    }

    assert role_model_allowed(**common)
    assert not role_model_allowed(**{**common, "move": "reflect"})
    assert not role_model_allowed(**{**common, "move": "summarise"})
    assert not role_model_allowed(**{**common, "need": UserNeed.vent})
    assert not role_model_allowed(**{**common, "safety_level": SafetyLevel.elevated})
    assert not role_model_allowed(**{**common, "crisis_seen": True})
    assert not role_model_allowed(**{**common, "use_turns": [1, 2]})


def test_explicit_request_does_not_bypass_the_two_use_limit() -> None:
    """Explicit requests remain blocked after two historical analogies in the session."""
    assert not role_model_allowed(
        safety_level=SafetyLevel.none,
        crisis_seen=False,
        phase=Phase.exploration,
        move="ask_clarifying",
        need=UserNeed.understand,
        explicit_request=True,
        use_turns=[2, 5],
        current_turn=8,
    )


def test_critic_accepts_grounded_analogy_and_rejects_unsupported_claims(
    draft_library: RoleModelLibrary,
) -> None:
    """The critic blocks unsupported names, dates, quotations, and equivalence claims."""
    card = draft_library.get("gandhi-pietermaritzburg")
    assert card is not None
    valid = (
        "Gandhi's unverified account offers a general lesson about responding constructively "
        "to an unjust experience. That lesson may be an analogy for your situation."
    )
    assert check_role_model_reply(valid, selected_card=card, used_card_id=card.id) == []

    invalid_replies = (
        "Gandhi was removed in 1947. That is an analogy for your response.",
        'Gandhi said "keep going". This analogy could be useful.',
        "Gandhi invented the telephone. This is an analogy.",
        "Gandhi travelled to London. This may offer an analogy.",
        "Gandhi worked with 12 people. This may offer an analogy.",
        "Gandhi's situation is exactly like yours.",
        "Churchill led a campaign. This may be an analogy.",
        "Vivekananda led a campaign. This may be an analogy.",
    )
    for reply_text in invalid_replies:
        assert check_role_model_reply(
            reply_text, selected_card=card, used_card_id=card.id
        )

    assert check_role_model_reply(
        valid, selected_card=card, used_card_id="a-different-card"
    )


@pytest.mark.parametrize(
    "reply",
    [
        "A useful analogy from Kalam's example for your situation is meaningful roles "
        "alongside experienced guidance.",
        "Kalam's lesson about meaningful roles and experienced guidance is a comparison "
        "for making room for someone to contribute.",
        "Kalam's lesson about meaningful roles and experienced guidance offers a parallel "
        "to what you are facing.",
        "Kalam's lesson about meaningful roles and experienced guidance resonates with "
        "your concern.",
        "Kalam's lesson about meaningful roles and experienced guidance can be compared "
        "with the support you need.",
        "Kalam's lesson about meaningful roles and experienced guidance is a useful "
        "analogy for your situation.",
        "Kalam's lesson about meaningful roles and experienced guidance is relevant to "
        "your situation. As an analogy for your situation, this could mean offering meaningful "
        "responsibility alongside experienced guidance.",
    ],
)
def test_critic_accepts_analogy_frames_connected_to_the_person(
    draft_library: RoleModelLibrary, reply: str
) -> None:
    """Clear analogy wording must connect the historical lesson to the person's context."""
    card = draft_library.get("kalam-empower-young-scientists")
    assert card is not None

    assert check_role_model_reply(reply, selected_card=card, used_card_id=card.id) == []


@pytest.mark.parametrize(
    "reply",
    [
        "Kalam's example is relevant to your situation.",
        "Kalam's lesson about meaningful roles and experienced guidance is a useful analogy.",
        "Kalam describes meaningful roles with experienced guidance. This may help with "
        "your situation.",
        "Kalam's example has parallels with other historical stories.",
        "Kalam's lesson offers an analogy for explaining its historical details.",
        "Kalam's story is one example. As an analogy, this can be useful.",
    ],
)
def test_critic_rejects_historical_mentions_without_an_analogy_connection(
    draft_library: RoleModelLibrary, reply: str
) -> None:
    """A historical mention or detached analogy label is not a user-facing connection."""
    card = draft_library.get("kalam-empower-young-scientists")
    assert card is not None

    assert "historical example must be clearly framed as an analogy" in check_role_model_reply(
        reply, selected_card=card, used_card_id=card.id
    )


@pytest.mark.parametrize(
    "name",
    ["A.P.J. Abdul Kalam", "APJ Abdul Kalam", "A P J Abdul Kalam", "A.P.J. Kalam", "Abdul Kalam", "Kalam"],
)
def test_critic_accepts_kalam_canonical_name_and_initial_variants(
    draft_library: RoleModelLibrary, name: str
) -> None:
    """Canonical Kalam name forms and initials identify the selected card."""
    card = draft_library.get("kalam-empower-young-scientists")
    assert card is not None
    reply = (
        f"{name}'s lesson about emerging contributors is a comparison for making room "
        "for someone to contribute with experienced guidance nearby."
    )

    assert check_role_model_reply(reply, selected_card=card, used_card_id=card.id) == []


@pytest.mark.parametrize("name", ["Marcus Aurelius", "Marcus", "Aurelius"])
def test_critic_accepts_marcus_canonical_name_and_variants(
    draft_library: RoleModelLibrary, name: str
) -> None:
    """Marcus Aurelius's full name and recognized variants identify the selected card."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None
    reply = (
        f"{name}'s reflection on cooperation is an analogy for protecting your boundary "
        "while leaving room for safe cooperation."
    )

    assert check_role_model_reply(reply, selected_card=card, used_card_id=card.id) == []


@pytest.mark.parametrize(
    "opening",
    ["Feeling", "Just", "Does", "Learning", "Growing"],
)
def test_critic_allows_ordinary_sentence_openings(
    draft_library: RoleModelLibrary, opening: str
) -> None:
    """Sentence-initial capitalization alone does not turn ordinary prose into a name."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None
    reply = (
        f"{opening} stuck in a conflict can be difficult. Marcus Aurelius's lesson "
        "offers an analogy for keeping room for cooperation where it is safe."
    )

    assert _unsupported_capitalized_names(reply, card) == []


def test_critic_reports_exact_unsupported_capitalized_tokens(
    draft_library: RoleModelLibrary,
) -> None:
    """Unsupported names remain blocked and are named in the critic diagnostic."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None
    reply = (
        "Marcus Aurelius's lesson is an analogy for this conflict. "
        "Churchill led a campaign."
    )

    assert _unsupported_capitalized_names(reply, card) == ["Churchill"]
    assert (
        "unsupported capitalized token(s): Churchill"
        in check_role_model_reply(reply, selected_card=card, used_card_id=card.id)
    )


def test_critic_rejects_an_unsupported_place_name(
    draft_library: RoleModelLibrary,
) -> None:
    """A place absent from the selected card is reported as an unsupported token."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None
    reply = (
        "Marcus Aurelius's lesson is an analogy for this conflict. "
        "London was the setting for the historical event."
    )

    assert _unsupported_capitalized_names(reply, card) == ["London"]
    assert (
        "unsupported capitalized token(s): London"
        in check_role_model_reply(reply, selected_card=card, used_card_id=card.id)
    )


def test_critic_allows_grammatical_openings_without_a_selected_card() -> None:
    """Ordinary copular sentence openings are not mistaken for historical figures."""
    assert check_role_model_reply(
        "That was a difficult week. There were many demands to manage.",
        selected_card=None,
        used_card_id=None,
    ) == []
    assert check_role_model_reply(
        "Churchill led a campaign.",
        selected_card=None,
        used_card_id=None,
    )


def test_critic_allows_names_explicitly_supported_by_the_card(
    draft_library: RoleModelLibrary,
) -> None:
    """Proper names included in the selected card remain valid."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None
    card_with_place = card.model_copy(
        update={"summary": f"{card.summary} The passage discusses a meeting in Paris."}
    )
    reply = (
        "Marcus Aurelius's lesson is a comparison for your situation in Paris, "
        "while leaving room for cooperation where it is safe."
    )

    assert check_role_model_reply(
        reply, selected_card=card_with_place, used_card_id=card_with_place.id
    ) == []


def test_critic_accepts_selected_name_and_normal_sentence_openings(
    draft_library: RoleModelLibrary,
) -> None:
    """The selected person's name and ordinary sentence openers are not false positives."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None
    valid = (
        "Marcus Aurelius reflects on cooperation as part of shared human life. "
        "Still, that may offer an analogy for protecting a boundary while leaving room "
        "for cooperation where it is safe."
    )

    assert check_role_model_reply(valid, selected_card=card, used_card_id=card.id) == []


def test_name_checker_ignores_sentence_openers_in_generated_marcus_reply(
    draft_library: RoleModelLibrary,
) -> None:
    """Ordinary capitalized sentence openings in a generated reply are not names."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None
    reply = (
        "Feeling frustrated when team dynamics stall is common; it is natural to want things "
        "to run smoothly. Marcus Aurelius once compared humans to parts of a body, like hands "
        "or feet, designed to work together. Just as he saw cooperation as our natural state, "
        "you might consider if your reaction can protect your standards while still leaving "
        "room for your team to contribute. Perhaps you could focus on one small change you can "
        "influence, rather than the parts of their behavior you cannot control. Does this way "
        "of looking at your team feel like it fits your situation?"
    )

    assert not _unsupported_capitalized_names(reply, card)


def test_critic_rejects_declared_use_without_the_selected_person(
    draft_library: RoleModelLibrary,
) -> None:
    """A card cannot be declared as used unless its selected person is mentioned."""
    card = draft_library.get("marcus-cooperation")
    assert card is not None

    assert "declares historical-card use without mentioning its person" in check_role_model_reply(
        "A historical analogy may help you consider cooperation.",
        selected_card=card,
        used_card_id=card.id,
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
