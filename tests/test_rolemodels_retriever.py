"""Offline tests for hybrid retrieval of historical role-model cards."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.contracts import EmotionalState, EmotionHypothesis, UserNeed
from app.rag.embedder import LocalEmbedder
from app.rolemodels.library import RoleModelLibrary
from app.rolemodels.models import Person
from app.rolemodels.retriever import RetrievalWeights, RoleModelRetriever

CARD_DIR = Path(__file__).parents[1] / "data" / "role_models"


class DeterministicTestEmbedder:
    """Provide scripted vectors for tests without claiming semantic model quality."""

    def __init__(
        self,
        *,
        target_titles: tuple[str, ...] = (),
        query_vector: tuple[float, ...] = (1.0, 0.0),
        target_vector: tuple[float, ...] = (1.0, 0.0),
    ) -> None:
        """Configure scripted semantic vectors for stable ranking assertions."""
        self.target_titles = set(target_titles)
        self.query_vector = query_vector
        self.target_vector = target_vector
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return fixed query/card vectors based on the input's first line."""
        self.calls.append(tuple(texts))
        vectors: list[list[float]] = []
        for index, text in enumerate(texts):
            if text.startswith("Title: "):
                title = text.splitlines()[0].removeprefix("Title: ")
                vector = self.target_vector if title in self.target_titles else (0.0, 1.0)
            else:
                vector = self.query_vector
            vectors.append(list(vector))
            if index == 0 and text.startswith("Title: "):
                raise AssertionError("query must precede card documents")
        return vectors


def _retriever(
    *,
    target_titles: tuple[str, ...] = (),
    query_vector: tuple[float, ...] = (1.0, 0.0),
    target_vector: tuple[float, ...] = (1.0, 0.0),
    minimum_relevance: float = 0.50,
    top_match_margin: float | None = None,
) -> tuple[RoleModelRetriever, DeterministicTestEmbedder]:
    """Build a retriever using the real all-card library and deterministic vectors."""
    embedder = DeterministicTestEmbedder(
        target_titles=target_titles,
        query_vector=query_vector,
        target_vector=target_vector,
    )
    return (
        RoleModelRetriever(
            directory=CARD_DIR,
            embedder=embedder,
            minimum_relevance=minimum_relevance,
            top_match_margin=top_match_margin,
        ),
        embedder,
    )


def test_default_retriever_loads_all_reviewed_and_unreviewed_cards() -> None:
    """All 24 cards are available and keep their original verification metadata."""
    retriever = RoleModelRetriever(embedder=DeterministicTestEmbedder())
    result = retriever.retrieve("A general question with no role-model theme.")

    assert len(result.ranked_matches) == 24
    cards = [item.card for item in result.ranked_matches]
    reviewed_cards = [card for card in cards if card.reviewed]
    unreviewed_cards = [card for card in cards if not card.reviewed]
    assert all(
        card.verification.status == "verified" for card in reviewed_cards
    )
    assert all(
        card.verification.status == "needs_verification"
        for card in unreviewed_cards
    )


def test_unreviewed_match_preserves_verification_and_source_metadata() -> None:
    """Retrieval exposes an unreviewed card without promoting or rewriting its metadata."""
    target_id = "gandhi-pietermaritzburg"
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    original = library.get(target_id)
    assert original is not None
    retriever = RoleModelRetriever(
        directory=CARD_DIR,
        embedder=DeterministicTestEmbedder(target_titles=(original.title,)),
    )

    result = retriever.retrieve("I was publicly mocked and feel furious.")

    assert result.match is not None
    assert result.match.card == original
    assert result.match.card.reviewed is False
    assert result.match.card.verification.status == "needs_verification"
    assert result.match.avoid_when == original.avoid_when
    assert result.match.source_locators == tuple(
        source.locator for source in original.sources
    )


def test_relevant_paraphrase_retrieves_the_expected_card() -> None:
    """Semantic evidence can recover a relevant card from a paraphrased concern."""
    target = "Responding to humiliation with purpose"
    retriever, _ = _retriever(target_titles=(target,))

    result = retriever.retrieve(
        "They mocked me in front of everyone; I want to turn that hurt into "
        "a constructive response."
    )

    assert result.match is not None
    assert result.match.card.id == "gandhi-pietermaritzburg"
    assert result.match.breakdown.semantic_similarity == 1.0
    assert result.match.breakdown.theme_overlap > 0.0


def test_unrelated_message_abstains_instead_of_returning_the_closest_card() -> None:
    """An unrelated query below the relevance threshold produces no match."""
    retriever, _ = _retriever()

    result = retriever.retrieve("What is the capital of Portugal?")

    assert result.match is None
    assert result.abstention_reason == "below_threshold"
    assert len(result.ranked_matches) == 24


def test_close_relevant_candidates_can_abstain_as_ambiguous() -> None:
    """The configured top-match margin rejects near-tied relevant candidates."""
    first = "Responding to humiliation with purpose"
    second = "Creating a pause within"
    retriever, _ = _retriever(
        target_titles=(first, second),
        top_match_margin=0.10,
    )

    result = retriever.retrieve("I need help thinking through a difficult situation.")

    assert result.match is None
    assert result.abstention_reason == "ambiguous"
    assert result.ranked_matches[0].score == result.ranked_matches[1].score


def test_explicit_person_restricts_candidates_but_does_not_bypass_threshold() -> None:
    """An explicit person narrows results but still must pass relevance scoring."""
    target = "Responding to humiliation with purpose"
    retriever, _ = _retriever(
        target_titles=(),
        minimum_relevance=1.0,
    )

    result = retriever.retrieve(
        "I feel angry and humiliated; could you share Gandhi's example?"
    )

    assert result.explicit_person == Person.gandhi
    assert result.match is None
    assert result.abstention_reason == "below_threshold"
    assert result.ranked_matches
    assert all(item.card.person == Person.gandhi for item in result.ranked_matches)
    assert any(item.card.title == target for item in result.ranked_matches)


def test_explicit_person_prioritizes_its_relevant_cards() -> None:
    """An explicit supported-person mention ranks only that person's cards."""
    target = "Responding to humiliation with purpose"
    retriever, _ = _retriever(target_titles=(target,))

    result = retriever.retrieve(
        "I feel angry and humiliated; could you share Gandhi's example?"
    )

    assert result.match is not None
    assert result.match.card.id == "gandhi-pietermaritzburg"
    assert len(result.ranked_matches) == 6
    assert all(item.card.person == Person.gandhi for item in result.ranked_matches)


def test_score_breakdown_matches_configured_contributions() -> None:
    """Every weighted score contribution is exposed and sums to the final score."""
    target = "Responding to humiliation with purpose"
    weights = RetrievalWeights()
    retriever = RoleModelRetriever(
        directory=CARD_DIR,
        embedder=DeterministicTestEmbedder(
            target_titles=(target,),
            target_vector=(0.8, 0.6),
        ),
        weights=weights,
    )

    result = retriever.retrieve(
        "I feel furious and humiliated after an unfair event and want to respond."
    )

    assert result.match is not None
    breakdown = result.match.breakdown
    assert breakdown.semantic_similarity == pytest.approx(0.8)
    assert breakdown.theme_overlap > 0
    assert breakdown.context_compatibility > 0
    assert breakdown.semantic_contribution == pytest.approx(
        breakdown.semantic_similarity * weights.semantic_similarity
    )
    assert breakdown.theme_contribution == pytest.approx(
        breakdown.theme_overlap * weights.theme_overlap
    )
    assert breakdown.context_contribution == pytest.approx(
        breakdown.context_compatibility * weights.context_compatibility
    )
    assert result.match.score == pytest.approx(
        breakdown.semantic_contribution
        + breakdown.theme_contribution
        + breakdown.context_contribution
    )


def test_recent_user_messages_and_state_are_included_without_assistant_history() -> None:
    """Only supplied user text and explicit state fields form the embedding query."""
    retriever, embedder = _retriever()
    current = "I am unsure what to do next."
    recent = "The deadline has made me feel pressured."
    state = EmotionalState(
        hypotheses=[EmotionHypothesis(label="anxiety_fear", probability=0.8)],
        intensity=0.7,
        trigger="work deadline",
        functional_impact="difficulty concentrating",
        need=UserNeed.understand,
        confidence=0.8,
    )

    retriever.retrieve(
        current,
        recent_user_messages=(recent,),
        conversation_state=state,
    )

    embedded_query = embedder.calls[0][0]
    assert current in embedded_query
    assert recent in embedded_query
    assert "work deadline" in embedded_query
    assert "anxiety_fear" in embedded_query
    assert "previous assistant analogy" not in embedded_query


def test_common_tags_receive_less_weight_than_rare_tags() -> None:
    """Inverse document frequency discounts tags found on many cards."""
    retriever, _ = _retriever()
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    frequencies = Counter(tag for card in library.all() for tag in set(card.tags))
    common = max(frequencies, key=frequencies.get)
    rare = min(frequencies, key=frequencies.get)

    assert frequencies[common] > frequencies[rare]
    assert retriever._tag_weights[common] < retriever._tag_weights[rare]


def test_results_are_deterministic_and_card_yaml_is_unchanged() -> None:
    """Repeated retrieval is stable and never modifies source YAML or card fields."""
    paths = sorted(CARD_DIR.glob("*.yaml"))
    before_bytes = {path.name: path.read_bytes() for path in paths}
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    before_cards = {card.id: card.model_dump(mode="json") for card in library.all()}
    target = "Responding to humiliation with purpose"
    retriever = RoleModelRetriever(
        directory=CARD_DIR,
        embedder=DeterministicTestEmbedder(target_titles=(target,)),
    )

    first = retriever.retrieve("I was mocked and want a purposeful response.")
    second = retriever.retrieve("I was mocked and want a purposeful response.")

    assert first == second
    assert first.match is not None
    assert first.match.card.model_dump(mode="json") == before_cards[first.match.card.id]
    assert {path.name: path.read_bytes() for path in paths} == before_bytes


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("optional embedding package is missing"),
        OSError("local model weights are unavailable"),
    ],
)
def test_missing_embedder_dependency_or_weights_abstains_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    """Unavailable optional embedding resources return a diagnosable no-match result."""

    def unavailable(
        _self: LocalEmbedder, _texts: Sequence[str]
    ) -> list[list[float]]:
        assert isinstance(_self, LocalEmbedder)
        assert len(_texts) > 1
        raise failure

    monkeypatch.setattr(LocalEmbedder, "embed", unavailable)
    retriever = RoleModelRetriever(
        directory=CARD_DIR,
        embedder=LocalEmbedder(),
    )

    result = retriever.retrieve("A synthetic concern for an offline test.")

    assert result.match is None
    assert result.ranked_matches == ()
    assert result.abstention_reason == "embedder_unavailable"
    assert result.embedder_error_type == type(failure).__name__
