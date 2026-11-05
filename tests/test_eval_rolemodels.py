"""Tests for the historical role-model retrieval evaluation dataset and metrics."""

from __future__ import annotations

from pathlib import Path

from app.rolemodels.library import RoleModelLibrary
from app.rolemodels.models import Person
from app.rolemodels.retriever import (
    DEFAULT_MINIMUM_RELEVANCE,
    RetrievalWeights,
    RoleModelRetrievalResult,
    RoleModelRetriever,
)
from app.rolemodels.selector import requested_person
from scripts.eval_rolemodels import Calibration, EvaluationCase, evaluate, load_cases
from tests.test_rolemodels_retriever import DeterministicTestEmbedder

CARD_DIR = Path(__file__).parents[1] / "data" / "role_models"


def test_evaluation_dataset_covers_all_cards_and_case_types() -> None:
    """The stable synthetic dataset includes all cards and positive/negative case types."""
    cases = load_cases()
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    target_ids = {case.target_card_id for case in cases if case.target_card_id}
    categories = {case.category for case in cases}

    assert target_ids == {card.id for card in library.all()}
    assert len(target_ids) == 24
    assert {
        "paraphrase",
        "explicit",
        "contrastive",
        "implicit",
        "implicit_confidence",
        "multi_turn",
    } <= categories
    assert {"unrelated", "ambiguous", "explicit_unrelated"} <= categories
    assert all(case.query.strip() for case in cases)
    assert sum(case.expected == "abstain" for case in cases) >= 15
    implicit_cases = [
        case
        for case in cases
        if case.category in {"implicit", "implicit_confidence"}
    ]
    assert implicit_cases
    assert all(requested_person(case.query) is None for case in implicit_cases)
    assert all(
        "example" not in case.query.casefold()
        and "analogy" not in case.query.casefold()
        and "historical" not in case.query.casefold()
        for case in implicit_cases
    )
    multi_turn_cases = [case for case in cases if case.category == "multi_turn"]
    assert len(multi_turn_cases) >= 5
    assert all(case.recent_user_messages for case in multi_turn_cases)
    explicit_people = {
        requested_person(case.query) for case in cases if case.category == "explicit"
    }
    assert explicit_people == {
        Person.gandhi,
        Person.kalam,
        Person.marcus_aurelius,
        Person.vivekananda,
    }


def test_metrics_measure_match_recall_abstention_and_review_coverage() -> None:
    """Metric aggregation distinguishes correct targets from abstentions and wrong cards."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    target = library.get("gandhi-pietermaritzburg")
    assert target is not None
    retriever = RoleModelRetriever(
        library=library,
        embedder=DeterministicTestEmbedder(target_titles=(target.title,)),
        minimum_relevance=0.0,
    )
    positive = EvaluationCase(
        "positive", "I was publicly mocked and want a constructive response.",
        target.id, "match", "paraphrase",
    )
    negative = EvaluationCase(
        "negative", "What is the capital of Portugal?", None, "abstain", "unrelated"
    )
    positive_result = retriever.retrieve(positive.query)
    negative_result = RoleModelRetrievalResult(None, (), "below_threshold", None)
    calibration = Calibration(RetrievalWeights(), 0.50, None)

    metrics = evaluate(
        (positive, negative),
        {positive.case_id: positive_result, negative.case_id: negative_result},
        library,
        calibration,
    )

    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["no_match_accuracy"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    reviewed = metrics["coverage_by_target_review_status"]["unreviewed"]
    assert reviewed == {
        "targets": 1,
        "top1_hits": 1,
        "recall_at_3_hits": 1,
        "top1_coverage": 1.0,
        "recall_at_3_coverage": 1.0,
    }
    assert metrics["per_person"]["gandhi"] == {
        "queries": 1,
        "top1_hits": 1,
        "recall_at_3_hits": 1,
    }
    assert metrics["by_category"]["unrelated"] == {
        "cases": 1,
        "expected_matches": 0,
        "correct_top1_matches": 0,
        "abstentions": 1,
        "false_positives": 0,
    }


def test_default_calibration_is_explicit_and_conservative() -> None:
    """The measured calibration is exposed as plain defaults, without a margin."""
    retriever = RoleModelRetriever(embedder=DeterministicTestEmbedder())

    assert retriever._weights == RetrievalWeights(0.90, 0.05, 0.05)
    assert retriever._minimum_relevance == DEFAULT_MINIMUM_RELEVANCE == 0.42
    assert retriever._top_match_margin is None
