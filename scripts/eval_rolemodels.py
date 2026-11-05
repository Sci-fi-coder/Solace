"""Evaluate historical role-model retrieval against a small synthetic dataset.

Run with ``python -m scripts.eval_rolemodels``. Embeddings are restricted to local
files; the command never downloads model weights or calls an external service.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.rag.embedder import LocalEmbedder  # noqa: E402
from app.rolemodels.library import RoleModelLibrary  # noqa: E402
from app.rolemodels.retriever import (  # noqa: E402
    DEFAULT_MINIMUM_RELEVANCE,
    RetrievalWeights,
    RoleModelMatch,
    RoleModelRetrievalResult,
    RoleModelRetriever,
)

_CARD_DIR = _REPO / "data" / "role_models"
_EVAL_PATH = _REPO / "tests" / "data" / "rolemodels_retrieval_eval.jsonl"
_LEGACY_CATEGORIES = frozenset(
    {
        "paraphrase",
        "explicit",
        "contrastive",
        "unrelated",
        "ambiguous",
        "explicit_unrelated",
    }
)


@dataclass(frozen=True)
class EvaluationCase:
    """A synthetic query with a target card or an expected abstention."""

    case_id: str
    query: str
    target_card_id: str | None
    expected: Literal["match", "abstain"]
    category: str
    recent_user_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Calibration:
    """A candidate hybrid scoring and abstention configuration."""

    weights: RetrievalWeights
    minimum_relevance: float
    top_match_margin: float | None


def load_cases(path: Path = _EVAL_PATH) -> tuple[EvaluationCase, ...]:
    """Load and validate the deterministic synthetic evaluation cases."""
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        expected = value["expected"]
        if expected not in ("match", "abstain"):
            raise ValueError(f"Invalid expectation on evaluation line {line_number}")
        target = value.get("target_card_id")
        if (expected == "match") != isinstance(target, str):
            raise ValueError(f"Invalid target card on evaluation line {line_number}")
        cases.append(
            EvaluationCase(
                case_id=value["id"],
                query=value["query"],
                target_card_id=target,
                expected=expected,
                category=value["category"],
                recent_user_messages=tuple(value.get("recent_user_messages", ())),
            )
        )
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return tuple(cases)


def _weighted_score(match: RoleModelMatch, weights: RetrievalWeights) -> float:
    """Recompute a candidate score from the embedder's raw, cached signals."""
    breakdown = match.breakdown
    return (
        breakdown.semantic_similarity * weights.semantic_similarity
        + breakdown.theme_overlap * weights.theme_overlap
        + breakdown.context_compatibility * weights.context_compatibility
    )


def _rank(
    result: RoleModelRetrievalResult,
    weights: RetrievalWeights,
) -> list[tuple[float, RoleModelMatch]]:
    """Rank raw retrieval signals under one candidate weight configuration."""
    ranked = [
        (_weighted_score(match, weights), match)
        for match in result.ranked_matches
    ]
    return sorted(
        ranked,
        key=lambda item: (
            -item[0],
            -item[1].breakdown.semantic_similarity,
            item[1].card.id,
        ),
    )


def _selected_match(
    ranked: list[tuple[float, RoleModelMatch]],
    calibration: Calibration,
) -> RoleModelMatch | None:
    """Apply the minimum-score and optional top-match-margin abstention rules."""
    if not ranked or ranked[0][0] < calibration.minimum_relevance:
        return None
    if (
        calibration.top_match_margin is not None
        and len(ranked) > 1
        and ranked[1][0] >= calibration.minimum_relevance
        and ranked[0][0] - ranked[1][0] < calibration.top_match_margin
    ):
        return None
    return ranked[0][1]


def evaluate(
    cases: tuple[EvaluationCase, ...],
    results: dict[str, RoleModelRetrievalResult],
    library: RoleModelLibrary,
    calibration: Calibration,
) -> dict[str, object]:
    """Measure top-1 hits, ranked top-3 recall, abstention, and card-status coverage.

    Top-1 precision counts an abstention on a relevant query as a miss. Returned-match
    precision is conditional on returning a card; recall@3 measures ranked candidates
    before the score threshold.
    """
    positive_cases = [case for case in cases if case.expected == "match"]
    negative_cases = [case for case in cases if case.expected == "abstain"]
    top1_hits = 0
    recall3_hits = 0
    false_positives = 0
    returned_positive_matches = 0
    correct_returned_positive_matches = 0
    reviewed_targets: dict[str, dict[str, int]] = {
        "reviewed": {"targets": 0, "top1_hits": 0, "recall_at_3_hits": 0},
        "unreviewed": {"targets": 0, "top1_hits": 0, "recall_at_3_hits": 0},
    }
    categories: dict[str, dict[str, int]] = {}
    per_card: dict[str, dict[str, int]] = {
        card.id: {"queries": 0, "top1_hits": 0, "recall_at_3_hits": 0}
        for card in library.all()
    }
    per_person: dict[str, dict[str, int]] = {
        person.value: {"queries": 0, "top1_hits": 0, "recall_at_3_hits": 0}
        for person in sorted({card.person for card in library.all()}, key=lambda item: item.value)
    }
    returned_status = {"reviewed": 0, "unreviewed": 0}

    for case in positive_cases:
        result = results[case.case_id]
        ranked = _rank(result, calibration.weights)
        target_id = case.target_card_id
        assert target_id is not None
        target = library.get(target_id)
        assert target is not None
        status = "reviewed" if target.reviewed else "unreviewed"
        recall3_hit = any(item.card.id == target_id for _, item in ranked[:3])
        selected = _selected_match(ranked, calibration)
        is_correct_return = selected is not None and selected.card.id == target_id
        top1_hits += is_correct_return
        recall3_hits += recall3_hit
        if selected is not None:
            returned_positive_matches += 1
            correct_returned_positive_matches += is_correct_return
        reviewed_targets[status]["targets"] += 1
        reviewed_targets[status]["top1_hits"] += is_correct_return
        reviewed_targets[status]["recall_at_3_hits"] += recall3_hit
        per_card[target_id]["queries"] += 1
        per_card[target_id]["top1_hits"] += selected is not None and selected.card.id == target_id
        per_card[target_id]["recall_at_3_hits"] += recall3_hit
        per_person[target.person.value]["queries"] += 1
        per_person[target.person.value]["top1_hits"] += (
            selected is not None and selected.card.id == target_id
        )
        per_person[target.person.value]["recall_at_3_hits"] += recall3_hit
        if selected is not None:
            returned_status["reviewed" if selected.card.reviewed else "unreviewed"] += 1

    for case in negative_cases:
        if _selected_match(_rank(results[case.case_id], calibration.weights), calibration):
            false_positives += 1

    for case in cases:
        category = categories.setdefault(
            case.category,
            {
                "cases": 0,
                "expected_matches": 0,
                "correct_top1_matches": 0,
                "abstentions": 0,
                "false_positives": 0,
            },
        )
        selected = _selected_match(
            _rank(results[case.case_id], calibration.weights), calibration
        )
        category["cases"] += 1
        category["expected_matches"] += case.expected == "match"
        category["correct_top1_matches"] += (
            case.expected == "match"
            and selected is not None
            and selected.card.id == case.target_card_id
        )
        category["abstentions"] += selected is None
        category["false_positives"] += (
            case.expected == "abstain" and selected is not None
        )

    positive_count = len(positive_cases)
    negative_count = len(negative_cases)
    return {
        "positive_cases": positive_count,
        "negative_cases": negative_count,
        "precision_at_1": top1_hits / positive_count if positive_count else 0.0,
        "returned_match_precision_at_1": (
            correct_returned_positive_matches / returned_positive_matches
            if returned_positive_matches
            else 0.0
        ),
        "positive_match_coverage": (
            returned_positive_matches / positive_count if positive_count else 0.0
        ),
        "recall_at_3": recall3_hits / positive_count if positive_count else 0.0,
        "no_match_accuracy": (
            (negative_count - false_positives) / negative_count if negative_count else 0.0
        ),
        "false_positive_rate": (
            false_positives / negative_count if negative_count else 0.0
        ),
        "false_positives": false_positives,
        "coverage_by_target_review_status": {
            status: {
                **coverage,
                "top1_coverage": (
                    coverage["top1_hits"] / coverage["targets"]
                    if coverage["targets"]
                    else 0.0
                ),
                "recall_at_3_coverage": (
                    coverage["recall_at_3_hits"] / coverage["targets"]
                    if coverage["targets"]
                    else 0.0
                ),
            }
            for status, coverage in reviewed_targets.items()
        },
        "returned_positive_matches_by_review_status": returned_status,
        "by_category": categories,
        "per_person": per_person,
        "per_card": per_card,
    }


def main() -> None:
    """Run offline real-embedding evaluation at the unchanged production calibration."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cases = load_cases()
    library = RoleModelLibrary.load(_CARD_DIR, include_unreviewed=True)
    target_ids = {case.target_card_id for case in cases if case.target_card_id}
    card_ids = {card.id for card in library.all()}
    if len(card_ids) != 24 or target_ids != card_ids:
        raise ValueError(
            f"Evaluation must cover all 24 cards; loaded={len(card_ids)}, "
            f"covered={len(target_ids)}"
        )

    retriever = RoleModelRetriever(
        directory=_CARD_DIR,
        embedder=LocalEmbedder(),
        minimum_relevance=0.0,
    )
    results: dict[str, RoleModelRetrievalResult] = {}
    for case in cases:
        results[case.case_id] = retriever.retrieve(
            case.query,
            recent_user_messages=case.recent_user_messages,
        )
    unavailable = {
        result.embedder_error_type
        for result in results.values()
        if result.abstention_reason == "embedder_unavailable"
    }
    if unavailable:
        print(
            "Real embedding evaluation unavailable in offline mode: "
            + ", ".join(sorted(error for error in unavailable if error))
        )
        return

    calibration = Calibration(RetrievalWeights(), DEFAULT_MINIMUM_RELEVANCE, None)
    legacy_cases = tuple(case for case in cases if case.category in _LEGACY_CATEGORIES)
    output = {
        "embedding_model": LocalEmbedder.DEFAULT_MODEL,
        "embedding_source": "local cached weights; Hugging Face offline mode enabled",
        "cases": len(cases),
        "cards": len(card_ids),
        "reviewed_cards": sum(card.reviewed for card in library.all()),
        "unreviewed_cards": sum(not card.reviewed for card in library.all()),
        "configuration": {
            "weights": {
                "semantic": calibration.weights.semantic_similarity,
                "theme": calibration.weights.theme_overlap,
                "context": calibration.weights.context_compatibility,
            },
            "minimum_relevance": calibration.minimum_relevance,
            "top_match_margin": calibration.top_match_margin,
        },
        "calibration_note": (
            "Hybrid score weights and the relevance threshold were held fixed. The retrieval "
            "representation was developed using this synthetic dataset, so these are "
            "development results rather than an independent holdout."
        ),
        "legacy_subset_case_count": len(legacy_cases),
        "legacy_52_case_subset": evaluate(
            legacy_cases,
            results,
            library,
            calibration,
        ),
        "expanded_dataset": evaluate(cases, results, library, calibration),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
