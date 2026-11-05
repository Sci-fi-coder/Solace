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
    RetrievalWeights,
    RoleModelMatch,
    RoleModelRetrievalResult,
    RoleModelRetriever,
)

_CARD_DIR = _REPO / "data" / "role_models"
_EVAL_PATH = _REPO / "tests" / "data" / "rolemodels_retrieval_eval.jsonl"
_WEIGHT_CANDIDATES = (
    RetrievalWeights(0.55, 0.25, 0.20),
    RetrievalWeights(0.70, 0.20, 0.10),
    RetrievalWeights(0.85, 0.10, 0.05),
    RetrievalWeights(),
)
_THRESHOLD_CANDIDATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
_MARGIN_CANDIDATES: tuple[float | None, ...] = (None, 0.03, 0.05, 0.08)


@dataclass(frozen=True)
class EvaluationCase:
    """A synthetic query with a target card or an expected abstention."""

    case_id: str
    query: str
    target_card_id: str | None
    expected: Literal["match", "abstain"]
    category: str


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
        "coverage_by_target_review_status": reviewed_targets,
        "returned_positive_matches_by_review_status": returned_status,
        "by_category": categories,
        "per_card": per_card,
    }


def _configuration_rows(
    cases: tuple[EvaluationCase, ...],
    results: dict[str, RoleModelRetrievalResult],
    library: RoleModelLibrary,
) -> list[tuple[Calibration, dict[str, object]]]:
    """Score a small predeclared grid without repeating any embedding calls."""
    rows = []
    for weights in _WEIGHT_CANDIDATES:
        for threshold in _THRESHOLD_CANDIDATES:
            for margin in _MARGIN_CANDIDATES:
                calibration = Calibration(weights, threshold, margin)
                rows.append(
                    (calibration, evaluate(cases, results, library, calibration))
                )
    return sorted(
        rows,
        key=lambda row: (
            row[1]["false_positive_rate"],
            -row[1]["precision_at_1"],
            -row[1]["returned_match_precision_at_1"],
            -row[1]["recall_at_3"],
            -row[1]["no_match_accuracy"],
            row[0].top_match_margin is not None,
            row[0].minimum_relevance,
        ),
    )


def main() -> None:
    """Run offline real-embedding evaluation and print actual metrics."""
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
        results[case.case_id] = retriever.retrieve(case.query)
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

    candidates = _configuration_rows(cases, results, library)
    initial = Calibration(RetrievalWeights(0.55, 0.25, 0.20), 0.50, None)
    chosen = candidates[0]
    output = {
        "embedding_model": LocalEmbedder.DEFAULT_MODEL,
        "embedding_source": "local cached weights; Hugging Face offline mode enabled",
        "cases": len(cases),
        "cards": len(card_ids),
        "reviewed_cards": sum(card.reviewed for card in library.all()),
        "unreviewed_cards": sum(not card.reviewed for card in library.all()),
        "initial_configuration": {
            "weights": {
                "semantic": initial.weights.semantic_similarity,
                "theme": initial.weights.theme_overlap,
                "context": initial.weights.context_compatibility,
            },
            "minimum_relevance": initial.minimum_relevance,
            "top_match_margin": initial.top_match_margin,
            "metrics": evaluate(cases, results, library, initial),
        },
        "calibrated_candidate": {
            "weights": {
                "semantic": chosen[0].weights.semantic_similarity,
                "theme": chosen[0].weights.theme_overlap,
                "context": chosen[0].weights.context_compatibility,
            },
            "minimum_relevance": chosen[0].minimum_relevance,
            "top_match_margin": chosen[0].top_match_margin,
            "metrics": chosen[1],
        },
        "top_calibration_candidates": [
            {
                "weights": (
                    row[0].weights.semantic_similarity,
                    row[0].weights.theme_overlap,
                    row[0].weights.context_compatibility,
                ),
                "minimum_relevance": row[0].minimum_relevance,
                "top_match_margin": row[0].top_match_margin,
                "precision_at_1": row[1]["precision_at_1"],
                "recall_at_3": row[1]["recall_at_3"],
                "positive_match_coverage": row[1]["positive_match_coverage"],
                "returned_match_precision_at_1": row[1][
                    "returned_match_precision_at_1"
                ],
                "no_match_accuracy": row[1]["no_match_accuracy"],
                "false_positive_rate": row[1]["false_positive_rate"],
            }
            for row in candidates[:10]
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
