"""Evaluate both retrievers on synthetic retrieval_eval.jsonl test cases.

Usage
-----
    python -m scripts.eval_retrieval

Reports:
- hit@1, hit@3, MRR for each retriever
- empty-result precision and recall
- Chroma threshold sweep (0.15 to 0.50, step 0.05)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.pipeline.retrieval import CardLibrary, KeywordCardRetriever  # noqa: E402
from app.rag.retriever import Retriever  # noqa: E402

_EVAL_PATH = _REPO / "tests" / "data" / "retrieval_eval.jsonl"


@dataclass
class EvalCase:
    """A single evaluation case from the JSONL file."""

    query: str
    relevant_ids: list[str]
    expect_empty: bool


@dataclass
class Metrics:
    """Aggregated retrieval metrics."""

    hit1: float = 0.0
    hit3: float = 0.0
    mrr: float = 0.0
    empty_precision: float = 0.0
    empty_recall: float = 0.0
    n_retrieval: int = 0
    n_empty: int = 0
    extra: dict = field(default_factory=dict)


def _load_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in _EVAL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cases.append(
            EvalCase(
                query=obj["query"],
                relevant_ids=obj.get("relevant_ids", []),
                expect_empty=obj.get("expect_empty", False),
            )
        )
    return cases


def _evaluate(
    cases: list[EvalCase],
    retriever_fn,
    min_sim: float = 0.30,
) -> Metrics:
    """Run evaluation for a retriever function.

    Args:
        cases: List of EvalCase instances.
        retriever_fn: A callable(query, k=3) → list[RetrievedItem].
        min_sim: Not used here directly; retriever_fn already encodes it.
    """
    retrieval_cases = [c for c in cases if not c.expect_empty]
    empty_cases = [c for c in cases if c.expect_empty]

    # Retrieval metrics
    hit1_total = 0
    hit3_total = 0
    mrr_total = 0.0

    for case in retrieval_cases:
        results = retriever_fn(case.query, k=3)
        doc_ids = [r.doc_id for r in results]
        relevant = set(case.relevant_ids)

        # hit@1
        if doc_ids and doc_ids[0] in relevant:
            hit1_total += 1

        # hit@3
        if any(d in relevant for d in doc_ids[:3]):
            hit3_total += 1

        # MRR
        for rank, d in enumerate(doc_ids[:3], 1):
            if d in relevant:
                mrr_total += 1.0 / rank
                break

    n_ret = len(retrieval_cases)
    hit1 = hit1_total / n_ret if n_ret else 0.0
    hit3 = hit3_total / n_ret if n_ret else 0.0
    mrr = mrr_total / n_ret if n_ret else 0.0

    # Empty precision/recall
    # Precision: of cases that returned empty, how many should be empty?
    # Recall: of cases that should be empty, how many returned empty?
    returned_empty = []
    should_empty = []
    for case in empty_cases:
        results = retriever_fn(case.query, k=3)
        returned_empty.append(len(results) == 0)
        should_empty.append(True)

    # Also check retrieval cases that returned empty (false negatives for retrieval)
    tp_empty = sum(1 for r in returned_empty if r)
    fp_empty = sum(1 for case in retrieval_cases if len(retriever_fn(case.query, k=3)) == 0)
    n_empty = len(empty_cases)

    prec = tp_empty / (tp_empty + fp_empty) if (tp_empty + fp_empty) > 0 else 1.0
    rec = tp_empty / n_empty if n_empty > 0 else 1.0

    return Metrics(
        hit1=hit1,
        hit3=hit3,
        mrr=mrr,
        empty_precision=prec,
        empty_recall=rec,
        n_retrieval=n_ret,
        n_empty=n_empty,
    )


def _chroma_threshold_sweep(cases: list[EvalCase], retriever: Retriever) -> list[dict]:
    """Sweep min_similarity from 0.15 to 0.50 and compute metrics for each."""
    thresholds = [round(t / 100, 2) for t in range(15, 55, 5)]
    rows: list[dict] = []

    retrieval_cases = [c for c in cases if not c.expect_empty]
    empty_cases = [c for c in cases if c.expect_empty]

    for thresh in thresholds:
        def make_fn(t):
            def fn(query: str, k: int = 3):
                return retriever.search(query, k=k, min_similarity=t)
            return fn  # noqa: B023
        fn = make_fn(thresh)
        n_ret = len(retrieval_cases)

        hit1 = hit3 = 0
        mrr = 0.0
        for case in retrieval_cases:
            results = fn(case.query)
            ids = [r.doc_id for r in results]
            relevant = set(case.relevant_ids)
            if ids and ids[0] in relevant:
                hit1 += 1
            if any(d in relevant for d in ids[:3]):
                hit3 += 1
            for rank, d in enumerate(ids[:3], 1):
                if d in relevant:
                    mrr += 1.0 / rank
                    break

        tp_empty = sum(1 for c in empty_cases if len(fn(c.query)) == 0)
        fp_empty = sum(1 for c in retrieval_cases if len(fn(c.query)) == 0)
        n_empty = len(empty_cases)

        prec = tp_empty / (tp_empty + fp_empty) if (tp_empty + fp_empty) > 0 else 1.0
        rec = tp_empty / n_empty if n_empty > 0 else 1.0

        rows.append(
            {
                "threshold": thresh,
                "hit@1": hit1 / n_ret if n_ret else 0.0,
                "hit@3": hit3 / n_ret if n_ret else 0.0,
                "mrr": mrr / n_ret if n_ret else 0.0,
                "empty_prec": prec,
                "empty_rec": rec,
            }
        )
    return rows


def _print_table(title: str, m: Metrics) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Retrieval cases : {m.n_retrieval}")
    print(f"  Empty cases     : {m.n_empty}")
    print(f"  hit@1           : {m.hit1:.3f}")
    print(f"  hit@3           : {m.hit3:.3f}")
    print(f"  MRR             : {m.mrr:.3f}")
    print(f"  empty precision : {m.empty_precision:.3f}")
    print(f"  empty recall    : {m.empty_recall:.3f}")


def main() -> None:
    """Run the evaluation and print results."""
    print("Loading evaluation cases …")
    cases = _load_cases()
    print(f"Loaded {len(cases)} cases ({sum(1 for c in cases if not c.expect_empty)} retrieval, "
          f"{sum(1 for c in cases if c.expect_empty)} empty-expected).")

    print("\nLoading card library …")
    library = CardLibrary.load(reviewed_only=True)

    # ── Keyword retriever ────────────────────────────────────────────────────
    print("Evaluating KeywordCardRetriever …")
    kw_retriever = KeywordCardRetriever(library)
    kw_m = _evaluate(cases, lambda q, k=3: kw_retriever.search_with_evidence(q, k=k))
    _print_table("KeywordCardRetriever", kw_m)

    # ── Chroma retriever ─────────────────────────────────────────────────────
    print("\nOpening Chroma retriever (this may download ~90 MB on first run) …")
    try:
        chroma_retriever = Retriever(library=library)
    except Exception as exc:
        print(f"ERROR: Could not open ChromaDB: {exc}")
        print("Run 'python -m scripts.kb_build' first.")
        sys.exit(1)

    chroma_m = _evaluate(cases, lambda q, k=3: chroma_retriever.search_with_evidence(q, k=k))
    _print_table("ChromaDB Retriever (min_sim=0.30)", chroma_m)

    # ── Threshold sweep ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ChromaDB Threshold Sweep")
    print(f"{'=' * 60}")
    print(f"  {'thresh':>7}  {'hit@1':>6}  {'hit@3':>6}  {'mrr':>6}  {'e_prec':>7}  {'e_rec':>6}")
    rows = _chroma_threshold_sweep(cases, chroma_retriever)
    for row in rows:
        print(
            f"  {row['threshold']:>7.2f}  {row['hit@1']:>6.3f}  {row['hit@3']:>6.3f}"
            f"  {row['mrr']:>6.3f}  {row['empty_prec']:>7.3f}  {row['empty_rec']:>6.3f}"
        )


if __name__ == "__main__":
    main()
