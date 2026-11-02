"""Tests for Layer 2b: ChromaDB store, embedder, retriever, and passage sanitizer.

All tests use FakeEmbedder and tmp_path-based Chroma directories.
No real model downloads, no real API calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.contracts import RetrievedItem
from app.pipeline.retrieval import CardLibrary
from app.rag.embedder import Embedder, FakeEmbedder
from app.rag.sanitizer import PageText, build_deny_set, chunk_text, sanitize_page
from app.rag.store import COLLECTION_CARDS, COLLECTION_PASSAGES, get_client, get_collection

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    """Return a deterministic FakeEmbedder for tests."""
    return FakeEmbedder()


@pytest.fixture()
def card_library() -> CardLibrary:
    """Load the real reviewed card library from data/kb/cards/."""
    return CardLibrary.load(reviewed_only=True)


@pytest.fixture()
def chroma_client(tmp_path: Path):
    """Open a fresh ChromaDB PersistentClient in a temp directory."""
    client = get_client(tmp_path / "chroma")
    return client


@pytest.fixture()
def cards_collection(chroma_client, fake_embedder: FakeEmbedder):
    """Return a fresh cards collection backed by FakeEmbedder."""
    return get_collection(chroma_client, COLLECTION_CARDS, fake_embedder)


@pytest.fixture()
def passages_collection(chroma_client, fake_embedder: FakeEmbedder):
    """Return a fresh passages collection backed by FakeEmbedder."""
    return get_collection(chroma_client, COLLECTION_PASSAGES, fake_embedder)


def _make_retriever(tmp_path: Path, fake_embedder: FakeEmbedder, library: CardLibrary):
    """Build a Retriever wired to a tmp_path chroma DB with FakeEmbedder."""
    from app.rag.retriever import Retriever

    return Retriever(chroma_path=tmp_path / "chroma", embedder=fake_embedder, library=library)


def _ingest_cards(collection, library: CardLibrary, fake_embedder: FakeEmbedder) -> int:
    """Helper: upsert all cards from library into collection. Returns count."""
    ids, documents, metadatas = [], [], []
    for card in library.all():
        embed_text = (
            card.title
            + " "
            + card.summary
            + " "
            + " ".join(card.when_to_use)
            + " "
            + " ".join(card.emotions)
        )
        ids.append(card.id)
        documents.append(embed_text)
        metadatas.append(
            {
                "card_id": card.id,
                "category": card.category,
                "emotions": "|".join(card.emotions),
                "role_affinity": "|".join(card.role_affinity),
                "source_ids": "|".join(s.source_id for s in card.sources),
                "reviewed": 1 if card.reviewed else 0,
            }
        )
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


# ── Test 1: FakeEmbedder ranking ──────────────────────────────────────────────


def test_fake_embedder_similar_texts_rank_higher(fake_embedder: FakeEmbedder) -> None:
    """Similar texts should produce higher cosine similarity than dissimilar texts."""
    anchor = "breathing exercise to calm anxiety"
    similar = "deep breathing technique for stress relief"
    dissimilar = "recipe for chocolate cake with butter"

    vecs = fake_embedder.embed([anchor, similar, dissimilar])
    a, s, d = vecs

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v, strict=False))

    sim_similar = dot(a, s)
    sim_dissimilar = dot(a, d)

    assert sim_similar > sim_dissimilar

    norms = [sum(x * x for x in v) ** 0.5 for v in vecs]
    for norm in norms:
        assert abs(norm - 1.0) < 1e-5, "FakeEmbedder vectors should be unit-length"


# ── Test 2: Filter fallback ───────────────────────────────────────────────────


def test_filter_fallback_returns_results(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """When emotion/role filter matches nothing, retriever falls back to unfiltered."""
    retriever = _make_retriever(tmp_path, fake_embedder, card_library)

    # Ingest cards into the retriever's collection
    _ingest_cards(retriever._cards_col, card_library, fake_embedder)

    # Use a non-existent emotion label to force filter to return nothing
    # then fall back to unfiltered (which should return results if cards exist)
    results = retriever.search(
        "feeling sad and low energy",
        emotions=["nonexistent_emotion_label_xyz"],
        k=3,
        min_similarity=0.0,  # accept everything so fallback is visible
    )
    # With fallback, should return results from unfiltered search
    assert isinstance(results, list)
    assert len(results) > 0, "Fallback should return unfiltered results when filter matches nothing"
    # All results are valid RetrievedItem instances
    for item in results:
        assert isinstance(item, RetrievedItem)
        assert 0.0 <= item.score <= 1.0


def test_filter_fallback_runs_when_filtered_results_miss_threshold(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """A filtered result below threshold triggers an unfiltered retry."""
    retriever = _make_retriever(tmp_path, fake_embedder, card_library)
    _ingest_cards(retriever._cards_col, card_library, fake_embedder)

    target = card_library.all()[0]
    unmatched_emotion = next(
        emotion
        for candidate in card_library.all()
        if candidate.id != target.id
        for emotion in candidate.emotions
        if emotion not in target.emotions
    )
    query = " ".join(
        [target.title, target.summary, *target.when_to_use, *target.emotions]
    )

    results = retriever.search(
        query,
        emotions=[unmatched_emotion],
        k=3,
        min_similarity=0.999,
    )

    assert results
    assert results[0].doc_id == target.id


def test_emotion_and_role_filters_are_both_applied(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """Emotion alternatives and role constraints use supported metadata filters."""
    retriever = _make_retriever(tmp_path, fake_embedder, card_library)

    where = retriever._build_where(
        ["anxiety_fear", "sadness"], "teacher", "cards"
    )
    expected_emotions = sorted(
        {
            "|".join(card.emotions)
            for card in card_library.all()
            if {"anxiety_fear", "sadness"}.intersection(card.emotions)
        }
    )
    expected_roles = sorted(
        {
            "|".join(card.role_affinity)
            for card in card_library.all()
            if "teacher" in card.role_affinity
        }
    )

    assert where == {
        "$and": [
            {"emotions": {"$in": expected_emotions}},
            {"role_affinity": {"$in": expected_roles}},
        ]
    }


def test_retriever_surfaces_collection_errors(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """Collection query failures remain visible instead of appearing as no-match."""

    class BrokenCollection:
        def query(self, **kwargs: object) -> dict[str, list[list[object]]]:
            raise RuntimeError("synthetic collection failure")

    retriever = _make_retriever(tmp_path, fake_embedder, card_library)
    retriever._cards_col = BrokenCollection()

    with pytest.raises(RuntimeError, match="synthetic collection failure"):
        retriever.search("stress")


# ── Test 3: Threshold returns empty ──────────────────────────────────────────


def test_threshold_returns_empty_below_min_similarity(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """Results below min_similarity threshold are filtered out → empty list."""
    retriever = _make_retriever(tmp_path, fake_embedder, card_library)
    _ingest_cards(retriever._cards_col, card_library, fake_embedder)

    # min_similarity=1.0 is impossible to reach (no exact duplicates of query)
    results = retriever.search(
        "I feel anxious",
        k=3,
        min_similarity=1.0,
    )
    assert results == [], "Setting min_similarity=1.0 should always return empty"


# ── Test 4: Idempotent rebuild ────────────────────────────────────────────────


def test_idempotent_rebuild(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """Upserting the same cards twice yields the same count."""
    chroma_path = tmp_path / "chroma"
    client = get_client(chroma_path)
    col = get_collection(client, COLLECTION_CARDS, fake_embedder)

    count1 = _ingest_cards(col, card_library, fake_embedder)
    count2 = _ingest_cards(col, card_library, fake_embedder)

    # Counts should be equal (upsert is idempotent)
    assert count1 == count2, "Upsert count should be stable across repeated ingestions"

    # Collection count should equal number of cards (no duplicates)
    actual_count = col.count()
    assert actual_count == count1, f"Expected {count1} docs in collection, got {actual_count}"


# ── Test 5: Passage sanitisation ─────────────────────────────────────────────


def test_passage_sanitisation_no_banned_content() -> None:
    """Sanitised passages must not contain phone numbers, URLs, or medication references."""
    page_with_phone = PageText(
        page_number=1,
        text=(
            "This is a long paragraph about coping strategies and mental health support. "
            "It contains useful information about stress management techniques. "
            "Please call our helpline at +1-800-123-4567 for immediate help. "
            "Remember to practice mindfulness every day and reach out to trusted people. "
            "Many resources are available to support your wellbeing and mental health journey. "
            "Community support and social connection are key protective factors for wellbeing."
        ),
    )
    clean, reason = sanitize_page(page_with_phone)
    if clean is not None:
        # The phone number line should be stripped
        assert "+1-800-123-4567" not in clean.text
    # Page may still be long enough; that is fine

    page_too_short = PageText(page_number=2, text="Too short.")
    clean2, reason2 = sanitize_page(page_too_short)
    assert clean2 is None
    assert reason2 is not None


def test_passage_sanitisation_deny_list() -> None:
    """Pages in the deny set for a source must be excluded before chunking."""
    deny_sections = [
        {"name": "self-harm module", "pages": "139-148"},
        {"name": "medication", "pages": "21-22"},
    ]
    denied = build_deny_set(deny_sections)
    assert 139 in denied
    assert 148 in denied
    assert 21 in denied
    assert 22 in denied
    assert 23 not in denied
    assert 138 not in denied


def test_chunk_text_overlap() -> None:
    """chunk_text should produce overlapping chunks of approximately the right size."""
    words = ["word"] * 500
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=200, overlap_words=40)

    assert len(chunks) >= 3, "500 words should yield at least 3 chunks with 200-word size"
    # Each chunk (except possibly the last) should have ~200 words
    for chunk in chunks[:-1]:
        n = len(chunk.split())
        assert 190 <= n <= 210, f"Chunk size {n} not near 200"


# ── Test 6: Reviewed-only ingestion ──────────────────────────────────────────


def test_reviewed_only_ingestion(
    tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    """Unreviewed cards must not appear in the store."""
    # Load with reviewed_only=True (default) and reviewed_only=False
    library_reviewed = CardLibrary.load(reviewed_only=True)
    library_all = CardLibrary.load(reviewed_only=False)

    # If all cards happen to be reviewed, we can still check counts match
    reviewed_count = len(library_reviewed)
    all_count = len(library_all)

    chroma_path = tmp_path / "chroma"
    client = get_client(chroma_path)

    col_reviewed = get_collection(client, "cards_reviewed", fake_embedder)
    col_all = get_collection(client, "cards_all", fake_embedder)

    n_reviewed = _ingest_cards(col_reviewed, library_reviewed, fake_embedder)
    n_all = _ingest_cards(col_all, library_all, fake_embedder)

    assert n_reviewed == reviewed_count
    assert n_all == all_count
    assert n_reviewed <= n_all, "Reviewed-only set must be a subset of all cards"

    # Collection counts should match
    assert col_reviewed.count() == reviewed_count
    assert col_all.count() == all_count


def test_kb_builder_selects_unreviewed_cards_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, card_library: CardLibrary
) -> None:
    """The build option maps to CardLibrary's reviewed-only boundary."""
    from scripts import kb_build

    reviewed_only_values: list[bool] = []

    def fake_load(cls, *, reviewed_only: bool) -> CardLibrary:
        reviewed_only_values.append(reviewed_only)
        return card_library

    monkeypatch.setattr(CardLibrary, "load", classmethod(fake_load))

    assert kb_build._load_card_library() is card_library
    assert kb_build._load_card_library(include_unreviewed=True) is card_library
    assert reviewed_only_values == [True, False]


# ── Test 7: Retriever satisfies RetrieverLike protocol ───────────────────────


def test_retriever_satisfies_retriever_like_protocol(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """Retriever must satisfy the RetrieverLike structural protocol."""
    from app.rag.retriever import Retriever

    retriever = Retriever(chroma_path=tmp_path / "chroma", embedder=fake_embedder, library=card_library)

    # RetrieverLike is a non-runtime-checkable Protocol, so we verify duck-typing
    # by confirming the method exists and is callable with all supported signatures.
    assert callable(getattr(retriever, "search_with_evidence", None)), (
        "Retriever must have a callable search_with_evidence method (RetrieverLike protocol)"
    )

    # Call the method with all supported signatures
    result = retriever.search_with_evidence("I feel stressed", k=3)
    assert isinstance(result, list)

    result2 = retriever.search_with_evidence("anxious and worried", emotions=["fear"], k=2)
    assert isinstance(result2, list)

    result3 = retriever.search_with_evidence("need help", role="counsellor", k=1)
    assert isinstance(result3, list)


# ── Test 8: Embedder satisfies protocol ──────────────────────────────────────


def test_fake_embedder_satisfies_embedder_protocol() -> None:
    """FakeEmbedder must satisfy the Embedder protocol."""
    fe = FakeEmbedder()
    assert isinstance(fe, Embedder), "FakeEmbedder must satisfy the Embedder Protocol"

    result = fe.embed(["hello world", "goodbye"])
    assert len(result) == 2
    assert all(len(v) == FakeEmbedder.DIM for v in result)


# ── Test 9: store get_collection returns correct collection ───────────────────


def test_get_collection_creates_cosine_collection(
    chroma_client, fake_embedder: FakeEmbedder
) -> None:
    """get_collection should create a collection with cosine distance metadata."""
    col = get_collection(chroma_client, "test_col", fake_embedder)
    assert col is not None
    # ChromaDB metadata should include cosine space
    meta = col.metadata or {}
    assert meta.get("hnsw:space") == "cosine"


# ── Test 10: Sanitizer rejects short pages ────────────────────────────────────


def test_sanitizer_rejects_page_shorter_than_minimum() -> None:
    """sanitize_page must reject pages under 200 characters after cleaning."""
    short_page = PageText(page_number=5, text="This is a short page.")
    clean, reason = sanitize_page(short_page)
    assert clean is None
    assert "too short" in (reason or "")


# ── Test 11: build_deny_set parses ranges correctly ──────────────────────────


def test_build_deny_set_complex_spec() -> None:
    """build_deny_set must handle comma-separated ranges and single pages."""
    deny_sections = [
        {"name": "section a", "pages": "139-148"},
        {"name": "section b", "pages": "21-22,36-40"},
        {"name": "section c", "pages": "100"},
    ]
    denied = build_deny_set(deny_sections)
    expected = set(range(139, 149)) | set(range(21, 23)) | set(range(36, 41)) | {100}
    assert denied == frozenset(expected)


# ── Test 12: RetrievedItem score is in [0, 1] ────────────────────────────────


def test_retrieved_items_scores_in_range(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """All returned RetrievedItem scores must be in [0.0, 1.0]."""
    from app.rag.retriever import Retriever

    retriever = Retriever(chroma_path=tmp_path / "chroma", embedder=fake_embedder, library=card_library)
    _ingest_cards(retriever._cards_col, card_library, fake_embedder)

    results = retriever.search("stress anxiety sadness", k=5, min_similarity=0.0)
    for item in results:
        assert 0.0 <= item.score <= 1.0, f"Score {item.score} out of range for {item.doc_id}"


# ── Test 13: Cards collection doc_id equals card.id ──────────────────────────


def test_cards_collection_doc_id_equals_card_id(
    tmp_path: Path, fake_embedder: FakeEmbedder, card_library: CardLibrary
) -> None:
    """doc_id in RetrievedItem for cards collection must equal the card's id."""
    from app.rag.retriever import Retriever

    retriever = Retriever(chroma_path=tmp_path / "chroma", embedder=fake_embedder, library=card_library)
    _ingest_cards(retriever._cards_col, card_library, fake_embedder)

    results = retriever.search("low mood sadness routine", k=3, min_similarity=0.0)
    valid_card_ids = {card.id for card in card_library.all()}

    for item in results:
        assert item.doc_id in valid_card_ids, (
            f"doc_id '{item.doc_id}' is not a valid card id"
        )
        assert item.collection == "cards"
