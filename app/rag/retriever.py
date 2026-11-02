"""ChromaDB-backed retriever implementing the RetrieverLike protocol."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from app.contracts import RetrievedItem

if TYPE_CHECKING:
    from app.pipeline.retrieval import CardLibrary
    from app.rag.embedder import Embedder


def _dist_to_sim(distance: float) -> float:
    """Convert ChromaDB cosine distance [0, 2] to similarity [0, 1].

    ChromaDB cosine distance: 0 = identical, 2 = maximally opposite.
    """
    sim = 1.0 - distance / 2.0
    return max(0.0, min(1.0, sim))


class Retriever:
    """ChromaDB-backed retriever. A zero-argument constructor works with defaults.

    Satisfies the :class:`~app.pipeline.retrieval.RetrieverLike` protocol.
    """

    def __init__(
        self,
        chroma_path: Path | None = None,
        embedder: Embedder | None = None,
        library: CardLibrary | None = None,
    ) -> None:
        """Initialise the retriever.

        Args:
            chroma_path: Path to the ChromaDB persistence directory.
                Defaults to ``data/chroma``.
            embedder: Embedder instance. Defaults to :class:`~app.rag.embedder.LocalEmbedder`.
            library: Pre-loaded :class:`~app.pipeline.retrieval.CardLibrary`.
                Loaded lazily from the default cards directory if not provided.
        """
        from app.rag.store import (  # noqa: PLC0415
            COLLECTION_CARDS,
            COLLECTION_PASSAGES,
            default_chroma_path,
            get_client,
            get_collection,
        )

        self._chroma_path = chroma_path or default_chroma_path()

        if embedder is None:
            from app.rag.embedder import LocalEmbedder  # noqa: PLC0415

            embedder = LocalEmbedder()
        self._embedder = embedder

        self._library = library  # lazy load deferred to property

        self._client = get_client(self._chroma_path)
        self._cards_col = get_collection(self._client, COLLECTION_CARDS, self._embedder)
        self._passages_col = get_collection(self._client, COLLECTION_PASSAGES, self._embedder)

    @property
    def _card_library(self) -> CardLibrary:
        if self._library is None:
            from app.pipeline.retrieval import CardLibrary  # noqa: PLC0415

            self._library = CardLibrary.load(reviewed_only=True)
        return self._library

    def search(
        self,
        query: str,
        *,
        emotions: Sequence[str] | None = None,
        role: str | None = None,
        k: int = 3,
        collection: str = "cards",
        min_similarity: float = 0.30,
    ) -> list[RetrievedItem]:
        """Search a single ChromaDB collection and return relevant items.

        Args:
            query: The user query string.
            emotions: Optional sequence of emotion labels for metadata filtering.
            role: Optional role name for metadata filtering.
            k: Maximum number of results to return.
            collection: Collection name (``"cards"`` or ``"passages"``).
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            A list of :class:`~app.contracts.RetrievedItem` instances, or an empty
            list when nothing reaches ``min_similarity`` (the "just listen" signal).
        """
        col = self._cards_col if collection == "cards" else self._passages_col

        where: dict | None = self._build_where(emotions, role, collection)

        results = self._query_collection(col, query, k=k, where=where)

        # Fallback: if filters yielded nothing, retry without them
        ids = results.get("ids", [[]])[0] if results else []
        if where is not None and not ids:
            results = self._query_collection(col, query, k=k, where=None)

        return self._to_retrieved_items(results, collection, min_similarity)

    def search_with_evidence(
        self,
        query: str,
        *,
        emotions: Sequence[str] | None = None,
        role: str | None = None,
        k: int = 3,
    ) -> list[RetrievedItem]:
        """Search the cards collection and return items above the similarity threshold.

        This method satisfies the :class:`~app.pipeline.retrieval.RetrieverLike` protocol.

        Args:
            query: The user query string.
            emotions: Optional sequence of emotion labels for metadata filtering.
            role: Optional role name for metadata filtering.
            k: Maximum number of results to return.

        Returns:
            A list of :class:`~app.contracts.RetrievedItem` instances, or an empty
            list when nothing reaches 0.30 similarity (the "just listen" signal).
        """
        return self.search(
            query,
            emotions=emotions,
            role=role,
            k=k,
            collection="cards",
            min_similarity=0.30,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_where(
        self,
        emotions: Sequence[str] | None,
        role: str | None,
        collection: str,
    ) -> dict | None:
        """Build a ChromaDB ``where`` clause from optional emotion/role filters.

        Returns None when no filters were requested.
        """
        if collection != "cards":
            return None  # passage collection has no emotion/role metadata

        conditions: list[dict] = []
        if emotions:
            # The metadata field stores emotions joined by "|"
            # ChromaDB supports $contains for string fields
            for emotion in emotions:
                conditions.append({"emotions": {"$contains": emotion}})
        if role:
            conditions.append({"role_affinity": {"$contains": role}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$or": conditions}

    def _query_collection(
        self,
        col: object,
        query: str,
        k: int,
        where: dict | None,
    ) -> dict:
        """Run a ChromaDB query and return the raw results dict."""
        kwargs: dict = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        try:
            return col.query(**kwargs)  # type: ignore[union-attr]
        except Exception:
            # If filtered query fails (e.g. no matching docs), return empty
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _to_retrieved_items(
        self,
        results: dict,
        collection: str,
        min_similarity: float,
    ) -> list[RetrievedItem]:
        """Convert raw ChromaDB results to RetrievedItem instances."""
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items: list[RetrievedItem] = []
        for doc_id, text, meta, dist in zip(ids, docs, metas, distances, strict=False):
            sim = _dist_to_sim(dist)
            if sim < min_similarity:
                continue

            if collection == "cards":
                source = meta.get("source_ids", doc_id)
                # doc_id IS the card id for the cards collection
                item = RetrievedItem(
                    collection=collection,
                    doc_id=doc_id,
                    source=source,
                    text=self._card_text(doc_id, text),
                    score=round(sim, 4),
                )
            else:
                source = meta.get("source_id", "")
                item = RetrievedItem(
                    collection=collection,
                    doc_id=doc_id,
                    source=source,
                    text=text or "",
                    score=round(sim, 4),
                )
            items.append(item)

        return items

    def _card_text(self, card_id: str, fallback_text: str) -> str:
        """Return a formatted card text string for RetrievedItem.text."""
        card = self._card_library.get(card_id)
        if card is None:
            return fallback_text or ""
        return f"{card.title}: {card.summary}"


__all__ = ["Retriever"]
