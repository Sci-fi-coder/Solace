"""Card library, a dependency-free keyword retriever, and citation helpers.

The ChromaDB retriever (Layer 2b) plugs in through the same ``search_with_evidence`` method.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import yaml

from app.contracts import RetrievedItem
from app.pipeline.types import Citation
from app.rag.cards import Card

logger = logging.getLogger(__name__)

SOURCE_LABELS = {
    "dwm_stress": "WHO: Doing What Matters in Times of Stress",
    "pm_plus": "WHO: Problem Management Plus",
    "mhgap_ig_v2": "WHO: mhGAP Intervention Guide",
    "nimh_stress": "NIMH: I'm So Stressed Out!",
}
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "so", "that", "the", "their", "them", "there", "they", "this", "to", "was", "we", "what", "when", "with", "you", "your", "feel", "feeling", "really", "very", "just"]
)


def default_cards_dir() -> Path:
    """data/kb/cards relative to the repository root."""
    return Path(__file__).resolve().parents[2] / "data" / "kb" / "cards"


class CardLibrary:
    """Loads reviewed knowledge cards from YAML files."""

    def __init__(self, cards: dict[str, Card]) -> None:
        self._cards = cards

    @classmethod
    def load(cls, directory: Path | None = None, *, reviewed_only: bool = True) -> CardLibrary:
        """Read every card file; skip unreviewed cards unless asked."""
        directory = directory or default_cards_dir()
        cards: dict[str, Card] = {}
        for path in sorted(directory.glob("*.yaml")):
            card = Card.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            if reviewed_only and not card.reviewed:
                continue
            cards[card.id] = card
        logger.info("Loaded %d knowledge cards", len(cards))
        return cls(cards)

    def get(self, card_id: str) -> Card | None:
        """Return a card by id."""
        return self._cards.get(card_id)

    def all(self) -> list[Card]:
        """All loaded cards."""
        return list(self._cards.values())

    def __len__(self) -> int:
        return len(self._cards)


class RetrieverLike(Protocol):
    """What the engine needs from a retriever (Chroma or keyword)."""

    def search_with_evidence(
        self,
        query: str,
        *,
        emotions: Sequence[str] | None = None,
        role: str | None = None,
        k: int = 3,
    ) -> list[RetrievedItem]:
        """Return relevant items; an empty list means 'nothing relevant, just listen'."""


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+", text.lower()) if t not in _STOP and len(t) > 2]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


class KeywordCardRetriever:
    """TF-IDF cosine over card text plus small emotion and role bonuses. A strong baseline."""

    def __init__(self, library: CardLibrary, *, min_score: float = 0.18) -> None:
        self._library = library
        self._min_score = min_score
        self._docs: dict[str, Counter[str]] = {}
        for card in library.all():
            text = " ".join([card.title, card.summary, *card.when_to_use, *card.emotions])
            self._docs[card.id] = Counter(_stem(t) for t in _tokens(text))
        n = max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for counts in self._docs.values():
            df.update(counts.keys())
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._norms = {
            cid: math.sqrt(sum((c * self._idf[t]) ** 2 for t, c in counts.items())) or 1.0
            for cid, counts in self._docs.items()
        }

    def search_with_evidence(
        self,
        query: str,
        *,
        emotions: Sequence[str] | None = None,
        role: str | None = None,
        k: int = 3,
    ) -> list[RetrievedItem]:
        """Rank cards for a query; return [] when nothing is relevant enough."""
        q = Counter(_stem(t) for t in _tokens(query))
        q_norm = math.sqrt(sum((c * self._idf.get(t, 1.0)) ** 2 for t, c in q.items())) or 1.0
        scored: list[tuple[float, Card]] = []
        for card in self._library.all():
            doc = self._docs[card.id]
            dot = sum(c * self._idf.get(t, 1.0) * doc.get(t, 0) * self._idf.get(t, 1.0) for t, c in q.items())
            score = dot / (q_norm * self._norms[card.id])
            lexical = score
            if lexical < 0.08:
                continue  # no real textual match: bonuses alone must never trigger retrieval
            if emotions and set(emotions) & set(card.emotions):
                score += 0.12
            if role and role in card.role_affinity:
                score += 0.05
            scored.append((min(score, 1.0), card))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedItem(
                collection="cards",
                doc_id=card.id,
                source=", ".join(s.source_id for s in card.sources),
                text=f"{card.title}: {card.summary}",
                score=round(score, 4),
            )
            for score, card in scored[:k]
            if score >= self._min_score
        ]


def citations_for(card_ids: Sequence[str], library: CardLibrary) -> list[Citation]:
    """Build UI citations (card title plus source pages) for the cards a reply used."""
    out: list[Citation] = []
    for cid in dict.fromkeys(card_ids):
        card = library.get(cid)
        if card is None:
            continue
        sources = [
            f"{SOURCE_LABELS.get(s.source_id, s.source_id)}, PDF p. " + ", ".join(map(str, s.pages))
            for s in card.sources
        ]
        out.append(Citation(card_id=card.id, title=card.title, sources=sources))
    return out
