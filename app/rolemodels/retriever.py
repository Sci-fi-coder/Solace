"""Hybrid retrieval for reviewed and unreviewed historical role-model cards."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from app.contracts import EmotionalState
from app.rag.embedder import Embedder, LocalEmbedder
from app.rolemodels.library import RoleModelLibrary
from app.rolemodels.models import PERSON_DISPLAY_NAMES, Person, RoleModelCard
from app.rolemodels.selector import requested_person

logger = logging.getLogger(__name__)

_TAG_TERMS: dict[str, tuple[str, ...]] = {
    "anger": ("anger", "angry", "furious", "irritated", "frustrated"),
    "calm": ("calm", "quiet", "settled", "peaceful"),
    "failure": ("fail", "failed", "failing", "failure", "mistake", "setback"),
    "fear": ("fear", "afraid", "anxious", "worried", "nervous", "panic", "stress"),
    "grief": ("grief", "grieving", "loss", "bereaved"),
    "hardship": ("hardship", "difficult time", "struggle", "adversity"),
    "humiliation": ("humiliated", "humiliation", "ridiculed", "embarrassed", "mocked"),
    "low_mood": ("low", "down", "sad", "empty", "hopeless", "lonely"),
    "pressure": ("pressure", "pressured", "overwhelmed", "deadline", "demand"),
    "purpose": ("purpose", "meaning", "values", "direction"),
    "responsibility": ("responsibility", "responsible", "duty", "accountable"),
    "routine": ("routine", "habit", "daily practice", "regular"),
    "self_doubt": ("self doubt", "doubt myself", "insecure", "inadequate", "not good enough"),
    "uncertainty": ("uncertain", "unsure", "uncertainty", "unclear"),
}


def _compile_tag_pattern(term: str) -> re.Pattern[str]:
    """Compile a bounded phrase pattern that accepts spaces, underscores, or hyphens."""
    escaped = re.escape(term).replace(r"\ ", r"[\s_-]+")
    return re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")


_TAG_PATTERNS = {
    tag: tuple(_compile_tag_pattern(term) for term in terms)
    for tag, terms in _TAG_TERMS.items()
}
_EMOTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "anger": re.compile(r"(?i)\b(?:ang(?:er|ry)|furious|irritat\w*|frustrat\w*)\b"),
    "anxiety_fear": re.compile(
        r"(?i)\b(?:anxious|anxiety|afraid|fear|fearful|nervous|worried|worry|panic|"
        r"overwhelmed|stress(?:ed)?)\b"
    ),
    "neutral": re.compile(r"(?i)\b(?:numb|flat|neutral|uncertain|unsure)\b"),
    "positive": re.compile(r"(?i)\b(?:hopeful|relieved|grateful|glad|better)\b"),
    "sadness": re.compile(r"(?i)\b(?:sad|low|lonely|grief|hopeless|empty|down)\b"),
    "shame_guilt": re.compile(
        r"(?i)\b(?:ashamed|guilty|shame|guilt|worthless|failure|humiliated)\b"
    ),
}
_STATE_FIELDS = (
    "trigger",
    "duration",
    "functional_impact",
    "need",
)


@dataclass(frozen=True)
class RetrievalWeights:
    """Calibrated hybrid-score weights for the current synthetic evaluation set."""

    semantic_similarity: float = 0.90
    theme_overlap: float = 0.05
    context_compatibility: float = 0.05

    def __post_init__(self) -> None:
        """Require finite weights in [0, 1] that sum to one."""
        values = (
            self.semantic_similarity,
            self.theme_overlap,
            self.context_compatibility,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("retrieval weights must be finite values between 0 and 1")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("retrieval weights must sum to 1.0")


@dataclass(frozen=True)
class RoleModelScoreBreakdown:
    """Raw hybrid signals and their weighted contributions for one card."""

    semantic_similarity: float
    theme_overlap: float
    context_compatibility: float
    semantic_contribution: float
    theme_contribution: float
    context_contribution: float

    @property
    def total(self) -> float:
        """Return the sum of the three weighted score contributions."""
        return (
            self.semantic_contribution
            + self.theme_contribution
            + self.context_contribution
        )


@dataclass(frozen=True)
class RoleModelMatch:
    """A ranked historical card with its original metadata and score explanation."""

    card: RoleModelCard
    score: float
    breakdown: RoleModelScoreBreakdown

    @property
    def avoid_when(self) -> str:
        """Expose the card's original prose condition without interpreting it."""
        return self.card.avoid_when

    @property
    def source_locators(self) -> tuple[str, ...]:
        """Return the source locators exactly as stored on the card."""
        return tuple(source.locator for source in self.card.sources)


@dataclass(frozen=True)
class RoleModelRetrievalResult:
    """Best match or an explicit reason for abstention, plus deterministic rankings."""

    match: RoleModelMatch | None
    ranked_matches: tuple[RoleModelMatch, ...]
    abstention_reason: Literal[
        "below_threshold", "ambiguous", "embedder_unavailable", "no_candidates"
    ] | None
    explicit_person: Person | None
    embedder_error_type: str | None = None


class RoleModelRetriever:
    """Retrieve historical cards with semantic, theme, and context evidence."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        library: RoleModelLibrary | None = None,
        embedder: Embedder | None = None,
        weights: RetrievalWeights | None = None,
        minimum_relevance: float = 0.45,
        top_match_margin: float | None = None,
    ) -> None:
        """Load all cards and configure an independently tunable relevance policy."""
        if not math.isfinite(minimum_relevance) or not 0.0 <= minimum_relevance <= 1.0:
            raise ValueError("minimum_relevance must be between 0.0 and 1.0")
        if top_match_margin is not None and (
            not math.isfinite(top_match_margin) or not 0.0 <= top_match_margin <= 1.0
        ):
            raise ValueError("top_match_margin must be between 0.0 and 1.0")

        self._library = (
            library
            if library is not None
            else RoleModelLibrary.load(directory, include_unreviewed=True)
        )
        self._cards = tuple(sorted(self._library.all(), key=lambda card: card.id))
        self._embedder = embedder if embedder is not None else LocalEmbedder()
        self._weights = weights if weights is not None else RetrievalWeights()
        self._minimum_relevance = minimum_relevance
        self._top_match_margin = top_match_margin
        self._card_texts = tuple(self._card_text(card) for card in self._cards)
        self._card_vectors: tuple[tuple[float, ...], ...] | None = None
        self._tag_weights = self._inverse_frequency_weights(
            [set(card.tags) for card in self._cards]
        )
        self._emotion_weights = self._inverse_frequency_weights(
            [set(card.emotions) for card in self._cards]
        )

    @property
    def cards(self) -> tuple[RoleModelCard, ...]:
        """Return the immutable set of loaded cards, preserving each card's metadata."""
        return self._cards

    def retrieve(
        self,
        current_user_message: str,
        *,
        recent_user_messages: Sequence[str] = (),
        conversation_state: EmotionalState | Mapping[str, object] | None = None,
        explicit_person: Person | str | None = None,
    ) -> RoleModelRetrievalResult:
        """Return a sufficiently relevant card, or an explicit no-match result.

        Only user messages and the supplied emotional/conversation state are considered;
        assistant-generated historical analogies are not accepted as retrieval context.
        """
        if not current_user_message.strip():
            raise ValueError("current_user_message must not be empty")

        user_messages = tuple(
            message.strip()
            for message in (current_user_message, *recent_user_messages)
            if message.strip()
        )
        state_values = self._state_values(conversation_state)
        state_text = self._state_text(state_values)
        query_text = "\n".join((*user_messages, state_text)).strip()
        person = self._resolve_person(explicit_person, "\n".join(user_messages))
        candidates = tuple(
            card for card in self._cards if person is None or card.person == person
        )
        if not candidates:
            return RoleModelRetrievalResult(None, (), "no_candidates", person)

        try:
            query_vector = self._query_vector(query_text)
        except (ImportError, OSError) as exc:
            logger.warning("Historical role-model embeddings unavailable (%s)", type(exc).__name__)
            return RoleModelRetrievalResult(
                None,
                (),
                "embedder_unavailable",
                person,
                embedder_error_type=type(exc).__name__,
            )

        user_text = "\n".join(user_messages)
        user_tags = self._extract_tags(user_text)
        state_tags = self._extract_tags(state_text)
        user_emotions = self._extract_emotions(user_text)
        user_emotions.update(self._state_emotions(state_values))

        matches: list[RoleModelMatch] = []
        for card, card_vector in zip(self._cards, self._get_card_vectors(), strict=True):
            if person is not None and card.person != person:
                continue
            semantic = self._cosine_similarity(query_vector, card_vector)
            theme = self._weighted_jaccard(user_tags, set(card.tags), self._tag_weights)
            context_signals: list[float] = []
            if user_emotions:
                context_signals.append(
                    self._weighted_jaccard(
                        user_emotions, set(card.emotions), self._emotion_weights
                    )
                )
            if state_tags:
                context_signals.append(
                    self._weighted_jaccard(state_tags, set(card.tags), self._tag_weights)
                )
            context = sum(context_signals) / len(context_signals) if context_signals else 0.0

            breakdown = RoleModelScoreBreakdown(
                semantic_similarity=semantic,
                theme_overlap=theme,
                context_compatibility=context,
                semantic_contribution=semantic * self._weights.semantic_similarity,
                theme_contribution=theme * self._weights.theme_overlap,
                context_contribution=context * self._weights.context_compatibility,
            )
            matches.append(RoleModelMatch(card, breakdown.total, breakdown))

        matches.sort(key=lambda item: (-item.score, -item.breakdown.semantic_similarity, item.card.id))
        ranked = tuple(matches)
        best = ranked[0]
        if best.score < self._minimum_relevance:
            return RoleModelRetrievalResult(None, ranked, "below_threshold", person)

        if (
            self._top_match_margin is not None
            and len(ranked) > 1
            and ranked[1].score >= self._minimum_relevance
            and best.score - ranked[1].score < self._top_match_margin
        ):
            return RoleModelRetrievalResult(None, ranked, "ambiguous", person)

        return RoleModelRetrievalResult(best, ranked, None, person)

    def _query_vector(self, query_text: str) -> tuple[float, ...]:
        """Embed one query and lazily cache validated vectors for every card."""
        if self._card_vectors is None:
            vectors = self._embedder.embed([query_text, *self._card_texts])
            if len(vectors) != len(self._cards) + 1:
                raise ValueError("embedder returned an unexpected number of vectors")
            query_vector = self._validated_vector(vectors[0])
            card_vectors = tuple(self._validated_vector(vector) for vector in vectors[1:])
            if any(len(vector) != len(query_vector) for vector in card_vectors):
                raise ValueError("embedder returned vectors with inconsistent dimensions")
            self._card_vectors = card_vectors
            return query_vector

        vectors = self._embedder.embed([query_text])
        if len(vectors) != 1:
            raise ValueError("embedder returned an unexpected number of vectors")
        query_vector = self._validated_vector(vectors[0])
        if len(query_vector) != len(self._card_vectors[0]):
            raise ValueError("query vector dimension does not match card vectors")
        return query_vector

    def _get_card_vectors(self) -> tuple[tuple[float, ...], ...]:
        """Return card vectors populated by the current query embedding call."""
        if self._card_vectors is None:
            raise RuntimeError("card embeddings were not initialized")
        return self._card_vectors

    @staticmethod
    def _validated_vector(vector: Sequence[float]) -> tuple[float, ...]:
        """Convert an embedder vector to finite floats and reject malformed output."""
        values = tuple(float(value) for value in vector)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("embedder returned an empty or non-finite vector")
        return values

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        """Return cosine similarity clamped to the retrieval score range [0, 1]."""
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        similarity = sum(a * b for a, b in zip(left, right, strict=True))
        return max(0.0, min(1.0, similarity / (left_norm * right_norm)))

    @staticmethod
    def _card_text(card: RoleModelCard) -> str:
        """Build semantic text from user-relevant card content, excluding avoid prose."""
        return "\n".join(
            (
                f"Title: {card.title}",
                f"Tags: {', '.join(card.tags)}",
                f"Emotions: {', '.join(card.emotions)}",
                card.summary,
                card.lesson,
                card.framing,
            )
        )

    @staticmethod
    def _inverse_frequency_weights(documents: Sequence[set[str]]) -> dict[str, float]:
        """Give less-common card vocabulary more weight than ubiquitous vocabulary."""
        frequencies: dict[str, int] = {}
        for document in documents:
            for value in document:
                frequencies[value] = frequencies.get(value, 0) + 1
        count = len(documents)
        return {
            value: math.log((count + 1) / (frequency + 1)) + 1.0
            for value, frequency in frequencies.items()
        }

    @classmethod
    def _weighted_jaccard(
        cls,
        left: set[str],
        right: set[str],
        weights: Mapping[str, float],
    ) -> float:
        """Calculate IDF-weighted Jaccard overlap for supported tags or emotions."""
        union = left | right
        if not union:
            return 0.0
        numerator = sum(weights.get(value, 1.0) for value in left & right)
        denominator = sum(weights.get(value, 1.0) for value in union)
        return numerator / denominator

    @staticmethod
    def _extract_tags(text: str) -> set[str]:
        """Map exact tags and bounded, curated paraphrases to the card tag vocabulary."""
        return {
            tag
            for tag, patterns in _TAG_PATTERNS.items()
            if any(pattern.search(text) for pattern in patterns)
        }

    @staticmethod
    def _extract_emotions(text: str) -> set[str]:
        """Extract coarse emotions from user text without treating them as safety signals."""
        return {
            emotion
            for emotion, pattern in _EMOTION_PATTERNS.items()
            if pattern.search(text)
        }

    @staticmethod
    def _state_values(
        state: EmotionalState | Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        """Read only the known emotional/context fields from an available state."""
        if state is None:
            return {}
        if isinstance(state, EmotionalState):
            return state.model_dump()
        return state

    @staticmethod
    def _state_text(state: Mapping[str, object]) -> str:
        """Render only explicit state fields, never a conversation transcript."""
        parts: list[str] = []
        hypotheses = state.get("hypotheses", ())
        if isinstance(hypotheses, Sequence) and not isinstance(hypotheses, (str, bytes)):
            for hypothesis in hypotheses:
                if isinstance(hypothesis, Mapping):
                    label = hypothesis.get("label")
                    if isinstance(label, str):
                        parts.append(label)
                elif isinstance(hypothesis, str):
                    parts.append(hypothesis)

        for field in _STATE_FIELDS:
            value = state.get(field)
            if isinstance(value, str) and value:
                parts.append(value.value if isinstance(value, StrEnum) else value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                parts.extend(item for item in value if isinstance(item, str) and item)
        return " ".join(parts)

    @staticmethod
    def _state_emotions(state: Mapping[str, object]) -> set[str]:
        """Collect supported coarse emotion labels from state hypotheses."""
        hypotheses = state.get("hypotheses", ())
        if not isinstance(hypotheses, Sequence) or isinstance(hypotheses, (str, bytes)):
            return set()
        return {
            label
            for hypothesis in hypotheses
            if isinstance(hypothesis, Mapping)
            and isinstance((label := hypothesis.get("label")), str)
            and label in _EMOTION_PATTERNS
        }

    @staticmethod
    def _resolve_person(
        explicit_person: Person | str | None,
        user_text: str,
    ) -> Person | None:
        """Resolve a directly supplied or user-mentioned supported person."""
        if explicit_person is None:
            return requested_person(user_text)
        if isinstance(explicit_person, Person):
            return explicit_person
        try:
            return Person(explicit_person.casefold())
        except ValueError:
            person = requested_person(explicit_person)
            if person is None:
                supported = ", ".join(PERSON_DISPLAY_NAMES.values())
                raise ValueError(
                    f"explicit_person must name a supported person: {supported}"
                ) from None
            return person


__all__ = [
    "RetrievalWeights",
    "RoleModelMatch",
    "RoleModelRetrievalResult",
    "RoleModelRetriever",
    "RoleModelScoreBreakdown",
]
