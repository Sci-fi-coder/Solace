"""YAML library for historical cards, independent of the self-help RAG library."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from app.rolemodels.models import Person, RoleModelCard

logger = logging.getLogger(__name__)


def default_role_models_dir() -> Path:
    """Return the repository's data/role_models directory."""
    return Path(__file__).resolve().parents[2] / "data" / "role_models"


class RoleModelLibrary:
    """Loads and indexes role-model cards without mixing them into self-help RAG."""

    def __init__(self, cards: dict[str, RoleModelCard]) -> None:
        """Store validated cards by their stable identifier."""
        self._cards = cards

    @classmethod
    def load(
        cls,
        directory: Path | None = None,
        *,
        include_unreviewed: bool = False,
    ) -> RoleModelLibrary:
        """Validate all YAML cards and return reviewed cards unless explicitly requested."""
        source_dir = directory or default_role_models_dir()
        cards: dict[str, RoleModelCard] = {}
        for path in sorted(source_dir.glob("*.yaml")):
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                card = RoleModelCard.model_validate(payload)
            except (OSError, yaml.YAMLError, ValueError) as exc:
                raise ValueError(f"Invalid role-model card {path}: {exc}") from exc
            if card.id in cards:
                raise ValueError(f"Duplicate role-model card id {card.id!r} in {path}")
            if include_unreviewed or card.reviewed:
                cards[card.id] = card
        logger.info(
            "Loaded %d role-model cards (include_unreviewed=%s)",
            len(cards),
            include_unreviewed,
        )
        return cls(cards)

    def get(self, card_id: str) -> RoleModelCard | None:
        """Return a card by stable id."""
        return self._cards.get(card_id)

    def all(self) -> list[RoleModelCard]:
        """Return the cards currently loaded into this library."""
        return list(self._cards.values())

    def for_person(self, person: Person) -> list[RoleModelCard]:
        """Return loaded cards for one supported person."""
        return [card for card in self._cards.values() if card.person == person]

    def __len__(self) -> int:
        """Return the number of loaded cards."""
        return len(self._cards)
