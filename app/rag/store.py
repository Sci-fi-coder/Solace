"""ChromaDB client, collection helpers, and a custom EmbeddingFunction wrapper.

The module guards all chromadb imports so they happen only when needed,
preventing import-time side effects in test environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb as _chromadb

    from app.rag.embedder import Embedder

# Resolved relative to the repo root (two parents above this file's package)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHROMA_PATH = _REPO_ROOT / "data" / "chroma"

COLLECTION_CARDS = "cards"
COLLECTION_PASSAGES = "passages"


def default_chroma_path() -> Path:
    """Return the default ChromaDB persistence directory (``data/chroma``)."""
    return _DEFAULT_CHROMA_PATH


def get_client(path: Path | None = None) -> _chromadb.PersistentClient:
    """Open (or create) a persistent ChromaDB client.

    Args:
        path: Directory for ChromaDB files; defaults to ``data/chroma``.
    """
    import chromadb  # noqa: PLC0415

    resolved = path or _DEFAULT_CHROMA_PATH
    resolved.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(resolved))


def get_collection(
    client: _chromadb.PersistentClient,
    name: str,
    embedder: Embedder,
) -> _chromadb.Collection:
    """Get-or-create a collection with cosine distance and a custom embedding function.

    Args:
        client: A ChromaDB PersistentClient.
        name: Collection name (``"cards"`` or ``"passages"``).
        embedder: An :class:`~app.rag.embedder.Embedder` instance.
    """

    ef = _EmbeddingFunctionAdapter(embedder)
    return client.get_or_create_collection(
        name=name,
        embedding_function=ef,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )


class _EmbeddingFunctionAdapter:
    """Wrap an :class:`~app.rag.embedder.Embedder` as a ChromaDB EmbeddingFunction.

    ChromaDB 1.1.x calls ``embedding_function.name()`` during
    ``get_or_create_collection`` to detect conflicts with persisted embedding
    configs.  We return ``"solace-custom"`` so the check never raises and
    repeated calls to ``get_or_create_collection`` are idempotent.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    @staticmethod
    def name() -> str:
        """Return a stable name used by ChromaDB conflict detection."""
        return "solace-custom"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a batch of texts and return float vectors."""
        return self._embedder.embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Embed a query batch."""
        return self.__call__(input)

    def default_space(self) -> str:
        """Return default distance space."""
        return "cosine"

    def supported_spaces(self) -> list[str]:
        """Return supported distance spaces."""
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, str]:
        """Return config dict for serialization."""
        return {}

    def is_legacy(self) -> bool:
        """Indicate this is not a legacy embedding function."""
        return False


__all__ = [
    "COLLECTION_CARDS",
    "COLLECTION_PASSAGES",
    "default_chroma_path",
    "get_client",
    "get_collection",
]
