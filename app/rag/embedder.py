"""Embedder protocol and a lazy sentence-transformers implementation.

Never use ChromaDB's default embedding function.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


@runtime_checkable
class Embedder(Protocol):
    """Minimal protocol every embedder must satisfy."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one float vector per text."""
        ...


class LocalEmbedder:
    """Lazy-loaded sentence-transformers embedder; CPU-only.

    The model is downloaded and loaded on first call to ``embed``; subsequent calls
    reuse the cached instance.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # lazy

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode *texts* and return a list of float vectors."""
        if self._model is None:
            st_cls = SentenceTransformer
            if st_cls is None:
                from sentence_transformers import SentenceTransformer as st_cls  # noqa: PLC0415

            self._model = st_cls(self._model_name, device="cpu")
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return [v.tolist() for v in vectors]


class FakeEmbedder:
    """Deterministic hash-based fake embedder for tests. No model downloads."""

    DIM = 384

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalised pseudo-random vectors derived from SHA-256 hashes."""
        result: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Build a 384-dim vector from repeating digest bytes
            raw: list[float] = []
            for _ in range((self.DIM + 31) // 32):
                raw.extend([(b - 128) / 128.0 for b in digest])
            vec = raw[: self.DIM]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            result.append([x / norm for x in vec])
        return result


__all__ = ["Embedder", "FakeEmbedder", "LocalEmbedder"]
