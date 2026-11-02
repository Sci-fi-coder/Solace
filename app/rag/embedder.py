"""Embedder protocol and a lazy sentence-transformers implementation.

Never use ChromaDB's default embedding function.
"""

from __future__ import annotations

import hashlib
import re
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
    """Deterministic hash-based token embedder for tests. No model downloads."""

    DIM = 384

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized feature-hash vectors with shared-token similarity."""
        result: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for token in re.findall(r"\w+", text.casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.DIM
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[index] += sign
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            result.append([x / norm for x in vec])
        return result


__all__ = ["Embedder", "FakeEmbedder", "LocalEmbedder"]
