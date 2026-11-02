"""RAG package — ChromaDB retrieval pipeline (Layer 2b).

Public API
----------
- :class:`~app.rag.embedder.Embedder` – protocol
- :class:`~app.rag.embedder.LocalEmbedder` – sentence-transformers embedder
- :class:`~app.rag.embedder.FakeEmbedder` – deterministic test embedder
- :class:`~app.rag.retriever.Retriever` – ChromaDB-backed retriever
- :func:`~app.rag.store.get_client` – open a ChromaDB PersistentClient
- :func:`~app.rag.store.get_collection` – get-or-create a collection
- :func:`~app.rag.store.default_chroma_path` – default DB path
- :func:`~app.rag.sanitizer.sanitize_page` – line-level sanitizer
- :func:`~app.rag.sanitizer.build_deny_set` – parse deny-section specs
- :func:`~app.rag.sanitizer.chunk_text` – word-based chunker
"""

from app.rag.cards import Card, CardCategory, CardSource, RoleAffinity, banned_content_reasons
from app.rag.embedder import Embedder, FakeEmbedder, LocalEmbedder
from app.rag.retriever import Retriever
from app.rag.sanitizer import PageText, build_deny_set, chunk_text, sanitize_page
from app.rag.store import (
    COLLECTION_CARDS,
    COLLECTION_PASSAGES,
    default_chroma_path,
    get_client,
    get_collection,
)

__all__ = [
    # cards
    "Card",
    "CardCategory",
    "CardSource",
    "RoleAffinity",
    "banned_content_reasons",
    # embedder
    "Embedder",
    "FakeEmbedder",
    "LocalEmbedder",
    # retriever
    "Retriever",
    # sanitizer
    "PageText",
    "build_deny_set",
    "chunk_text",
    "sanitize_page",
    # store
    "COLLECTION_CARDS",
    "COLLECTION_PASSAGES",
    "default_chroma_path",
    "get_client",
    "get_collection",
]
