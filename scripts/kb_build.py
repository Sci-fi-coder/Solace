"""Knowledge-base builder: ingest cards and passages into ChromaDB.

Usage
-----
    python -m scripts.kb_build           # idempotent upsert
    python -m scripts.kb_build --rebuild # wipe then recreate

The first run downloads ~90 MB of model weights. If your environment blocks
outbound network access, run ``pip install sentence-transformers`` and download
the ``sentence-transformers/all-MiniLM-L6-v2`` model manually first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# ── Repo-root bootstrap ──────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.pipeline.retrieval import CardLibrary  # noqa: E402
from app.rag.embedder import LocalEmbedder  # noqa: E402
from app.rag.sanitizer import PageText, build_deny_set, chunk_text, sanitize_page  # noqa: E402
from app.rag.store import (  # noqa: E402
    COLLECTION_CARDS,
    COLLECTION_PASSAGES,
    get_client,
    get_collection,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_KB_WORK_DIR = _REPO / "data" / "kb_work"
_SOURCES_YAML = _REPO / "data" / "kb" / "sources.yaml"


def _load_card_library(*, include_unreviewed: bool = False) -> CardLibrary:
    """Load reviewed cards unless the caller explicitly includes unreviewed cards."""
    return CardLibrary.load(reviewed_only=not include_unreviewed)


def _load_deny_sets() -> dict[str, frozenset[int]]:
    """Parse sources.yaml and return a mapping of source_id → denied page numbers."""
    raw = yaml.safe_load(_SOURCES_YAML.read_text(encoding="utf-8"))
    result: dict[str, frozenset[int]] = {}
    for source in raw.get("sources", []):
        sid = source["id"]
        deny_sections = source.get("deny_sections", [])
        result[sid] = build_deny_set(deny_sections)
    return result


def _ingest_cards(collection: object, library: CardLibrary) -> int:
    """Upsert all reviewed cards into the cards collection.

    Args:
        collection: ChromaDB Collection for cards.
        library: Loaded CardLibrary.

    Returns:
        Number of cards upserted.
    """
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

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
                "reviewed": card.reviewed,
            }
        )

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore[union-attr]
    return len(ids)


def _ingest_passages(
    collection: object,
    deny_sets: dict[str, frozenset[int]],
) -> dict[str, int]:
    """Chunk and upsert sanitized passages from all kb_work JSON files.

    Args:
        collection: ChromaDB Collection for passages.
        deny_sets: Mapping of source_id → denied page numbers.

    Returns:
        A dict mapping source_id → number of chunks upserted.
    """
    counts: dict[str, int] = defaultdict(int)

    for json_path in sorted(_KB_WORK_DIR.glob("*.json")):
        source_id = json_path.stem
        if source_id not in deny_sets:
            raise ValueError(f"Source {source_id!r} is missing from data/kb/sources.yaml")
        denied = deny_sets[source_id]

        raw_pages: list[dict] = json.loads(json_path.read_text(encoding="utf-8"))
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for raw in raw_pages:
            page_num: int = raw["page_number"]
            if page_num in denied:
                continue

            page = PageText(page_number=page_num, text=raw["text"])
            clean, reason = sanitize_page(page)
            if clean is None:
                logger.debug("Skipping %s p%d: %s", source_id, page_num, reason)
                continue

            chunks = chunk_text(clean.text)
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = f"{source_id}_p{page_num}_c{chunk_idx}"
                ids.append(doc_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "source_id": source_id,
                        "pdf_page": page_num,
                        "chunk_index": chunk_idx,
                    }
                )

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore[union-attr]
        counts[source_id] = len(ids)
        logger.info("  %s: %d chunks", source_id, len(ids))

    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the kb_build script."""
    parser = argparse.ArgumentParser(description="Build the Solace ChromaDB knowledge base.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe both collections before rebuilding (default: idempotent upsert).",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include cards that have not been human-reviewed.",
    )
    args = parser.parse_args(argv)

    logger.info("Opening ChromaDB client ...")
    embedder = LocalEmbedder()
    client = get_client()

    if args.rebuild:
        logger.info("--rebuild: deleting existing collections ...")
        existing_collections = {
            collection.name for collection in client.list_collections()
        }
        for collection_name in (COLLECTION_CARDS, COLLECTION_PASSAGES):
            if collection_name in existing_collections:
                client.delete_collection(collection_name)

    cards_col = get_collection(client, COLLECTION_CARDS, embedder)
    passages_col = get_collection(client, COLLECTION_PASSAGES, embedder)

    # ── Cards ────────────────────────────────────────────────────────────────
    logger.info("Loading cards ...")
    library = _load_card_library(include_unreviewed=args.include_unreviewed)
    logger.info("Ingesting %d cards ...", len(library))
    n_cards = _ingest_cards(cards_col, library)
    print(f"cards collection: {n_cards} documents")
    card_source_counts: dict[str, int] = defaultdict(int)
    for card in library.all():
        for source_id in {source.source_id for source in card.sources}:
            card_source_counts[source_id] += 1
    for source_id, count in sorted(card_source_counts.items()):
        print(f"  {source_id}: {count} cards")

    # ── Passages ─────────────────────────────────────────────────────────────
    logger.info("Loading deny sets ...")
    deny_sets = _load_deny_sets()
    logger.info("Ingesting passages ...")
    passage_counts = _ingest_passages(passages_col, deny_sets)
    total_passages = sum(passage_counts.values())
    print(f"passages collection: {total_passages} chunks total")
    for sid, cnt in sorted(passage_counts.items()):
        print(f"  {sid}: {cnt} chunks")


if __name__ == "__main__":
    main()
