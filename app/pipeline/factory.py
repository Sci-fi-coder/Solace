"""Build a ready-to-use engine from application settings (used by the CLI and the UI)."""

from __future__ import annotations

import logging
import os

from app.pipeline.engine import SolaceEngine
from app.pipeline.retrieval import CardLibrary, KeywordCardRetriever, RetrieverLike
from app.privacy import install_privacy_filter

logger = logging.getLogger(__name__)


def _make_retriever(library: CardLibrary, mode: str) -> RetrieverLike:
    """'keyword' = dependency-free baseline; 'chroma' = Layer 2b retriever; 'auto' tries Chroma first."""
    if mode in ("auto", "chroma"):
        try:
            from app.rag.retriever import Retriever  # type: ignore[import-not-found]

            return Retriever()  # type: ignore[no-any-return]
        except Exception as exc:
            if mode == "chroma":
                raise
            logger.warning("Chroma retriever unavailable (%s); using the keyword retriever.", type(exc).__name__)
    return KeywordCardRetriever(library)


def build_engine_from_settings(
    *,
    use_emotion: bool | None = None,
    separate_safety_call: bool = False,
    retriever_mode: str | None = None,
) -> SolaceEngine:
    """Wire the gateway, storage, cards, retriever and safety gate together."""
    from app.config import settings
    from app.llm.gateway import LLMGateway
    from app.safety.classifier import GatewaySafetyClassifier
    from app.safety.sentinel import SafetyGate
    from app.storage import repository as repo

    db = repo.build_engine(settings.database_url)
    factory = repo.make_session_factory(db)
    gateway = LLMGateway.from_settings(db)
    install_privacy_filter(gateway)
    library = CardLibrary.load()
    retriever = _make_retriever(library, retriever_mode or os.environ.get("SOLACE_RETRIEVER", "auto"))
    if use_emotion is None:
        use_emotion = os.environ.get("SOLACE_EMOTION", "0") == "1"
    emotion = None
    if use_emotion:
        from app.emotion.recognizer import EmotionRecognizer

        emotion = EmotionRecognizer()
    gate = SafetyGate(classifier=GatewaySafetyClassifier(gateway) if separate_safety_call else None)
    return SolaceEngine(
        gateway, factory, library=library, retriever=retriever, emotion_reader=emotion, safety_gate=gate
    )
