"""Pipeline integration tests for optional historical inspiration context."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.contracts import Phase, RetrievedItem, TaskKind
from app.pipeline.engine import SolaceEngine
from app.pipeline.retrieval import CardLibrary
from app.pipeline.types import ReplyOutput
from app.rag.cards import Card, CardSource
from app.rolemodels.library import RoleModelLibrary
from app.storage import repository as repo


class ScriptedGateway:
    """Return one deterministic plan and reply without network access."""

    def __init__(self, plan: dict[str, object], reply: ReplyOutput) -> None:
        self.plan = plan
        self.reply = reply
        self.calls: list[tuple[TaskKind, list[dict[str, str]]]] = []

    def generate(self, task, messages, *, json_schema=None, temperature=0.7, max_tokens=800):
        self.calls.append((task, messages))
        value = self.plan if task == TaskKind.extract else self.reply
        return SimpleNamespace(parsed=value, text="", provider="fake", model="offline")


class SingleCardRetriever:
    """Return one supplied self-help card as the advice grounding source."""

    def __init__(self, card: Card) -> None:
        self.card = card

    def search_with_evidence(self, query, *, emotions=None, role=None, k=3):
        return [
            RetrievedItem(
                collection="cards",
                doc_id=self.card.id,
                source="synthetic",
                text=self.card.summary,
                score=0.9,
            )
        ]


def _technique_card() -> Card:
    """Build a reviewed synthetic technique card for the pipeline test."""
    return Card(
        id="synthetic-grounding",
        title="A small grounding pause",
        category="technique",
        summary="Notice a steady point of contact. Return attention to the present moment.",
        when_to_use=["When stress feels distracting."],
        when_not_to_use=["When the person does not want a technique."],
        steps=["Notice one point of contact."],
        emotions=["anxiety_fear", "anger"],
        role_affinity=["mentor", "counsellor"],
        sources=[CardSource(source_id="synthetic", pages=[1])],
        reviewed=True,
    )


def _engine(tmp_path: Path, gateway: ScriptedGateway) -> tuple[SolaceEngine, str]:
    """Construct an isolated engine with optional tested role-model data."""
    db = repo.build_engine(f"sqlite:///{tmp_path / 'rolemodels.db'}")
    factory = repo.make_session_factory(db)
    self_help_card = _technique_card()
    role_models = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    engine = SolaceEngine(
        gateway,
        factory,
        library=CardLibrary({self_help_card.id: self_help_card}),
        retriever=SingleCardRetriever(self_help_card),
        role_model_library=role_models,
    )
    session_id = engine.start_session("synthetic-rolemodel-user").session_id
    runtime = engine._runtimes[session_id]
    runtime.phase = Phase.guidance
    runtime.turn_count = 2
    return engine, session_id


def _plan(*, safety_level: str = "none") -> dict[str, object]:
    """Return an action-oriented plan that can pass the existing phase guard."""
    return {
        "safety_level": safety_level,
        "emotions": [{"label": "anger", "probability": 0.6}],
        "intensity": 0.6,
        "need": "act",
        "confidence": 0.9,
        "phase": "guidance",
        "speaker": "mentor",
        "reason": "The person asked for practical support.",
        "move": "offer_technique",
        "retrieve": True,
        "retrieve_query": "humiliation anger purpose",
        "wants_new_help": True,
    }


def test_pipeline_passes_separate_card_and_emits_provenance_only_when_used(
    tmp_path: Path,
) -> None:
    """A grounded analogy is passed separately and provenance reflects actual use."""
    reply = ReplyOutput(
        reply=(
            "Gandhi was removed from a first-class railway compartment despite holding a ticket. "
            "That may offer an analogy for responding constructively to an unjust experience."
        ),
        explanation="A historical parallel was relevant to the person's theme.",
        used_card_ids=["synthetic-grounding"],
        used_role_model_id="gandhi-pietermaritzburg",
    )
    gateway = ScriptedGateway(_plan(), reply)
    engine, session_id = _engine(tmp_path, gateway)

    output = engine.handle_turn(
        session_id,
        "I feel angry and humiliated by an unfair experience. What practical step could help?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == "gandhi-pietermaritzburg"
    assert output.historical_inspiration.source_locators
    assert engine._runtimes[session_id].role_model_use_turns == [3]
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "SEPARATE OPTIONAL HISTORICAL INSPIRATION" in reply_call[-1]["content"]
    assert "gandhi-pietermaritzburg" in reply_call[-1]["content"]


def test_pipeline_omits_provenance_when_reply_does_not_use_the_card(tmp_path: Path) -> None:
    """A selected but unused card never appears as response provenance."""
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply="A grounding pause may help you notice one steady point. Would you like to try it?",
            used_card_ids=["synthetic-grounding"],
        ),
    )
    engine, session_id = _engine(tmp_path, gateway)

    output = engine.handle_turn(
        session_id,
        "I feel angry and humiliated by an unfair experience. What practical step could help?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is None
    assert engine._runtimes[session_id].role_model_use_turns == []


def test_elevated_safety_suppresses_role_model_context_and_provenance(tmp_path: Path) -> None:
    """Elevated safety never passes historical context to reply generation."""
    gateway = ScriptedGateway(
        _plan(safety_level="elevated"),
        ReplyOutput(
            reply="That sounds heavy, and I'm glad you told me. We can take this slowly.",
        ),
    )
    engine, session_id = _engine(tmp_path, gateway)

    output = engine.handle_turn(
        session_id,
        "I feel angry and humiliated by an unfair experience.",
        enable_role_models=True,
    )

    assert output.historical_inspiration is None
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "SEPARATE OPTIONAL HISTORICAL INSPIRATION" not in reply_call[-1]["content"]
