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
from app.rolemodels.retriever import (
    RoleModelMatch,
    RoleModelRetrievalResult,
    RoleModelRetriever,
    RoleModelScoreBreakdown,
)
from app.rolemodels.selector import requested_person, select_role_model
from app.storage import repository as repo

CARD_DIR = Path(__file__).parents[1] / "data" / "role_models"


class ScriptedGateway:
    """Return one deterministic plan and reply without network access."""

    def __init__(
        self,
        plan: dict[str, object],
        reply: ReplyOutput,
        *,
        retry_reply: ReplyOutput | None = None,
    ) -> None:
        self.plan = plan
        self.reply = reply
        self.retry_reply = retry_reply
        self.reply_calls = 0
        self.calls: list[tuple[TaskKind, list[dict[str, str]]]] = []

    def generate(self, task, messages, *, json_schema=None, temperature=0.7, max_tokens=800):
        self.calls.append((task, messages))
        if task == TaskKind.extract:
            value = self.plan
        else:
            value = self.retry_reply if self.reply_calls and self.retry_reply else self.reply
            self.reply_calls += 1
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


class EmptyRetriever:
    """Return no self-help cards for testing historical-only requests."""

    def search_with_evidence(self, query, *, emotions=None, role=None, k=3):
        return []


class ScriptedRoleModelRetriever:
    """Use the pre-existing selector deterministically for legacy reply-prompt tests."""

    def __init__(self, library: RoleModelLibrary) -> None:
        self.library = library
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        current_user_message: str,
        *,
        recent_user_messages=(),
        conversation_state=None,
        explicit_person=None,
    ) -> RoleModelRetrievalResult:
        """Record user-only context and return the selector's deterministic test candidate."""
        self.calls.append(
            {
                "current": current_user_message,
                "recent": tuple(recent_user_messages),
                "state": conversation_state,
                "explicit_person": explicit_person,
            }
        )
        labels = (
            [hypothesis.label for hypothesis in conversation_state.hypotheses]
            if conversation_state is not None
            else []
        )
        card = select_role_model(
            self.library,
            "\n".join((*recent_user_messages, current_user_message)),
            emotion_labels=labels,
        )
        if card is None:
            return RoleModelRetrievalResult(
                None, (), "below_threshold", requested_person(current_user_message)
            )
        breakdown = RoleModelScoreBreakdown(1.0, 0.0, 0.0, 0.55, 0.0, 0.0)
        match = RoleModelMatch(card, breakdown.total, breakdown)
        return RoleModelRetrievalResult(
            match, (match,), None, requested_person(current_user_message)
        )


class FixtureRoleModelEmbedder:
    """Return fixed vectors for one named card without model downloads."""

    def __init__(self, target_title: str | None) -> None:
        self.target_title = target_title
        self.queries: list[str] = []

    def embed(self, texts):
        """Make only the named card semantically similar to every test query."""
        vectors = []
        target_card = False
        section_prefixes = (
            "Situation described by the card:",
            "Lesson:",
            "Framing:",
            "Card themes and emotions:",
        )
        for text in texts:
            if text.startswith("Title: "):
                title = text.splitlines()[0].removeprefix("Title: ")
                target_card = title == self.target_title
                vectors.append([1.0, 0.0] if target_card else [0.0, 1.0])
            elif text.startswith(section_prefixes):
                vectors.append([1.0, 0.0] if target_card else [0.0, 1.0])
            else:
                self.queries.append(text)
                vectors.append([1.0, 0.0])
        return vectors


def _hybrid_retriever(
    library: RoleModelLibrary,
    *,
    target_id: str | None = None,
    minimum_relevance: float = 0.5,
) -> RoleModelRetriever:
    """Build the real hybrid retriever with deterministic offline vectors."""
    target = library.get(target_id) if target_id is not None else None
    return RoleModelRetriever(
        library=library,
        embedder=FixtureRoleModelEmbedder(target.title if target is not None else None),
        minimum_relevance=minimum_relevance,
    )


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


def _engine(
    tmp_path: Path,
    gateway: ScriptedGateway,
    role_model_library: RoleModelLibrary | None = None,
    *,
    empty_retrieval: bool = False,
    role_model_retriever=None,
) -> tuple[SolaceEngine, str]:
    """Construct an isolated engine with optional tested role-model data."""
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        retriever=(
            EmptyRetriever() if empty_retrieval else SingleCardRetriever(self_help_card)
        ),
        role_model_library=(
            role_model_library if role_model_library is not None else role_models
        ),
        role_model_retriever=(
            role_model_retriever
            if role_model_retriever is not None
            else ScriptedRoleModelRetriever(
                role_model_library if role_model_library is not None else role_models
            )
        ),
    )
    session_id = engine.start_session("synthetic-rolemodel-user").session_id
    runtime = engine._runtimes[session_id]
    runtime.phase = Phase.guidance
    runtime.turn_count = 2
    return engine, session_id


def _plan(
    *,
    safety_level: str = "none",
    need: str = "act",
    move: str = "offer_technique",
    phase: str = "guidance",
    confidence: float = 0.9,
) -> dict[str, object]:
    """Return an action-oriented plan that can pass the existing phase guard."""
    return {
        "safety_level": safety_level,
        "emotions": [{"label": "anger", "probability": 0.6}],
        "intensity": 0.6,
        "need": need,
        "confidence": confidence,
        "phase": phase,
        "speaker": "mentor",
        "reason": "The person asked for practical support.",
        "move": move,
        "retrieve": True,
        "retrieve_query": "humiliation anger purpose",
        "wants_new_help": True,
    }


def test_pipeline_passes_separate_card_and_emits_provenance_only_when_used(
    tmp_path: Path,
) -> None:
    """A grounded analogy is passed separately and provenance reflects actual use."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    card = library.get("marcus-cooperation")
    assert card is not None
    reply = ReplyOutput(
        reply=(
            "Marcus Aurelius's unverified lesson about cooperation can offer an analogy for "
            "protecting a boundary while leaving room for safe cooperation."
        ),
        explanation="A historical parallel was relevant to the person's theme.",
        used_card_ids=["synthetic-grounding"],
        used_role_model_id=card.id,
    )
    gateway = ScriptedGateway(_plan(), reply)
    engine, session_id = _engine(tmp_path, gateway, RoleModelLibrary({card.id: card}))

    output = engine.handle_turn(
        session_id,
        "I feel angry about a team conflict and want to protect a boundary. "
        "Could Marcus Aurelius offer a relevant lesson?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == card.id
    assert output.historical_inspiration.source_locators
    assert engine._runtimes[session_id].role_model_use_turns == [3]
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "USER-REQUESTED HISTORICAL CARD (optional when relevant and appropriate)" in (
        reply_call[-1]["content"]
    )
    assert "marcus-cooperation" in reply_call[-1]["content"]
    assert "use is optional" in reply_call[0]["content"]


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
        "I feel angry and humiliated by an unfair experience. Could Marcus Aurelius help?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is None
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "Card id: marcus-cooperation" not in reply_call[-1]["content"]


def test_explicit_marcus_request_reaches_prompt_as_required_context(
    tmp_path: Path, monkeypatch
) -> None:
    """An explicit Marcus request passes its flag and selected card into reply prompting."""
    from app.pipeline import engine as engine_module

    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("marcus-cooperation")
    assert card is not None
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply=(
                "Marcus Aurelius's unverified lesson about cooperation can be an analogy. "
                "As an analogy, this may support protecting your boundary while leaving room "
                "for cooperation where it is safe."
            ),
            used_card_ids=["synthetic-grounding"],
            used_role_model_id=card.id,
        ),
    )
    engine, session_id = _engine(
        tmp_path, gateway, RoleModelLibrary({card.id: card})
    )
    prompt_flags: list[bool] = []
    build_messages = engine_module.build_reply_messages

    def capture_prompt(*args, **kwargs):
        prompt_flags.append(kwargs["explicit_request"])
        return build_messages(*args, **kwargs)

    monkeypatch.setattr(engine_module, "build_reply_messages", capture_prompt)
    output = engine.handle_turn(
        session_id,
        "I feel angry about this conflict. Could you relate it to Marcus Aurelius?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == "marcus-cooperation"
    assert prompt_flags == [True]
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "USER-REQUESTED HISTORICAL CARD (optional when relevant and appropriate)" in (
        reply_call[-1]["content"]
    )
    assert "Person: Marcus Aurelius" in reply_call[-1]["content"]
    assert "cooperation where it is safe" in reply_call[-1]["content"]


def test_explicit_kalam_request_is_selected_alongside_clarifying_question(
    tmp_path: Path,
) -> None:
    """An explicit Kalam request can coexist with the planner's clarifying question."""
    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("kalam-empower-young-scientists")
    assert card is not None
    question = "What part of this responsibility feels hardest to carry?"
    plan = {
        **_plan(),
        "phase": "exploration",
        "move": "ask_clarifying",
        "need": "act",
        "confidence": 0.4,
        "retrieve": False,
        "clarifying_question": question,
    }
    gateway = ScriptedGateway(
        plan,
        ReplyOutput(
            reply=(
                "You are carrying a lot of responsibility. A.P.J. Abdul Kalam's unverified "
                "lesson about meaningful roles and experienced guidance offers an analogy "
                "for your situation. What part of this responsibility feels hardest to carry?"
            ),
            used_role_model_id=card.id,
        ),
    )
    engine, session_id = _engine(tmp_path, gateway, RoleModelLibrary({card.id: card}))
    engine._runtimes[session_id].phase = Phase.exploration

    output = engine.handle_turn(
        session_id,
        "I feel unsure about taking responsibility. Could A.P.J. Abdul Kalam offer a lesson?",
        enable_role_models=True,
    )

    assert output.move == "ask_clarifying"
    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == card.id
    assert question in output.reply
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert question in reply_call[-1]["content"]
    assert "Person: A.P.J. Abdul Kalam" in reply_call[-1]["content"]
    assert f"role_model: eligible {card.id}" in output.debug_steps


def test_explicit_marcus_request_is_selected_when_exploration_advice_becomes_question(
    tmp_path: Path,
) -> None:
    """Guardrail-converted advice remains a clarifying turn with Marcus context selected."""
    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("marcus-cooperation")
    assert card is not None
    question = "What boundary would feel important to protect here?"
    plan = {
        **_plan(),
        "phase": "exploration",
        "move": "psychoeducation",
        "need": "understand",
        "confidence": 0.4,
        "retrieve": True,
        "clarifying_question": question,
    }
    gateway = ScriptedGateway(
        plan,
        ReplyOutput(
            reply=(
                "This conflict seems to be weighing on you. Marcus Aurelius's unverified lesson "
                "on cooperation can be an analogy for keeping room for cooperation without "
                "giving up a needed boundary. What boundary would feel important to protect here?"
            ),
            used_role_model_id=card.id,
        ),
    )
    engine, session_id = _engine(tmp_path, gateway, RoleModelLibrary({card.id: card}))
    engine._runtimes[session_id].phase = Phase.exploration

    output = engine.handle_turn(
        session_id,
        "I'm worried about a conflict. Could Marcus Aurelius offer a relevant perspective?",
        enable_role_models=True,
    )

    assert output.phase == Phase.exploration
    assert output.move == "ask_clarifying"
    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == card.id
    assert question in output.reply
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert question in reply_call[-1]["content"]
    assert "Person: Marcus Aurelius" in reply_call[-1]["content"]
    assert f"role_model: eligible {card.id}" in output.debug_steps


def test_explicit_kalam_request_passes_the_selected_card_and_name(tmp_path: Path) -> None:
    """An explicit Kalam request gets the reviewed Kalam card and canonical display name."""
    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("kalam-empower-young-scientists")
    assert card is not None
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply=(
                "A.P.J. Abdul Kalam's unverified lesson about meaningful roles and experienced "
                "guidance offers an analogy for your self-doubt about responsibility."
            ),
            used_card_ids=["synthetic-grounding"],
            used_role_model_id=card.id,
        ),
    )
    engine, session_id = _engine(
        tmp_path, gateway, RoleModelLibrary({card.id: card})
    )

    output = engine.handle_turn(
        session_id,
        "I feel self-doubt about responsibility. Could A.P.J. Abdul Kalam offer an analogy?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == "kalam-empower-young-scientists"
    assert "role_model: eligible kalam-empower-young-scientists" in output.debug_steps
    assert "role_model_critic: first attempt passed" in output.debug_steps
    assert (
        "role_model_validation: complete reply validated on first attempt"
        in output.debug_steps
    )
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "USER-REQUESTED HISTORICAL CARD (optional when relevant and appropriate)" in (
        reply_call[-1]["content"]
    )
    assert "Person: A.P.J. Abdul Kalam" in reply_call[-1]["content"]
    assert "Avoid when:" in reply_call[-1]["content"]
    assert "First address the person's actual concern" in reply_call[0]["content"]
    assert "use is optional" in reply_call[0]["content"]
    assert "Do not give a biography" in reply_call[0]["content"]
    assert "self-doubt about responsibility" in reply_call[-1]["content"]


def test_explicit_kalam_request_is_available_without_self_help_cards(
    tmp_path: Path,
) -> None:
    """An explicit historical request stays eligible when unrelated RAG returns no cards."""
    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("kalam-empower-young-scientists")
    assert card is not None
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply=(
                "A.P.J. Abdul Kalam's unverified lesson about meaningful roles and experienced "
                "guidance offers an analogy for your situation: your team can have ownership "
                "while support stays "
                "available."
            ),
            used_role_model_id=card.id,
        ),
    )
    engine, session_id = _engine(
        tmp_path, gateway, RoleModelLibrary({card.id: card}), empty_retrieval=True
    )

    output = engine.handle_turn(
        session_id,
        "I'm struggling to keep my team motivated and help them take ownership of their work. "
        "Could you share a relevant lesson from A.P.J. Abdul Kalam about empowering a team?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == card.id
    assert output.move == "reflect"
    assert "retrieval: 0 cards" in output.debug_steps
    assert any(step == f"role_model: eligible {card.id}" for step in output.debug_steps)
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "USER-REQUESTED HISTORICAL CARD (optional when relevant and appropriate)" in (
        reply_call[-1]["content"]
    )
    assert "Only if the selected card is relevant" in reply_call[-1]["content"]
    assert "Preserve the planned question" in reply_call[-1]["content"]
    for card_text in (card.title, card.summary, card.lesson, card.framing):
        assert card_text in reply_call[-1]["content"]


def test_explicit_request_without_available_card_does_not_invent_history(
    tmp_path: Path,
) -> None:
    """If no reviewed card is available, counselling proceeds without historical context."""
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply="A grounding pause may help you notice one steady point. Would you like to try it?",
            used_card_ids=["synthetic-grounding"],
        ),
    )
    engine, session_id = _engine(tmp_path, gateway, RoleModelLibrary({}))

    output = engine.handle_turn(
        session_id,
        "I feel angry and would like a Marcus Aurelius analogy.",
        enable_role_models=True,
    )

    assert output.historical_inspiration is None
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "No sufficiently relevant historical card was selected" in reply_call[0]["content"]
    assert "marcus-cooperation" not in reply_call[-1]["content"]


def test_explicit_historical_failure_retries_with_actionable_feedback(
    tmp_path: Path,
) -> None:
    """An invalid historical draft is retried with concrete name and analogy directions."""
    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("marcus-cooperation")
    assert card is not None
    invalid = ReplyOutput(
        reply=(
            "Marcus Aurelius's unverified lesson mentions cooperation. It mentions Churchill "
            "as well. "
            "This may be helpful."
        ),
        used_card_ids=["synthetic-grounding"],
        used_role_model_id=card.id,
    )
    valid_retry = ReplyOutput(
        reply=(
            "Marcus Aurelius's unverified lesson about cooperation can be an analogy. "
            "As an analogy, this may support protecting your boundary while leaving room "
            "for cooperation where it is safe."
        ),
        used_card_ids=["synthetic-grounding"],
        used_role_model_id=card.id,
    )
    gateway = ScriptedGateway(_plan(), invalid, retry_reply=valid_retry)
    engine, session_id = _engine(
        tmp_path, gateway, RoleModelLibrary({card.id: card})
    )

    output = engine.handle_turn(
        session_id,
        "Could you relate this conflict to Marcus Aurelius?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    reply_calls = [messages for task, messages in gateway.calls if task == TaskKind.converse]
    assert len(reply_calls) == 2
    retry_prompt = reply_calls[1][-1]["content"]
    assert "Remove unsupported capitalized token(s): Churchill" in retry_prompt
    assert "Explicitly frame the historical example as an analogy" in retry_prompt
    assert any(
        step == "role_model_critic: first attempt rejected: "
        "unsupported capitalized token(s): Churchill; "
        "historical example must be clearly framed as an analogy"
        for step in output.debug_steps
    )
    assert "role_model_critic: retry attempt passed" in output.debug_steps
    assert (
        "role_model_validation: complete reply validated on retry" in output.debug_steps
    )


def test_missing_historical_person_name_gets_specific_retry_feedback(
    tmp_path: Path,
) -> None:
    """A declared card without its person's name triggers an explicit naming instruction."""
    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("marcus-cooperation")
    assert card is not None
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply="A historical analogy may help you consider cooperation.",
            used_card_ids=["synthetic-grounding"],
            used_role_model_id=card.id,
        ),
        retry_reply=ReplyOutput(
            reply=(
                "Marcus Aurelius's unverified lesson about cooperation can be an analogy. "
                "As an analogy, this may support protecting your boundary while leaving room "
                "for cooperation where it is safe."
            ),
            used_card_ids=["synthetic-grounding"],
            used_role_model_id=card.id,
        ),
    )
    engine, session_id = _engine(
        tmp_path, gateway, RoleModelLibrary({card.id: card})
    )

    output = engine.handle_turn(
        session_id,
        "Could you relate this conflict to Marcus Aurelius?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    reply_calls = [messages for task, messages in gateway.calls if task == TaskKind.converse]
    assert len(reply_calls) == 2
    assert "Name Marcus Aurelius explicitly in the reply" in reply_calls[1][-1]["content"]


def test_failed_explicit_historical_response_cannot_bypass_critic(
    tmp_path: Path,
) -> None:
    """Two rejected historical drafts lead to the safe fallback without provenance."""
    library = RoleModelLibrary.load(
        Path(__file__).parents[1] / "data" / "role_models",
        include_unreviewed=True,
    )
    card = library.get("marcus-cooperation")
    assert card is not None
    invalid = ReplyOutput(
        reply="Marcus Aurelius reflects on cooperation. Churchill led a campaign.",
        used_card_ids=["synthetic-grounding"],
        used_role_model_id=card.id,
    )
    gateway = ScriptedGateway(_plan(), invalid)
    engine, session_id = _engine(
        tmp_path, gateway, RoleModelLibrary({card.id: card})
    )

    output = engine.handle_turn(
        session_id,
        "Could you relate this conflict to Marcus Aurelius?",
        enable_role_models=True,
    )

    assert output.degraded
    assert output.historical_inspiration is None
    assert output.reply != invalid.reply
    assert any(step.startswith("fallback:") for step in output.debug_steps)
    assert any(
        step.startswith("role_model_critic: first attempt rejected: ")
        and "Churchill" in step
        for step in output.debug_steps
    )


def test_hybrid_retriever_runs_for_every_turn_and_has_all_24_cards(tmp_path: Path) -> None:
    """The conversation engine's default historical retriever loads every card."""
    gateway = ScriptedGateway(
        _plan(move="reflect", need="understand"),
        ReplyOutput(reply="That sounds difficult. What part feels most important right now?"),
    )
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    retriever = _hybrid_retriever(library)
    engine, session_id = _engine(
        tmp_path, gateway, library, role_model_retriever=retriever
    )

    output = engine.handle_turn(
        session_id,
        "I am unsure what to do next.",
        enable_role_models=False,
    )

    assert len(engine._role_model_retriever.cards) == 24
    assert any(step.startswith("role_model_match:") for step in output.debug_steps)
    assert "role_model: " not in " ".join(output.debug_steps)


def test_hybrid_match_is_used_when_relevant_and_eligible(tmp_path: Path) -> None:
    """A relevant hybrid match reaches generation and is retained as provenance."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    card = library.get("marcus-cooperation")
    assert card is not None
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply=(
                "You are trying to protect a boundary in a difficult team conflict. "
                "Marcus Aurelius's unverified lesson about cooperation can be an analogy for "
                "your situation, while keeping your boundary clear."
            ),
            used_card_ids=["synthetic-grounding"],
            used_role_model_id=card.id,
        ),
    )
    retriever = _hybrid_retriever(library, target_id=card.id)
    engine, session_id = _engine(
        tmp_path, gateway, library, role_model_retriever=retriever
    )

    output = engine.handle_turn(
        session_id,
        "I'm worried about a team conflict and want to protect a boundary. "
        "Could Marcus Aurelius offer a relevant lesson?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == card.id
    assert f"role_model_match: {card.id} score=" in " ".join(output.debug_steps)
    assert f"role_model: eligible {card.id}" in output.debug_steps
    assert output.historical_inspiration.source_locators == [
        source.locator for source in card.sources
    ]


def test_no_relevant_hybrid_match_keeps_normal_counselling_and_no_history(
    tmp_path: Path,
) -> None:
    """An explicit person request cannot force a card below the configured threshold."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    gateway = ScriptedGateway(
        _plan(move="reflect", need="understand"),
        ReplyOutput(
            reply="That sounds frustrating. What would feel most useful to focus on first?"
        ),
    )
    retriever = _hybrid_retriever(library, minimum_relevance=1.0)
    engine, session_id = _engine(
        tmp_path, gateway, library, role_model_retriever=retriever
    )

    output = engine.handle_turn(
        session_id,
        "I have a routine question about my grocery list. Could Marcus Aurelius help?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is None
    assert "role_model_match: below_threshold" in output.debug_steps
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "No sufficiently relevant historical card was selected" in reply_call[0]["content"]
    assert "Card id: marcus-cooperation" not in reply_call[-1]["content"]


def test_unreviewed_card_can_be_used_only_as_explicitly_unverified_lesson(
    tmp_path: Path,
) -> None:
    """Unreviewed evidence retains its status and cannot support asserted historical events."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    card = library.get("gandhi-pietermaritzburg")
    assert card is not None and not card.reviewed
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply=(
                "Gandhi's unverified card offers a lesson about choosing a constructive response "
                "after humiliation. That lesson can be an analogy for your situation without "
                "equating your experience with his."
            ),
            used_card_ids=["synthetic-grounding"],
            used_role_model_id=card.id,
        ),
    )
    retriever = _hybrid_retriever(library, target_id=card.id)
    engine, session_id = _engine(
        tmp_path, gateway, library, role_model_retriever=retriever
    )

    output = engine.handle_turn(
        session_id,
        "I was mocked unfairly and feel furious. What could help me respond?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == card.id
    assert card.reviewed is False
    assert card.verification.status == "needs_verification"
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "Review status: unreviewed" in reply_call[-1]["content"]
    assert "Verification status: needs_verification" in reply_call[-1]["content"]
    assert "UNVERIFIED" in reply_call[0]["content"]


def test_zero_counselling_cards_does_not_disable_historical_matching(tmp_path: Path) -> None:
    """Historical matching runs independently when counselling RAG returns no cards."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    card = library.get("kalam-empower-young-scientists")
    assert card is not None
    gateway = ScriptedGateway(
        _plan(),
        ReplyOutput(
            reply=(
                "You want your team to have meaningful ownership with support available. "
                "Kalam's lesson offers an analogy for your situation: room to contribute can "
                "coexist with experienced guidance."
            ),
            used_role_model_id=card.id,
        ),
    )
    retriever = _hybrid_retriever(library, target_id=card.id)
    engine, session_id = _engine(
        tmp_path,
        gateway,
        library,
        empty_retrieval=True,
        role_model_retriever=retriever,
    )

    output = engine.handle_turn(
        session_id,
        "I want my team to take more ownership while still having experienced support. "
        "Could Kalam offer a relevant lesson?",
        enable_role_models=True,
    )

    assert output.historical_inspiration is not None
    assert output.historical_inspiration.card_id == card.id
    assert "retrieval: 0 cards" in output.debug_steps
    assert f"role_model: eligible {card.id}" in output.debug_steps


def test_venting_and_safety_sensitive_turns_suppress_a_hybrid_candidate(
    tmp_path: Path,
) -> None:
    """Matching remains observable but venting/elevated policy blocks historical use."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    card = library.get("marcus-cooperation")
    assert card is not None

    for index, plan in enumerate(
        (_plan(need="vent", move="reflect"), _plan(safety_level="elevated"))
    ):
        gateway = ScriptedGateway(
            plan,
            ReplyOutput(reply="I hear how difficult this feels. We can take it slowly."),
        )
        retriever = _hybrid_retriever(library, target_id=card.id)
        engine, session_id = _engine(
            tmp_path / str(index),
            gateway,
            library,
            role_model_retriever=retriever,
        )
        output = engine.handle_turn(
            session_id,
            "I feel stressed about a team conflict and want to protect a boundary.",
            enable_role_models=True,
        )

        assert any(step.startswith("role_model_match:") for step in output.debug_steps)
        assert output.historical_inspiration is None
        reply_call = next(
            messages for task, messages in gateway.calls if task == TaskKind.converse
        )
        assert "Card id: marcus-cooperation" not in reply_call[-1]["content"]


def test_non_explicit_clarification_is_not_interrupted_by_a_historical_match(
    tmp_path: Path,
) -> None:
    """A retrieved card does not replace the planned clarifying question."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    card = library.get("marcus-cooperation")
    assert card is not None
    question = "What boundary feels most important to protect?"
    gateway = ScriptedGateway(
        {
            **_plan(move="ask_clarifying", phase="exploration", confidence=0.4),
            "retrieve": False,
            "clarifying_question": question,
        },
        ReplyOutput(reply=f"What feels difficult about this conflict? {question}"),
    )
    retriever = _hybrid_retriever(library, target_id=card.id)
    engine, session_id = _engine(
        tmp_path, gateway, library, role_model_retriever=retriever
    )
    engine._runtimes[session_id].phase = Phase.exploration

    output = engine.handle_turn(
        session_id,
        "I'm worried about a conflict and unsure about a boundary.",
        enable_role_models=True,
    )

    assert output.move == "ask_clarifying"
    assert question in output.reply
    assert output.historical_inspiration is None
    reply_call = next(messages for task, messages in gateway.calls if task == TaskKind.converse)
    assert "Card id: marcus-cooperation" not in reply_call[-1]["content"]


def test_use_limits_and_spacing_still_suppress_a_retrieved_candidate(
    tmp_path: Path,
) -> None:
    """The two-use cap and five-turn spacing still govern actual historical use."""
    library = RoleModelLibrary.load(CARD_DIR, include_unreviewed=True)
    card = library.get("marcus-cooperation")
    assert card is not None

    for index, use_turns in enumerate(([2, 3], [2])):
        gateway = ScriptedGateway(
            _plan(),
            ReplyOutput(
                reply="A practical step can be to name the boundary you want to keep.",
                used_card_ids=["synthetic-grounding"],
            ),
        )
        engine, session_id = _engine(
            tmp_path / f"limit-{index}",
            gateway,
            library,
            role_model_retriever=_hybrid_retriever(library, target_id=card.id),
        )
        runtime = engine._runtimes[session_id]
        runtime.turn_count = 5
        runtime.role_model_use_turns = use_turns
        output = engine.handle_turn(
            session_id,
            "I feel pressure in a team conflict and want to protect a boundary.",
            enable_role_models=True,
        )

        assert any(step.startswith("role_model_match:") for step in output.debug_steps)
        assert output.historical_inspiration is None
        assert "Card id: marcus-cooperation" not in next(
            messages for task, messages in gateway.calls if task == TaskKind.converse
        )[-1]["content"]
        assert gateway.reply_calls == 1
