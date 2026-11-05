"""Pipeline tests with a scripted gateway: no network, no model downloads."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.contracts import Phase, RetrievedItem, Role, SafetyLevel, TaskKind, UserNeed
from app.pipeline.critic import check_reply, fallback_reply
from app.pipeline.engine import SolaceEngine
from app.pipeline.guardrails import (
    MAX_CLARIFYING,
    SessionRuntime,
    enforce_plan,
    next_phase,
    render_state_summary,
    state_from_plan,
)
from app.pipeline.retrieval import CardLibrary, KeywordCardRetriever
from app.pipeline.types import EmotionGuess, PlannerOutput, ProfileUpdate, ReplyOutput
from app.privacy import install_privacy_filter, scrub_pii
from app.rolemodels.retriever import RoleModelRetrievalResult
from app.safety.responses import CRISIS_TEMPLATE, ELEVATED_CHECK_IN
from app.storage import repository as repo


class ScriptedGateway:
    """Returns queued planner/reply objects and records every call."""

    def __init__(self, plans=None, replies=None):
        self.plans = list(plans or [])
        self.replies = list(replies or [])
        self.calls: list[tuple[TaskKind, list[dict[str, str]]]] = []
        self.filters: list = []

    def add_filter(self, fn):
        self.filters.append(fn)

    def generate(self, task, messages, *, json_schema=None, temperature=0.7, max_tokens=800):
        self.calls.append((task, messages))
        queue = self.plans if json_schema is PlannerOutput else self.replies
        if not queue:
            raise RuntimeError("scripted gateway ran out of responses")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(parsed=item, text="", provider="fake", model="m", attempts=[])


class NoHistoricalMatchRetriever:
    """Keep general pipeline tests offline without loading embedding model weights."""

    def retrieve(
        self,
        current_user_message,
        *,
        recent_user_messages=(),
        conversation_state=None,
    ):
        """Return a deterministic no-match result for unrelated pipeline behavior."""
        return RoleModelRetrievalResult(None, (), "below_threshold", None)


def plan(**kw) -> PlannerOutput:
    base = {
        "emotions": [EmotionGuess(label="anxiety_fear", probability=0.6),
                     EmotionGuess(label="sadness", probability=0.4)],
        "intensity": 0.6,
        "need": UserNeed.unknown,
        "confidence": 0.3,
        "phase": Phase.exploration,
        "speaker": Role.counsellor,
        "reason": "test",
        "move": "ask_clarifying",
        "clarifying_question": "What has felt hardest this week?",
    }
    base.update(kw)
    return PlannerOutput(**base)


def reply(text="That sounds really heavy, and I'm glad you told me. What has felt hardest this week?", **kw):
    return ReplyOutput(reply=text, explanation="Responded warmly.", **kw)


@pytest.fixture()
def factory(tmp_path):
    engine = repo.build_engine(f"sqlite:///{tmp_path / 't.db'}")
    return repo.make_session_factory(engine)


@pytest.fixture(scope="module")
def library():
    return CardLibrary.load()


def make_engine(factory, gateway, library=None, retriever=None):
    return SolaceEngine(
        gateway,
        factory,
        library=library,
        retriever=retriever,
        role_model_retriever=NoHistoricalMatchRetriever(),
    )


def start(engine, user="u1"):
    return engine.start_session(user, "Sam").session_id


# ── Safety ────────────────────────────────────────────────────────────────────


def test_crisis_rules_skip_every_llm_call(factory):
    gw = ScriptedGateway()
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "thinking of ending it all tonight")
    assert out.safety_level == SafetyLevel.crisis
    assert out.reply == CRISIS_TEMPLATE
    assert out.speaker is None
    assert gw.calls == []


def test_planner_can_escalate_to_crisis_and_it_sticks(factory):
    gw = ScriptedGateway(plans=[plan(safety_level=SafetyLevel.crisis), plan(), ], replies=[reply()])
    engine = make_engine(factory, gw)
    sid = start(engine)
    first = engine.handle_turn(sid, "I just can't see a way forward anymore")
    assert first.reply == CRISIS_TEMPLATE
    second = engine.handle_turn(sid, "ok I'm a bit calmer")
    assert second.safety_level == SafetyLevel.elevated  # sticky escalation after a crisis


def test_elevated_forces_counsellor_no_retrieval_and_fixed_check_in(factory, library):
    gw = ScriptedGateway(
        plans=[plan(safety_level=SafetyLevel.elevated, speaker=Role.peer, move="offer_technique",
                    phase=Phase.guidance, retrieve=True, need=UserNeed.act, confidence=0.9)],
        replies=[reply("That sounds exhausting, and I'm really glad you shared it with me.")],
    )
    spy = SimpleNamespace(calls=0)

    class Retriever:
        def search_with_evidence(self, *a, **k):
            spy.calls += 1
            return []

    engine = make_engine(factory, gw, library, Retriever())
    out = engine.handle_turn(start(engine), "everyone would be better off without me")
    assert out.safety_level == SafetyLevel.elevated
    assert out.speaker == Role.counsellor
    assert out.reply.endswith(ELEVATED_CHECK_IN)
    assert spy.calls == 0
    assert out.move == "reflect"


# ── Ask before advising ───────────────────────────────────────────────────────


def test_first_turn_asks_one_question_without_advice(factory):
    gw = ScriptedGateway(plans=[plan()], replies=[reply()])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "I've been so stressed lately")
    assert out.move == "ask_clarifying"
    assert out.reply.count("?") == 1
    assert out.phase in (Phase.checkin, Phase.exploration)


def test_early_advice_is_coerced_into_a_question(factory, library):
    gw = ScriptedGateway(
        plans=[plan(move="offer_technique", retrieve=True, retrieve_query="breathing")],
        replies=[reply()],
    )
    spy = SimpleNamespace(calls=0)

    class Retriever:
        def search_with_evidence(self, *a, **k):
            spy.calls += 1
            return []

    engine = make_engine(factory, gw, library, Retriever())
    out = engine.handle_turn(start(engine), "I'm so anxious about exams")
    assert out.move == "ask_clarifying"
    assert spy.calls == 0
    assert "ask before advising" in out.explanation


def test_venting_user_is_listened_to(factory):
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.understanding, turn_count=3)
    p = plan(move="offer_technique", need=UserNeed.vent, phase=Phase.understanding, confidence=0.8)
    state = state_from_plan(p, None)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state)
    assert decision.move == "reflect"
    assert decision.retrieve is False


def test_question_budget_forces_a_summary(factory):
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=4,
                        questions_asked=MAX_CLARIFYING)
    p = plan()
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state_from_plan(p, None))
    assert decision.move == "summarise"


def test_high_confidence_moves_exploration_to_understanding():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=2)
    p = plan(confidence=0.8, trigger="exams", duration="2 weeks", functional_impact="poor sleep",
             need=UserNeed.understand)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state_from_plan(p, None))
    assert decision.phase == Phase.understanding
    assert decision.move == "summarise"


def test_phase_jumps_are_limited():
    rt = SessionRuntime(session_id="s", user_id="u")
    p = plan()
    assert next_phase(Phase.checkin, Phase.guidance, p, rt) == Phase.exploration
    assert next_phase(Phase.exploration, Phase.closing, p, rt) == Phase.closing
    act = plan(need=UserNeed.act, confidence=0.7)
    assert next_phase(Phase.exploration, Phase.guidance, act, rt) == Phase.guidance


def test_session_turn_limit_closes():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.guidance, turn_count=13)
    p = plan(move="reflect", phase=Phase.guidance)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state_from_plan(p, None))
    assert decision.move == "close" and decision.phase == Phase.closing


# ── Retrieval and grounding ───────────────────────────────────────────────────


def _guidance_runtime(engine, sid):
    rt = engine._runtimes[sid]
    rt.phase = Phase.understanding
    rt.turn_count = 3
    return rt


def test_technique_turn_uses_a_card_and_cites_it(factory, library):
    gw = ScriptedGateway(
        plans=[plan(move="offer_technique", phase=Phase.guidance, need=UserNeed.act, confidence=0.8,
                    retrieve=True, speaker=Role.teacher,
                    retrieve_query="anxiety fear breathing slow down panic")],
        replies=[reply("One simple thing is to slow your breathing down. Want to try it together?",
                       used_card_ids=["slow-breathing-for-stress"])],
    )
    engine = make_engine(factory, gw, library, KeywordCardRetriever(library))
    sid = start(engine)
    _guidance_runtime(engine, sid)
    out = engine.handle_turn(sid, "my chest gets tight before exams, what can I do")
    assert out.move == "offer_technique"
    assert [c.card_id for c in out.citations] == ["slow-breathing-for-stress"]
    assert "Problem Management Plus" in out.citations[0].sources[0]
    assert "Grounded in" in out.explanation


def test_no_relevant_card_means_listening_not_improvising(factory, library):
    gw = ScriptedGateway(
        plans=[plan(move="psychoeducation", phase=Phase.guidance, need=UserNeed.understand,
                    confidence=0.8, retrieve=True, retrieve_query="zzzz qqqq")],
        replies=[reply("That sounds like a lot to carry. What feels most important to talk about?")],
    )
    engine = make_engine(factory, gw, library, KeywordCardRetriever(library))
    sid = start(engine)
    _guidance_runtime(engine, sid)
    out = engine.handle_turn(sid, "explain it")
    assert out.move == "reflect"
    assert out.citations == []


def test_keyword_retriever_finds_cards_and_returns_nothing_for_small_talk(library):
    retriever = KeywordCardRetriever(library)
    hits = retriever.search_with_evidence("anxiety fear breathing slow down panic", emotions=["anxiety_fear"])
    assert hits and hits[0].doc_id.startswith("slow-breathing")
    assert retriever.search_with_evidence("hello thanks bye", emotions=["neutral"]) == []


# ── Critic ────────────────────────────────────────────────────────────────────


GOOD = "That sounds really hard, and it makes sense that you feel this way. What feels heaviest right now?"


def test_good_reply_passes():
    verdict, issues = check_reply(GOOD, role=Role.counsellor, move="ask_clarifying",
                                  retrieved_ids=[], used_card_ids=[])
    assert verdict.passed and not issues


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        ("It sounds like you have depression, and that is hard for anyone. What is going on now?", "diagnosis"),
        ("You could ask about medication for this. What do you think about that idea?", "medication"),
        ("Call 98765 43210 for help right away. Is that something you could do today?", "phone number"),
        ("Please just relax, everything will be fine. What is on your mind today?", "minimising"),
        ("When I was your age I felt the same way. What is on your mind today?", "lived experience"),
        ("That is tough. Why now? What happened at home? Who did you talk to about it?", "more than one question"),
    ],
)
def test_critic_catches_unsafe_or_off_role_replies(bad, fragment):
    verdict, issues = check_reply(bad, role=Role.counsellor, move="ask_clarifying",
                                  retrieved_ids=[], used_card_ids=[])
    assert not verdict.passed
    assert any(fragment in i for i in issues)


def test_critic_requires_card_grounding_for_techniques():
    text = "A slow breath can help you settle. Would you like to try it together right now?"
    verdict, _ = check_reply(text, role=Role.teacher, move="offer_technique",
                             retrieved_ids=["a"], used_card_ids=[])
    assert not verdict.grounded
    verdict, _ = check_reply(text, role=Role.teacher, move="offer_technique",
                             retrieved_ids=["a"], used_card_ids=["a"])
    assert verdict.passed


def test_peer_cannot_use_clinical_vocabulary_and_counsellor_cannot_direct():
    v, _ = check_reply("Cognitive therapy could help with this. Want to try it with me?",
                       role=Role.peer, move="reflect", retrieved_ids=[], used_card_ids=[])
    assert not v.role_fidelity
    v, _ = check_reply("You should talk to your parents about this. Does that feel possible to you?",
                       role=Role.counsellor, move="reflect", retrieved_ids=[], used_card_ids=[])
    assert not v.role_fidelity


def test_fallback_replies_pass_their_own_checks():
    for move, q in [("reflect", None), ("summarise", None), ("close", None),
                    ("ask_clarifying", "What has felt hardest this week?")]:
        text = fallback_reply(move, q)
        verdict, issues = check_reply(text, role=Role.counsellor, move=move,
                                      retrieved_ids=[], used_card_ids=[])
        assert verdict.passed, (move, issues)


def test_engine_regenerates_once_after_a_failed_check(factory):
    gw = ScriptedGateway(plans=[plan()], replies=[
        reply("It sounds like you have depression, which is hard. What is on your mind?"),
        reply(GOOD),
    ])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "I can't get out of bed")
    assert out.reply == GOOD
    assert len([c for c in gw.calls if c[0] == TaskKind.converse]) == 2
    assert not out.degraded


def test_engine_falls_back_to_a_safe_template_after_two_bad_drafts(factory):
    bad = reply("You have depression. Why? When did it start? Who knows?")
    gw = ScriptedGateway(plans=[plan()], replies=[bad, bad])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "I can't get out of bed")
    assert "What has felt hardest this week?" in out.reply
    assert out.degraded


def test_planner_failure_degrades_gracefully(factory):
    gw = ScriptedGateway(plans=[RuntimeError("boom")], replies=[reply()])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "hello, I'm not doing great")
    assert out.degraded
    assert out.move in ("ask_clarifying", "reflect")


def test_planner_fallback_updates_state_and_safely_records_failure(factory):
    """A planner exception uses heuristic state and omits the raw user message from diagnostics."""
    user_text = "I've been stressed about final exams"
    gw = ScriptedGateway(
        plans=[RuntimeError(f"{user_text} {'x' * 180}")],
        replies=[reply()],
    )
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), user_text)
    out = engine.handle_turn(start(engine), user_text)

    assert out.degraded
    assert out.state is not None
    assert out.state.hypotheses[0].label == "anxiety_fear"
    assert out.state.trigger == "exams"
    assert "Rule-based planning" in out.explanation
    assert not any(user_text in step for step in out.debug_steps)
    diagnostic = next(step for step in out.debug_steps if step.startswith("planner:"))
    assert diagnostic.startswith("planner: failed (RuntimeError: ")
    exception_message = diagnostic.split("RuntimeError: ", maxsplit=1)[1].rsplit(
        "): heuristic plan", maxsplit=1
    )[0]
    assert len(exception_message) <= 140


def test_nine_message_demo_completes_when_every_planner_call_fails(factory, library):
    """The checked-in synthetic demo script completes through the rule-based planner."""
    script = [
        line.strip()
        for line in Path("docs/demo_script.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    replies = [
        reply("The pressure of final exams sounds heavy. How long has this been going on?"),
        reply(
            "The lack of sleep sounds hard. What would help most right now: talking it through, "
            "understanding it, or finding something practical?"
        ),
        reply("That's a lot to carry with everyone expecting so much. I'm here; say more if you like."),
        reply(
            "Exam stress can keep your body on alert, which can make sleep harder. This is a common "
            "response to pressure. Does that fit your experience?",
            used_card_ids=["stress-versus-anxiety"],
        ),
        reply("I'm glad that feels close to your experience. We can take this at your pace."),
        reply(
            "One small option is to breathe out a little more slowly than you breathe in. "
            "Would you like to try it together?",
            used_card_ids=["slow-breathing-for-stress"],
        ),
        reply(
            "Let's take it gently: notice your breath, then let the next exhale be a little slower. "
            "How does that feel?",
            used_card_ids=["slow-breathing-for-stress"],
        ),
        reply("I'm glad it helped a bit. We can leave it there and you can return to it when useful."),
        reply(
            "You did something meaningful by talking this through. You might keep the slower exhale "
            "as an option, and you're welcome back anytime."
        ),
    ]
    gateway = ScriptedGateway(
        plans=[RuntimeError("synthetic planner outage") for _ in script],
        replies=replies,
    )

    class ScriptRetriever:
        """Return the reviewed card appropriate for each state-derived query."""

        def search_with_evidence(self, query, *, emotions=None, role=None, k=3):
            card_id = (
                "stress-versus-anxiety"
                if "understand" in query
                else "slow-breathing-for-stress"
            )
            card = library.get(card_id)
            assert card is not None
            return [
                RetrievedItem(
                    collection="cards",
                    doc_id=card.id,
                    source=", ".join(source.source_id for source in card.sources),
                    text=f"{card.title}: {card.summary}",
                    score=1.0,
                )
            ]

    engine = make_engine(factory, gateway, library, ScriptRetriever())
    session_id = start(engine, "demo-user")
    outputs = [engine.handle_turn(session_id, message) for message in script]
    moves = [output.move for output in outputs]

    assert len(outputs) == 9
    assert all(output.degraded for output in outputs)
    assert any(output.phase != Phase.checkin for output in outputs)
    assert moves == [
        "ask_clarifying",
        "reflect",
        "reflect",
        "psychoeducation",
        "reflect",
        "offer_technique",
        "follow_up",
        "reflect",
        "close",
    ]
    assert [output.phase for output in outputs] == [
        Phase.checkin,
        Phase.exploration,
        Phase.understanding,
        Phase.understanding,
        Phase.understanding,
        Phase.guidance,
        Phase.guidance,
        Phase.guidance,
        Phase.closing,
    ]
    assert {"ask_clarifying", "psychoeducation", "offer_technique", "follow_up", "close"} <= set(
        moves
    )
    assert outputs[0].speaker == Role.counsellor
    assert outputs[-1].phase == Phase.closing
    assert outputs[3].citations
    assert outputs[5].citations
    assert outputs[6].citations
    assert outputs[-1].state is not None
    assert outputs[-1].state.hypotheses[0].label == "anxiety_fear"


def test_total_llm_failure_still_returns_a_safe_reply(factory):
    gw = ScriptedGateway(plans=[RuntimeError("x")], replies=[RuntimeError("y")])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "hello, I'm not doing great")
    assert out.reply and out.degraded and out.provider_used == "fallback"


# ── Roles, handoffs, state ────────────────────────────────────────────────────


def test_handoff_line_and_throttle(factory):
    gw = ScriptedGateway(
        plans=[plan(speaker=Role.counsellor, phase=Phase.understanding, move="reflect", need=UserNeed.vent),
               plan(speaker=Role.mentor, phase=Phase.understanding, move="reflect", need=UserNeed.vent),
               plan(speaker=Role.teacher, phase=Phase.understanding, move="reflect", need=UserNeed.vent)],
        replies=[reply("That sounds heavy, and it makes sense. I'm here, say more if you like."),
                 reply("Carrying that much is tiring. I'm here, say more if you like."),
                 reply("Expectations can pile up fast. I'm here, say more if you like.")],
    )
    engine = make_engine(factory, gw)
    sid = start(engine)
    engine._runtimes[sid].phase = Phase.understanding
    first = engine.handle_turn(sid, "I'm overwhelmed with college")
    assert first.handoff is None and first.speaker_name == "Anika"
    second = engine.handle_turn(sid, "mostly the deadlines")
    assert second.speaker == Role.mentor and "Rohan" in (second.handoff or "")
    third = engine.handle_turn(sid, "and my parents' expectations")
    assert third.speaker == Role.mentor and third.handoff is None  # throttled


def test_state_persists_across_turns_and_summary_is_readable(factory):
    gw = ScriptedGateway(
        plans=[plan(trigger="final exams", duration="two weeks"),
               plan(trigger=None, duration=None, functional_impact="not sleeping")],
        replies=[reply(), reply()],
    )
    engine = make_engine(factory, gw)
    sid = start(engine)
    engine.handle_turn(sid, "exams are crushing me")
    out = engine.handle_turn(sid, "I can't sleep either")
    assert out.state.trigger == "final exams" and out.state.functional_impact == "not sleeping"
    assert "final exams" in out.state_summary and "anxiety or fear" in out.state_summary
    assert "correct me" in render_state_summary(out.state)


# ── Memory, storage, privacy ──────────────────────────────────────────────────


def test_profile_memory_and_resume_note(factory):
    gw = ScriptedGateway(
        plans=[plan(trigger="exam pressure",
                    profile_update=ProfileUpdate(preferred_role=Role.mentor, themes=["Exams"]))],
        replies=[reply()],
    )
    engine = make_engine(factory, gw)
    sid = start(engine)
    engine.handle_turn(sid, "exam pressure is huge, I'd prefer to talk to the mentor")
    engine.end_session(sid)
    again = engine.start_session("u1", "Sam")
    assert again.resume_note == "Last time we spoke about exam pressure."
    assert "Last time we spoke about exam pressure." in again.greeting
    profile = repo.get_profile("u1", factory=factory)
    assert profile.preferred_role == Role.mentor and "exams" in profile.recurring_themes


def test_greeting_discloses_ai_and_limits(factory):
    engine = make_engine(factory, ScriptedGateway())
    greeting = engine.start_session("u2").greeting
    assert "AI companion" in greeting and "not a therapist" in greeting


def test_turn_is_traced_and_persisted(factory):
    gw = ScriptedGateway(plans=[plan()], replies=[reply()])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "I feel stuck")
    trace = repo.get_trace(out.turn_id, factory=factory)
    assert trace is not None and trace.decision is not None
    assert [s.name for s in trace.steps][:2] == ["safety", "planner"]
    assert trace.critic is not None and trace.critic.passed


def test_delete_user_data_erases_everything(factory):
    gw = ScriptedGateway(plans=[plan()], replies=[reply()])
    engine = make_engine(factory, gw)
    sid = start(engine)
    engine.handle_turn(sid, "I feel stuck")
    engine.delete_user_data("u1")
    assert repo.get_user("u1", factory=factory) is None
    assert sid not in engine._runtimes


def test_empty_message_is_rejected(factory):
    engine = make_engine(factory, ScriptedGateway())
    with pytest.raises(ValueError):
        engine.handle_turn(start(engine), "   ")


def test_scrub_pii_and_gateway_filter():
    text = "call me on +91 98765 43210, mail a.b@c.com or see https://x.org/y, id ABCDE1234F"
    cleaned = scrub_pii(text)
    for secret in ("98765", "a.b@c.com", "https://", "ABCDE1234F"):
        assert secret not in cleaned
    gw = ScriptedGateway()
    install_privacy_filter(gw)
    assert len(gw.filters) == 1
    assert "[email]" in gw.filters[0]([{"role": "user", "content": "me@x.com"}])[0]["content"]


# ── Real gateway parsing path with sloppy model output ───────────────────────


def test_planner_schema_tolerates_sloppy_model_output():
    messy = {
        "safety_level": "None", "emotions": [{"label": "Anxiety", "probability": 70},
                                             {"label": "Stress", "probability": 20},
                                             {"label": "banana", "probability": 10}],
        "intensity": 60, "trigger": None, "need": "ACT", "confidence": 0.5, "open_questions": "how long?",
        "phase": "Exploration", "speaker": "Mentor", "move": "Ask_Clarifying",
        "clarifying_question": "How long has this been going on?", "retrieve": None,
        "profile_update": {"preferred_role": None, "themes": ["exams"], "values_note": None},
    }
    parsed = PlannerOutput.model_validate(messy)
    assert parsed.speaker == Role.mentor and parsed.need == UserNeed.act and parsed.intensity == 0.6
    assert [e.label for e in parsed.emotions] == ["anxiety_fear"]
    assert parsed.open_questions == ["how long?"] and parsed.retrieve is False


def test_engine_works_through_the_real_gateway_class(factory):
    import json

    from app.llm.gateway import LLMGateway

    plan_json = json.dumps({"emotions": [{"label": "sadness", "probability": 0.8}], "move": "reflect",
                            "speaker": "peer", "phase": "exploration", "confidence": 0.3, "need": "vent"})
    reply_json = json.dumps({"reply": "That sounds so heavy, and it makes sense you feel worn down. What has been weighing on you most?",
                             "explanation": "Validated the feeling.", "used_card_ids": []})
    gw = LLMGateway.from_mock(fixed_output=plan_json)
    engine = SolaceEngine(
        gw, factory, role_model_retriever=NoHistoricalMatchRetriever()
    )
    sid = start(engine)
    out = engine.handle_turn(sid, "I just feel so low")  # reply call also returns plan JSON -> falls back safely
    assert out.degraded and out.reply
    gw2 = LLMGateway.from_mock(fixed_output=reply_json)
    engine2 = SolaceEngine(
        gw2, factory, role_model_retriever=NoHistoricalMatchRetriever()
    )
    out2 = engine2.handle_turn(start(engine2, "u9"), "I just feel so low")
    assert out2.degraded  # planner JSON missing -> safe plan, but the reply parsed through the real gateway
    assert "weighing on you most" in out2.reply


# ── Fixes found in the first real conversation ───────────────────────────────


def test_a_thank_you_does_not_erase_the_feelings_so_far():
    first = state_from_plan(plan(emotions=[EmotionGuess(label="anxiety_fear", probability=0.9),
                                           EmotionGuess(label="sadness", probability=0.1)]), None)
    thanks = plan(emotions=[EmotionGuess(label="positive", probability=0.6),
                            EmotionGuess(label="anxiety_fear", probability=0.4)])
    second = state_from_plan(thanks, first)
    assert second.hypotheses[0].label == "anxiety_fear"


def test_unclear_picture_turns_a_bare_reflection_into_one_question():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=1)
    p = plan(move="reflect", confidence=0.3)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state_from_plan(p, None))
    assert decision.move == "ask_clarifying" and decision.question


def test_no_two_questions_in_a_row_and_no_questions_for_someone_venting():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=2,
                        last_move="ask_clarifying")
    p = plan(move="reflect", confidence=0.3)
    assert enforce_plan(p, rt, safety_level=SafetyLevel.none,
                        state=state_from_plan(p, None)).move == "reflect"
    rt.last_move = None
    vent = plan(move="reflect", confidence=0.3, need=UserNeed.vent)
    assert enforce_plan(vent, rt, safety_level=SafetyLevel.none,
                        state=state_from_plan(vent, None)).move == "reflect"


def test_entering_understanding_summarises_before_any_advice():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=2)
    p = plan(move="reflect", confidence=0.7, phase=Phase.understanding, trigger="exams",
             duration="2 weeks", functional_impact="poor sleep", need=UserNeed.understand)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state_from_plan(p, None))
    assert decision.move == "summarise" and decision.phase == Phase.understanding


def test_a_second_technique_right_after_one_becomes_a_follow_up_on_the_same_card(factory, library):
    gw = ScriptedGateway(
        plans=[plan(move="offer_technique", phase=Phase.guidance, need=UserNeed.act, confidence=0.8,
                    retrieve=True, speaker=Role.mentor, wants_new_help=False)],
        replies=[reply("Glad that helped. How did the slower breathing feel for you?",
                       used_card_ids=["slow-breathing-grounding"])],
    )
    engine = make_engine(factory, gw, library, KeywordCardRetriever(library))
    sid = start(engine)
    rt = _guidance_runtime(engine, sid)
    rt.last_move = "offer_technique"
    rt.techniques_offered = ["slow-breathing-grounding"]
    rt.last_speaker = Role.mentor
    out = engine.handle_turn(sid, "thanks, that helps")
    assert out.move == "follow_up"
    assert [c.card_id for c in out.citations] == ["slow-breathing-grounding"]


def test_asking_for_something_new_allows_a_new_technique():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.guidance, turn_count=5,
                        last_move="offer_technique")
    p = plan(move="offer_technique", phase=Phase.guidance, need=UserNeed.act, confidence=0.8,
             retrieve=True, wants_new_help=True)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state_from_plan(p, None))
    assert decision.move == "offer_technique" and decision.retrieve


def test_critic_flags_a_repeated_opening_and_long_step_lists():
    prev = "It sounds like a lot of pressure is on you right now. What feels heaviest?"
    now = "It sounds like the sleep loss is making everything harder. What has helped before?"
    verdict, issues = check_reply(now, role=Role.counsellor, move="ask_clarifying",
                                  retrieved_ids=[], used_card_ids=[], previous_reply=prev)
    assert not verdict.passed and any("vary the opening" in i for i in issues)
    steps = "Here is one way:\n1. Notice.\n2. Breathe out.\n3. Pause.\n4. Look around.\nWant to try it?"
    verdict, issues = check_reply(steps, role=Role.mentor, move="offer_technique",
                                  retrieved_ids=["a"], used_card_ids=["a"])
    assert not verdict.role_fidelity and any("list of 4" in i for i in issues)


# ── Fixes found in the second real conversation ──────────────────────────────


def test_confidence_cannot_exceed_the_evidence():
    first = state_from_plan(plan(confidence=0.9, trigger="exams"), None)
    assert first.confidence <= 0.35
    full = state_from_plan(plan(confidence=0.95, trigger="exams", duration="2 weeks",
                                functional_impact="no sleep", need=UserNeed.act), None)
    assert full.confidence == pytest.approx(0.8)


def test_positive_emotion_is_smoothed_with_a_lower_alpha():
    """A positive model reading does not displace a prior anxiety hypothesis."""
    first = state_from_plan(
        plan(
            emotions=[
                EmotionGuess(label="anxiety_fear", probability=0.9),
                EmotionGuess(label="sadness", probability=0.1),
            ]
        ),
        None,
    )
    positive = plan(
        emotions=[EmotionGuess(label="positive", probability=1.0)],
        trigger=None,
        duration=None,
        functional_impact=None,
        need=UserNeed.unknown,
    )

    state = state_from_plan(positive, first)

    assert state.hypotheses[0].label == "anxiety_fear"
    assert next(item for item in state.hypotheses if item.label == "positive").probability == 0.2


@pytest.mark.parametrize("move", ["offer_technique", "psychoeducation"])
def test_advice_after_follow_up_requires_an_explicit_new_request(move):
    """A fresh advice move after follow-up needs a new request for help."""
    runtime = SessionRuntime(
        session_id="s",
        user_id="u",
        phase=Phase.guidance,
        turn_count=6,
        last_move="follow_up",
    )
    requested = plan(
        move=move,
        phase=Phase.guidance,
        need=UserNeed.act,
        confidence=0.8,
    )
    state = state_from_plan(requested, None)

    decision = enforce_plan(
        requested,
        runtime,
        safety_level=SafetyLevel.none,
        state=state,
    )
    assert decision.move == "reflect"

    new_help = requested.model_copy(update={"wants_new_help": True})
    decision = enforce_plan(
        new_help,
        runtime,
        safety_level=SafetyLevel.none,
        state=state,
    )
    assert decision.move == move


def test_asks_what_would_help_before_summarising_when_need_is_unknown():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=2,
                        questions_asked=1, last_move="reflect")
    p = plan(move="reflect", phase=Phase.understanding, confidence=0.9, trigger="exams",
             duration="2 weeks", functional_impact="no sleep", need=UserNeed.unknown)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state_from_plan(p, None))
    assert decision.phase == Phase.exploration
    assert decision.move == "ask_clarifying" and "talk it through" in (decision.question or "")


def test_exploration_is_led_by_the_counsellor_unless_the_user_chose_someone():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=1)
    p = plan(speaker=Role.mentor, move="reflect", need=UserNeed.vent)
    state = state_from_plan(p, None)
    assert enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state).speaker == Role.counsellor
    rt.preferred_role = Role.mentor
    assert enforce_plan(p, rt, safety_level=SafetyLevel.none, state=state).speaker == Role.mentor


def test_elevated_turns_are_never_turned_into_narrowing_questions():
    rt = SessionRuntime(session_id="s", user_id="u", phase=Phase.exploration, turn_count=1)
    p = plan(move="reflect", confidence=0.2)
    decision = enforce_plan(p, rt, safety_level=SafetyLevel.elevated, state=state_from_plan(p, None))
    assert decision.move == "reflect"


def test_role_preference_needs_an_explicit_statement():
    from app.pipeline.memory import explicit_role_preference

    assert explicit_role_preference("can I talk to the mentor instead") == Role.mentor
    assert explicit_role_preference("I'd rather have a friend to talk to") == Role.peer
    assert explicit_role_preference("can you give me something practical to try?") is None
    assert explicit_role_preference("my mentor at college is great") is None


def test_a_technique_request_is_never_answered_with_an_explainer_card(factory, library):
    class MixedRetriever:
        def search_with_evidence(self, query, *, emotions=None, role=None, k=3):
            from app.contracts import RetrievedItem

            return [RetrievedItem(collection="cards", doc_id=cid, source="x", text=cid, score=0.9)
                    for cid in ("stress-versus-anxiety", "slow-breathing-for-stress")]

    gw = ScriptedGateway(
        plans=[plan(move="offer_technique", phase=Phase.guidance, need=UserNeed.act, confidence=0.8,
                    trigger="exams", duration="2 weeks", functional_impact="no sleep", retrieve=True,
                    speaker=Role.mentor)],
        replies=[reply("A slow breath can steady you. Want to try it together right now?",
                       used_card_ids=["slow-breathing-for-stress"])],
    )
    engine = make_engine(factory, gw, library, MixedRetriever())
    sid = start(engine)
    _guidance_runtime(engine, sid)
    out = engine.handle_turn(sid, "can you give me something practical to try?")
    assert [c.card_id for c in out.citations] == ["slow-breathing-for-stress"]


def test_follow_up_fallback_fits_a_thank_you_and_debug_steps_are_exposed(factory):
    verdict, issues = check_reply(fallback_reply("follow_up"), role=Role.mentor, move="follow_up",
                                  retrieved_ids=[], used_card_ids=[])
    assert verdict.passed, issues
    assert "hard" not in fallback_reply("follow_up")
    gw = ScriptedGateway(plans=[plan()], replies=[reply()])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "I feel stuck")
    assert out.debug_steps and out.debug_steps[0].startswith("safety")


# ── Record/replay cache ──────────────────────────────────────────────────────


class _CountingGateway:
    def __init__(self):
        self.calls = 0

    def generate(self, task, messages, *, json_schema=None, temperature=0.7, max_tokens=800, **kw):
        self.calls += 1
        return SimpleNamespace(parsed=reply("Recorded answer. What feels heaviest right now?"),
                               text="", provider="live", model="m", attempts=[])


def _msgs(text="hello"):
    return [{"role": "user", "content": text}]


def test_cache_records_then_replays_without_calling_the_model(tmp_path):
    from app.pipeline.cache import CachingGateway

    path = tmp_path / "c.json"
    live = _CountingGateway()
    rec = CachingGateway(live, path, "record")
    first = rec.generate(TaskKind.converse, _msgs(), json_schema=ReplyOutput, temperature=0.6)
    assert live.calls == 1 and first.parsed.reply.startswith("Recorded answer")
    replay = CachingGateway(live, path, "replay")
    second = replay.generate(TaskKind.converse, _msgs(), json_schema=ReplyOutput, temperature=0.6)
    assert live.calls == 1  # no new live call
    assert second.parsed == first.parsed and second.provider == "cache:live"


def test_replay_miss_raises_and_auto_fills_the_cache(tmp_path):
    from app.pipeline.cache import CachingGateway

    path = tmp_path / "c.json"
    live = _CountingGateway()
    with pytest.raises(LookupError):
        CachingGateway(live, path, "replay").generate(TaskKind.converse, _msgs(), json_schema=ReplyOutput)
    auto = CachingGateway(live, path, "auto")
    auto.generate(TaskKind.converse, _msgs("a"), json_schema=ReplyOutput)
    auto.generate(TaskKind.converse, _msgs("a"), json_schema=ReplyOutput)
    auto.generate(TaskKind.converse, _msgs("b"), json_schema=ReplyOutput)
    assert live.calls == 2 and auto.hits == 1  # one repeat hit, two distinct prompts


def test_cache_key_changes_with_the_prompt_and_off_mode_passes_through(tmp_path):
    from app.pipeline.cache import CachingGateway

    live = _CountingGateway()
    off = CachingGateway(live, tmp_path / "c.json", "off")
    off.generate(TaskKind.converse, _msgs("a"), json_schema=ReplyOutput)
    off.generate(TaskKind.converse, _msgs("a"), json_schema=ReplyOutput)
    assert live.calls == 2 and not (tmp_path / "c.json").exists()
    with pytest.raises(ValueError):
        CachingGateway(live, tmp_path / "x.json", "bogus")


def test_a_replayed_conversation_works_end_to_end_with_no_live_model(factory, tmp_path):
    from app.pipeline.cache import CachingGateway

    path = tmp_path / "c.json"
    script = lambda: ScriptedGateway(plans=[plan()], replies=[reply()])  # noqa: E731
    recorded = SolaceEngine(
        CachingGateway(script(), path, "record"),
        factory,
        role_model_retriever=NoHistoricalMatchRetriever(),
    )
    out1 = recorded.handle_turn(start(recorded), "I've been stressed about exams")
    replayed = SolaceEngine(
        CachingGateway(ScriptedGateway(), path, "replay"),
        factory,
        role_model_retriever=NoHistoricalMatchRetriever(),
    )
    out2 = replayed.handle_turn(start(replayed, "u7"), "I've been stressed about exams")
    assert out2.reply == out1.reply and not out2.degraded
    assert out2.provider_used.startswith("cache:")


def test_failure_details_are_visible_in_debug_steps(factory):
    gw = ScriptedGateway(plans=[RuntimeError("503 UNAVAILABLE high demand")], replies=[reply()])
    engine = make_engine(factory, gw)
    out = engine.handle_turn(start(engine), "I feel stuck")
    assert any("RuntimeError" in s and "503" in s for s in out.debug_steps)
