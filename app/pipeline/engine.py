"""SolaceEngine: one call per user turn, safety first, structured session, local critic."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from app.contracts import (
    ConductorDecision,
    CriticVerdict,
    EmotionalState,
    EmotionReading,
    Message,
    RetrievedItem,
    Role,
    SafetyLevel,
    SafetyResult,
    StepRecord,
    TaskKind,
    TurnTrace,
)
from app.pipeline import memory
from app.pipeline.critic import check_reply, fallback_reply
from app.pipeline.guardrails import (
    Decision,
    SessionRuntime,
    enforce_plan,
    render_state_summary,
    state_from_plan,
)
from app.pipeline.heuristic import heuristic_plan
from app.pipeline.prompts import build_planner_messages, build_reply_messages
from app.pipeline.retrieval import CardLibrary, RetrieverLike, citations_for
from app.pipeline.roles import ROLES, handoff_line
from app.pipeline.types import PlannerOutput, ReplyOutput, SessionInfo, TurnOutput
from app.rolemodels.critic import check_role_model_reply, has_historical_reference
from app.rolemodels.library import RoleModelLibrary
from app.rolemodels.models import PERSON_DISPLAY_NAMES, RoleModelCard, RoleModelProvenance
from app.rolemodels.retriever import RoleModelRetrievalResult, RoleModelRetriever
from app.rolemodels.selector import requested_person, role_model_allowed
from app.safety.responses import CRISIS_TEMPLATE, ELEVATED_CHECK_IN, action_for
from app.safety.sentinel import SafetyGate
from app.storage import repository as repo

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
_RANK = {SafetyLevel.none: 0, SafetyLevel.elevated: 1, SafetyLevel.crisis: 2}
MAX_INPUT_CHARS = 2000


class GatewayLike(Protocol):
    """The slice of LLMGateway the engine uses (lets tests inject a scripted fake)."""

    def generate(
        self,
        task: TaskKind,
        messages: list[dict[str, str]],
        *,
        json_schema: type[T] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Any:
        """Return an object with .parsed, .text, .provider, .model."""


class EmotionReader(Protocol):
    """EmotionRecognizer-compatible reader (optional)."""

    def read(self, message: str) -> EmotionReading:
        """Return an emotion reading."""


class HistoricalRetriever(Protocol):
    """The historical-retrieval interface used independently of counselling RAG."""

    def retrieve(
        self,
        current_user_message: str,
        *,
        recent_user_messages: Sequence[str] = (),
        conversation_state: EmotionalState | None = None,
    ) -> RoleModelRetrievalResult:
        """Return the best sufficiently relevant historical card, if any."""


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _parse(result: Any, schema: type[T]) -> T:
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    if isinstance(parsed, dict):
        return schema.model_validate(parsed)
    return schema.model_validate_json(getattr(result, "text", "") or "{}")


class SolaceEngine:
    """Runs the per-turn pipeline: safety, plan, guardrails, retrieval, reply, critic, memory."""

    def __init__(
        self,
        gateway: GatewayLike,
        factory: sessionmaker,
        *,
        library: CardLibrary | None = None,
        retriever: RetrieverLike | None = None,
        emotion_reader: EmotionReader | None = None,
        safety_gate: SafetyGate | None = None,
        role_model_library: RoleModelLibrary | None = None,
        role_model_retriever: HistoricalRetriever | None = None,
    ) -> None:
        self._gateway = gateway
        self._factory = factory
        self._library = library or CardLibrary({})
        self._retriever = retriever
        self._emotion = emotion_reader
        self._gate = safety_gate or SafetyGate(classifier=None)
        self._role_model_retriever = (
            role_model_retriever
            if role_model_retriever is not None
            else RoleModelRetriever(library=role_model_library)
            if role_model_library is not None
            else RoleModelRetriever()
        )
        self._runtimes: dict[str, SessionRuntime] = {}

    # ── Sessions ──────────────────────────────────────────────────────────────

    def start_session(self, user_id: str, display_name: str | None = None) -> SessionInfo:
        """Create (or resume) a user and open a new session with a greeting."""
        if repo.get_user(user_id, factory=self._factory) is None:
            repo.create_user(user_id, display_name, factory=self._factory)
        session_id = uuid.uuid4().hex[:12]
        repo.create_session(session_id, user_id, factory=self._factory)
        profile = repo.get_profile(user_id, factory=self._factory)
        note = memory.resume_note(profile)
        rt = SessionRuntime(
            session_id=session_id,
            user_id=user_id,
            preferred_role=profile.preferred_role if profile else None,
            themes=list(profile.recurring_themes) if profile else [],
            resume_note=note,
        )
        text = memory.greeting(display_name or (profile.display_name if profile else None), note)
        hello = Message(role="assistant", content=text, speaker_role=Role.counsellor)
        rt.history.append(hello)
        repo.append_message(session_id, hello, factory=self._factory)
        self._runtimes[session_id] = rt
        return SessionInfo(session_id=session_id, user_id=user_id, greeting=text, resume_note=note)

    def end_session(self, session_id: str) -> dict[str, Any]:
        """Store a deterministic summary for the next session and forget the runtime."""
        rt = self._runtimes.pop(session_id)
        summary = memory.session_summary(rt)
        profile = repo.get_profile(rt.user_id, factory=self._factory)
        base = memory.merge_profile(
            profile, user_id=rt.user_id, display_name=None, update=_empty_update()
        )
        prefs = dict(base.communication_prefs)
        prefs["last_session"] = summary
        repo.save_profile(base.model_copy(update={"communication_prefs": prefs}), factory=self._factory)
        return summary

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Erase everything stored about a user (privacy requirement)."""
        for sid in [s for s, r in self._runtimes.items() if r.user_id == user_id]:
            self._runtimes.pop(sid)
        return repo.delete_user_data(user_id, factory=self._factory)

    # ── One turn ──────────────────────────────────────────────────────────────

    def handle_turn(
        self,
        session_id: str,
        user_text: str,
        *,
        enable_role_models: bool = False,
    ) -> TurnOutput:
        """Process one user message and return the reply plus the panels' contents."""
        started = time.perf_counter()
        text = user_text.strip()[:MAX_INPUT_CHARS]
        if not text:
            raise ValueError("Empty message.")
        rt = self._runtimes[session_id]
        turn_id = uuid.uuid4().hex[:12]
        steps: list[StepRecord] = []

        t0 = time.perf_counter()
        gate = self._gate.check(text, rt.history, rt.gate_state)
        steps.append(StepRecord(name="safety", summary=f"rules: {gate.level.value}", latency_ms=_ms(t0)))
        if gate.level == SafetyLevel.crisis:
            return self._finish_crisis(rt, text, turn_id, gate, steps, started)

        emotion = self._read_emotion(text, steps)
        plan, _, planner_ok = self._plan(rt, text, emotion, steps)
        safety = self._merge_safety(gate, plan, planner_ok, rt)
        if safety.level == SafetyLevel.crisis:
            return self._finish_crisis(rt, text, turn_id, safety, steps, started, emotion=emotion)

        state_before = rt.state
        state = state_from_plan(plan, state_before)
        decision = enforce_plan(plan, rt, safety_level=safety.level, state=state)

        explicit_role_model_request = requested_person(text) is not None
        role_model_candidate = self._match_role_model(text, state, rt, steps)
        role_model_eligibility_move = decision.move
        items, cards = self._retrieve(decision, state, steps, rt)
        if (decision.retrieve or decision.move == "follow_up") and not cards:
            decision = _listen_only(decision)

        role_model: RoleModelCard | None = None
        if enable_role_models and role_model_candidate is not None:
            allowed = role_model_allowed(
                safety_level=safety.level,
                crisis_seen=bool(rt.gate_state.get("crisis_seen")),
                phase=decision.phase,
                move=(
                    role_model_eligibility_move
                    if explicit_role_model_request
                    else decision.move
                ),
                need=state.need,
                explicit_request=explicit_role_model_request,
                use_turns=rt.role_model_use_turns,
                current_turn=rt.turn_count + 1,
            )
            if allowed:
                role_model = role_model_candidate
                summary = f"eligible {role_model.id}"
            else:
                summary = "suppressed by safety or session guardrails"
            steps.append(StepRecord(name="role_model", summary=summary, latency_ms=0.0))
        elif enable_role_models:
            steps.append(StepRecord(name="role_model", summary="no relevant match", latency_ms=0.0))

        reply_out, verdict, draft, provider_used, reply_ok = self._reply(
            rt,
            decision,
            text,
            cards,
            items,
            safety.level == SafetyLevel.elevated,
            steps,
            role_model=role_model,
            explicit_role_model_request=explicit_role_model_request,
        )
        final = reply_out.reply
        if safety.level == SafetyLevel.elevated:
            final = f"{final}\n\n{ELEVATED_CHECK_IN}"

        citations = citations_for(
            [c for c in reply_out.used_card_ids if any(i.doc_id == c for i in items)], self._library
        )
        role_model_provenance = (
            RoleModelProvenance(
                card_id=role_model.id,
                person=role_model.person,
                title=role_model.title,
                source_locators=[source.locator for source in role_model.sources],
            )
            if role_model is not None
            and safety.level == SafetyLevel.none
            and reply_ok
            and reply_out.used_role_model_id == role_model.id
            else None
        )
        handoff = (
            handoff_line(decision.speaker)
            if rt.last_speaker is not None and decision.speaker != rt.last_speaker
            else None
        )
        degraded = (not planner_ok) or (not reply_ok) or safety.degraded
        explanation = self._explain(decision, plan, reply_out, citations)
        if role_model_provenance is not None:
            explanation += (
                f"\nHistorical inspiration: {role_model_provenance.title} was shared as an analogy."
            )
        summary_text = render_state_summary(state)

        trace = TurnTrace(
            turn_id=turn_id,
            session_id=session_id,
            user_message=text,
            safety=safety,
            emotion=emotion,
            state_before=state_before,
            state_after=state,
            decision=ConductorDecision(
                phase=decision.phase,
                speaker=decision.speaker,
                reason=(plan.reason or "; ".join(decision.notes) or decision.move)[:300],
                retrieve=decision.retrieve,
                retrieve_query=decision.retrieve_query,
            ),
            retrieved=items,
            draft=draft,
            critic=verdict,
            final_reply=final,
            explanation=explanation,
            steps=steps,
            provider_used=provider_used,
        )
        self._commit(
            rt,
            session_id,
            text,
            final,
            decision,
            state,
            plan,
            reply_out,
            handoff,
            trace,
            role_model_provenance=role_model_provenance,
        )
        return TurnOutput(
            reply=final,
            speaker=decision.speaker,
            speaker_name=ROLES[decision.speaker].name,
            handoff=handoff,
            phase=decision.phase,
            safety_level=safety.level,
            safety_action=safety.action,
            state=state,
            state_summary=summary_text,
            explanation=explanation,
            citations=citations,
            historical_inspiration=role_model_provenance,
            move=decision.move,
            degraded=degraded,
            provider_used=provider_used,
            latency_ms=_ms(started),
            turn_id=turn_id,
            debug_steps=[f"{x.name}: {x.summary}" for x in steps],
        )

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _match_role_model(
        self,
        text: str,
        state: EmotionalState,
        rt: SessionRuntime,
        steps: list[StepRecord],
    ) -> RoleModelCard | None:
        """Retrieve an all-card historical candidate independently of counselling RAG."""
        recent_user_messages = tuple(
            message.content for message in rt.history if message.role == "user"
        )[-4:]
        started = time.perf_counter()
        try:
            result = self._role_model_retriever.retrieve(
                text,
                recent_user_messages=recent_user_messages,
                conversation_state=state,
            )
        except Exception as exc:
            logger.warning("Historical role-model retrieval failed (%s).", type(exc).__name__)
            steps.append(
                StepRecord(
                    name="role_model_match",
                    summary=f"unavailable ({type(exc).__name__})",
                    latency_ms=_ms(started),
                )
            )
            return None

        candidate = result.match
        summary = (
            f"{candidate.card.id} score={candidate.score:.3f}"
            if candidate is not None
            else result.abstention_reason or "no relevant match"
        )
        steps.append(
            StepRecord(name="role_model_match", summary=summary, latency_ms=_ms(started))
        )
        return candidate.card if candidate is not None else None

    def _read_emotion(self, text: str, steps: list[StepRecord]) -> EmotionReading | None:
        if self._emotion is None:
            return None
        t0 = time.perf_counter()
        try:
            reading = self._emotion.read(text)
        except Exception as exc:  # optional signal; never block a turn
            logger.warning("Emotion reader failed (%s).", type(exc).__name__)
            return None
        steps.append(StepRecord(name="emotion", summary=f"hint: {reading.label}", latency_ms=_ms(t0)))
        return reading

    def _plan(
        self, rt: SessionRuntime, text: str, emotion: EmotionReading | None, steps: list[StepRecord]
    ) -> tuple[PlannerOutput, Any, bool]:
        t0 = time.perf_counter()
        try:
            result = self._gateway.generate(
                TaskKind.extract,
                build_planner_messages(rt, text, emotion),
                json_schema=PlannerOutput,
                temperature=0.2,
                max_tokens=900,
            )
            plan = _parse(result, PlannerOutput)
            steps.append(
                StepRecord(
                    name="planner",
                    summary=f"{plan.move}/{plan.speaker.value}/{plan.phase.value}",
                    latency_ms=_ms(t0),
                )
            )
            return plan, result, True
        except Exception as exc:
            logger.warning("Planner failed (%s); using rule-based planning.", type(exc).__name__)
            diagnostic = str(exc)
            if text:
                diagnostic = re.sub(
                    re.escape(text), "[user text]", diagnostic, flags=re.IGNORECASE
                )
            steps.append(
                StepRecord(
                    name="planner",
                    summary=f"failed ({type(exc).__name__}: {diagnostic[:140]}): heuristic plan",
                    latency_ms=_ms(t0),
                )
            )
            return heuristic_plan(rt, text, emotion), None, False

    def _merge_safety(
        self, gate: SafetyResult, plan: PlannerOutput, planner_ok: bool, rt: SessionRuntime
    ) -> SafetyResult:
        level = max(gate.level, plan.safety_level, key=_RANK.__getitem__)
        if level == gate.level and planner_ok:
            return gate.model_copy(update={"degraded": False})
        if level == gate.level:
            return gate
        rt.gate_state["crisis_seen"] = rt.gate_state.get("crisis_seen") or level == SafetyLevel.crisis
        reasons = [*gate.reasons, f"planner:{plan.safety_level.value}"]
        return SafetyResult(
            level=level,
            reasons=reasons,
            source="rule+classifier" if gate.level != SafetyLevel.none else "classifier",
            action=action_for(level),
            degraded=False,
        )

    def _previous_card(
        self, rt: SessionRuntime, steps: list[StepRecord]
    ) -> tuple[list[RetrievedItem], list[Any]]:
        card = self._library.get(rt.techniques_offered[-1]) if rt.techniques_offered else None
        if card is None:
            return [], []
        steps.append(StepRecord(name="retrieval", summary="reused previous card", latency_ms=0.0))
        item = RetrievedItem(
            collection="cards",
            doc_id=card.id,
            source=", ".join(src.source_id for src in card.sources),
            text=f"{card.title}: {card.summary}",
            score=1.0,
        )
        return [item], [card]

    def _retrieve(
        self,
        decision: Decision,
        state: EmotionalState,
        steps: list[StepRecord],
        rt: SessionRuntime,
    ) -> tuple[list[RetrievedItem], list[Any]]:
        if decision.move == "follow_up":
            return self._previous_card(rt, steps)
        if not decision.retrieve or self._retriever is None:
            return [], []
        t0 = time.perf_counter()
        emotions: Sequence[str] = [h.label for h in state.hypotheses[:2]]
        try:
            found = self._retriever.search_with_evidence(
                decision.retrieve_query or "", emotions=emotions, role=decision.speaker.value, k=6
            )
        except Exception as exc:
            logger.warning("Retrieval failed (%s).", type(exc).__name__)
            found = []
        wanted = "technique" if decision.move == "offer_technique" else "psychoeducation"
        items = [i for i in found if i.collection == "cards"]
        right_kind = [
            i for i in items if (c := self._library.get(i.doc_id)) is not None and c.category == wanted
        ]
        items = right_kind  # a technique request must not be answered with an explainer card, or vice versa
        cards = [c for c in (self._library.get(i.doc_id) for i in items) if c is not None][:2]
        items = [i for i in items if any(c.id == i.doc_id for c in cards)]
        steps.append(
            StepRecord(name="retrieval", summary=f"{len(cards)} cards", latency_ms=_ms(t0))
        )
        return items, cards

    def _reply(
        self,
        rt: SessionRuntime,
        decision: Decision,
        text: str,
        cards: list[Any],
        items: list[RetrievedItem],
        elevated: bool,
        steps: list[StepRecord],
        *,
        role_model: RoleModelCard | None = None,
        explicit_role_model_request: bool = False,
    ) -> tuple[ReplyOutput, CriticVerdict, str | None, str, bool]:
        retrieved_ids = [i.doc_id for i in items]
        previous_bot = next((m.content for m in reversed(rt.history) if m.role == "assistant"), None)
        feedback: list[str] | None = None
        first_draft: str | None = None
        provider_used = "fallback"
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                result = self._gateway.generate(
                    TaskKind.converse,
                    build_reply_messages(
                        rt,
                        role=decision.speaker,
                        move=decision.move,
                        question=decision.question,
                        user_text=text,
                        cards=cards,
                        role_model=role_model,
                        explicit_request=explicit_role_model_request,
                        elevated=elevated,
                        feedback=feedback,
                    ),
                    json_schema=ReplyOutput,
                    temperature=0.6,
                    max_tokens=600,
                )
                out = _parse(result, ReplyOutput)
                provider_used = f"{getattr(result, 'provider', '?')}/{getattr(result, 'model', '?')}"
            except Exception as exc:
                logger.warning("Reply generation failed (%s).", type(exc).__name__)
                steps.append(
                    StepRecord(
                        name="reply",
                        summary=f"failed ({type(exc).__name__}: {str(exc)[:140]})",
                        latency_ms=_ms(t0),
                    )
                )
                break
            verdict, issues = check_reply(
                out.reply,
                role=decision.speaker,
                move=decision.move,
                retrieved_ids=retrieved_ids,
                used_card_ids=out.used_card_ids,
                previous_reply=previous_bot,
            )
            historical_issues = (
                check_role_model_reply(
                    out.reply,
                    selected_card=role_model,
                    used_card_id=out.used_role_model_id,
                )
                if out.used_role_model_id is not None or has_historical_reference(out.reply)
                else []
            )
            if historical_issues:
                verdict = verdict.model_copy(
                    update={
                        "passed": False,
                        "grounded": False,
                        "issues": [*verdict.issues, *historical_issues],
                    }
                )
                issues.extend(historical_issues)
                issues.extend(_role_model_retry_feedback(historical_issues, role_model))
            if role_model is not None or out.used_role_model_id is not None:
                attempt_label = "first" if attempt == 0 else "retry"
                steps.append(
                    StepRecord(
                        name="role_model_critic",
                        summary=(
                            f"{attempt_label} attempt passed"
                            if not historical_issues
                            else f"{attempt_label} attempt rejected: "
                            + "; ".join(historical_issues)[:240]
                        ),
                        latency_ms=_ms(t0),
                    )
                )
            steps.append(
                StepRecord(
                    name="critic",
                    summary="passed" if verdict.passed else "issues: " + "; ".join(issues)[:120],
                    latency_ms=_ms(t0),
                )
            )
            if first_draft is None:
                first_draft = out.reply
            if verdict.passed:
                if role_model is not None and out.used_role_model_id == role_model.id:
                    steps.append(
                        StepRecord(
                            name="role_model_validation",
                            summary=(
                                "complete reply validated on first attempt"
                                if attempt == 0
                                else "complete reply validated on retry"
                            ),
                            latency_ms=0.0,
                        )
                    )
                return out, verdict, first_draft if attempt else None, provider_used, True
            feedback = issues
        safe = fallback_reply(decision.move, decision.question)
        verdict, _ = check_reply(
            safe, role=decision.speaker, move=decision.move, retrieved_ids=[], used_card_ids=[]
        )
        steps.append(StepRecord(name="fallback", summary="safe template used", latency_ms=0.0))
        return ReplyOutput(reply=safe, explanation="A safe standard reply was used."), verdict, first_draft, provider_used, False

    # ── Output helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _explain(
        decision: Decision,
        plan: PlannerOutput,
        reply: ReplyOutput,
        citations: Sequence[Any],
    ) -> str:
        policy = ROLES[decision.speaker]
        why = plan.reason or "keeping the conversation steady"
        lines = [
            f"{policy.name} ({decision.speaker.value}) is speaking: {why}",
            f"Approach: {policy.paradigm}.",
            f"What this turn does: {decision.move.replace('_', ' ')} (phase: {decision.phase.value}).",
        ]
        if reply.explanation:
            lines.append(reply.explanation)
        if decision.notes:
            lines.append("Guardrails: " + "; ".join(decision.notes) + ".")
        if citations:
            lines.append("Grounded in: " + "; ".join(c.title for c in citations) + ".")
        return "\n".join(lines)

    def _finish_crisis(
        self,
        rt: SessionRuntime,
        text: str,
        turn_id: str,
        safety: SafetyResult,
        steps: list[StepRecord],
        started: float,
        *,
        emotion: EmotionReading | None = None,
    ) -> TurnOutput:
        rt.gate_state["crisis_seen"] = True
        explanation = "Your message suggested you may be in serious distress, so a fixed safety message was shown."
        trace = TurnTrace(
            turn_id=turn_id,
            session_id=rt.session_id,
            user_message=text,
            safety=safety,
            emotion=emotion,
            final_reply=CRISIS_TEMPLATE,
            explanation=explanation,
            steps=steps,
            provider_used="fixed-template",
        )
        rt.history.extend([Message(role="user", content=text), Message(role="assistant", content=CRISIS_TEMPLATE)])
        rt.turn_count += 1
        repo.append_message(rt.session_id, rt.history[-2], factory=self._factory)
        repo.append_message(rt.session_id, rt.history[-1], factory=self._factory)
        repo.save_trace(trace, factory=self._factory)
        return TurnOutput(
            reply=CRISIS_TEMPLATE,
            speaker=None,
            phase=rt.phase,
            safety_level=SafetyLevel.crisis,
            safety_action=safety.action,
            state=rt.state,
            state_summary=render_state_summary(rt.state),
            explanation=explanation,
            degraded=False,
            provider_used="fixed-template",
            latency_ms=_ms(started),
            turn_id=turn_id,
            debug_steps=[f"{x.name}: {x.summary}" for x in steps],
        )

    def _commit(
        self,
        rt: SessionRuntime,
        session_id: str,
        text: str,
        final: str,
        decision: Decision,
        state: EmotionalState,
        plan: PlannerOutput,
        reply: ReplyOutput,
        handoff: str | None,
        trace: TurnTrace,
        *,
        role_model_provenance: RoleModelProvenance | None = None,
    ) -> None:
        user_msg = Message(role="user", content=text)
        bot_msg = Message(role="assistant", content=final, speaker_role=decision.speaker)
        rt.history.extend([user_msg, bot_msg])
        rt.state = state
        rt.phase = decision.phase
        rt.turn_count += 1
        if role_model_provenance is not None:
            rt.role_model_use_turns.append(rt.turn_count)
        if decision.move == "ask_clarifying":
            rt.questions_asked += 1
        rt.last_move = decision.move
        rt.turns_since_handoff = 0 if handoff else rt.turns_since_handoff + 1
        rt.last_speaker = decision.speaker
        if decision.move in ("offer_technique", "psychoeducation") and reply.used_card_ids:
            rt.techniques_offered.extend(reply.used_card_ids)
        for theme in plan.profile_update.themes:
            tag = " ".join(theme.lower().split()[:3])
            if tag and tag not in rt.themes:
                rt.themes.append(tag)
        explicit_pref = memory.explicit_role_preference(text)
        if explicit_pref is not None:
            rt.preferred_role = explicit_pref
        try:
            repo.append_message(session_id, user_msg, factory=self._factory)
            repo.append_message(session_id, bot_msg, factory=self._factory)
            repo.save_trace(trace, factory=self._factory)
            profile = repo.get_profile(rt.user_id, factory=self._factory)
            merged = memory.merge_profile(
                profile,
                user_id=rt.user_id,
                display_name=None,
                update=plan.profile_update.model_copy(update={"preferred_role": explicit_pref}),
            )
            repo.save_profile(merged, factory=self._factory)
        except Exception as exc:  # storage trouble must not break the conversation
            logger.error("Persisting a turn failed (%s).", type(exc).__name__)


def _empty_update() -> Any:
    from app.pipeline.types import ProfileUpdate

    return ProfileUpdate()


def _listen_only(decision: Decision) -> Decision:
    return Decision(
        phase=decision.phase,
        speaker=decision.speaker,
        move="reflect",
        question=None,
        retrieve=False,
        retrieve_query=None,
        notes=(*decision.notes, "no relevant card found: staying with listening"),
    )


def _role_model_retry_feedback(
    issues: list[str], role_model: RoleModelCard | None
) -> list[str]:
    """Translate historical critic failures into concrete instructions for the next draft."""
    person = (
        PERSON_DISPLAY_NAMES[role_model.person]
        if role_model is not None
        else "the selected historical figure"
    )
    feedback: list[str] = []
    for issue in issues:
        if "declares historical-card use without mentioning its person" in issue:
            feedback.append(f"Name {person} explicitly in the reply.")
        elif "historical example must be clearly framed as an analogy" in issue:
            feedback.append("Explicitly frame the historical example as an analogy.")
        elif issue.startswith("unsupported capitalized token(s):"):
            tokens = issue.partition(":")[2].strip()
            feedback.append(
                f"Remove unsupported capitalized token(s): {tokens}; use only names present "
                "in the selected card."
            )
        elif "introduces an unsupported historical date or number" in issue:
            feedback.append(
                "Remove unsupported dates, numbers, and statistics; use only details stated in the card."
            )
        elif "uses quotation marks in historical context" in issue:
            feedback.append("Remove historical quotations and quotation marks.")
        elif "claims the user's situation is the same as the historical person's" in issue:
            feedback.append("Remove any claim that the two situations are identical or equivalent.")
        elif "historical claim is not grounded in the selected card" in issue:
            feedback.append(
                "Remove unsupported historical facts and use only the selected card's information."
            )
        elif "introduces a historical person outside the selected card" in issue:
            feedback.append("Remove every historical figure other than the selected person.")
        elif "claims use of a historical card that was not selected" in issue:
            feedback.append(
                "Set used_role_model_id only to the selected card id when the reply genuinely uses it."
            )
        elif "mentions a historical person without declaring the selected card as used" in issue:
            feedback.append(
                "Either remove the historical mention or declare the selected card id only if it is used."
            )
        elif "explicitly requested historical card was not used" in issue:
            feedback.append(
                f"Use {person}'s selected card: name {person}, connect its lesson to the person's "
                "situation, frame it as an analogy, and declare its id only if genuinely used."
            )
    return feedback


__all__ = ["EmotionReader", "GatewayLike", "SolaceEngine"]
