"""Prompt builders. Kept short: free-tier providers have tight token-per-minute limits."""

from __future__ import annotations

import json

from app.contracts import EmotionReading, Message, Role, SafetyLevel, UserNeed
from app.pipeline.guardrails import MAX_CLARIFYING, SessionRuntime
from app.pipeline.roles import ROLES
from app.rag.cards import Card
from app.rolemodels.models import RoleModelCard

PLANNER_SYSTEM = """You are the planning brain of Solace, a supportive, non-clinical counselling companion. \
You do NOT write the reply. Read the person's latest message in context and return ONE JSON object only.

SAFETY: safety_level = "crisis" only for explicit intent or a plan to end their life or to harm someone. \
"elevated" for hopelessness, a passive wish to disappear or not wake up, feeling like a burden, or urges to \
self-harm without intent. Otherwise "none". Idioms ("this exam is killing me") are "none". Never diagnose.

UNDERSTAND: update the emotional picture. emotions = up to 4 of [anger, anxiety_fear, neutral, others, \
positive, sadness, shame_guilt] with probabilities. intensity 0-1. Fill trigger, duration, functional_impact \
only if the person has said it (otherwise null). need = vent (wants to be heard), understand, act (wants \
steps) or unknown. confidence 0-1 = how well you understand their situation. open_questions = up to 3 unknowns.

DECIDE: phase (checkin, exploration, understanding, guidance, closing); speaker (counsellor = feelings and \
meaning; mentor = goals and one small next step; teacher = explain what is happening and teach one skill; \
peer = casual validation); move (ask_clarifying, reflect, offer_technique, psychoeducation, summarise, close, follow_up); \
clarifying_question = ONE short question that best separates your top two hypotheses or fills the biggest \
unknown (always provide your best one, even if you choose another move); retrieve = true only for \
offer_technique or psychoeducation; wants_new_help = true only if the person asks for something new or says \
the last suggestion did not work; \
retrieve_query = a short phrase about the feeling, trigger and need; profile_update = {preferred_role only if \
the person states a preference, themes up to 3 short tags, values_note only if stated}.

RULES: ask before advising. If the person mainly wants to vent, reflect. If you just offered a technique and \
the person reacts to it (yes, thanks, it helped), use move follow_up (never a second technique) or close if \
they are done. reason = one short neutral sentence in the third person. Choose the speaker whose style fits \
their need and keep the same speaker unless the need changes. Stop asking once you have asked 3 questions or \
confidence is 0.7 or more: summarise or move on.

Return JSON with exactly these keys: safety_level, emotions [{label, probability}], intensity, trigger, \
duration, functional_impact, need, confidence, open_questions, phase, speaker, reason, move, \
clarifying_question, retrieve, retrieve_query, wants_new_help, profile_update {preferred_role, themes, values_note}."""


def _recent(history: list[Message], n: int = 4, width: int = 280) -> str:
    items = history[-n:]
    if not items:
        return "(none)"
    return "\n".join(
        f"{'Person' if m.role == 'user' else 'Solace'}: {m.content[:width]}" for m in items
    )


def build_planner_messages(
    rt: SessionRuntime, user_text: str, emotion: EmotionReading | None
) -> list[dict[str, str]]:
    """Context plus the latest message for the planner call."""
    state = rt.state.model_dump(mode="json", exclude={"open_questions"}) if rt.state else None
    context = {
        "turn": rt.turn_count + 1,
        "phase_now": rt.phase.value,
        "questions_asked_so_far": rt.questions_asked,
        "question_budget": MAX_CLARIFYING,
        "last_speaker": rt.last_speaker.value if rt.last_speaker else None,
        "preferred_role": rt.preferred_role.value if rt.preferred_role else None,
        "known_state": state,
        "themes_from_past": rt.themes[-3:],
        "last_session_note": rt.resume_note,
    }
    hint = ""
    if emotion is not None:
        hint = (
            f"\nWeak hint from a small emotion model (often wrong, never a safety signal): "
            f"{emotion.label} {emotion.score:.2f}"
        )
    user = (
        f"Context: {json.dumps(context)}\n\nRecent turns:\n{_recent(rt.history)}{hint}\n\n"
        f"Latest message from the person:\n{user_text}"
    )
    return [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}]


_MOVE_BRIEF = {
    "ask_clarifying": (
        "Briefly reflect what they said, then ask this one question in your own words: \"{q}\". "
        "Give no advice."
    ),
    "reflect": (
        "Reflect the feeling and what seems to matter underneath it. No advice. You may end with one "
        "gentle, open invitation to say more."
    ),
    "reflect_vent": (
        "The person wants to vent. Reflect the feeling and what seems to matter underneath it. No advice and "
        "NO question; end with a short invitation such as 'I'm here, say more if you like.'"
    ),
    "follow_up": (
        "You just offered the technique on the card below. Respond to what they said about it. If they want "
        "to try it, guide them through only the next one or two steps together. If they said thanks or that "
        "it helped, acknowledge that warmly and ask how it felt or whether they would like to wrap up. Do "
        "NOT introduce a new technique."
    ),
    "summarise": (
        "Give a short, human summary of what you understand (feelings, what set it off, what they seem to "
        "need) and ask whether it fits. No advice."
    ),
    "offer_technique": (
        "Offer the ONE technique from the card below, in your own words and conversationally. Give only the "
        "first {n} steps, then say you can carry on together if they would like, and ask if they want to try "
        "it. Use only what the card says."
    ),
    "psychoeducation": (
        "Explain in 3-4 plain sentences, using only the card below, what they may be experiencing and that "
        "it is common. Then ask if that matches their experience."
    ),
    "close": (
        "Wrap up warmly: acknowledge their effort in talking, restate one thing they could try if a technique "
        "came up, and invite them back. No new advice."
    ),
}


def _card_block(cards: list[Card]) -> str:
    parts = []
    for card in cards:
        parts.append(
            f"CARD id={card.id}\nTitle: {card.title}\nSummary: {card.summary}\n"
            f"Steps: {' | '.join(card.steps)}\nWhen not to use: {' | '.join(card.when_not_to_use)}"
        )
    return "\n\n".join(parts)


def build_reply_messages(
    rt: SessionRuntime,
    *,
    role: Role,
    move: str,
    question: str | None,
    user_text: str,
    cards: list[Card],
    role_model: RoleModelCard | None = None,
    elevated: bool,
    feedback: list[str] | None = None,
) -> list[dict[str, str]]:
    """Messages for the role reply call."""
    policy = ROLES[role]
    system = (
        f"You are {policy.name}, the {role.value} voice of Solace, an AI companion (not a human, not a "
        f"therapist). Approach: {policy.paradigm}. Style: {policy.style}\n"
        f"Do: {' '.join(policy.moves)}\nAvoid: {'; '.join(policy.avoid)}.\n"
        "HARD RULES: never diagnose or name a disorder for the person; never mention medication; no phone "
        "numbers, links or helplines; no 'just relax' or 'don't worry'; never claim personal experiences; "
        "ask at most ONE question; under 110 words; warm, plain English (mirror a few Hinglish words only if "
        "the person used them); speak to the person directly; vary your openings and avoid starting with "
        "'It sounds like' more than once in a conversation."
    )
    if elevated:
        system += (
            " The person may be struggling more than usual: be extra warm, give no advice, and do not "
            "mention suicide or self-harm. A separate message with support options follows automatically."
        )
    if role_model is not None:
        system += (
            " A separate historical inspiration card may be used only as an analogy, never as clinical "
            "advice or as the same situation. Use only facts in that card; do not introduce other people, "
            "events, dates, numbers, or quotations. Do not use quotation marks. If it does not fit naturally, "
            "leave it unused."
        )
    system += (
        '\nReturn JSON only: {"reply": "...", "explanation": "one or two sentences on why you replied this '
        'way", "used_card_ids": [], "used_role_model_id": null}. List a card id in used_card_ids only if '
        "you used that self-help card. Set used_role_model_id to the historical card id only if your reply "
        "actually uses it; otherwise leave it null."
    )
    brief_key = "reflect_vent" if move == "reflect" and rt.state and rt.state.need == UserNeed.vent else move
    brief = _MOVE_BRIEF[brief_key].format(q=question or "", n=policy.max_numbered_steps)
    state_line = ""
    if rt.state is not None:
        top = ", ".join(f"{h.label} {h.probability:.0%}" for h in rt.state.hypotheses[:3])
        state_line = (
            f"What you know: feelings {top or 'unclear'}; trigger {rt.state.trigger or 'unknown'}; "
            f"need {rt.state.need.value}.\n"
        )
    cards_text = f"\n\n{_card_block(cards)}" if cards else ""
    role_model_text = ""
    if role_model is not None:
        role_model_text = (
            "\n\nSEPARATE OPTIONAL HISTORICAL INSPIRATION (not a self-help card):\n"
            f"Card id: {role_model.id}\n"
            f"Person: {role_model.person.value}\n"
            f"Title: {role_model.title}\n"
            f"Summary: {role_model.summary}\n"
            f"Lesson: {role_model.lesson}\n"
            f"Framing: {role_model.framing}"
        )
    user = (
        f"{state_line}Recent turns:\n{_recent(rt.history)}\n\nLatest message from the person:\n{user_text}\n\n"
        f"Your task now: {brief}{cards_text}{role_model_text}"
    )
    last_bot = next((m.content for m in reversed(rt.history) if m.role == "assistant"), None)
    if last_bot:
        user += f"\n\nYour last message began: \"{' '.join(last_bot.split()[:6])}...\". Begin differently."
    if feedback:
        user += "\n\nYour previous draft broke these rules, so rewrite it: " + "; ".join(feedback)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def is_elevated(level: SafetyLevel) -> bool:
    """Convenience for callers."""
    return level == SafetyLevel.elevated
