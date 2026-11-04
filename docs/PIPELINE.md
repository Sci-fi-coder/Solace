# Solace pipeline (written in chat; start here before touching app/pipeline)

One user message = `SolaceEngine.handle_turn(session_id, text) -> TurnOutput` (app/pipeline/engine.py).

## Flow (2 LLM calls per turn, 3 on a critic retry)
1. **Safety rules** (`SafetyGate`, rules-only). `crisis` -> fixed template, **zero LLM calls**.
2. **Planner** (one structured call, `TaskKind.extract`): reads the message, updates the emotional state
   (hypotheses, trigger, duration, impact, need, confidence), judges semantic safety, and proposes
   phase / speaker / move / retrieval query. Final safety level = max(rules, planner); crisis sticks.
   If the planner call fails, a deterministic regex-based planner updates the same state and proposes a
   conservative turn; the rule-based SafetyGate remains authoritative and the turn is marked degraded.
3. **Guardrails** (`guardrails.py`, deterministic): phase machine, ask-before-advise, question budget (4),
   "vent means listen", narrowing (an unclear picture turns a bare reflection into one question, never two in a row),
   summarise before advising when entering `understanding`, `follow_up` instead of stacking a second technique
   (reuses the previous card unless the planner sets `wants_new_help`), handoff throttle (min 2 turns),
   elevated -> counsellor + no advice, session turn limit. Feelings are smoothed across turns (EMA). Confidence
   is capped by evidence (trigger, duration, impact, need), so one message can never reach 'fairly sure'. The
   counsellor leads exploration; the 'what would help most?' question is asked before summarising; a technique
   request is only answered with a card of category `technique` (an explainer card needs `psychoeducation`).
   A preferred role is stored only if the person explicitly asks for one.
4. **Retrieval** only for `offer_technique` / `psychoeducation`. No relevant card => the move becomes `reflect`.
5. **Role reply** (one structured call, `TaskKind.converse`): persona policy in `roles.py`
   (Anika counsellor, Rohan mentor, Meera teacher, Kabir peer).
6. **Critic** (`critic.py`, local regex/rules, no LLM): no diagnosis, no medication, no numbers/links, no
   minimising, no fake lived experience, one question max, grounded in retrieved cards, role fidelity.
   One regeneration with feedback, then a fixed safe fallback reply.
7. **Memory**: trace + messages persisted; profile themes/preferred role merged; `end_session()` stores a
   deterministic summary used as "Last time we spoke about ..." next session.

Provider requests use `PROVIDER_TIMEOUT_SECONDS` (default 25). Gemini receives the timeout in
milliseconds; OpenAI-compatible SDK retries are disabled so transient retries stay gateway-owned.

## TurnOutput (what the UI shows)
`reply`, `speaker`/`speaker_name`, `handoff` (shown when the speaker changes), `phase`, `safety_level`,
`state` + `state_summary` (the "what I understand so far" panel), `explanation` (the "why I said this" panel),
`citations` (card title + WHO/NIMH source + PDF page), `move`, `degraded`, `provider_used`, `latency_ms`, `turn_id`.

## Entry points
- `build_engine_from_settings()` in `app/pipeline/factory.py` (real gateway, storage, cards, retriever).
- `python -m scripts.chat_cli` for a terminal chat. `/state`, `/why`, `/sources`, `/end`, `/quit`.
- Retriever: `RetrieverLike.search_with_evidence(query, *, emotions, role, k) -> list[RetrievedItem]`.
  `KeywordCardRetriever` is the baseline; the Layer 2b Chroma `Retriever` plugs in with the same method
  (engine only uses items with `collection == "cards"` and looks the card up by `doc_id` in `CardLibrary`).

## Rules for anyone editing this layer
- Never let an LLM write crisis text; never put helplines/numbers in generated replies (critic blocks them).
- Emotion-model output is a weak hint only, never a safety signal.
- Do not change the contracts in app/contracts without telling the project owner.
