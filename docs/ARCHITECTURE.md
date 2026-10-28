# ARCHITECTURE.md — Solace System Architecture

## Module Dependency Rules

The following dependency graph is **enforced** — violations must be discussed in DECISIONS.md before being introduced.

```
app/contracts/          ← no internal imports; pure Pydantic + stdlib
    ▲   ▲   ▲   ▲
    │   │   │   │
config  │   │   │       ← pydantic-settings; imports contracts
    ▲   │   │   │
    │   │   │   │
  llm/ storage/ tracing/   ← import contracts + config only
       ▲   ▲   ▲
       │   │   │
     agents/ (Layer 4)   ← imports llm, rag, storage, tracing, contracts
                         ← agents NEVER import each other; Conductor mediates
```

### Allowed import matrix

| Module | May import |
|--------|-----------|
| `contracts` | `pydantic`, `enum`, `datetime`, stdlib |
| `config` | `contracts`, `pydantic_settings`, stdlib |
| `llm` | `contracts`, `config`, stdlib, third-party LLM SDKs |
| `storage` | `contracts`, `config`, `sqlalchemy`, stdlib |
| `tracing` | `contracts`, `config`, stdlib |
| `safety` | `contracts`, `config`, `llm` (structured classify only), `tracing`, `PyYAML`, stdlib |
| `emotion` | `contracts`, `tracing`, `transformers` (lazy, optional `[ml]` extra) |
| `rag` | `contracts`, `config`, `storage`, `chromadb`, `sentence_transformers` |
| `agents` | `contracts`, `config`, `llm`, `rag`, `storage`, `tracing`, `safety` |
| `main` | all of the above |

---

## Data Flow (per turn)

```
HTTP POST /chat
    │
    ▼
app/main.py
    │  raw user text
    ▼
safety/sentinel.py ──────────────────────── SafetyResult
    │ if crisis → immediate response (no LLM)
    │
    ▼
safety/emotion.py ───────────────────────── EmotionReading
    │
    ▼
storage/repository.get_profile() ─────────── UserProfile
storage/repository.get_recent_messages() ─── Message[]
    │
    ▼
agents/explorer.py ──────────────────────── EmotionalState (updated)
    │
    ▼
agents/conductor.py ─────────────────────── ConductorDecision
    │ if retrieve:
    ▼
rag/retriever.py ────────────────────────── RetrievedItem[]
    │
    ▼
agents/<role>.py ────────────────────────── draft reply (str)
    │
    ▼
agents/critic.py ────────────────────────── CriticVerdict
    │ if not passed → retry once, then escalate
    │
    ▼
storage/repository.save_trace() ─────────── TurnTrace persisted
storage/repository.append_message()
    │
    ▼
HTTP response: { reply, explanation, trace_id }
```

---

## Database Schema (Layer 0)

```sql
users          (id, display_name, created_at, updated_at)
sessions       (id, user_id→users, started_at, ended_at, phase)
messages       (id, session_id→sessions, role, content,
                speaker_role, timestamp)
turn_traces    (id, session_id→sessions, turn_id, data JSON,
                created_at)
profiles       (id, user_id→users UNIQUE, data JSON,
                created_at, updated_at)
provider_usage (id, provider, date, call_count,
                UNIQUE(provider, date))
```

---

## LLM Gateway Resilience

```
dispatch(task, messages)
    │
    for provider in fallback_chain[task]:
        │
        ├─ provider key missing?  → skip (WARN)
        ├─ provider in cooldown?  → skip
        ├─ daily cap exceeded?    → skip
        │
        ├─ call provider (with tenacity retry on 429/5xx/timeout)
        │       ├─ 429 → set cooldown, raise for fallthrough
        │       ├─ 5xx/timeout → backoff+jitter, retry up to max_retries
        │       └─ success → return LLMResult
        │
        └─ all retries failed → try next provider
    │
    └─ all providers failed → raise GatewayExhaustedError
```

---

## Structured Output Flow

```
gateway.generate(..., json_schema=MyModel)
    │
    ├─ send request asking for JSON
    ├─ parse response as JSON
    ├─ validate against MyModel
    │       └─ ValidationError →
    │           ├─ send repair prompt with error
    │           ├─ parse + validate again
    │           └─ still fails → raise StructuredOutputError
    └─ return LLMResult(parsed=MyModel instance)
```

---

## Session Phase State Machine

```
checkin ──► exploration ──► understanding ──► guidance ──► closing
   ▲                                                          │
   └──────────────────── (new session) ◄────────────────────-┘
```

Transitions are decided by the **Conductor** based on EmotionalState confidence and UserNeed.

---

## Privacy Design

- `delete_user_data(user_id)` cascades: messages → sessions → turn_traces → profiles → users
- No PII in logs above DEBUG level
- PII scrubber hooks into `pre_send_filter` (Layer 5)
- Free-tier LLM providers: only synthetic data in dev/test

---

## Layer 1 — Safety and Emotion

`SafetyGate.check(message, history, session_state)` always runs the YAML-backed rule layer
first. It then invokes the structured `TaskKind.classify` gateway classifier when configured.
The final level is the maximum of the rule and classifier levels; classifier failure or cap
exhaustion logs a warning, marks the result degraded, and preserves the rule result. A session
that has reached `crisis` remains at least `elevated` on every later turn.

The classifier receives the current user turn and the six most recent user turns. Its result
records the serving provider/model and preceding gateway attempts. Crisis output is never drafted
by an LLM: `responses.py` supplies a fixed crisis template and returns `crisis_override`.
Elevated results return `soft_check` guidance so orchestration can continue with a gentle
counsellor check-in.

Rule patterns are maintained in `data/safety/patterns_en.yaml` and
`data/safety/patterns_hinglish.yaml`. Text is lowercased, common leetspeak and typos are mapped,
long repeated letters are reduced, and punctuation noise is removed before matching. Negation is
limited to nearby intent phrases so constructions such as "I'm not going to hurt myself" do not
become crisis matches. Idioms such as "this exam is killing me" and "I could die of
embarrassment" do not match the explicit self-harm/intent patterns. This rule set is a safety
floor, not a clinically validated detector.

`EmotionRecognizer` lazily loads `SamLowe/roberta-base-go_emotions` on first inference and accepts
an injected classifier for deterministic tests. It returns the fine label plus an aggregated
coarse label from the 28 GoEmotions classes. The model's score is an uncalibrated classification
score; it is **not** a calibrated measure of emotional intensity or clinical severity.
Install optional CPU inference dependencies with
`python -m pip install ".[ml]" --extra-index-url https://download.pytorch.org/whl/cpu`.
