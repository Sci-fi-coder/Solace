# AGENTS.md — Solace Project Context

> **AI coding agent:** Read this file in full at the start of every session.
> Follow the HANDOFF PROTOCOL at the end of this file without exception.

---

## 1. Project Overview

**Solace** is a session-based, multi-agent mental-health counselling system inspired by human role models (counsellor, mentor, teacher, peer). It is explicitly **NOT** a generic chatbot.

### What makes it different

- Runs a **structured session**: Check-in → Exploration → Understanding → Guidance → Closing
- Asks **narrowing questions** before advising; never jumps to advice
- Maintains a visible **Emotional State Model** of the user
- Uses **four role agents** with distinct reasoning paradigms (see §3)
- **One speaker per turn** with explicit handoffs orchestrated by the Conductor
- Grounds replies in a **curated local RAG knowledge base**
- A **rule-based Safety Sentinel** runs *outside* the LLM on every message — before anything else
- A **Critic** verifies every drafted reply (grounded, no diagnosis, role style)
- Keeps **memory across sessions**
- Every reply carries an **explanation** of why it was said (TurnTrace)

---

## 2. Per-Turn Pipeline

```
User Message
    │
    ▼
[Safety Sentinel]  ← rule-based, outside LLM, runs FIRST
    │  SafetyResult
    ▼
[Emotion Classifier]  ← EmotionReading
    │
    ▼
[Memory / Profile Lookup]  ← UserProfile, recent messages
    │
    ▼
[Explorer]  ← updates EmotionalState, decides next question
    │
    ▼
[Conductor]  ← picks Role, Phase, retrieve-or-listen → ConductorDecision
    │
    ▼
[RAG]  ← if retrieve: query ChromaDB → RetrievedItem[]
    │
    ▼
[Role Agent]  ← drafts reply using role prompt + state + retrieved docs
    │
    ▼
[Critic]  ← CriticVerdict: grounded, no_diagnosis, role_fidelity
    │
    ▼
[Save Memory + Trace]  ← TurnTrace persisted to SQLite
    │
    ▼
Final Reply + Explanation → User
```

---

## 3. Role Agents

| Role | Paradigm | When used |
|------|----------|-----------|
| **Counsellor** | Person-centred (Rogers): unconditional positive regard, reflection, empathy | Default for venting / distress |
| **Mentor** | Solution-focused / GROW model: goal → reality → options → will | When user wants to act |
| **Teacher** | CBT-style psychoeducation: name the pattern, explain, normalise | When user wants to understand |
| **Peer** | Lived-experience support: "I've felt that too" | When user needs validation |

---

## 4. Layer Plan

| Layer | Scope | Status |
|-------|-------|--------|
| **0 – Foundation** | Repo skeleton, contracts, LLM gateway, storage, tracing, /health | ✅ Complete |
| **1 – Safety + Emotion** | Safety Sentinel (rule-based), emotion classifier, escalation protocol | ⬜ Next |
| **2 – Knowledge Base / RAG** | ChromaDB, sentence-transformers, ingestion pipeline | ⬜ Planned |
| **3 – Explorer + State Model** | Emotional-state model, narrowing-question logic | ⬜ Planned |
| **4 – Role Agents + Conductor + Critic** | All four role agents, Conductor, Critic | ⬜ Planned |
| **5 – Memory / Profile / Privacy** | Cross-session memory, user profile, GDPR delete | ⬜ Planned |
| **6 – Frontend** | Session UI with explanation panel | ⬜ Planned |
| **7 – Evaluation** | Automated evaluation suite using TurnTraces | ⬜ Planned |

---

## 5. Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| API server | FastAPI |
| Data validation | Pydantic v2 |
| ORM / DB | SQLAlchemy 2.0 + SQLite |
| LLMs | Gemini (google-genai), Groq, OpenRouter, Mistral, Cerebras (free-tier, OpenAI-compat) |
| RAG (Layer 2) | ChromaDB + local sentence-transformers |
| Tests | pytest |

> ⚠️ Free-tier LLM providers may log request data.
> **All dev/test must use synthetic data only. Never send real user content during development.**

---

## 6. Coding Conventions

### General
- **Python 3.11+** — use `match`, `|` unions, `tomllib`, etc. where appropriate
- Type-hint everything; use `from __future__ import annotations` at top of every file
- No `Any` unless genuinely unavoidable; comment why
- Docstrings on every public class and function (one-line summary + params if non-obvious)
- Max line length: 100 characters
- Imports: stdlib → third-party → local, separated by blank lines

### Pydantic
- All models in `app/contracts/__init__.py` — **do not scatter them**
- Use `model_config = ConfigDict(...)` not class-level `Config`
- Validate at the boundary (gateway output, storage read); don't re-validate internally

### SQLAlchemy
- Use 2.0 style (`select()`, `session.execute()`, `session.scalars()`)
- All mutations go through repository functions in `app/storage/repository.py`
- Never import ORM models in contracts or gateway layers

### LLM Gateway
- **Never log user message content at INFO level** — use DEBUG with a truncated hash
- All provider credentials come from environment variables only
- Providers with missing keys are skipped with a WARNING log, not an error
- Every outgoing message list passes through `pre_send_filter` hooks before dispatch

### Safety
- Safety Sentinel runs **before** any LLM call, every turn, no exceptions
- Crisis-level safety results must never reach an LLM agent — handle immediately

### Privacy
- `delete_user_data(user_id)` must remove all rows for that user across **all** tables
- No PII in log files at any level above DEBUG

### Tests
- Use `MockProvider` for all LLM calls in tests — no real API calls
- Use `pytest` fixtures; no global state
- Tests live in `tests/`; mirror the `app/` structure (e.g., `tests/test_gateway.py`)
- Every test function has a docstring

### Dependency management
- Add dependencies to `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`
- Record *why* each non-obvious dependency was chosen in `docs/DECISIONS.md`

---

## 7. Architecture Notes

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module dependency diagram and data-flow notes.

Key invariants:
1. `app/contracts/` has **zero** internal imports — it is the shared interface
2. `app/llm/` imports only `contracts` and `config` — no storage, no agents
3. `app/storage/` imports only `contracts` and `config`
4. `app/tracing/` imports only `contracts` and `config`
5. Agents import `contracts`, `llm`, `rag`, `storage`, `tracing` — never each other directly (Conductor mediates)

---

## 8. HANDOFF PROTOCOL

**Every AI coding session must follow this protocol without exception.**

### Session Start
1. Read this file (`AGENTS.md`) in full
2. Read `docs/HANDOFF.md` — understand what was done last and what is next
3. Check `docs/DECISIONS.md` for constraints already decided
4. Run `pytest` to confirm baseline health before touching code

### During Session
- Update `docs/HANDOFF.md` incrementally as you complete sub-tasks
- When you make a non-obvious choice (library, schema design, workaround), append it to `docs/DECISIONS.md` immediately
- Never weaken a contract to make a test pass — fix the implementation

### Session End
1. Update `docs/HANDOFF.md`:
   - **Done**: bullet list of what was completed
   - **Next**: the concrete next step(s) for the following session
   - **Known issues**: anything incomplete, broken, or needing revisit
2. Append any new non-obvious choices to `docs/DECISIONS.md`
3. Run `pytest` — all tests must pass before committing
4. Commit with a clear message: `layer-N: <what>` (e.g., `layer-0: scaffold contracts and llm gateway`)
5. Push to the correct branch (current: `layer-0-foundation`)
