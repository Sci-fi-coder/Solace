# Solace

**A session-based, multi-agent mental-health counselling system.**

Solace is a capstone project exploring how multiple AI agents—each embodying a distinct human counselling role—can collaborate within a structured therapeutic session. It is explicitly *not* a generic chatbot.

---

## Key Design Principles

| Principle | Detail |
|-----------|--------|
| **Structured sessions** | Every conversation follows: Check-in → Exploration → Understanding → Guidance → Closing |
| **Role agents** | Counsellor (person-centred), Mentor (GROW/solution-focused), Teacher (CBT psychoeducation), Peer (lived-experience support) |
| **Emotional state model** | A visible, updated model of the user's emotional state drives agent selection |
| **Safety-first** | A rule-based Safety Sentinel runs *outside* the LLM on every message |
| **Grounded replies** | A local RAG knowledge base; replies cite sources |
| **Critic gate** | Every drafted reply is checked before delivery |
| **Explainability** | Every reply carries a `TurnTrace` explaining why it was said |
| **Privacy** | `delete_user_data()` removes all user data across all tables |

---

## Layer Roadmap

| Layer | Scope |
|-------|-------|
| **0 – Foundation** *(this branch)* | Repo skeleton, contracts, LLM gateway, storage, tracing, health endpoint |
| **1 – Safety + Emotion** | Safety Sentinel (rule-based), emotion classifier, escalation protocol |
| **2 – Knowledge Base / RAG** | ChromaDB, sentence-transformers, ingestion pipeline |
| **3 – Explorer + State Model** | Emotional-state model, narrowing-question logic |
| **4 – Role Agents + Conductor + Critic** | All four role agents, Conductor, Critic |
| **5 – Memory / Profile / Privacy** | Cross-session memory, user profile, user data deletion |
| **6 – Frontend** | Session UI with explanation panel |
| **7 – Evaluation** | Automated evaluation suite using TurnTraces |

---

## Quick Start

```bash
# 1. Clone and create venv
git clone <repo>
cd solace
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env
# Edit .env and add at least one LLM API key

# 4. Run
uvicorn app.main:app --reload

# 5. Health check
curl http://localhost:8000/health

# 6. Smoke-test LLM gateway
python -m scripts.smoke_llm
# Optional: target one provider/task; model catalogs are also available separately.
python -m scripts.smoke_llm --provider gemini --task converse
python -m scripts.check_models
```

Daily call caps are conservative local guardrails configured with the `*_DAILY_CAP`
environment variables; they are not claims about provider quota limits.

---

## Stack

- **Python 3.11+**, FastAPI, Pydantic v2, SQLAlchemy 2.0 + SQLite
- **LLMs**: Gemini (google-genai), Groq, OpenRouter, Mistral, Cerebras (all free-tier, OpenAI-compatible)
- **RAG** *(Layer 2)*: ChromaDB + local sentence-transformers
- **Tests**: pytest

> ⚠️ Free-tier LLM providers may log request data. Development and testing use **synthetic data only**.

---

## Safety and limitations

Solace is not a substitute for professional care and does not diagnose or provide
clinical treatment. If you are in crisis or immediate danger, contact local emergency
services or a crisis hotline in your area. Free-tier LLM providers may log request data;
use synthetic data only during development and testing.

---

## Project Structure

```
solace/
├── app/
│   ├── contracts/      # Shared Pydantic v2 models (the cross-layer interface)
│   ├── llm/            # LLM gateway, providers, resilience
│   ├── storage/        # SQLAlchemy ORM + repository functions
│   ├── tracing/        # TraceRecorder, TurnTrace assembly
│   ├── agents/         # Role agents (Layer 4)
│   ├── rag/            # Retrieval pipeline (Layer 2)
│   └── safety/         # Safety Sentinel (Layer 1)
├── docs/               # HANDOFF.md, DECISIONS.md, ARCHITECTURE.md
├── scripts/            # Developer utilities (smoke_llm, etc.)
├── tests/              # pytest suite
├── AGENTS.md           # Full project context for AI coding agents
└── pyproject.toml
```

---

## Contributing / Agent Handoff

See [AGENTS.md](./AGENTS.md) for the full project context, coding conventions, and the mandatory **handoff protocol** that every AI coding session must follow.
