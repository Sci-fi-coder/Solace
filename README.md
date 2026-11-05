# Solace

Solace is a session-based counselling conversation prototype. It guides a conversation through
check-in, exploration, understanding, guidance, and closing instead of answering as a general
purpose chatbot. It is an educational software project, not a therapist, diagnostic system, or
clinically validated intervention.

## What is implemented

- A rule-based safety gate that runs before model calls and routes crisis messages to a fixed
  response.
- A structured conversation planner, deterministic phase and listening guardrails, and four
  speaker roles: Counsellor, Mentor, Teacher, and Peer.
- A local counselling knowledge base with retrieval for self-help cards and, when configured,
  ChromaDB passages.
- Separate optional historical role-model retrieval across 24 YAML cards.
- A local reply critic, persistent sessions and traces, and a Streamlit chat interface.
- A FastAPI application exposing a health endpoint; the interactive chat is provided by
  Streamlit.

## Conversation flow

For each message, `SolaceEngine.handle_turn()`:

1. Runs the deterministic safety rules first. A crisis result returns a fixed response without an
   LLM call; elevated safety also suppresses historical context.
2. Uses a structured planner call to propose an emotional-state update, phase, role, move, and
   counselling-retrieval request. A deterministic planner is used if planning fails.
3. Applies deterministic guardrails for session phase, listening, clarification, advice, safety,
   and frequency limits.
4. Retrieves counselling cards/passages when the planned move calls for them. The baseline
   retriever is keyword-based; ChromaDB is optional and can be selected by configuration.
5. Independently evaluates historical cards. Historical context is passed to generation only
   when the UI toggle is enabled, a relevant card passes the score threshold, and the historical
   eligibility rules permit it.
6. Generates a role-specific reply and checks it with local critic rules. A failed draft can be
   retried once with feedback; repeated failure produces a safe fallback.
7. Persists the conversation, state, retrieval evidence, decision, critic verdict, and reply in a
   `TurnTrace`. The UI shows the reply, session phase, state summary, explanation, and any
   validated provenance.

The counselling knowledge base and historical cards are separate sources. Historical cards are
not counselling techniques, are not inserted into counselling RAG, and do not replace the
selected speaker's response.

## Historical inspirations

The sidebar option **Enable historical inspirations** is off by default. Retrieval is local and
hybrid:

- Semantic similarity uses the local `sentence-transformers/all-MiniLM-L6-v2` embedder.
- Theme overlap and emotional/context compatibility use weighted overlap with supported card
  tags and emotion labels.
- The current configuration weights semantic/theme/context evidence at `0.90 / 0.05 / 0.05`,
  requires a minimum score of `0.45`, and has no separate top-match margin.
- The query uses the current and up to four recent user messages plus selected emotional-state
  fields. It does not use assistant-generated text.
- Naming a supported person narrows candidate cards to that person but does not raise the score
  with an explicit bonus; the name remains part of the semantic query. It does not bypass the
  relevance threshold, so an irrelevant named-person request can still abstain.

Historical use is suppressed for elevated or crisis safety, after a crisis in the session, while
venting, for reflect/follow-up/closing moves, and when per-session use limits apply. Clarifying
turns are eligible only with an explicit request. Without an explicit request, use is limited to
understanding/guidance and separated by at least five turns. An explicit request can relax phase
and spacing restrictions and permit a clarifying turn, but cannot bypass relevance, safety,
venting/listening restrictions, blocked moves, or the two-use session cap.

The local critic checks the selected person's name, analogy framing, unsupported names,
detectable factual claims, numbers, quotations, and whether the reply declares the selected card
as used. For an unreviewed card, the reply must label the material as unverified, and detectable
historical fact claims are rejected. The critic is a deterministic guardrail, not a comprehensive
historical fact-checker. Provenance appears only when a selected card is declared as used and the
reply passes validation; it contains the card title and source locator(s).

### Card inventory and review state

There are six cards for each of Gandhi, Vivekananda, Marcus Aurelius, and A.P.J. Abdul Kalam.
The table reports status in the current working tree. The baseline implementation commit does
not include the pre-existing YAML edits; those files remain outside that commit.

| Person | Card ID | Title | Current review / verification status |
|---|---|---|---|
| Mahatma Gandhi | `gandhi-noakhali-peace` | Taking patient steps toward reconciliation | Reviewed / verified |
| Mahatma Gandhi | `gandhi-pausing-campaign` | Pausing when a method causes harm | Reviewed / verified |
| Mahatma Gandhi | `gandhi-pietermaritzburg` | Responding to humiliation with purpose | Unreviewed / needs_verification |
| Mahatma Gandhi | `gandhi-satyagraha-strategy` | Choosing a deliberate response to injustice | Unreviewed / needs_verification |
| Mahatma Gandhi | `gandhi-school-truth` | Truthfulness despite insecurity | Unreviewed / needs_verification |
| Mahatma Gandhi | `gandhi-shared-work` | Building solidarity through shared work | Reviewed / verified |
| A.P.J. Abdul Kalam | `kalam-empower-young-scientists` | Making room for emerging contributors | Reviewed / verified |
| A.P.J. Abdul Kalam | `kalam-ethical-leadership` | Keeping purpose and ethics connected | Unreviewed / needs_verification |
| A.P.J. Abdul Kalam | `kalam-long-term-capability` | Building capability one stage at a time | Unreviewed / needs_verification |
| A.P.J. Abdul Kalam | `kalam-own-failure-share-success` | Owning failure and sharing success | Unreviewed / needs_verification |
| A.P.J. Abdul Kalam | `kalam-programme-ownership` | Building ownership across a team | Unreviewed / needs_verification |
| A.P.J. Abdul Kalam | `kalam-structured-reviews` | Using review to learn and improve | Unreviewed / needs_verification |
| Marcus Aurelius | `marcus-control-response` | Focus on the part you can choose | Unreviewed / needs_verification |
| Marcus Aurelius | `marcus-cooperation` | Remembering the value of cooperation | Reviewed / verified |
| Marcus Aurelius | `marcus-inner-retreat` | Creating a pause within | Unreviewed / needs_verification |
| Marcus Aurelius | `marcus-judgement` | Separating an event from your judgement | Unreviewed / needs_verification |
| Marcus Aurelius | `marcus-morning-duty` | Beginning despite reluctance | Unreviewed / needs_verification |
| Marcus Aurelius | `marcus-temper-response` | Choosing a measured response | Reviewed / verified |
| Swami Vivekananda | `vivekananda-chicago` | Taking a step into the unknown | Reviewed / verified |
| Swami Vivekananda | `vivekananda-family-hardship` | Finding direction during family hardship | Reviewed / verified |
| Swami Vivekananda | `vivekananda-service-others` | Connecting purpose with helping others | Unreviewed / needs_verification |
| Swami Vivekananda | `vivekananda-small-steady-work` | Respecting steady, ordinary effort | Unreviewed / needs_verification |
| Swami Vivekananda | `vivekananda-strength-self-belief` | Rebuilding confidence through self-belief | Unreviewed / needs_verification |
| Swami Vivekananda | `vivekananda-training-leaders` | Allowing others room to grow | Unreviewed / needs_verification |

Review status is preserved from each card; retrieval does not promote a card. Eight cards are
reviewed in the current working tree and sixteen remain unreviewed. Check each card's YAML
metadata for its source locator and verification notes.

## Technology

- Python 3.11+
- FastAPI and Uvicorn (the current API exposes `/health`)
- Streamlit for the interactive UI
- Pydantic v2, SQLAlchemy 2, and SQLite
- Configurable LLM gateway for Gemini and OpenAI-compatible providers, including Groq,
  OpenRouter, Mistral, and Cerebras
- Local sentence-transformers embeddings for historical retrieval; optional ChromaDB and
  sentence-transformers support for counselling RAG
- pytest and Ruff

## Setup

From the repository root, using PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rag]"
python -m pip install -r requirements-ui.txt
```

For normal provider-backed use, copy `.env.example` to `.env` and configure at least one
provider key. Settings are loaded from environment variables or `.env`; keep credentials out of
source control. Provider requests may be logged by free-tier services, so use synthetic text
during development and testing.

The local embedding model must already be available for semantic historical retrieval. The
embedder loads lazily and may fetch weights on first use if they are not cached. For an offline
run, make the model available locally first; if embeddings cannot load, historical retrieval
abstains. The evaluator forces offline mode and requires cached model weights.

## Run

Start the Streamlit application in its local scripted demo mode (no provider keys or network
calls):

```powershell
$env:SOLACE_UI_DEMO = "1"
streamlit run ui/app.py
```

The demo is scripted; it demonstrates UI behavior but does not exercise historical retrieval or
provide a live safety assessment. For the provider-backed application, configure a provider key
and run `streamlit run ui/app.py` without `SOLACE_UI_DEMO=1`. The command-line client is
available with `python -m scripts.chat_cli`.

The FastAPI health-only service can be started separately:

```powershell
uvicorn app.main:app --reload
```

Then visit `http://127.0.0.1:8000/health`.

## Tests and retrieval evaluation

```powershell
python -m pytest -q
ruff check .
```

To print the deterministic historical retrieval evaluation report:

```powershell
python -m scripts.eval_rolemodels
```

The evaluator uses cached local model weights in offline mode; it does not download weights or
call an external service. Its small synthetic English-language dataset produced:

- Precision@1: `9/34` (`26.5%`)
- Recall@3: `34/34` (`100%`)
- No-match accuracy: `18/18` (`100%`)
- False-positive rate: `0/18` (`0%`)
- Relevant-query match coverage: `11/34` (`32.4%`)
- Precision among returned relevant-query matches: `9/11` (`81.8%`)
- Correct top-1 results on `8/24` distinct card targets

These are results on a small synthetic dataset, not validated semantic accuracy or evidence of
clinical effectiveness. Matching coverage is limited, especially without explicit names; broader
independently authored evaluation and human review are needed.

## Limitations and safety

Solace does not diagnose, provide clinical treatment, or replace professional care. The rule-based
safety gate and local critics are safeguards, not a clinical safety guarantee. Historical cards
are paraphrased, source review is required, and the historical critic cannot establish that a
source claim is true. The retrieval evaluation is small, synthetic, and English-only. External
LLM providers may retain or log requests; do not use real or sensitive user content for
development or testing.
