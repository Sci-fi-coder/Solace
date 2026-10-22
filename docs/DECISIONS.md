# DECISIONS.md — Architecture & Dependency Decisions

> Append non-obvious choices here with date, rationale, and alternatives considered.
> Never delete entries — only add.

---

## 2026-10-23 — Layer 0 Foundation

### D-001: SQLite over PostgreSQL for development
**Decision:** Use SQLite (`sqlite:///./data/solace.db`) as the default database.
**Rationale:** Zero infrastructure required; trivially portable; sufficient for a capstone with < 100 concurrent sessions. The entire swap to PostgreSQL is a single `DATABASE_URL` change — SQLAlchemy abstracts the rest.
**Alternative:** PostgreSQL via Docker. Adds operational complexity with no benefit at this stage.

### D-002: `tenacity` for retry / backoff
**Decision:** Use the `tenacity` library for LLM provider retry logic rather than hand-rolling exponential backoff.
**Rationale:** Battle-tested, composable decorators; supports jitter out of the box; integrates cleanly with the provider class pattern. Saves ~50 lines of fragile custom code.
**Alternative:** `asyncio.sleep` loops. Rejected: error-prone, harder to test.

### D-003: `google-genai` (not `google-generativeai`)
**Decision:** Use `google-genai>=1.0.0` (the new unified Gemini SDK) not the older `google-generativeai`.
**Rationale:** `google-genai` is Google's current recommended SDK for Gemini; `google-generativeai` is deprecated. Method: `client.models.generate_content(model=..., contents=...)`.
**Alternative:** `google-generativeai`. Rejected: deprecated, inconsistent API surface.

### D-004: OpenAI SDK for all non-Gemini providers
**Decision:** Use the official `openai` Python SDK (≥1.35) to talk to Groq, OpenRouter, Mistral, and Cerebras by setting `base_url` and `api_key`.
**Rationale:** All four providers expose an OpenAI-compatible `/chat/completions` endpoint. One client, four providers, zero extra dependencies.
**Alternative:** `httpx` calls directly. Rejected: the `openai` SDK handles streaming, retries, and structured output better.

### D-005: Synchronous storage layer
**Decision:** Storage repository functions are synchronous (use `sqlalchemy` Session, not `AsyncSession`).
**Rationale:** FastAPI with sync endpoints is simpler and sufficient for a single-user capstone. Async SQLAlchemy requires `aiosqlite` and `run_sync` wrappers that add complexity without benefit at this scale. `aiosqlite` is still declared as a dependency to smooth the upgrade path.
**Alternative:** Full async with `AsyncSession`. Deferred to Layer 5+ if needed.

### D-006: MockProvider for tests
**Decision:** `MockProvider` returns scripted, deterministic outputs keyed by `TaskKind`. Tests never call real APIs.
**Rationale:** Free-tier API keys are rate-limited; real calls in tests would be flaky, slow, and could leak synthetic data to external logs.
**Alternative:** VCR cassettes (record/replay). Rejected: more setup; MockProvider is simpler and fully deterministic.

### D-007: `pre_send_filter` as a list of callables
**Decision:** `LLMGateway` holds a `pre_send_filter: list[Callable]` applied to message lists before dispatch.
**Rationale:** Clean plugin point for a PII scrubber (Layer 5) without touching provider code. Identity by default (no overhead).
**Alternative:** Middleware pattern. Rejected: overkill for a list of message dicts.

### D-008: Per-provider per-day call counter in SQLite
**Decision:** `provider_usage` table tracks `(provider, date, call_count)`. Gateway checks this before dispatching and skips providers over their daily cap.
**Rationale:** Free-tier providers have hard daily limits. Soft cap in config + DB counter = no surprise rate-limit failures mid-session.
**Alternative:** In-memory counter. Rejected: resets on server restart; counter would be wrong after a crash.

### D-009: Contracts package has zero internal imports
**Decision:** `app/contracts/__init__.py` imports only from `pydantic`, `enum`, and `datetime`. No imports from `app.llm`, `app.storage`, etc.
**Rationale:** Contracts are the shared interface between all layers. Circular import risk is eliminated by design. Any layer can safely import contracts.
**Alternative:** Scatter models across packages. Rejected: creates coupling and circular import risk.

### D-010: `pydantic-settings` for configuration
**Decision:** `app/config.py` uses `pydantic-settings` `BaseSettings` with `.env` file support.
**Rationale:** Type-safe, validated config with automatic env-var loading. All provider secrets, caps, and gateway params in one place.
**Alternative:** `os.getenv` calls scattered through the code. Rejected: untestable, no type safety, easy to miss a variable.

### D-011: Gateway-managed retry and cooldown
**Decision:** Keep the gateway's small retry loop and remove the unused `tenacity` dependency.
**Rationale:** Rate limits must immediately set provider cooldown and fall through; transient
provider errors use the gateway's configured backoff. Keeping these transitions explicit avoids
an additional retry abstraction that was not used by the implementation.
**Alternative:** `tenacity`. Rejected for Layer 0 because the gateway already owns fallback and
cooldown decisions.

### D-012: Defer async SQLite dependency
**Decision:** Remove `aiosqlite` until the storage layer actually adopts asynchronous SQLAlchemy.
**Rationale:** All Layer 0 repository operations are synchronous, so the driver is unused.
**Alternative:** Retain a future-use dependency. Rejected to keep installed dependencies aligned
with the implemented foundation.

### D-013: Inject the gateway usage database
**Decision:** Require a SQLAlchemy engine when building a configured gateway.
**Rationale:** The configured gateway must persist provider daily-cap reservations. Injecting the
engine keeps the LLM layer independent of storage implementation details.
**Alternative:** Let the gateway construct its own engine. Rejected to preserve the architecture's
LLM-to-storage dependency boundary.
