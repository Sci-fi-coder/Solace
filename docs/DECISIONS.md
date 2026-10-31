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

### D-014: Classify provider errors at the gateway boundary
**Decision:** Map recognized HTTP authentication, bad-request/model, rate-limit, and transient
failures to explicit provider exception types. Retry only transient errors, cool down 429/quota
failures, and disable authentication-failing providers for the process. Unknown exceptions
propagate unchanged.
**Rationale:** A provider's HTTP status and underlying message must remain visible; stale models
and authentication failures must not be mislabeled as rate limits or hidden by a fallback.
Fallback results carry structured diagnostics for preceding failed or skipped providers.
**Alternative:** Retry all `ProviderError` instances. Rejected because it retried permanent
request/authentication failures and obscured their causes.

### D-015: Keep provider models current and overridable
**Decision:** Use `gemini-2.5-flash`, `llama-3.1-8b-instant`,
`mistralai/mistral-small-2603`, and `qwen-3.8-27b` as the Gemini, Groq, OpenRouter, and
Cerebras defaults; retain `mistral-small-latest` for Mistral. All remain overridable by
environment settings and fallback-chain `provider/model` entries.
**Rationale:** Direct Groq and Cerebras calls proved their prior configured models unavailable.
Gemini's configured model was confirmed in its live catalog, and the public OpenRouter catalog
confirmed its prior default was absent and the selected replacement present. Groq and Cerebras
catalog requests were blocked by Cloudflare HTTP 403/Error 1010 in this environment; their
replacement IDs are listed in their official supported-model catalogs. Mistral's model-list API
requires authentication and no Mistral key is configured, so its existing alias was retained.
**Alternative:** Guess replacements without checking provider evidence. Rejected; inaccessible
catalogs are reported rather than treated as successful checks.

### D-016: Conservative local daily-call guardrails
**Decision:** Set each provider's default local daily cap to 20 calls, configurable through its
`*_DAILY_CAP` environment variable.
**Rationale:** These low defaults reduce accidental usage while model/provider compatibility is
being validated. They are application guardrails, not claims about provider quotas or rate limits.
**Alternative:** Encode provider free-tier limits as caps. Rejected because limits vary by
account, plan, and time and were not verified for every provider.

### D-017: Make provider configuration and fallback diagnostics observable
**Decision:** Log the absolute dotenv path and configured provider names, warn only with names of
settings that conflict between process environment and dotenv, and expose model-catalog checking
at smoke-test startup. Smoke tests select one task by default and accept explicit provider/task
arguments.
**Rationale:** Configuration provenance and fallback causes must be debuggable without exposing
credential values or spending multiple calls per smoke run.
**Alternative:** Log raw configuration and run every task in each smoke test. Rejected for
credential safety and avoidable usage.

---

## 2026-10-26 — Gateway Fallthrough and Resilience

### D-018: Catch unknown provider exceptions and continue the fallback chain
**Decision:** Catch unhandled/unexpected provider exceptions in the gateway fallback loop, log a warning, record them in `attempts` (`{provider, model, outcome: "failed", error_summary: "<Type>: <redacted message>"}`), and continue to the next provider. If all providers fail, `GatewayExhaustedError` raises with the full history of all attempts.
**Rationale:** This replaces the earlier D-014 rule ("unknown exceptions surface unchanged"). An unhandled exception or non-standard status code (such as a 402 or client status error) from one provider should not kill the entire fallback chain when healthy fallback providers are available.
**Alternative:** Surface unknown exceptions immediately. Rejected because a single odd failure halts the entire session rather than falling through.

### D-019: Classify 402 and quota/billing errors as BillingOrQuota and disable provider
**Decision:** Map HTTP 402 and quota/billing errors to a dedicated `BillingOrQuota(ProviderError)` exception class. When encountered, disable the provider process-wide and fall through to the next provider.
**Rationale:** Account credit exhaustion and payment requirements are persistent conditions that do not resolve with backoff or cooldowns. Disabling the provider process-wide avoids repeatedly hammering a known-exhausted account on later turns while clearly reporting the cause.
**Alternative:** Treat 402 as a transient error or rate limit. Rejected because 402 requires manual billing/account intervention and retrying wastes time and latency.

### D-020: Use provider SDKs / OpenAI client for model catalog checking
**Decision:** In `scripts/check_models.py`, use the official `google-genai` and `openai` clients to retrieve model catalogs rather than raw `urllib` HTTP requests. Default output displays warnings only, reserving full catalog dumps for `--verbose`.
**Rationale:** Raw HTTP requests triggered Cloudflare HTTP 403 / Error 1010 on Groq and Cerebras endpoints in this environment. The SDK clients handle TLS and User-Agent headers properly. Defaulting to warning-only output keeps smoke CLI runs clean and uncluttered.
**Alternative:** Header spoofing with raw HTTP clients. Rejected as fragile when official client libraries are already installed dependencies.

## 2026-10-28 — Verified Gemini default

### D-021: Use `gemini-3.1-flash-lite` as the Gemini default
**Decision:** Keep `gemini-3.1-flash-lite` as the Gemini default. It is the first passing model among the three requested candidates in priority order, and its name contains neither `preview` nor `latest`.
**Verification:** Ran `python -m scripts.smoke_llm --provider gemini` with each candidate selected through the app configuration, using the synthetic prompt from the smoke script:
- `gemini-2.5-flash`: **FAIL**, HTTP 404 `NOT_FOUND`; exact API message: `This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash for the latest features and improvements. We recommend you to use the Interactions API (https://ai.google.dev/gemini-api/docs/get-started).`
- `gemini-2.5-flash-lite`: **FAIL**, HTTP 404 `NOT_FOUND`; exact API message: `This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemini-3.5-flash-lite for the latest features and improvements. We recommend you to use the Interactions API (https://ai.google.dev/gemini-api/docs/get-started).`
- `gemini-3.1-flash-lite`: **PASS**, HTTP 200; provider `gemini` returned the exact synthetic reply `Smoke test OK.` in 3779 ms.
**Rationale:** Catalog visibility did not establish that the account could generate with the older models. The requested live synthetic generation test resolved that conflict; the third candidate is the first available and replying model.

### D-022: Keep Groq behind Gemini for small tasks
**Decision:** Use `openai/gpt-oss-20b` as the Groq default and retain `openai/gpt-oss-120b` only as a documented optional alternative. Converse remains Gemini-first. Classify, judge, and extract try the verified `gemini-3.1-flash-lite` first and Groq GPT-OSS 20B second.
**Rationale:** The Gemini Flash Lite candidate passed a real synthetic generation call. Groq's free-tier token-per-minute limit makes it a fallback for small calls rather than the primary route. No default chain includes Cerebras.

### D-023: Preserve contract enum string formatting while satisfying Ruff
**Decision:** Use Python 3.11 `StrEnum` for contract enums through a shared base that keeps the prior `str(EnumMember)` representation.
**Rationale:** The repository-wide Ruff requirement flags the previous `str, Enum` pattern. The custom representation preserves the prior public formatting while retaining string equality and serialization behavior; a contract test guards that behavior.

### D-024: Bound GPT-OSS reasoning retries and keep reasoning private
**Decision:** Send configured reasoning effort only for model IDs beginning `openai/gpt-oss`. If that parameter is rejected, retry once without it. Exclude separate reasoning fields and tagged reasoning from `LLMResult.text`. If GPT-OSS returns an empty answer with finish reason `length`, retry once with a larger token limit, then raise `EmptyResponseError` so the gateway can fall through.
**Verification:** Mocked provider tests cover all four task effort defaults, unsupported-parameter retry, reasoning exclusion, successful larger-budget retry, and exhaustion falling through to a backup provider.

### D-025: Honor provider retry hints without logging error content
**Decision:** Use numeric or HTTP-date `Retry-After` headers, millisecond retry headers, and structured or textual retry hints in error bodies as cooldown duration. A zero-second hint is honored. Cooldown logs state that a retry hint was used but never log the provider error message.
**Verification:** Mocked tests cover header/body parsing and assert a 17-second cooldown while confirming synthetic error content is absent from logs.

### D-026: Keep blank example settings safe
**Decision:** Set `env_ignore_empty=True` and leave every `.env.example` value blank, including model, provider-enable, limit, reasoning, and fallback variables.
**Rationale:** This meets the empty-template requirement while allowing the application defaults in `Settings` to remain effective when a user copies the example unchanged.
**Verification:** Tests compare all `Settings` fields to the example keys and confirm representative blank model, boolean, and integer values preserve their defaults.

## 2026-10-28 — Layer 1 Safety and Emotion

### D-027: Keep deterministic safety rules as the floor
**Decision:** Run reviewable YAML rule patterns before the gateway safety classifier, take the
maximum resulting risk level, and treat classifier failure/cap exhaustion as a degraded rules-only
result. A crisis result sets a sticky session flag that keeps later turns at least elevated.
**Rationale:** A model outage or erroneous downgrade must not erase a deterministic safety signal.
Patterns remain in data files so reviewers can audit language and update them without code edits.
Nearby intent negation is scoped to the matched phrase, while idioms are excluded by using
explicit self/other-harm language rather than broad death/violence keywords.
**Alternative:** Ask the LLM to arbitrate or replace rule matches. Rejected because it could
downgrade a rule hit and fail open during provider outages.

### D-028: Keep crisis language non-generative and helplines unverified by default
**Decision:** Crisis replies are fixed constants in `app/safety/responses.py`; no crisis reply
uses an LLM. Helpline entries use `TODO_VERIFY`, `verified: false`, and emit a warning when
loaded. The crisis template links to a location-specific directory rather than inventing a phone
number.
**Rationale:** A locally reviewed response avoids generation variability, while country-specific
numbers must be verified before they are shown as reliable resources.

### D-029: Add safety diagnostics and fine/coarse emotion to contracts
**Decision:** Add degradation/provider/model/attempt fields to `SafetyResult`, and fine/coarse
label fields to `EmotionReading` while preserving existing fields and defaults.
**Rationale:** The trace must reveal provider fallback and degraded operation, and emotion callers
need both the original GoEmotions label and a coarser category. `TraceRecorder` stores summary-only
step records; no message text is included in log messages or step summaries.

### D-030: Keep local GoEmotions inference lazy and optional
**Decision:** Put `torch` and `transformers` in the optional `[ml]` extra, default to the requested
`SamLowe/roberta-base-go_emotions` model, load it only on first inference, and force CPU
execution. Tests inject a classifier and do not download weights.
**Rationale:** Layer 0 installs remain lightweight and tests deterministic; CPU inference works
without requiring a GPU. The score is not calibrated as emotional intensity. The optional
dependency remains separate from the core installation; CPU-only installation uses PyTorch's CPU
package index.
**Verification:** The official CPU index listed Python 3.13 Windows wheels starting with torch
2.6.0; the installed Python 3.13.2 environment loaded the selected model and evaluated all
synthetic examples successfully.

### D-031: Make LLM evaluation opt-in and cache successful calls
**Decision:** Safety evaluation defaults to rules-only; `--with-llm` enables gateway calls and
stores successful structured results under `.cache/`, keyed by a SHA-256 digest of model chain,
current message, and recent user context.
**Rationale:** Synthetic evaluation should not spend free-tier quota on repeated runs. Including
recent context prevents a prior-turn change from incorrectly reusing a multi-turn classification.

## 2026-10-29 — Unicode safety normalization

### D-032: Normalize Unicode before safety rule matching
**Decision:** Normalize incoming safety text with NFKC before matching; canonicalize Unicode
apostrophe and quotation characters to ASCII apostrophe, convert Unicode whitespace to ASCII
spaces, map U+200B ZERO WIDTH SPACE to an ASCII space, and remove other zero-width format
characters and default-ignorable marks (including variation selectors and the combining
grapheme joiner).
**Rationale:** Visually equivalent punctuation and fullwidth text must not bypass deterministic
safety patterns or negation checks. Treating U+200B as whitespace preserves word boundaries,
while removing other format characters recovers words split internally by invisible formatting.
**Alternative:** Add pattern-specific Unicode spellings or delete every invisible separator
without regard to token boundaries. Rejected because pattern-specific variants are brittle and
deleting a word separator can merge tokens and evade a rule.
**Pattern audit:** English safety expressions use only ASCII apostrophes; Hinglish expressions
have no apostrophes; neither YAML file contains non-ASCII punctuation. No pattern edits were
necessary.

## 2026-10-29 — Safety phrase families and emotion isolation

### D-033: Expand safety coverage with phrase families and keep emotion out of safety decisions
**Decision:** Add reviewable English and Hinglish YAML families for passive ideation and explicit
self-harm expressions, including normalized leetspeak and bare fragments. Suppress explicit-family
crisis matches when the phrase is locally framed as an elevated urge, preserving existing
non-crisis self-harm-urge behavior. Keep the GoEmotions output entirely separate from safety
classification; it must never be used to soften, redirect, or override the Safety Sentinel level.
**Rationale:** Evaluation found missing phrase coverage, not a leetspeak normalization defect:
`k1ll mys3lf` already normalizes to `kill myself`, but the previous crisis patterns required
surrounding intent or additional context. The GoEmotions model can map passive-ideation text to
positive/desire labels, so its emotion result is not safe to treat as a safety signal.
**Verification:** Added positive paraphrases and hard negatives for both phrase families. The
final 129-example rules-only evaluation achieved crisis recall 1.000 (36/36), crisis precision
1.000 (36/36), and none false-positive rate 0.000 (0/42).

### D-034: Make Layer 1 smoke-test model use explicit
**Decision:** Keep the smoke script rules-only for safety by default, enable the LLM safety
classifier only with `--with-llm`, support skipping emotion inference with `--no-emotion`, and
reserve complete output for `--verbose`.
**Rationale:** Local safety-rule checks should not require an LLM call, while model downloads and
full diagnostic output should be explicit user choices. Existing provider and model defaults are
unchanged.

### D-035: Isolate smoke-test inputs unless a session is named
**Decision:** Use fresh sessions per input by default; `--session SESSION_ID` explicitly reuses
sticky state, while `--fresh` makes the default isolation explicit.
**Rationale:** Manual phrase checks must expose the current rule result without prior crisis turns
masking it through sticky escalation.

### D-036: Initialize the temporary evaluation gateway database
**Decision:** Create the existing SQLAlchemy tables before initializing an LLM gateway in the
Layer 1 smoke and safety-evaluation scripts.
**Rationale:** The gateway loads and enforces its provider daily-call counter from `provider_usage`;
an in-memory SQLite engine without that table aborts live evaluation before classification.
The initializer reuses the storage schema and leaves provider/model defaults unchanged.

## 2026-10-30 — Layer 2a knowledge cards

### D-037: Use PyMuPDF for local page-preserving extraction
**Decision:** Use PyMuPDF (`fitz`) for extracting one text record per PDF page and for the
synthetic-PDF extraction test.
**Rationale:** PyMuPDF is installed in the Python 3.13 environment and can preserve 1-based page
boundaries while handling the illustrated WHO guide. PDFs remain outside the repository; extracted
JSON is written under ignored `data/kb_work/`.
**Alternative:** pypdf. Rejected for this change because it is not installed in the environment;
the selected library satisfies the Python 3.13 compatibility requirement.

### D-038: Match source PDFs by the exact registry filename
**Decision:** Only extract files whose names exactly match `sources.yaml`; report all other files
as unmatched and all expected files that are absent.
**Rationale:** The PM Plus mapping requires `WHO_MSD_MER_18_5_eng.pdf`, while the local directory
contains `WHO_MSD_MER_18.5_eng.pdf`. Since the mapping explicitly forbids guessing or renaming,
the dotted filename is not treated as PM Plus.

### D-039: Keep card drafting resumable and isolated from dotenv files
**Decision:** Cache each successful structured response by SHA-256 of source id, PDF page numbers,
and the complete prompt. The drafting CLI initializes the configured gateway from process
environment settings while its working directory is a new empty temporary directory.
**Rationale:** Successful sections can be reused after quota exhaustion without resending source
text, and gateway initialization cannot load the repository's `.env`. Provider request usage is
measured from the gateway's in-memory SQLite counter; defaults and model names remain unchanged.
