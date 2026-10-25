# HANDOFF.md — Solace Session Log

> **AI coding agent:** Read this file after AGENTS.md at every session start.
> Update it at session end per the HANDOFF PROTOCOL.

---

## Current Branch
`fix-gateway-fallthrough`

---

## Session: 2026-10-26 — Gateway fallthrough resilience and error classification

### Done
- Replaced the rule "unknown exceptions surface unchanged": unexpected exceptions from any provider
  now log a warning, are recorded in `attempts` (`{provider, model, outcome: "failed", error_summary: "<Type>: <redacted message>"}`),
  and continue fallback to the next provider instead of terminating the chain.
- Added `BillingOrQuota` error class (`app/llm/providers.py`), classifying HTTP 402 and quota/billing
  exhaustion errors. The gateway now disables that provider process-wide, logs a clear warning, and
  falls through to the next provider.
- `GatewayExhaustedError` now preserves all provider attempts both in its message and as a structured
  `.attempts` attribute.
- Re-architected `scripts/check_models.py` to use `google-genai` and `openai` clients directly instead
  of raw `urllib` HTTP calls. This resolves Cloudflare HTTP 403 / Error 1010 blocks for Groq and
  Cerebras catalogs. Default output now prints warnings only; complete model lists are displayed
  with `--verbose`.
- Fixed pre-existing Ruff `UP017` in `app/main.py` by using `datetime.now(UTC)` and confirmed all checks pass.
- Added comprehensive unit tests in `tests/test_gateway.py` covering unknown exception fallthrough,
  HTTP 402 provider disabling, and all-fail attempts listing.
- Observed provider usability:
  - **Groq is currently unusable on this account**: Configured model `llama-3.1-8b-instant` returns
    HTTP 404 `model_not_found`. The catalog accessible to this key contains only specialized models
    (whisper, prompt-guard, orpheus, allam) and no general llama chat models.
  - **Cerebras is currently unusable on this account**: Model `qwen-3.8-27b` is present in catalog,
    but completion calls fail with HTTP 402 `payment_required` ("Payment required to access this resource.
    Visit your billing tab."), indicating exhausted credits / billing setup required.
  - **Gemini**: Serving successfully using `gemini-3-flash-preview` (`gemini-2.5-flash` returned HTTP 404
    indicating deprecation for new users).

### Next
1. Start **Layer 1** only: implement `app/safety/sentinel.py` as the rule-based Safety Sentinel
   returning `SafetyResult`; add focused tests before wiring any LLM path.

### Known Issues
- Groq and Cerebras are currently unusable with this account's credentials due to missing models / billing limits.
- Mistral and OpenRouter have no credentials configured in this environment.
- Layer 1 safety and emotion processing are not implemented; Layer 0 exposes only `/health`.
- The optional `[rag]` dependency extra has not been installed or validated.

### Verification
- `pytest -q`: 72 passed.
- `ruff check app/main.py app/llm/ scripts/check_models.py tests/test_gateway.py`: all checks passed.
- `python -m scripts.check_models`: warned on groq configured model missing, no Cloudflare 403 blocks.
- `python -m scripts.check_models --verbose`: displayed live catalogs for gemini (60), groq (11), cerebras (2).
- `python -m scripts.smoke_llm`: passed via fallback to gemini.
- `python -m scripts.smoke_llm --provider cerebras`: cleanly disabled cerebras after HTTP 402 and reported attempt.

---

## Session: 2026-10-23 — Layer 0 gateway diagnostics and fixes

### Done
- Reproduced the direct provider failures: Groq returned HTTP 400 `model_decommissioned` for
  `llama3-8b-8192`; Cerebras returned HTTP 404 `model_not_found` for `llama3.1-8b`.
- Added provider error classes and gateway handling for rate limits, authentication, bad
  requests/model errors, transient failures, and unknown exceptions. Authentication failures
  disable that provider process-wide; unknown exceptions propagate unchanged.
- Added structured fallback attempts to `LLMResult`, including explicit skipped reasons, and
  retained `provider/model` fallback-chain overrides.
- Treated blank and placeholder credentials as unconfigured; added safe dotenv/provider startup
  diagnostics and conflict warnings that disclose variable names only.
- Added `scripts/check_models.py`; updated model defaults/examples, conservative configurable
  local daily caps, README usage, and mock-based retry/fallback tests.
- Updated `scripts/smoke_llm.py` with `--provider`, `--task`, and one-task default behavior.
- Ran the Gemini smoke call successfully with the synthetic prompt; it served on
  `gemini-2.5-flash`.

### Next
1. Start **Layer 1** only: implement `app/safety/sentinel.py` as the rule-based Safety Sentinel
   returning `SafetyResult`; add focused tests before wiring any LLM path.

### Known Issues
- Groq and Cerebras model-list requests are blocked by Cloudflare HTTP 403/Error 1010 in this
  environment. Their current replacement model IDs are present in the official catalogs, but
  `scripts/check_models.py` should be rerun where those endpoints are reachable.
- Mistral's model-list endpoint requires authentication; no Mistral key is configured, so the
  existing `mistral-small-latest` alias could not be checked against the authenticated catalog.
- Layer 1 safety and emotion processing are not implemented; Layer 0 exposes only `/health`.
- The optional `[rag]` dependency extra has not been installed or validated.

### Verification
- Python: 3.13.2.
- `pytest -q`: 70 passed.
- `python -m scripts.smoke_llm --provider gemini`: passed; provider `gemini`, model
  `gemini-2.5-flash`, synthetic response `Smoke test OK.`.
- Ruff checks on changed Python files: passed; `app/main.py`'s pre-existing `UP017` style finding
  was excluded.

---

## Session: 2026-10-23 — Layer 0 verification and completion

### Done
- Verified the baseline was on `layer-0-foundation` at `cf79897` with a clean worktree.
- Fixed provider alias handling so fallback results, cooldowns, and usage counts use the
  configured provider name.
- Made daily-cap claims atomic and UTC-dated; SQLite persistence failures now surface instead
  of silently switching to memory-only behavior.
- Required an engine for configured gateways and wired `scripts/smoke_llm.py` to the SQLite
  engine; added a persistence/restart test.
- Ensured structured-output repair is filtered and counted as a provider call; verified INFO
  logs do not include the input message.
- Fixed detached ORM results and SQLite database-directory creation; contract timestamps are
  timezone-aware.
- Removed unused `tenacity`, `aiosqlite`, and runtime `httpx` dependencies; kept `httpx` in the
  dev extra for FastAPI testing.
- Updated README safety limitations and replaced “GDPR delete” with “user data deletion”.
- Confirmed the build backend is `setuptools.build_meta`; package discovery includes `app` and
  `scripts`.

### Next
1. Start **Layer 1** only: implement `app/safety/sentinel.py` as the rule-based Safety Sentinel
   returning `SafetyResult`; add focused tests before wiring any LLM path.

### Known Issues
- Real Gemini/OpenAI-compatible provider requests were not run; provider SDK/model compatibility
  still needs a synthetic-only smoke test with configured credentials.
- Layer 1 safety and emotion processing are not implemented yet; Layer 0 exposes only `/health`.
- The optional `[rag]` dependency extra was not installed or validated as part of this Layer 0
  verification.

### Verification
- Python: 3.13.2
- `pip install -e ".[dev]"`: passed.
- `pytest -q`: 58 passed.
- Uvicorn `GET /health`: `{"status":"ok","environment":"development",...}`.
- Focused Ruff import/unused-symbol check: passed.

## Decisions Log (brief; full entries in DECISIONS.md)
- SQLite chosen over PostgreSQL for zero-infrastructure dev; swap is one `DATABASE_URL` change.
- Retry/cooldown transitions are explicit in the gateway; see D-011 for why unused `tenacity`
  was removed.
- `google-genai` (not `google-generativeai`) — the new unified Gemini SDK.
- MockProvider uses scripted deterministic outputs keyed by `TaskKind` to keep tests hermetic.
