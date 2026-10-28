# HANDOFF.md — Solace Session Log

> **AI coding agent:** Read this file after AGENTS.md at every session start.
> Update it at session end per the HANDOFF PROTOCOL.

---

## Current Branch
`fix-gateway-fallthrough`

---

## Session: 2026-10-28 — Gateway fallthrough audit and completion

### Done
- Completed the required read-only audit before edits: the branch was clean, Python compilation
  succeeded, and the baseline suite passed 72 tests. The initial full Ruff run found 29
  violations, which were fixed without excluding files.
- Tested Gemini candidates in order with the app smoke script and synthetic input:
  `gemini-2.5-flash` and `gemini-2.5-flash-lite` returned HTTP 404 for this account;
  `gemini-3.1-flash-lite` returned HTTP 200 and `Smoke test OK.` The latter is the verified
  non-preview code default.
- Set cheap-task defaults to try verified Gemini Flash Lite before Groq GPT-OSS 20B. Converse
  remains Gemini-first. Groq's code default is `openai/gpt-oss-20b`; `openai/gpt-oss-120b`
  remains an optional alternative.
- Added and passed deterministic provider tests for task-configured reasoning effort, parameter
  rejection retry, reasoning-text exclusion, empty length-limited retry/fallback, and
  Retry-After headers/body hints.
- Kept Cerebras disabled in default chains; the adapter remains available. The HTTP 402 is a
  reported account limitation and the billing-failure fallback path is covered by
  `test_billing_or_quota_402_disables_provider`.
- Made `.env.example` list every settings field as an empty value and confirmed empty values
  preserve application defaults. `.env` was not opened, copied, or modified.
- `python -m scripts.check_models` reached Gemini, Groq, and Cerebras catalogs; it warned that
  the configured Groq model was missing. The code-default Groq model was present (no default
  warning) and its explicit synthetic smoke call returned `Smoke test OK.`.
- Removed `scratch_gemini.py`, which was tracked in the accepted Antigravity commit; no
  `scratch_settings.py` was present.
- Confirmed `test_unknown_exception_surfaces_unchanged` was replaced by
  `test_unknown_exception_falls_through_and_appears_in_attempts`: provider errors now fall
  through rather than propagate unchanged. The same commit added
  `test_billing_or_quota_402_disables_provider` and
  `test_all_fail_error_contains_every_attempt` (three additions, one removal; net +2, from
  70 to 72).

### Next
- Do not begin Layer 1 in this session. Start it only in a separately authorized task.

### Known issues
- Groq has a low free-tier tokens-per-minute limit and should remain a small-call fallback.
- The unmodified `--provider groq` smoke failed because the configured model was not in the
  live catalog; selecting the code default explicitly passed. No dotenv file was changed.
- The unmodified Gemini smoke passed using the configured model reported at runtime; a separate
  run explicitly selecting the code default `gemini-3.1-flash-lite` also passed.
- Cerebras HTTP 402 is reported for this account; it was not re-probed during this audit and
  remains disabled by default.
- Mistral and OpenRouter had no configured keys, so their catalogs could not be checked.
- No Layer 1 safety or emotion work was performed.

### Verification
- `python -m compileall app scripts tests`: passed.
- `ruff check .`: passed with no exclusions.
- `pytest -q`: 90 passed.
- Gemini candidates: two HTTP 404 failures, followed by `gemini-3.1-flash-lite` HTTP 200.
- `python -m scripts.smoke_llm --provider gemini`: passed; explicit code-default run also passed.
- `python -m scripts.smoke_llm --provider groq`: failed on the configured stale model; explicit
  `openai/gpt-oss-20b` run passed.
- `python -m scripts.check_models`: warned only for the configured stale Groq model.

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
