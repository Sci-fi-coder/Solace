# HANDOFF.md — Solace Session Log

> **AI coding agent:** Read this file after AGENTS.md at every session start.
> Update it at session end per the HANDOFF PROTOCOL.

---

## Current Branch
`layer-1-safety-emotion`

---

## Session: 2026-10-30 — Final Layer 1 phrase coverage

### Done
- Added the three reported misses, common paraphrases, per-family hard negatives, and required
  regression phrases to the synthetic safety evaluation data.
- Added YAML-only disappearance, better-off-without-me, and intentional ending-it phrase families.
- Updated `try_layer1.py` to show pre-sticky `rule_level` and `final_level`, use a fresh session per
  input by default, and support explicit `--fresh` or sticky `--session SESSION_ID` testing.
- Added evaluation output for classifier-caught rule misses and classifier-introduced false alarms.
- Initialized the existing provider-usage schema in LLM-enabled Layer 1 scripts so live evaluation
  can run with its in-memory SQLite gateway.
- Rules cannot cover all paraphrases. The LLM classifier is the semantic layer; rules are the
  floor; when the classifier is unavailable the system runs degraded and this is a known limitation.

### Verification
- Baseline `pytest -q`: 167 passed, 1 deselected.
- Before YAML patterns, rules-only evaluation: 151 examples; crisis recall 0.900 (36/40),
  crisis precision 1.000 (36/36), none false-positive rate 0.000 (0/53).
- After YAML patterns, rules-only evaluation: 151 examples; crisis recall 1.000 (40/40),
  crisis precision 1.000 (40/40), none false-positive rate 0.000 (0/53).
- Focused safety/CLI/eval tests: 105 passed, 1 deselected; `ruff check .` and
  `git diff --check` passed.
- Rules-only 40-example sample: crisis recall/precision 1.000 (25/25), none false-positive rate
  0.000 (0/11).
- Live rules+LLM 40-example sample: same metrics; no classifier-caught rules misses and no new
  false alarms. 8 successful Gemini results and 10 Groq fallback results; 22 classifier failures
  degraded to rules-only.
- Final `pytest -q`: 195 passed, 1 deselected; `ruff check .` and `git diff --check` passed.

### Known issues
- The live sample used provider fallback and degraded operation for 22/40 cases; results reflect
  the configured provider availability during this run.
- Safety rules and model outputs are screening aids, not clinically validated detectors.

### Next
- Layer 1 final fix is complete. Begin Layer 2 only in a separate session after this change is
  committed and pushed.

---

## Session: 2026-10-29 — Safety phrase-family coverage

### Done
- Added English and Hinglish passive-ideation families for sleep/not waking, disappearance,
  better-off-dead/no-reason-to-live, inability to continue, and gender/spelling variants.
- Added English, Hinglish, and normalized-leetspeak explicit self-harm expression families,
  including bare fragments. Preserved elevated classifications for urge-framed self-harm language.
- Added hard-negative and positive-paraphrase cases to the safety evaluation dataset and regression
  tests. None of the hard negatives fired.
- Updated `scripts/try_layer1.py` with explicit `--with-llm`, `--no-emotion`, and `--verbose`
  options. Default results contain only safety level, reasons, degraded status, classifier provider,
  and classifier model.
- Documented that GoEmotions labelled “I just want to sleep and not wake up” as fine label
  `desire` / coarse label `positive`, and “everyone would be better off without me” as `positive`.
  Emotion output is never a safety signal and must never soften, redirect, or override safety.

### Verification
- Test-first, before pattern/code changes: 105 examples; crisis recall 0.966 (28/29), crisis
  precision 1.000 (28/28), none false-positive rate 0.000 (0/35). The confusion matrix showed
  two elevated misses and one crisis miss.
- After fixes: 129 examples; crisis recall 1.000 (36/36), crisis precision 1.000 (36/36), and
  none false-positive rate 0.000 (0/42).
- Focused safety and CLI tests: 75 passed, 1 deselected.
- Full `pytest -q`: 167 passed, 1 deselected. `ruff check .` and `git diff --check`: passed.

### Known issues
- `k1ll mys3lf` normalized successfully to `kill myself`; the original miss was a missing bare
  expression pattern, not a normalization failure.
- The normalizer translates only the configured ASCII leetspeak characters (`0`, `1`, `3`, `4`,
  `5`, `$`). Other substitutions, spaced-letter obfuscation, and cross-script lookalikes remain
  outside its coverage. Removing non-U+200B format characters at a word boundary can also join
  adjacent words.
- Safety rules remain a screening floor, not a clinically validated detector.

### Next
- Continue Layer 1 safety/emotion work only. Review unsupported obfuscation and Unicode-confusable
  handling as a separate task; do not begin Layer 2.

---

## Session: 2026-10-29 — Unicode safety normalization

### Done
- Fixed rules normalization to apply NFKC, normalize Unicode apostrophe/quotation variants,
  normalize Unicode whitespace, and remove zero-width format characters before rule and negation
  matching. U+200B is mapped to a normal space so it cannot merge adjacent words.
- Added synthetic evaluation and regression cases for straight/curly/left/modifier/backtick/acute
  apostrophes, negated intent, fullwidth text, zero-width characters, and non-breaking spaces.
- Verified every evaluation phrase has the same exact rule result after straight apostrophes are
  programmatically changed to curly apostrophes.
- Audited both safety pattern YAML files: English contains 16 expressions with ASCII apostrophes;
  Hinglish has no apostrophes; neither file contains non-ASCII punctuation. No YAML changes were
  needed.

### Verification
- Before the fix, the 102-example rules-only evaluation had crisis recall 0.964 (27/28),
  precision 1.000 (27/27), and none false-positive rate 0.000 (0/35). Seven added Unicode
  examples failed: five apostrophe variants and the fullwidth elevated and crisis phrases.
- After the fix, rules-only evaluation: 102 examples; crisis recall 1.000 (28/28), precision
  1.000 (28/28), none false-positive rate 0.000 (0/35).
- `pytest -q`: 132 passed, 1 deselected.
- `ruff check .` and `git diff --check`: passed.

### Known issues
- NFKC and punctuation/space normalization do not address visually confusable letters from
  different scripts. Other zero-width format characters are stripped; if one replaces a word
  boundary rather than being inserted within a word, stripping may join adjacent words.
- Safety rules are not clinically validated.

### Next
- Continue Layer 1 safety/emotion work only; do not begin Layer 2 without authorization. Review
  Unicode-confusable and invisible-separator robustness as a separate Layer 1 task.

---

## Session: 2026-10-28 — Layer 1 Safety and Emotion

### Done
- Created `layer-1-safety-emotion` from `main`.
- Added YAML-backed English and Hinglish rule patterns, normalization, negation handling, the
  sticky `SafetyGate`, structured gateway classification, fixed crisis templates, and unverified
  helpline metadata.
- Added lazy GoEmotions inference with injectable classifier, fine/coarse labels, evaluation
  scripts, synthetic datasets, and focused unit tests.
- Updated architecture and decision documentation for Layer 1.

### Verified
- `pytest -q`: 116 passed, 1 deselected (the marked slow model-download test).
- `ruff check .` and `git diff --check`: passed.
- Safety rules-only evaluation: 88 synthetic examples; crisis recall 1.000, crisis precision
  1.000, and none-class false-positive rate 0.000.
- Emotion evaluation: 48 synthetic examples; coarse-label accuracy 0.917.
- `scripts/try_layer1.py` completed an interactive synthetic turn and recorded safety/emotion
  StepRecords.
- A wheel build confirmed all three safety YAML files are packaged.
- The official PyTorch CPU index listed CPython 3.13 Windows wheels beginning at 2.6.0; the local
  Python 3.13.2 model evaluation also completed.

### Known gaps
- Rules+LLM evaluation was not run: it remains explicitly opt-in and no provider credentials were
  configured in the isolated verification environment. Gateway fallback/degraded behavior and
  result caching are covered with deterministic tests.
- Safety patterns and emotion outputs are not clinically validated.

### Next
- Begin Layer 2 (Knowledge Base / RAG) in a separate session after this layer is committed.

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
