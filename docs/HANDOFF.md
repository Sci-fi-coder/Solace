# HANDOFF.md — Solace Session Log

> **AI coding agent:** Read this file after AGENTS.md at every session start.
> Update it at session end per the HANDOFF PROTOCOL.

---

## Session: 2026-11-05 — Historical retrieval evaluation and calibration (Step 3)

### Done
- Added a deterministic 52-case synthetic dataset covering every one of the 24 cards,
  explicit requests for all four people, six similar-theme distinctions, unrelated requests,
  ambiguous messages, and explicit but irrelevant requests. No card content or review state
  was changed.
- Added `scripts/eval_rolemodels.py` to evaluate actual local embeddings offline, report
  Precision@1, Recall@3, no-match accuracy, false-positive rate, per-card results, and
  reviewed/unreviewed coverage, and compare a bounded calibration grid without repeating
  embedding calls.
- Evaluated cached `sentence-transformers/all-MiniLM-L6-v2` weights with offline mode enabled.
  Calibrated to semantic/theme/context weights 0.90/0.05/0.05, minimum relevance 0.45, and no
  top-match margin. The previous 0.55/0.25/0.20 and 0.50 threshold had 1/34 P@1; the selected
  setting had 9/34 P@1, 34/34 Recall@3, and 18/18 no-match accuracy.
- Kept explicit-person relevance checks, card metadata, safety/eligibility logic, and the
  conversation pipeline unchanged.

### Verification
- Focused role-model, evaluation, and historical pipeline tests: 36 passed.
- Full suite: 440 passed, 1 deselected.
- `ruff check .`: passed.
- `git diff --check` reports only the pre-existing extra blank line at EOF in the user-modified
  `data/role_models/marcus-control-response.yaml:27`; it was left untouched.
- No model weights were downloaded and no external API calls were made.

### Known issues
- The 52 queries are synthetic and English-only; the evaluation is a calibration signal, not a
  semantic quality guarantee. Returned-match precision on relevant queries was 9/11 (81.8%),
  and only 11/34 relevant queries passed the threshold.
- At evaluation time, the working tree contained pending user card-metadata edits (8 reviewed,
  16 unreviewed), which are not part of the implementation baseline. Under that metadata,
  reviewed-target top-1 coverage was 3/11 and unreviewed-target top-1 coverage was 6/23.
  Every card's relevant queries had the target in its top three; correct top-1 retrieval occurred
  on 8 of 24 distinct cards. Explicit-person requests matched the expected card on 2/4 examples.
  Card ranking needs broader human-authored testing.

### Next
- Step 3 is complete. Review the calibrated retrieval behavior against a broader, independently
  authored set before further threshold changes. Do not commit or push.

## Current Branch
`layer-7-rolemodels`

## Session: 2026-11-05 — Historical role-model pipeline integration

### Done
- Integrated the standalone hybrid retriever into `handle_turn()` after safety, emotion
  extraction, planning, and state update. Matching is independent of counselling RAG and receives
  the current and recent user messages plus updated state; assistant-generated analogies are not
  added to the retrieval query.
- Wired the production factory to the all-card retriever. The live retriever loads all 24 cards
  with reviewed/unreviewed and verification metadata intact.
- Kept a retrieved candidate separate from permission to use it. Existing safety, venting,
  listening, phase (including the explicit clarification exception), session-cap, and spacing
  checks gate generation; historical evidence is a separate prompt input and provenance is
  recorded only when the validated reply actually uses it.
- Updated generation instructions and historical validation so use is optional, no-card replies
  remain ordinary counselling, and unreviewed cards are identified as unverified and cannot
  support asserted historical facts. `avoid_when` remains exposed as prose rather than being
  treated as machine-interpreted policy.
- Added offline integration coverage for all-card availability, relevance and abstention,
  explicit-person threshold behavior, empty counselling retrieval, eligibility restrictions,
  unreviewed cards, and provenance.

### Verification
- Focused role-model, critic, and pipeline tests: 129 passed.
- Full suite: 437 passed, 1 deselected, 19 warnings.
- `ruff check .`: passed.
- `git diff --check` reports only the existing trailing blank line in the user-modified
  `data/role_models/marcus-control-response.yaml`. That YAML and the other pre-existing user
  changes were left untouched.
- No model weights were downloaded and no live provider calls were made.

### Known issues
- Retrieval weights and the minimum relevance threshold remain experimental, and real semantic
  quality depends on the optional local model being available. If embedding is unavailable, the
  retriever reports no match while retaining its loaded card metadata.
- The historical critic is deterministic and conservative; it does not prove historical accuracy
  or parse arbitrary `avoid_when` prose.

### Next
- Step 2 is complete. Stop before any further pipeline integration or unrelated changes. No commit
  or push requested.

## Session: 2026-11-05 — Standalone hybrid historical retriever

### Done
- Added `app/rolemodels/retriever.py` as a standalone retriever that explicitly loads all
  historical cards, including unreviewed drafts, without changing card YAML or approval metadata.
- Added semantic, inverse-frequency-weighted tag, and emotion/state-context scoring with a
  configurable experimental relevance threshold and optional ambiguity margin.
- Added focused offline tests using deterministic mocked embeddings; no model weights or external
  APIs are used by the tests. The retriever is not integrated into `engine.py`.

### Verification
- Baseline suite: 416 passed, 1 deselected.
- Focused retriever tests: 13 passed.
- Full suite: 429 passed, 1 deselected, 19 warnings.
- `ruff check .`: passed.
- `git diff --check` reports only the pre-existing trailing blank line in the user-modified
  `data/role_models/marcus-control-response.yaml`.

### Known issues
- Retrieval weights and the 0.50 minimum score are initial experimental settings, not validated
  relevance thresholds. Semantic quality depends on the optional local model being available.
- The full suite retains 19 existing warnings from `fitz` and Streamlit/Pandas.

### Next
- Step 1 is complete. Keep the retriever standalone; do not integrate it into `engine.py` without
  separate authorization. No commit or push requested.

## Session: 2026-11-05 — Kalam analogy framing

### Done
- Found that `_ANALOGY_FRAME` relied on broad keyword presence: detached labels such as "this is
  an analogy" could pass, while a connection without a listed cue could fail.
- Updated explicit historical-request prompts to ask for a clear, natural sentence linking the
  selected example to the person's concern, with wording variation rather than a fixed template.
- Tightened analogy validation to require a visible relation to the person's situation, concern,
  or applicable action. Left factual, name, date, quotation, provenance, selector, and safety
  checks unchanged.
- Added accepted and rejected analogy-frame tests, prompt coverage for the reflective/no-RAG path,
  and a clearer connected Kalam fixture. Existing Marcus coverage still passes.

### Verification
- Baseline suite: 403 passed, 1 deselected.
- Focused role-model/pipeline suite: 60 passed.
- Full suite: 416 passed, 1 deselected, 19 warnings.
- `ruff check .`: passed.
- `git diff --check` reports only the pre-existing trailing blank line in the user-modified
  `data/role_models/marcus-control-response.yaml`.
- No live Gemini calls were made.

### Known issues
- The live Kalam response still needs provider verification; no live request was made in this
  session.

### Next
- Verify a live explicit Kalam response passes the role-model critic and main critic, then confirm
  Marcus provenance behavior remains unchanged. No commit or push requested.

## Session: 2026-11-05 — Explicit role-model clarification turns

### Done
- Identified the suppression: `enforce_plan()` can convert early-phase advice into
  `ask_clarifying`, and `role_model_allowed()` rejected that move unconditionally even for an
  explicit historical-person request. Explicit requests already bypass the ordinary phase filter.
- Allowed an explicit request through the `ask_clarifying` move gate only. Safety, sticky crisis,
  venting, other listening moves, and the two-use session cap remain enforced.
- Added offline tests for explicit Kalam clarification, explicit Marcus selection in Exploration
  after advice is converted to clarification, preservation of the planner's question, and retained
  automatic/safety/listening/frequency restrictions.

### Verification
- Baseline suite: 400 passed, 1 deselected.
- Focused role-model/pipeline/UI suite: 60 passed.
- Full suite: 403 passed, 1 deselected.
- `ruff check .`: passed.
- No live Gemini calls were made; response generation still requires live verification.

### Known issues
- None introduced by this change; role-model generation still requires live provider verification.

### Next
- Review the explicit clarification-turn change. No commit or push requested.

## Session: 2026-11-05 — Explicit historical role-model requests

### Done
- Passed the existing explicit-person request flag through reply generation into the prompt builder.
- Made explicit selected cards required when safe to use, while automatic selections remain optional;
  prompts now use canonical names, include the card's avoid-when condition, and forbid unsupported
  historical details or figures when no card is selected.
- Added actionable historical-critic retry feedback and retained the two-attempt critic/fallback
  path. Added regression coverage for explicit Marcus Aurelius and A.P.J. Abdul Kalam requests,
  safety suppression, missing cards, provenance, unsupported claims, and failed retries.
- Kept explicit historical requests eligible when self-help retrieval is empty, while retaining
  the reflect-only decision and all existing safety, venting, clarification, follow-up, and session
  guards. Automatic historical inspirations remain optional.
- Added a prompt path for a requested card presented reflectively when no self-help card is
  available.
- Replaced the growing sentence-opener allowlist with sentence-position-aware capitalized-token
  diagnostics. Sentence splitting now preserves dotted initials; selected-card names and supported
  card terminology remain allowed, while unsupported names still fail validation.
- Made explicit-request prompts require a concise concern acknowledgment, the canonical figure
  name, the card's lesson, and a clear analogy without a biography.
- Added exact unsupported-token diagnostics to retry feedback and development trace summaries.
  Trace steps distinguish first/retry critic results, complete validation, and safe fallback
  without logging the generated reply.
- Added offline regressions for Kalam and Marcus name variants, sentence openings, card-supported
  locations, exact unsupported-name diagnostics, explicit/optional prompting, retry, and fallback.

### Verification
- Baseline suite: 382 passed, 1 deselected.
- Focused role-model/pipeline/UI suite: 57 passed.
- Full suite: 400 passed, 1 deselected.
- `ruff check .`: passed.
- `git diff --check` reports only the pre-existing trailing blank line in the user-modified
  `data/role_models/marcus-control-response.yaml`; no task changes were made to that file.
- No live Gemini calls were made; response generation still requires live verification.

### Known issues
- The historical critic remains a conservative deterministic check, not proof of historical
  accuracy; reviewed cards still require human source verification.
- The live failed drafts were not supplied, so their exact tokens cannot be reconstructed; the
  diagnostic path is covered with the known sentence-opening false positives and an unsupported
  name in scripted drafts.

### Next
- Review the final diff. No commit or push requested.

## Session: 2026-11-04 — Historical role-model knowledge

### Done
- Created a separate Pydantic role-model schema, YAML library, deterministic
  theme/emotion/person selector, and local critic without changing counselling
  roles or the self-help card schema.
- Integrated optional historical context after the existing safety and phase
  guards. The feature is opt-in, reviewed-only in the normal factory, capped at
  two actual uses per session, and records provenance only for a reply that
  declares and passes validation for card use.
- Added the Streamlit opt-in toggle and provenance panel. Elevated/crisis states
  suppress context and panel display; the offline demo remains independent of
  providers and unreviewed cards.
- Validated the supplied YAML structure and recorded the source-locator review
  flags in [ROLE_MODELS.md](./ROLE_MODELS.md). Corrected the local role-model
  README to point at `data/role_models/`; raw PDFs were not modified.

### Verification
- Baseline with automatic dotenv disabled: 354 passed, 1 deselected.
- Full suite with automatic dotenv disabled: 372 passed, 1 deselected.
- UI-specific tests with automatic dotenv disabled: 13 passed.
- Focused role-model/pipeline/UI tests: 27 passed.
- `ruff check .`, `git diff --check`, and `compileall`: passed.
- Offline YAML schema validation: 24 valid, six per supported person; all remain
  unreviewed with `needs_verification`.

### Known issues
- The source locators and historical claims still need human comparison with the
  supplied PDFs. See [ROLE_MODELS.md](./ROLE_MODELS.md); do not mark drafts
  reviewed before that review.
- Since no reviewed cards are available, the feature currently returns no
  historical inspiration by design.
- The deterministic critic cannot prove historical accuracy; it is a
  conservative response check, not a replacement for source verification.

### Next
- Human-verify and approve each card against its cited source before marking it
  reviewed. Do not commit raw PDFs or unrelated untracked files.

## Session: 2026-11-05 — Streamlit UI integration

### Done
- Integrated the existing Streamlit application, offline demo engine, UI tests, dependency file,
  and launch documentation from the UI worktree without merging its branch.
- Added a new-session control after ending a session; a new session keeps the selected user and
  display name while clearing the previous turn state.
- Kept normal mode wired to `build_engine_from_settings()` and demo mode isolated from provider
  initialization. Added generic UI handling for engine initialization and turn failures.
- Made phase, emotional state, safety state, citations, explanation, degraded mode, and trace
  information visible. Recognizable credentials and bearer tokens are redacted in debug steps;
  crisis turns use a fixed, visually distinct safety presentation.
- Documented root-level launch commands and the relevant configuration variable names in
  [UI.md](./UI.md).

### Verification
- UI tests: 11 passed.
- Full suite with automatic dotenv loading disabled: 354 passed, 1 deselected.
- `ruff check .` and `git diff --check`: passed.
- Offline Streamlit server health endpoint: HTTP 200.

### Known issues
- Normal mode still requires a functioning local Solace configuration and an available provider;
  provider/network failures are displayed as generic recoverable UI errors.
- The offline demo is scripted and its crisis turn is only a presentation example, not a live
  safety assessment.

### Next
- Commit the UI integration separately. Do not push or merge either milestone.

## Session: 2026-11-04 — Pipeline reliability and planner fallback

### Done
- Added a shared `PROVIDER_TIMEOUT_SECONDS` environment setting (25-second default), passed it
  into Gemini and OpenAI-compatible clients, disabled OpenAI SDK retries, and classified timeout
  and deadline exceptions as transient for gateway-owned retries.
- Added a deterministic planner fallback for planner exceptions. It updates emotional state,
  preserves known context, selects conservative moves, and remains visibly degraded; the
  SafetyGate remains authoritative.
- Prevented new advice immediately after a follow-up unless the user explicitly asks for more
  help. Positive-only state updates now use alpha 0.2.
- Added offline provider, heuristic, guardrail, fallback, and nine-message demo regression tests.
- Documented the fallback and provider timeout behavior in the pipeline guide and recorded the
  design decisions below.

### Verification
- Baseline suite: 307 passed, 1 deselected.
- Focused provider, heuristic, and pipeline tests: 106 passed.
- Final full suite: 343 passed, 1 deselected. The suite was invoked with default dotenv loading
  disabled to avoid implicit `.env` access; explicit temporary dotenv tests remain active.
- `ruff check .`: passed.
- The exact nine-message script completed with moves
  `ask_clarifying -> reflect -> reflect -> psychoeducation -> reflect -> offer_technique ->`
  `follow_up -> reflect -> close`; phases were
  `checkin -> exploration -> understanding -> understanding -> understanding ->`
  `guidance -> guidance -> guidance -> closing`.
  Every turn was degraded; the expected advice turns carried citations; the final top emotion was
  `anxiety_fear`.

### Known issues
- The planner fallback uses a small English regex vocabulary. It is coarse, can miss paraphrases,
  and must not be used as a safety classifier; crisis and elevated handling remain with the
  existing SafetyGate.
- Advice and psychoeducation still require a matching reviewed knowledge card; if retrieval has
  no usable card, the existing engine stays in listening mode.
- I did not directly inspect or print `.env`. The baseline pytest process imported the app's
  default settings singleton, which is configured to load `.env`; it may have implicitly loaded
  the file. Later validation disabled automatic dotenv loading.

### Next
- Review the uncommitted reliability changes. No commit, merge, or push was made.

---

## Session: 2026-10-31 — Layer 2a extraction resilience and source selection

### Done
- Changed `kb_draft_cards` to use the normal `app.config.settings` loader and report configured
  provider names only. Failures now show one safe error line and per-provider attempts; traceback
  output is opt-in with `--debug`.
- Made structured-output parse and repair failures provider-local gateway failures. The gateway
  records each failed provider and continues through the chain; all-provider failure retains the
  complete attempts list.
- Probed Groq `openai/gpt-oss-20b` with `reasoning_effort=low`, `max_tokens=8192`, and a strict
  JSON Schema shaped from the card schema. Strict schema mode succeeded once and failed once on
  the card id pattern, so this model is excluded from extraction only. Other chains and defaults
  remain unchanged.
- Replaced page-wide source exclusion with line-level contact/medication sanitisation and a
  200-character minimum. Added mhGAP deny-list ranges based on PDF section headings and
  `include_pages: "13-20,35"` to prioritize essential care, communication, and psychosocial
  material.
- Dry-run diagnostics now list selected PDF pages and every excluded page grouped by reason.

### Verification
- Baseline `pytest -q`: 215 passed, 1 deselected.
- Focused gateway/config/KB tests: 46 passed.
- Pre-change dry run: mhGAP 14 pages, PM+ 13, DWM stress 35, NIMH 1.
- Post-change dry run: mhGAP 13 pages, PM+ 11, DWM stress 35, NIMH 2.
- Groq synthetic structured probe: one of two repeated strict-schema card responses validated;
  the other failed with `json_validate_failed` for the card-id pattern. No generated card was
  saved.
- Exact DWM dry-run reproduction selected pages 21, 24, and 25 (818 sanitized characters); it
  made no gateway/API calls.
- Final `pytest -q`: 224 passed, 1 deselected. `ruff check .` and `git diff --check` passed.

### Known issues
- PM+ still uses the existing local extracted JSON while its exact-mapped raw PDF filename remains
  absent; no source registry mapping was changed.
- Card generation/collection validation remains unverified; this task did not make real drafting
  calls.

### Next
- Review the proposed mhGAP page ranges and dry-run exclusions, then continue Layer 2a card review
  only. Do not begin Layer 2b.

---

## Session: 2026-10-30 — Layer 2a source registry and card drafts

### Done
- Created `layer-2a-cards` directly from `main`; confirmed the Layer 1 merge is present on
  `main` before starting.
- Inventoried `KB_RAW_DIR` by exact filename. Matched mhGAP (173 pages), DWM stress (132 pages),
  and NIMH stress (2 pages); the mapped PM Plus filename is missing. The similarly named
  `WHO_MSD_MER_18.5_eng.pdf` is unmatched and was not extracted.
- Added the source registry, page-preserving PyMuPDF extraction, card schema, cached structured
  drafting workflow, YAML validator, and offline tests. No PDF was copied into the repository.
- Extracted the three exact-matched PDFs to ignored `data/kb_work/`; DWM had no extractable text
  on PDF pages 2, 6, and 129–131. mhGAP and NIMH had no blank pages.
- Added safe-page selection and line-level removal of phone/URL/helpline contact material before
  sending selected text to the gateway. Invalid citations remain in YAML with `needs_review`.

### Verification
- Baseline `pytest -q`: 195 passed, 1 deselected.
- Focused Layer 2a tests: 20 passed, including synthetic PDF extraction, banned-content patterns,
  citation review, cache reuse, and quota-stop behavior.
- Final `pytest`: 215 passed, 1 deselected. `ruff check .`, `git diff --check`, and
  `compileall` passed. Tests were run without implicit dotenv loading.
- Drafting was attempted, but no enabled provider credentials were available in the process
  environment. The CLI now exits before provider dispatch in that case; real API calls: 0,
  cards drafted: 0.

### Known issues
- PM Plus remains unavailable until the exact mapped filename is supplied; the unmatched
  similarly named PDF is intentionally ignored.
- No card collection exists yet, so the collection validator and required emotion/category
  coverage remain unverified. The implementation is not committed or pushed.

### Next
- Configure an enabled LLM provider through process environment settings (without sharing key
  values in chat), rerun `python -m scripts.kb_draft_cards --limit 30`, inspect generated cards,
  run `python -m scripts.kb_validate`, then commit and push `layer-2a-cards`.

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

---

## Session: 2026-11-02 — Layer 2b Chroma audit and completion

### Done
- Audited the unaudited Layer 2b WIP commit and completed the local embedder, Chroma store,
  builder, retriever, passage-safety checks, synthetic evaluation set, and offline tests.
- Replaced the fake embedder's whole-text hashes with deterministic token feature hashing so
  shared terms produce meaningful test ranking without model downloads.
- Added `--include-unreviewed`; default builds include reviewed cards only. Rebuilds now delete
  only the two target collections and surface failures instead of suppressing them.
- Made Chroma metadata filters use supported `$in` comparisons over joined metadata values,
  require both emotion and role filters when both are supplied, and retry unfiltered when the
  filtered results do not meet the similarity threshold.
- Corrected cosine similarity to `1 - distance` (clamped to [0, 1]); the threshold sweep now
  changes retrieval and empty-result metrics.
- Added eight first-person synthetic retrieval cases without changing or removing existing cases.

### Verification
- `python -m compileall app scripts tests`: passed.
- `ruff check .`: passed.
- `python -m pytest -q`: 302 passed, 1 deselected.
- Rebuild and subsequent upsert both produced 30 card documents and 652 passage chunks:
  DWM stress 123, mhGAP 212, NIMH stress 5, and PM+ 312. Card source references: DWM 14,
  mhGAP 5, NIMH 3, PM+ 10.
- `Retriever()` returned three valid card IDs for the planner-style anxiety/breathing query with
  emotion and role filters.
- The final passage scan found zero phone-like strings, URLs, restricted helpline/lifeline/
  medication terms, medication/dosage/helpline patterns, or chunks in mhGAP deny ranges.
- The evaluation file has 68 cases: 54 retrieval cases, 14 expected-empty cases, 42 first-person
  cases, 9 Hinglish-flavoured cases, and 8 short planner-style queries. No near-verbatim card
  title cases were found.
- Final evaluation: Chroma hit@1 0.648, hit@3 0.778, MRR 0.713, empty precision 0.600, empty
  recall 0.857. Keyword baseline: 0.426, 0.444, 0.435, 0.364, 0.857 respectively.

### Known issues
- At the default 0.30 threshold, Chroma's empty precision remains 0.600; the full threshold
  sweep is recorded in the task report. Chroma hit@3 remains above the keyword baseline.

### Next
- Layer 2b audit and implementation are complete. Review retrieval quality before proceeding to
  the next project layer.
