# Historical role-model inspirations

Historical examples are an optional, separately sourced context. They are analogies
only; they are not counselling roles, self-help techniques, or clinical advice.
The four counselling roles and the self-help RAG collection are unchanged.

## Data location and card shape

Keep YAML cards in `data/role_models/` and original source PDFs in
`data/role_models/raw/`. The loader reads only top-level `*.yaml` files from the
card directory. It does not read, edit, or merge the PDFs into the self-help
knowledge base.

Supported `person` values are `gandhi`, `vivekananda`, `marcus_aurelius`, and
`kalam` (A.P.J. Abdul Kalam). A card contains:

- `id`, `person`, `title`
- `tags` from the package's finite theme vocabulary and `emotions` from the
  existing coarse emotion vocabulary
- paraphrased `summary`, `lesson`, `framing`, and `avoid_when`
- one or more `sources`, each with `title`, optional `author` and `year`,
  a non-empty `locator`, and optional HTTP(S) `url`
- `verification` (`status`, optional `checked_by`, and `notes`)
- `reviewed`, defaulting to `false`

Unknown fields (including quotation fields), unsupported people/tags/emotions,
URLs in summaries, and cards without source locators fail Pydantic validation.
A card can be marked reviewed only after its verification status is `verified`.
The 24 supplied drafts remain unchanged: `reviewed: false` and
`verification.status: needs_verification`.

`RoleModelLibrary.load()` validates every YAML card but exposes reviewed cards
only. `include_unreviewed=True` is an explicit validation/testing option; do not
use it when constructing the application engine. The normal factory uses the
reviewed-only default. When no reviewed cards exist, the feature returns no
historical inspiration.

## Draft validation and source review

All 24 YAML files currently validate, with six cards each for Gandhi,
Vivekananda, Marcus Aurelius, and A.P.J. Abdul Kalam. All required source
locators are present and non-empty. This structural check is not a substitute
for matching every claim and printed page against its source.

Human review flags from the supplied README and locator text:

- `vivekananda-family-hardship.yaml`: chapter and section are named, but the
  printed page is not; confirm it in the supplied edition before approval.
- All six Marcus Aurelius cards need section-level spot-checking against the
  supplied George Long 1880 translation. In particular,
  `marcus-control-response.yaml`, `marcus-cooperation.yaml`, and
  `marcus-temper-response.yaml` explicitly request locator confirmation.
- `kalam-long-term-capability.yaml` cites printed page 1 in two web-derived
  sources; confirm that each supplied source copy has a stable, matching page
  reference.
- `kalam-own-failure-share-success.yaml` correctly attributes public
  responsibility after the SLV-3 failure and sharing success with the team to
  Satish Dhawan, not Kalam. Preserve this distinction and verify it against the
  cited AIMA address.

The other locators are present and syntactically readable, but still require
source-level human verification before their cards are marked reviewed. No
summaries, lessons, framing, or source details were rewritten during schema
validation.

## Selection and safety

Selection is deterministic and local: regexes match card themes and coarse
emotions, or a uniquely mentioned supported person. Ambiguous/no-match text
returns no card; historical examples are never forced into a turn.

The existing Safety Sentinel remains authoritative and runs first. A card is
never selected unless the final safety level is `none`, and a previous crisis
in the session suppresses selection even if a later turn is otherwise clear.
Elevated and crisis replies receive no historical context or provenance.

Automatic selection is limited to understanding/guidance when the user is not
venting and the engine is not listening, reflecting, following up, closing, or
asking a clarifying question. An explicit request naming one supported person
may bypass the phase and minimum-spacing rules only. It never bypasses safety,
the listening/clarifying restrictions, or the hard maximum of two uses per
session. Automatic uses must be separated by at least four complete turns.

The selected card is passed to the response generator as a separate optional
context object, not as a self-help card. The local critic checks person names,
analogy framing, quotation marks, historical numbers, equivalence claims, and
lexical grounding against the selected card. Provenance is emitted only when
the reply declares that card as used and passes the checks. The critic is a
deterministic guardrail, not a historical fact verifier; source review remains
required.

## Streamlit

Use the **Enable historical inspirations** sidebar toggle to opt in. It is off
by default. A **Historical inspiration** panel appears only when the engine
reports actual, validated use, and includes the person, card title, source
locator, and the notice:

> Shared as an analogy, not as the same situation.

The offline demo remains provider-independent and does not load unreviewed
cards. Because the supplied cards are all unreviewed, the feature is currently
safely inactive in normal and demo modes until human verification and approval
are complete.
