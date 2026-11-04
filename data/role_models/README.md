# Solace historical role-model cards — draft set

24 YAML cards: 6 each for Gandhi, Vivekananda, Marcus Aurelius, and A.P.J. Abdul Kalam.

All cards are intentionally `reviewed: false` and `verification.status: needs_verification`.
Do not activate them until the source locators and claims have been checked against the PDFs.

Place the YAML files in `data/role_models/` beside this README. Keep original
source PDFs in the `raw/` subdirectory; application code only loads YAML files
from this directory and never modifies the PDFs.

Important review notes:
- Vivekananda family-hardship card locator needs exact printed-page confirmation.
- Marcus cards require section-level spot-checking against the George Long 1880 translation before approval; the inner-retreat, morning-duty, and judgement cards use Book IV.3, Book V.1, and Book VIII.47 respectively.
- Kalam SLV-3 card attributes the press-conference responsibility and success-sharing to Satish Dhawan, not Kalam.
- Kalam's AIMA address is the primary source for the leadership examples.
- These cards are paraphrases; no quotations are included.
- These are historical analogies, not counselling roles or clinical advice.

Schema follows the earlier Solace card shape: id, person, title, tags, emotions, summary, lesson, framing, avoid_when, sources, verification, reviewed.
