# Adjudication: references and scientific-accuracy review, 7 October 2026

- **Request:** `manuscript-rendered.md` (rendered text of the working draft),
  review kind `research-note-scientific`, model `gpt-6-astra` (Pro mode, xhigh
  effort, 34,000 output tokens), response
  `resp_0269efe86b1a75e4016ac6c522911c87d0bcb73ff596414c0b`, completed. Usage:
  64,080 input and 32,023 output tokens (25,395 reasoning). Conservative cost
  $2.24. Artifacts: `../neural-shrimp-data/consults/2026-10-07-references/execute-001/`.
- **Verdict:** "NOT READY FOR READER REVIEW" (four blocking findings).
- **Framing authority:** the human delegated framing decisions to the executing
  agent on 7 October ("you decide for me, but don't let pro turn us into timid
  losers"). Rule adopted: keep ambitious claims when a concrete test can earn
  them; state precisely what has passed; never drop a claim silently, never
  keep one without a test.

## Findings and decisions

| # | Finding | Decision |
|---|---|---|
| B01 | Loss-score success is not parameter convergence; "rarely converges" asserts prevalence | Accepted: report "87.1%-99.97% of score-positive cells were not classified as fixed points after 20,000 updates"; distinguish loss reduction, parameter-state classification and orbit validation; the score "conflates dynamical regimes" rather than "misclassifies"; drop the prevalence sentence |
| B02 | Multiplier crossings are signatures, not sufficient mechanisms | Accepted in wording ("Neimark-Sacker candidates" until transversality, nondegeneracy and an attracting circle are documented); the ancestry stream now tests exactly these conditions, so the stronger wording can be earned |
| B03 | Projected ordering is overinterpreted; some period-denominator agreement is definitional (p/gcd(p,2) two-step points) | Accepted. The rotation-map "91% locked" statistic was withdrawn as definitional (research log, 7 October); the manuscript keeps the Birkhoff test with its null, states which relations are tested and which are definitional, and says odd windows "did not pass this diagnostic" |
| B04 | Central attributions need content checks; remove internal verification notes from the published bibliography | Done for the most consequential: Hofmann 2026 read at the source (`docs/novelty-hofmann-2026-10-07.md`); Metaxas-Sapsis abstract confirms invariant curves and phase-locking Arnold tongues in a multiplicative-weights learning rule (games, not descent on one loss); Sveleba et al. abstract confirms maps from a two-dimensional error projection (our "without full-state verification" stands). Van der Maas et al. 1990 is paywalled: library check pending. Internal notes to move to a ledger before release |
| S01-S03 | Design table, operational definition of "numerically certified" with tolerances, compact provenance and audit-independence summary, union-bound wording for nulls, reconciliation of 65/115, 31/61 and 35/56 | Accepted for the revision |
| R01 | Missing adjacent work: Damian-Nichani-Lee (self-stabilization), Agarwala-Pedregosa-Pennington (progressive sharpening), Lewkowycz et al. (catapult), Sato-Akiyama-Farmer (chaos in learning a game) | Accepted; all four verified at arXiv/Crossref (arXiv:2209.15594, 2210.04860, 2003.02218; PNAS 99:4748, 2002) |
| Pruning | Drop Ruelle-Takens from the single-circle breakdown claim; drop Stoop et al. (spirals untested); remove Jones 2012 as an explanation of odd windows | Accepted for Ruelle-Takens and Stoop. Jones 2012 is kept, but only as the reference for periodicity-hub organization in an explicitly open question ("whether the odd windows are organized as in periodicity hubs is open"), not as an explanation |
| F01-F02 | Figure 1 caption mixes unconditional and conditional fractions; route chronology in Figures 3-4 oversimplified (rotation falls 0.2180 to 0.2159; chaos between 2/9 and 1/4) | Accepted |
| S04 | Placeholders, authorship, disclosure, release links | Kept as release gates for the human |
