# Task13: manuscript consistency review, 13 October 2026

This review is bound to commit `1491f37972e706426d3343ce5b69fa91ab89da01`, `paper/llm/main.tex` SHA256 `5723058082549c2bc1ae142f32b88a4e98fc5d84aa36a97a81fdd6c3e1d777c8`, `paper/llm/references.bib` SHA256 `4a062ece54122fb866387dba2dff13b132426792388520a21bd4f39d36eb4c1f`, and research-log SHA256 `46c93e871c04adafca2bc4f859fc0ab7f0265b7dc24a41601c485a1926a9f5c8`. All line numbers below refer to this manuscript. Claude's checkout/manuscript are unchanged.

Scope: full manuscript reading; mathematical/source consistency; reuse of accepted Task2/5/6/7 and Huginn design reviews; direct transcription of selected completed producer JSONs, with their exact bytes/hashes retained in `saved-number-transcription.json`. This is not a fresh model execution, refit, complete raw-artifact audit, or independent reproduction of every figure. Producer completion/gate fields in that transcription are not independently rerun. No Huginn held-out test outcomes, test-question arrays, model or cache files were read. Task11's second-classifier conclusions and the still-running Task1 certificate are not pre-empted. The citation pass is separate.

## Corrections that affect the mathematical or empirical claim

### C01 — passage coordinate sign is missing (193–197, 298–324, 379–384)

The displayed raw coefficients are negative for both philosophy folds and Task5 B. With `u'=u+a eps+b u²`, `a<0`, `b<0`, positive eps produces decreasing u; it cannot implement the stated entrance `-U` and exit `+U`. The computations actually reorient the critical modes for positive quadratic coefficient. The numerical results survive; the mathematical description must say so.

Suggested replacement after the raw coefficient definitions:

> We retain the signed coefficients in the original eigenvector gauge. For passage calculations we set `v=sign(b)u` and choose the ghost side `p-p_c=sign(ab)epsilon`, with `epsilon>0`. Then `v'=v+|a|epsilon+|b|v²` to quadratic order. Full and midpoint passages start at `v=-U` and `v=0`, respectively, and end at the first strict crossing `v>U`, with `U=0.1/|b|`.

Replace the table caption's raw-sign iteration with that oriented equation. Define `p=alpha` for the original/A folds and `p=log10 T` for B and the hostile fold. The latter parameter units apply to **a**, not b. State that newer cross-checks use nested AD, while the original coefficient used a finite-difference plateau. Source: accepted `codex/crosscheck/` and `codex/confirm-fold/` reviews, Task7 scientific review, and the saved original `passage.json`'s raw negative a/b and positive entrance/exit convention. No change to recorded counts or to ab is proposed.

### C02 — cascade caption understates actual flip errors and overstates device coverage (514–516, 577–582, 1003)

The five H100 C2 rows have `|mu+1|` equal to approximately `4.73e-12, 1.29e-12, 5.39e-12, 2.65e-11, 3.58e-10`. C3's repeated P48 value is `1.75e-10`. Therefore “real -1 (to 1e-11)” is false for the displayed set. These values still pass the numerical algorithm's declared 1e-6 crossing gate.

The completed CPU JSON contains only periods 3, 6, 12 and 24. It does not substantiate a fifth CPU row. Across those four shared CPU/H100 rows the largest parameter difference is `3.816e-15`, slightly above the stated `3e-15`. The CPU P24 Newton residual is `5.284e-13`; the displayed H100 C2 rows themselves are below `4e-13`.

Suggested caption/body:

> For the five displayed H100 flips, the critical Ritz value is real and differs from -1 by at most 4e-10; the orbit residuals are below 4e-13. The first four parameters agree between the completed CPU and H100 computations to 4e-15. The consumer-GPU comparison and any fifth-row coverage should be listed separately with its completed artifact.

Use the exact finite coverage in the evidence ladder; do not infer missing CPU output from a run intention. The JSON transcription includes every accepted flip row of C2/C3 and the completed CPU attempt. A newer local-GPU file was seen during inventory but was **not** admitted as completed evidence here; no stronger claim relies on it.

### C03 — six Ritz values are described as a complete spectrum (200–205, 541–543, 577–578)

`tools/llm_feigenbaum.py:103–121` computes a projected `Q.T @ M @ Q` spectrum, six values by default, and critical/significant Ritz residuals. The acceptance test at 237–241 checks the next **computed** modulus. No full-spectrum or complementary-tail bound occurs in this path. Low Ritz residuals do not prove that omitted modes have smaller modulus.

Replace “every/all other multiplier” with “the other computed Ritz values”; replace “transverse multipliers below…” with “the other computed Ritz moduli below…”. Describe these as estimates of the dominant spectrum. This is a finite scope correction, not a demand for a new dense or interval computation. The separately audited full 896-dimensional Task7 fold spectrum is unaffected.

### C04 — flip extent is conflated with superstable extent (49–50, 107–109)

The displayed flip table ends at the 48→96 crossing. The superstable table extends to period384. The final ratios4.478 and4.625 belong to the superstable sequence. Replace with:

> We locate flips through 48→96 and superstable points through period384. The latter sequence's final interval ratios are4.478 and4.625.

The saved superstable centres independently give3.190324…,4.019369…,4.477936…,4.625068…; the printed table values match their rounding. Finite-order convergence evidence is correctly distinguished from proving the limit at973–974; retain that distinction.

### C05 — estimated folds and cusp candidates acquire stronger labels elsewhere (343; 437–439, 469–485; 47, 105–106)

The stripe folds are extrapolations of `(1-mu)^2`, as the caption itself discloses; replace “Exact stripe edges” by “Numerically estimated stripe edges.” All16 printed edge locations/types match the saved eight-row `edges.json` at their displayed precision; no numeric changes are needed.

The fold tangent reversals and quadratic-coefficient sign changes support cusp candidates. The same draft explicitly leaves the cubic nondegeneracy coefficient uncomputed. Replace the categorical “so both are cusps” and unqualified abstract/contribution wording by “numerical cusp candidates, supported by tangent reversal and a sign change of the quadratic coefficient.” The missing nondegeneracy calculation should have one consistent status throughout. This does not discard the continued skeleton.

### C06 — Huginn development association still inherits the old map/clock mismatch (675–701)

The accepted source-only review at576b0b86 showed that the historical spectrum used the gold-answer prefill, while the development residual was a masked aggregate over all answer-choice rows; separate tokenization could differ. The old trace's last native residual is evaluated at the **input** to recurrence call32, `R(s31)`, while the answer readout uses `s32`. Float32 Ritz estimates and bfloat16 trajectory arithmetic are also distinct. Draftv3.1 repairs the prospective source; it does not retroactively change development arrays.

Pending a separately documented development recalculation, replace693 and699–701 with:

> In the historical development analysis, gold-prefill spectral features were associated with the all-choice aggregate residual entering recurrence call32. This is a predictive association across related maps and precisions, not a same-state contraction identity. The frozen test uses matched gold-choice token IDs and a per-row residual with an explicit clock.

Describe `rho_hat(lambda)=max` over the returned Ritz values as an **estimated spectral feature**, not the exact full spectral radius or a guaranteed stability boundary. The exact relaxation criterion is conditional: all eigenvalues must satisfy `Re(mu)<1` for a nonempty strictly stable positive-lambda interval. “No fixed relaxation can speed convergence much” should be limited to the inspected spectral estimates and measured prompts. No demand for exhaustive spectra is made for an association experiment. No test outcomes were inspected.

### C07 — cited TAPS under-relaxation comparison is wrong for GSM8K (688–690)

[Wang et al., arXiv:2609.36653v1, Table3](https://arxiv.org/html/2609.36653v1) reports Huginn GSM8K19.41% for eta1, **19.79%** for eta0.8, and8.34% for eta1.2. Under-relaxation slightly lowers its four-benchmark mean38.87→38.71, not GSM8K. The source also has a progress/fluctuation sensitivity theory and fixed-point toy analysis, so “without an explanation” is too broad.

Suggested replacement:

> Wang et al. report a Huginn GSM8K drop from19.41% to8.34% at a constant step1.2. At0.8 GSM8K is19.79%, while the four-task mean falls slightly. Their trajectory-based scheduling analysis does not compute the prompt-specific spectral thresholds studied here.

The1.2 numbers currently quoted round correctly. This review checked the primary paper's table and relevant theory sections, not their experimental replay.

### C08 — hostile arctangent error is rounded into a false bound (775)

The largest absolute error is **0.9225409889** iterates, from119 versus118.0774590111. “Within0.92” is not a valid bound. Replace by “within0.923 iterates” or “less than0.93”. Retain the primary quadratic result: **5/6 pass, aggregate fails**. The six observed/predicted rows at772–774 are correct. Source: accepted `codex/hostile-fold-crosscheck/passage-001-scientific-review.{md,json}` and its source-bound saved audit. Arctan agreement remains secondary, not a replacement endpoint.

### C09 — hostile fold evidence ladder is stale (1004–1005)

The hostile primitive P2 fold and normal form have now been independently reproduced with a separate float64 map and nested AD, full spectrum, both-phase closure/divisor and composed-precision checks. Body769–771 already says this. Move these items to rung2. Keep other hostile edges and global window organization at their actual evidence level. The controlled-passage result is also independently checked but its frozen all-six forecast criterion failed. Do not promote either to an interval certificate.

### C10 — finite Monte Carlo warning is interpreted as an attainable ceiling (67–69, 935–948)

The true conditional event probability ranks states optimally for its information set. A32-continuation estimate and a heuristic inverse-K extrapolation are not that oracle. Overlapping confidence intervals do not establish equivalence to it or prove that the jump time is mostly unpredictable.

The quoted AUC point estimates match the saved completed `committor.json`: finite-K0.7226/0.7102/0.7016; extrapolated0.7470/0.7287/0.7164; state0.7275/0.7239/0.7061. Finite-K minus state CIs are `[-.05251,.04243]`, `[-.04633,.01791]`, `[-.03578,.02757]`. Extrapolated-minus-state CIs are `[-.03271,.06914]`, `[-.02938,.03822]`, `[-.02237,.04403]`. Thus even a difference above.05 is not excluded at horizon1 by the extrapolated comparison. All saved undefined-bootstrap counts are zero; the calibration and coverage fields report success. This is a saved-report check, not new rollout or bootstrap validation.

Suggested replacement for946–948 and abstract67–69:

> In this finite Monte Carlo experiment the committor estimate did not provide detectably better warning than the in-context probe. The heuristic extrapolation leaves appreciable uncertainty, so these results do not identify the Bayes warning ceiling. They motivate evaluating reactive control alongside prediction, rather than proving that prediction cannot improve.

Do not present success of a separate detect-and-replace experiment as a causal consequence of this nonsignificant comparison.

### C11 — pending classifier sensitivity materially affects specificity claims (60, 114–115, 880–883, 981)

These statements must be scoped to the original classifier until Task11 is complete. Claude's16:00 message reports a sizeable second-classifier change for the generic-kindness control; this is provisional producer context, **not independently accepted here**. The phrase “does not lower the hostile share at all” and the deduction that the push “must” use the contrast direction should be held for Task11. Report both classifiers when independently audited, including disagreements; do not adjust thresholds after seeing them.

### C12 — small scope/terminology corrections (971–973, 985, 1010–1015)

- “One model family (two sizes)” omits the already-described coarse7B maps at633–638. Suggested: “One model family, with detailed orbit studies at0.5B/1.5B and coarse maps at7B.”
- “One prospective test” omits the new separately frozen hostile six-row passage test; use “prospectively frozen passage tests, including failed all-row criteria.”
- “Such sets can be computed exactly” at985 should be “defined precisely and computed numerically.”
- “Exact residual below1e-210” should be “a rigorous upper bound on the point residual below1e-210.” The fixed radius is a proposed isolating domain; it is not a certificate before inclusion, primitive-period and attraction gates plus independent review all pass.
- The cited legacy receipt list is dated12October and contains278 files. It does not index many13October results. Add the compact Task5/6/7/Task1 receipts and current producer manifests to the reproducibility pointers; do not imply the old list covers every current claim.

## Checked values that do not need changing

- Main stripe map:65,536 cells,35,759 positive-exponent cells, declared float64 transient300/measure300, maximum labelled period32; matches the saved scan summary. This confirms the count, not that every finite-time positive cell is a rigorously chaotic invariant set.
- Original controlled passage table: all12 displayed measured/quadratic counts match the saved18-row original JSON; six development rows remain separate. The missing coordinate orientation is C01.
- Stripe-edge table: all16 locations/types match the saved rows; fold extrapolation status is C05.
- Superstable centres/ratios: all six printed centres, last four ratio roundings and saved clean-bracket widths match. These are six-mode computations, C03.
- Task5 table's23/24 outcome and failed A-midpoint row agree with the accepted independent report; preserve the failure.
- Task6 corrected susceptibility CI, L7 state increments and L8 confirmation AUC/CI and window-based sensitivity/FPR wording at917–933 agree with the accepted saved-evidence audit. The earlier stale upperCI.020 is already corrected to.002.
- Task7 fold location/product agreement and all six hostile controlled counts are consistent with the accepted reports. No inference about natural reinjection follows from these controlled entrances.
- L10 saved report agrees with the rounded original-classifier values at889–894: unmodified.59397; regeneration-only would-be.121875, shown.0098214, mean dose.004875; mild constant.1069196/dose.020; hold3 would-be.0823661/dose.0133393. “Quarter dose” is a steering-dose comparison, not a compute-saving claim. The same saved report gives12.1875% regenerated turns overall in the regeneration-only arm (14.53% early,13.36% late), so “about one in seven” is defensible only for those subwindows; “about one in eight overall” is the clearer replacement at893. Replacement adds generation calls and the omitted originals cannot be retrospectively rescored with classifier2; Task11 handles that constraint.

## Completeness and editorial handoff

The abstract is about660 whitespace-separated source words, far too much detail for a useful abstract. Shorten after applying substantive corrections; a short proposed version will accompany the final packet. This is an editorial recommendation, not a claim about a universal arXiv word limit.

Unverified here: all individual pixels/trajectories, all cusp coefficient rows/continuation curves, historical Huginn development raw arrays, the gratitude appendix's fitted state assignments, complete fresh classifier2 results, and the underway final interval gates. These need their existing source-bound producer/cross-check receipts or pending assigned audits; this note does not silently mark them independently reproduced. Every discrepancy actually found in the checked scope is listed above, including small bound/coverage errors rather than only headline changes.
