# Task40 supplemental Sec. V raw-evidence review

Verdict for this scope: **GO WITH FIXES**.
Four findings: one must-fix, two should-fix, one clarification note.
This supplements `raw-review.md`; it does not repeat the A1/A5–A8 audit or address the separate DITCH comparison.

Article: `bb42a82189da91c37f50804c077832df9e2237ae`, `/data2/chaos-journal-neural-shrimp/main.tex`, read only. Research reference: `c567545ca93f90324dae75be2c333c8ac452dab5`. All line numbers below are from that exact article.

## Execution and evidence identity

The independently written `fold_saved_audit.py` was frozen at private checkpoint `52eb95f1add11586ccc5ccd53244f12827bc4098`, after source review and before the authorized single execution. Source SHA256 `63527011fcd65a63ac13d31bbb89c65a0ad7e17e8099f5421c905a2d2ef23ea7`, 15,967 bytes.

The bounded `fold-001` execution completed with exit 0, no timeout, child 1624756 reaped; its receipt UTC is `2026-10-20T03:30:51.215443+00:00`. Launch-through-reap time was 0.1657307674176991 s; the script's internal saved-data work was 0.050401945132762194 s. Limits were one numerical thread, nice increment 10, 4 GiB address space, 90 CPU seconds, 120 wall seconds, offline Hugging Face and empty CUDA visibility. No model calls, eigenvalue solves, trajectory reruns or old-audit reruns occurred. Installed `uv` was launched with the explicitly authorized sandbox escalation.

Exact immutable results:

| File | Bytes | SHA256 |
|---|---:|---|
| `fold-evidence.json` | 787504 | `a36547bf00c025e7ee5c3b4bac4c20929775ece476862ff577bf425f816d0b3c` |
| `fold-001.receipt.json` | 1121 | `3612deb564e0970453315c1ef719fda22d99570682604bd0d177523e03d636cb` |

The result's `manifest` pins all 22 original inputs by absolute path, byte length and full SHA256. Its distinction between freshly reduced saved arrays and reused closed audit receipts is substantive: no claim is made that all historical model results were newly independently replayed.

## Findings

### S1. The 4–8% statement is a median, and its channel differs from Fig. 14d [must-fix]

**Where:** Sec. V, “Passages and the window law,” lines 927–929; Table IV caption/first row, lines 948–960. Compare Fig. 14 caption, lines 817–818.

**Claim:** “From each measured entry point the normal form predicts the length of the laminar phase that follows within 4–8% for eps <= 10^-4.”

**Evidence:** The original `passage-natural-p2-t-0.115/{passage.json,u.npy,transverse.npy}`, reduced in `natural.original_3U_transverse005`, gives the following complete-phase results. Each row uses four free trajectories, excludes runs censored at either observation endpoint, and defines the phase by both `|u| < 3U` and transverse distance `< 0.05`. Error is `(observed - predicted) / predicted`.

| eps | Complete phases | Full normal-form passage | Mean observed | Mean fraction | Median signed error | Largest absolute per-phase error |
|---:|---:|---:|---:|---:|---:|---:|
| 3e-6 | 22 | 157 | 96.136364 | 0.6123335263 | -4.8782% | 12.5000% |
| 1e-5 | 29 | 83 | 55.344828 | 0.6668051516 | -4.0816% | 64.9351% |
| 3e-5 | 44 | 45 | 27.477273 | 0.6106060606 | -7.4074% | 38.6364% |
| 1e-4 | 110 | 22 | 13.109091 | 0.5958677686 | -7.6923% | 70.5882% |

Every phase has a scalar forecast; there are no silently omitted missing predictions. The two larger saved offsets, 3e-4 and 1e-3, are also retained in the result but are outside the article's stated range.

The narrower figure definition, `|u| < U` alone, is independently reduced in `natural.article_figure_U_only`. It gives 22, 30, 45 and 110 phases, full passages 144, 71, 34 and 14, and median errors -4.2329%, -3.8839%, -4.7619% and -11.1111%. Thus merely inserting “median” while implicitly keeping the figure's channel would still fail to reproduce the stated 4–8%.

**Replacement:** “For this comparison we define a complete laminar phase by `|u| < 3U` and transverse distance below 0.05, excluding phases cut by an observation endpoint. At the four tested distances up to 10^-4, the median signed error `(observed-predicted)/predicted` is between -7.7% and -4.1%; individual phases deviate more. Their mean lengths are approximately 0.60–0.67 of the normal form's full passage through that channel.”

Table IV should name that phase definition, either directly or by an unambiguous nearby reference. The displayed 0.60–0.67 is reasonable rounded descriptive reporting of the means; it must not be read as an exact lower bound of 0.60. The figure's separately stated `|u| < U` criterion and its 40–70 counts can stay.

### S2. Round explicit precision bounds outward [should-fix]

**Where:** Sec. V, “The fold,” lines 850–866.

**Claim:** The critical multiplier equals 1 “to within 3×10^-13”; independent coefficients agree “to ... 2.2×10^-6 relatively.”

**Evidence:** `normal-form-p2-t-0.115/normal-form.json` saves the critical multiplier `1.0000000000003006`, giving error `3.006483950684924e-13`. The closed independent `codex/crosscheck/audits/fold-001.json`, freshly hash-bound but not rerun, records the relative b discrepancy `2.2068638969017986e-6`. The alpha discrepancy is `5.800915303666443e-15`; the a discrepancy is `2.5979218776228663e-12` absolute, or `5.374017139945206e-13` relative.

**Replacement:** Use “within 3.1×10^-13.” If the coefficient comparison is also a bound, use “within 6×10^-15 in alpha, 3×10^-12 in a, and 2.3×10^-6 relatively in b,” explicitly distinguishing the absolute a tolerance. Alternatively label the b discrepancy as approximately 2.2×10^-6.

The underlying fold remains supported: saved orbit residual `3.754226018299182e-14`, next modulus `0.03698467452797282`, nonzero a and b. This is a wording/rounding correction, not evidence against the fold.

### S3. A grid-resolution error model is presented as a location bound [should-fix]

**Where:** Sec. V, limits of the window law, lines 1006–1010.

**Claim:** “the edges are located to within 2×10^-4 of their distances from the fold.”

**Evidence:** The original P16–20 `pod64/edges/stripes-t-0.115/edges.json` stores extrapolated folds, obtained from four stable-side multiplier values using a linear fit to `(1-mu)^2`; it does not store solved unit-circle crossing records. The P22–32 evidence is the `pod64/hidden/wp-line/hidden-chaos.json` finite grid. Its widest contiguous period-label runs have neighboring-cell brackets, not verified fold crossings.

The exact research reference `c567545c:docs/research-log.md`, lines 29254–29257, identifies the estimate behind the claim: approximately 2e-8 for continuation folds and `1e-7/sqrt(12)` for WP grid edges. The latter is a uniform-within-cell standard-deviation convention, not a worst-case bound. The original WP selection rule is recorded at lines 23776–23787 and is independently reproduced here, including all alternate runs.

**Replacement:** “Estimated localization scales are about 2×10^-5–2×10^-4 of the distance from the fold, using extrapolated P16–20 fold locations and a uniform-within-cell error model for the P22–32 grid boundaries.” Do not say “within” without a separately justified enclosing error bound.

The computed -1.99806 log-log slope and the 1.35–3.51 bottleneck-width range are reproducible from these saved estimates. The free-exponent sensitivity interval 2.13–2.45 and the nonlinear fit misfits were not freshly refitted in this supplement; their status remains partial, not silently passed.

### S4. State the window-prediction denominator and reinjection-band comparison [note]

**Where:** Sec. V, lines 943–945.

**Claim:** The frozen law placed the P22–30 edges “within 3% in distance from the fold”; a fixed reinjection model “did not, except at period 30.”

**Evidence:** The frozen `docs/wp-window-predictions.json` contains three fixed reinjection choices, `v_r = -0.0003, -0.00049, -0.00074`, with one fitted offset per choice. The source result compares the measured distance with their envelope. All 3200 original WP grid coordinates match the declared input cells; the unique widest run is selected per period, and alternate runs are retained.

| Period | Measured grid-edge distance | Frozen law | Signed law error / forecast | Reinjection prediction range | Measured inside range? |
|---:|---:|---:|---:|---:|---|
| 22 | 0.000397880587684 | 0.000400466709323 | -0.645777% | 0.000379993981951–0.000390304994295 | no |
| 24 | 0.000317555486089 | 0.000317716265042 | -0.050605% | 0.000296902675567–0.000308389423469 | no |
| 26 | 0.000253035323539 | 0.000258196025140 | -1.998753% | 0.000238635779564–0.000250387061992 | no |
| 28 | 0.000211822444514 | 0.000213959661365 | -0.998888% | 0.000196180594264–0.000207762635797 | no |
| 30 | 0.000174910909659 | 0.000180189334282 | -2.929377% | 0.000164276287605–0.000175484562692 | yes |

The frozen article's direction is correct: it says **did not, except at P30**, not the reverse. However, inclusion in a band spanning three reinjection points is not agreement with one specific median reinjection prediction. For example, the median-choice P30 prediction is `0.0001692386050516969`, distinct from the measured grid distance.

**Suggested wording:** “The frozen law placed the five grid boundaries within 3% of the forecast distance from the fold. Of the five observed boundaries, only P30 lay within the prediction band obtained from the three tested reinjection points.”

## Completed claim ledger

| Article location | Evidence and outcome | Scope |
|---|---|---|
| Fig. 14a,b, lines 808–820 | All 27 saved branch rows have residual <=1e-10 and `multipliers_converged:true`; maximum saved residual 9.47881404323669e-11. Saved Newton counts range 0–7. Critical-coordinate projections recomputed from saved states and left vector; attracting and unstable sides retained. | Fresh saved projection and scalar gates; original float32 branch spectra not recomputed. The caption already discloses their precision. |
| Fold, lines 850–866 | alpha_c=0.06279827217667086; a=-4.834227003692353; b=-25.42851228205931; ab=122.92720073765378; unit right-vector norm and left/right normalization recovered. | Fresh scalar/vector reductions; no derivative or spectrum rerun. |
| Independent ab=123.1, line 865 | Closed branch audit, SHA `ed21ac6c0309332acc976b4544dadd26d51fe5a74616c0524593b2818606d2c1c0379`: 18 selected rows, quadratic intercept 123.08319025741714; linear-fit sensitivity 120.57184165076972 retained. Strict 1e-11 screen failed row 3, and the disclosed producer 1e-10 screen passed. | Exact closed receipt reused; failure/amendment not erased. |
| Independent fold implementation, line 866 | Closed fold audit, SHA `ba3b72e9c5b5993deb259e47075603f85d0ec95572e986126e564387292ffeb5`: augmented solve converged after iterations 0–2; final augmented residual 5.4353123048745106e-14. Coefficient discrepancies above. | Exact closed receipt reused. |
| Table III, lines 896–909; Fig. 14c; passage text 877–880 | All 12 printed observed/predicted pairs match the saved trajectories and newly iterated scalar normal form: full 144/144,71/71,34/34,14/14; midpoint 73/74,36/37,17/18,7/7; displaced 143/144,70/71,33/34,13/14. All 18 saved rows, including six earlier-development rows, were checked and retained; every one is within one iterate. | Fresh first-exit reductions; one already saved transverse direction only, not a statement about arbitrary perturbations. |
| Fig. 14d, lines 817–818 | In the displayed first 600 iterates of natural-3 at eps=1e-5, complete `|u|<U` phase lengths are 40,40,70,47,41,42,62. | Fresh saved-array check; the 40–70 caption passes. |
| Natural phases and Table IV first row | Full per-phase ledger for both definitions and all six saved eps values; the four quoted eps results are tabulated under S1. | Fresh reduction. Must state definition and median. |
| Centers, figure `fig:windowlaw` and Table IV | N=4–7 products 0.02631564517,0.02646819558,0.02648620164,0.02629966334; inverse-root slope 6.165330797; log-log slope -1.999348510; effective fractions 0.57233–0.57436. Full prefactor 0.08028820588, half 0.02007205147. | Fresh reductions of saved orbit coordinates. All eight table orbits preserve converged/period/divisor/residual metadata in the result. |
| “midpoints ... on a float32 cut,” lines 939–940 | Original 700-cell cut was checked. For each period, the selected run contains the declared table center; all alternate runs retained. Saved table centers are close representative rounded coordinates, not bit-identical exact midpoint values. E.g. P14 original midpoint .0644420600858 versus saved .064443. Differences are below the grid spacing and do not undermine the quoted approximate slope. | Fresh grid check; no new float64 window location. The earlier `window_centers.limit` string in the result predates the disclosed pre-run addition; `original_float32_cut` is the actual added coverage. |
| P16–20 edges / Table IV | Stored extrapolations independently recovered: .0638000727075607,.06350339885055502,.06331825756131203; four fit points each. Slope6.129556848; full slope3.529182551; ratio1.736820569; f=.575764715; C=.02661594225,N0=.1508783142. | Fresh scalar fits; no crossing solve. Rounded table values 6.13,3.53,1.74,.58,.0266,.15 pass. |
| WP forecast, lines 943–945 | All five comparisons and all alternate period runs retained; quantitative result under S4. | Fresh finite-grid reduction; not a formal fold validation. |
| Width and combined slope, lines 1006–1008 | N=5–13 width/U values 3.50920,2.94408,2.52818,2.21149,1.97566,1.76355,1.61354,1.46621,1.35143; combined slope -1.998062633 using the original alpha_c=.0627983 convention. | Fresh reduction of saved edge estimates. |
| Table V, lines 967–999 | All 48 article measurement entries match the original closed Task5/5B audit records, and all forecast entries were compared with those records. All failed forecast criteria retained. | Reused exact audits, not new trajectory replay. |
| First prospective stage, lines 1025–1030 | Discrete primary 23/24 within one, including seven exact forecasts <=5; one A midpoint failure61 versus59. Continuous24/24. Seven nonzero discrete residuals negative. | Closed Task5 audit SHA `76217ba867aca5e43ef801dce45263a200a5c0854c2e3dd2b84f0d4e9d12eab6`. |
| Second stage / pooled, lines 1032–1040 | Continuous18/24; discrete16/24. C12/12 continuous, D midpoint4/4. Six primary misses retained. Pooled continuous42/48. D full/displaced shortfall extrema .4013846143 and5.7448209515; article0.40–5.74 is nearest rounding, not a strict upper bound. The six relative misses use forecast denominators and round to3.1–5.2%. | Closed Task5B audit SHA `e906239572e15002e9811b217b1c87aa2e0018755a6ed29741666bb7a6a15e7d`; original failed D01 setup preserved in receipt provenance. |

## Partial and skipped

- The nonlinear free-exponent fits yielding 2.13–2.45 and the 1–3% fit-misfit range were not rerun. Only their original source description, saved input coverage and the independent simple log-log slope were checked.
- No fresh full-state periodic closure, primitive-period, composed derivative, eigenvalue, normal-form differentiation or finite-difference precision test was performed. Saved convergence flags and historical independent audits cannot be promoted to a new runtime qualification.
- The WP labels and contiguous-run boundaries are finite-horizon classifications. Multiple runs and fragmented P24/P28 labels remain visible in the result; no continuity or basin assertion is inferred from them.
- The declared original pre-outcome freeze chronology of Task5/5B is inherited from their exact closed receipts. It was not reconstructed through every historical raw file or process.
- Article text examples and the earlier coexistence/chaotic-text paragraphs before the fold rewrite were outside this supplemental assignment. The initial A1/A5–A8 findings remain in `raw-review.md`; no previously completed controls or trajectories were repeated.
