# Task38 — independent review of the night’s results and their presentation

**Disposition:** the available-evidence review is complete. Five issues need correction before using the current wording; ten further corrections or qualifications and the explicit evidence gaps below remain for producer triage. The saved geometry supports the reported numerical kite and six-cusp structure. The new scalar-coding interpretation is substantially more qualified than parts of the text imply. None of these findings identifies a flaw in the previously closed interval certificates, which were not rerun or re-audited.

Task38 is a saved-data/source/publication review, with zero target LLM model calls. The separate Task37 independent manual-map R-pair is running under its own exact source and limits; it supplies no completed result to this review. No paper, blog, producer tool, shared log, other-owner job or public state was edited.

## Exact reviewed versions and later deltas

- Article main: `b8413881e7ce97a9e58adcf1bf7d78a8cebb0514:main.tex`, SHA256 `d18cb53e2b11117bb8b5494f8b2ff53537a0f8da7497972880767da5311c518d`.
- Alternative p6-only: `d49718d3294f180e67f94de8194536458655e918:main.tex`, SHA256 `9df885c80e7cf3fa96edcf13ec00d3b7dc643c4cf0c577eee5f7e012291cb139`.
- Report: `651a50059a145e2bb0526f9675bcc41e9cb895a8:paper/llm/main.tex`, SHA256 `a903c8a0373333f01a2bdbc3e4f66cfcdafc2101005077a076a75611661cc912`.
- Read-only blog: `chaos-bifurcations-language-model-feedback/index.md`, SHA256 `4c6646cae55b08d5d3d85dd19ef9d0ad11348a82bef504fc20f3be92ecd21926`. This is an uncommitted snapshot, not an assumed Git revision.

File:line references below refer to these snapshots. `snapshot-retention.json` binds durable copies under `/data2/neural-shrimp-data/scratch/audit-codex-1019/task38-review-snapshots-001`. Exact raw paths, sizes, hashes, every failed/partial extractor attempt and audit sources/results are retained in the geometry, symbolic and PV manifests and `raw-evidence-retention.json`.

Concurrent producer changes were ordinarily merged. The narrow `b4e31cfb` delta repairs the return-map figure's pooled gate, the combined word legend's checked-status distinction, and a stale report limitation. It does not repair the substantive symbolic findings below. New hook material in `0d71b711` postdates the assigned snapshot and is not silently included in this review. The alternative branch and blog have not been presumed synchronized.

## Must fix


### M1. DSS6: the selected top-two product is often nonreal

**Where/quote:** report 2446–2447, blog 4960–4961: “The product delta = mu1 mu2 of the two leading multipliers is negative in 151 of the 529 cells”. The following text treats delta as a real two-dimensional block determinant and discusses its zero curve.

**Evidence:** Original `pod64/dss/p6-body/grid.npz`, saved `multipliers` in descending modulus. At 240/529 solved cells, the first multiplier is real and the second is only one member of a complex subleading conjugate pair: their product has a nonzero imaginary part (maximum absolute imaginary part 0.00696132006799271). The count 151 is reproduced by **Re(mu1 mu2) < 0**, not by a negative real determinant. Eight of those 151 products are nonreal. The top two form a real/conjugate-closed pair in 289 cells; 143 of those products are negative. Example [row 0, col 12]: multipliers -0.3030628280 and 0.0016020178 + 0.0030695243i, product -0.0004855121 - 0.0009302587i. The explicit conditional-model paragraph does not cure this preceding empirical claim.

**Replacement:** “At 289 solved cells the two largest-modulus multipliers form a real or complex-conjugate pair; their real product is negative at 143. At the other 240 cells the leading real multiplier is followed by a complex conjugate pair, so selecting only the first two does not define a real two-dimensional determinant. We do not infer a determinant-zero curve from that selection.” Retain the separately qualified conditional model. Exact 240 rows: `geometry-details-001.json`, `DSS_nonreal_top2_product`.


### M2. C labels do not locate superstable curves

**Location/quote:** article:1251–1255, “The 153 cells on the curves themselves, where the leading pair is complex, have no defined last symbol”; caption article:1277–1278, “magenta on the window's superstable curve (complex leading pair)”. Equivalent report:2522–2526, 2548–2549; table caption report:2562. Article:1248–1251 also presents every inferred sign boundary as an established superstable curve.

**Evidence:** `cell_word`, producer lines 179–182, returns C when neither final symbol receives two thirds of the votes among phases selecting the winning W. It computes no multiplier or zero-crossing condition. Independent replay of every original SD-W `word_cos` array reproduces all 153 C labels. Only two selected C cells were Newton checked. Their stored leading eigenvalues are −0.0022710989196211414 + 0.024287213446202938i and −0.0009749062774780157 + 0.004396420374427073i, with nonzero moduli about 0.0243932 and 0.00450322. A complex nonzero pair supplies neither a scalar derivative sign nor a zero multiplier. The two observations cannot establish that the other 151 C cells have complex leading pairs. The map's pale/full sign colors are predictions from word parity, not a dense eigenspectrum census.

**Proposed text:** “The scan assigns C to 153 cells whose usable phases do not agree sufficiently on the final symbol. At two selected C cells, Newton calculations report small, nonzero complex leading pairs. These labels identify ambiguous scalar coding; they do not locate superstable curves. Pale and full colors indicate the multiplier sign predicted by whole-word parity, checked at the selected real-multiplier cells.” Replace the magenta legend and the table's parenthetical claim accordingly. In the scalar-map analogy, describe sign-changing curves as candidates unless a zero multiplier has actually been located.


### M3. The return-map error bound is pooled, not phasewise

**Location/quote:** article:1231–1232, “with at most 4.7 % of the rarer symbol on the wrong side ... in every phase”; report:2502–2503. Assigned return-map caption article:1264–1265/report:2535–2536 implicitly uses the same green/red classification as the text.

**Evidence:** Independent replay from original `return_projection` and `word_cos` arrays gives 58 chaotic six-band cells, 48 with pooled `sum(errors)/sum(minority) <= 0.05`, 23 with zero errors, and a largest passing pooled ratio 0.04697156983930779. The passing cell at alpha=0.07340851063829787, log10(T)=−0.15410638297872342 has a usable phase with 36/166 = 0.21686746987951808 errors/minority. Thus the claimed per-phase upper bound is false. The producer's original `unimodal_all_phases` flag instead tests 99% separation among visits in every usable phase and passes only 44 cells; the assigned figure code colors by that flag. The old plot and prose therefore used different tests.

**Proposed text:** “Pooling classification errors and minority-symbol counts across the usable phases of each cell, 48 of 58 chaotic six-band cells have an error ratio below 5%; the largest passing ratio is 4.7%, and 23 cells have no errors. This pooled test does not impose a 4.7% bound on each phase.” State the denominator and pooled criterion in the figure caption/legend. If desired, disclose that the largest usable-phase ratio among passing cells is 21.7%.

**Narrow later delta:** read-only diff `11895a13..b4e31cfb` changes the figure color gate to `errors_over_minority <= args.max_errors` and adds `--max-errors=0.05`. This repairs the code-level 44-versus-48 figure mismatch; it does not repair the quoted all-phase prose. No new scientific rerun or regenerated-image authentication was performed in this delta review.


### M4. Singer’s theorem needs its endpoint alternative

**Location/quote:** article:1236–1238/report:2507–2509, “A unimodal map with negative Schwarzian derivative has at most one attracting periodic orbit”.

**Evidence and limit:** Singer's theorem concerns a C3 interval self-map with negative Schwarzian away from critical points: an attracting periodic orbit's immediate basin contains a critical point **or an interval boundary point**. Unimodality gives the stated uniqueness only after excluding the boundary alternative. The report already states the qualification correctly at 4001–4009, as does the frozen blog at 4123–4138. The original [Singer publisher record](https://epubs.siam.org/doi/10.1137/0135020) verifies the article; the explicit endpoint statement was checked in Proposition 2.6 of [Ou's 2018 research thesis, PDF page 24](https://www.math.stonybrook.edu/alumni/2018-Dyi-Shing-Ou.pdf). The latter is a primary research work restating Singer, not a claim that the original 1978 full theorem was accessible. Neither negative Schwarzian nor an invariant scalar interval has been established for these LLM return maps.

**Proposed text:** “For a C3 unimodal interval map with negative Schwarzian derivative, at most one attracting periodic orbit can have an immediate basin avoiding the interval endpoints. The additional turning points observed below the kite are consistent with a scalar explanation of nearby coexistence; they do not establish a global scalar reduction or its Schwarzian sign.” Keeping the modest consistency observation is fair; it is not a demonstrated cause of coexistence at another parameter.


### M5. Use consistent names for the exponent-only masks

Locations: article529–550, report5211–5234, blog2013–2035; producer tools/pv_summary.py chaos indicator and tools/plot_pv_frequency.py. Every PV summary and frequency uses only exponent>0.001. It includes3–18 cells per map with a detected period and positive finite-window exponent. The gray complement includes134–465 cells per map with no detected period and exponent<=0.001. Consequently the caption “gray: periodic” is not accurate for all gray cells, and these percentages must not silently inherit the general paper's “no detected period and positive exponent” chaos definition. The report/blog already describe finite-time limitations elsewhere; preserve them.

Replacement retaining the verified numbers: “Here colors and shares use the finite-window threshold lambda>10^-3; gray denotes its complement, including unresolved nonperiodic cells. This indicator can overlap a detected period at a small number of cells.” Alternatively regenerate all PV summaries/figures with the general period-first label and recheck every number. This is a label/consistency issue; no new scientific run is needed.

**Where/quote:** article HP5 caption 1015–1020; report corresponding `fig:realshrimp` caption; `tools/plot_real_shrimp.py:142–144`: `chaos = clam > 1e-3`, then family and other-periodic masks exclude `chaos`.

**Evidence:** The displayed HP5 panels agree visually with the stated 512x512 layout, axes, cyan followed-domain edge, four cusp markers, magenta fold self-crossings and blue fourth-leg samples. But the plotting source assigns positive-exponent/detected-period cells the chaotic color rather than the stated period-family/other-periodic colors. The already-completed original-array audit established 33 such cells inside the nearest-cell FA mask, 32 in the P5 family; the global stored HP5 `chaotic` flag uses the no-period criterion. This extends G7 from the brief's percentages to a paper figure. No new array reduction was performed here.

**Replacement/action:** Use the declared period-aware chaos mask for panels (a,b), or explicitly caption the colors as positive finite-time exponents that override detected periods. A period-aware redraw should accompany corrected G7 percentages. The separate trace/Type-A and census panels were visually checked only; their older numerical audits were not repeated.


## Should fix


### S1. DSS6: the stated endpoint rule gives 2/43, not 7/38

**Where/quote:** report 2447–2449, blog 4962–4967: “45 times; by an endpoint rule, 7 ... (|mu1| within a factor 3 of |mu2| on both sides) and 38 ... exchanges of dominance”.

**Evidence:** Original DSS6 grid contains exactly 45 horizontally/vertically neighboring solved pairs whose leading multipliers are real and change sign. Only **2** satisfy |mu1|/|mu2| <= 3 at **both** endpoints. They are [(11,17),(12,17)] with ratios [1.174574394,2.712749153], and [(14,18),(15,18)] with [1.396159214,1.470955436]. The other 43 do not satisfy the stated rule. Using “either endpoint” instead gives 22, not seven. Saved `mu`, `mu_imag`, `next_modulus` give the same answer as the saved multiplier triples. Endpoint values alone also do not establish zero crossings versus exchanges of identity.

**Replacement:** “The real leading multiplier changes sign across 45 neighboring solved pairs. Only two have leading and next moduli within a factor three at both endpoints; this endpoint test does not determine whether a continuously tracked multiplier crosses zero or exchanges dominance.” All 45 ratios retained in `geometry-results-001.json`, `dss.endpoint_rows`.


### S2. Disclose four continuation points outside the stated tolerance

**Where/quote:** article main 1144–1145, p6-only 1006–1007, report 2400–2402, blog 4834–4835: “with every point gated ... critical multiplier real and within 10^-7 of +/-1”.

**Evidence:** All 18 relevant P6/kite curve files were checked row by row (1564 stored rows). Four explicitly rejected rows remain excluded. Of 1560 nonrejected rows, four exceed 1e-7: `p6s-tu-fold-wide-minus/curve.json` indexes 106 and 109, deviations 2.55664474257955e-7 and 2.8779428795644435e-7; `p6s-tu-fold-wide-plus/curve.json` indexes 100 and 104, deviations 1.0302396213912601e-7 and 1.2035009988586864e-7. Their residuals are below 1e-10; all critical multipliers are real. They pass the figure's looser 1e-3 filter and are drawn. The eight main body curves and all four kite curves pass the stated strict gate after rejected rows are excluded.

**Replacement:** Either restrict the sentence to the body/kite curves actually satisfying 1e-7 and disclose the extended curves' four exceptions, or state the observed bound “all accepted drawn points have residual below 1e-10 and real critical multiplier within 2.9e-7 of +/-1; four extended-curve points exceed 1e-7”. Do not describe a changed bound as the original admission criterion. Per-file, per-row gates: `geometry-curves-004.json`, `curve_gates`.


### S3. Synchronize or retire the older p6-only geometry

**Where/quote:** p6-only 1016–1022 gives “4.1 and 0.7 grid steps ... Fig. teaser” and “How far the two orbits coexist and where the kite's cusp lies are not established here”; 1046–1051 still gives three cusps and unresolved coexistence/cusp/symbolic sequences. Main has six cusps and the new kite continuation.

**Evidence:** The original flip curves' nearest saved points to the requested rounded R are 1.7398972838 and 0.2587736504 steps of Fig.1 (alpha spacing .015/511, temperature spacing .045/511), rounding to **1.7 and 0.3**. The old numbers used HR9 spacing. The six original cusp records and new S branch exist and reproduce. The alternative branch is not main with only the P5 discussion removed.

**Replacement:** Synchronize the new geometry, corrected Fig.1 distances and current figures into the alternative version, or label the branch explicitly as superseded. Preserve its intended P6-focused scope. This finding does not require a new numerical run.


### S4. Round cusp C’s upper bound upward

**Where/quote:** article main 1162, report 2430–2431, blog 4910: critical multiplier “within 7.6 x 10^-9 of +1”.

**Evidence:** Original `pod64/kite/locate-la-plus-14/locate.json`, final multiplier 0.9999999923760712, gives **7.623928821587356e-9**. The six-cusp table's overall 8.2e-9 bound is valid.

**Replacement:** “within 7.7 x 10^-9 of +1”. The coordinates, c, determinant, next modulus and table rows otherwise reproduce at their displayed precision.


### S5. Do not assign the measured exponent offsets to a single unisolated cause

**Where/quote:** report 809–813: the random-start twin alignment “shifts the exponent ... offsets of +0.0035 and -0.0073”.

**Evidence:** `pod64/hr11/scan.npz` cells [317,298] and [318,298] and `llm-soft-loop/kite/basin-check/basin-check.json` reproduce offsets **+0.0034821545633140216** and **-0.007252607850311968**. Scan source initializes the random twin **after** the 300-step base transient, then immediately starts its 300-step average, so alignment is indeed included. The two comparisons do not separately measure alignment, finite perturbation bias, roundoff, or residual base-trajectory transient effects; no saved per-step alignment ablation is supplied.

**Replacement:** “The averaging window includes the twin's alignment from a random direction. At these two periodic cells the finite-window estimates differ from ln|mu1|/6 by +0.0035 and -0.0073; these offsets are consistent with that transient, whose contribution has not been isolated.” Period-based labels remain unchanged by this comparison.


### S6. Disclose the MSS filter and the eligible periodic denominator

**Location/quote:** article:1244–1245/report:2515–2516, “44,577 ... 23 distinct words in all, each a superstable word of the quadratic family”; article:1256–1257/report:2527–2528, “Of the six-band cells, 3 % carry no word; 63 % of them lie within two cells of a window's edge.”

**Evidence:** The producer explicitly discards non-MSS W at lines 224–229. Raw-array replay reproduces all 409,600 statuses and every assigned label, with zero mismatches: 44,577 retained codes; 1,469 eligible periodic cells rejected, comprising 1,227 phase-vote failures, two too-few-usable cases and **240 otherwise phase-qualified non-MSS words**. Thus 44,817 cells passed the phase stage, and 44,577/44,817 survived MSS filtering. “All retained words are MSS” is true by construction, not independent evidence for universal unimodal organization. The rejected labels are L|L (224), LL|L (12), RR|R (2), RRR|R (2). Failure proportion is 1,469/(44,577+1,469) = about 3.19%, among eligible **periodic** six-band cells, not all chaotic and periodic six-band cells. 922/1,469 = about 62.76% are within Euclidean distance two grid cells of the complement of that eligible domain. A chessboard-distance definition would instead give 1,014/1,469; specify the distance.

**Proposed text:** “After phase voting and an explicit MSS-admissibility filter, 44,577 periodic six-band cells receive 23 distinct window labels. The filter rejects 240 additional phase-qualified cells with non-MSS labels. In total, 1,469 of 46,046 eligible periodic six-band cells fail a coding gate (3.2%); 922 of these failures (63%) lie within Euclidean distance two grid cells of the eligible domain's edge.”

The independent quadratic check itself succeeds: parity-lexicographic enumeration through period nine gives 1,1,1,2,3,5,9,16,28. Separately bracketing roots of f_a^k(0)=0 on 131,073 parameter nodes in [0,2], refining with Brent, and excluding proper-divisor roots recovers each enumerated itinerary in the expected parameter order; maximum return residual 2.0445867221496883e−12. This checks the finite enumeration numerically, not by proof of completeness. It does not validate transferring every label to an LLM window. Primary background: [MSS](https://www.sciencedirect.com/science/article/pii/0097316573900332), [Milnor–Thurston](https://link.springer.com/chapter/10.1007/BFb0082847).


### S7. Present scalar coding as an approximation, not an established reduction

**Location/quote:** article:1220–1225/report:2491–2496, “The window's organization follows from a one-dimensional map ... in most of the region, one-dimensional with a single turning point ... traces a parabola”; article:1240–1243/report:2511–2514, identifies generic orbit words as “the itinerary of the critical point”.

**Evidence:** The retained return maps are finite two-coordinate PCA projections and threshold classifications. The phase gate uses the **95th percentile** of local projected PCA-direction angles <45 degrees, weak-visit fraction <=10%, and at least five minority visits. It does not bound all local directions, establish an invariant one-dimensional manifold or conjugacy, or prove a polynomial parabola. A cosine of six-step-separated finite twin offsets has the scalar derivative sign only conditional on an aligned scalar tangent and a consistently oriented band with the relevant acute angle; the numerical checks are evidence for this approximation, not a theorem in 896 dimensions. Seven of the ten pooled-test failures have more than one robust turning point in the straightest usable projected phase (16 bins, at least four visits per bin, changes accumulated beyond 3% of the y-range). That is the reproduced finite-projection result; the remaining three failures do not become demonstrations of non-unimodality.

The cited [Jones preprint, arXiv:1201.4343v1](https://arxiv.org/pdf/1201.4343) explicitly assumes conjugacy for selected Rössler slices on pp.1–2 and reports numerical ordering checks through period seven. It does not prove this LLM map's reduction. The primary paper's full text is more qualified than its abstract. No peer-reviewed successor was established in the lookup.

**Proposed text:** “Motivated by Jones's unimodal interpretation of selected Rössler return maps, we test an approximate scalar coding of the six-band attractor. Many sampled projected return maps have a single visible turning point. ... We label a periodic orbit by W|X: W is matched to the corresponding quadratic superstable itinerary WC, and X records the orbit's inferred final branch symbol. Away from a demonstrated superstable parameter, this is an orbit label, not a measured critical-point itinerary.” Use “fails the coding test” for red cells; reserve “has multiple resolved turning points” for the seven supported cases. The existing Discussion article:1308 statement that no global one-dimensional reduction is established should govern this paragraph too.


### S8. Disclose the subsequent clean-phase rescoring and its limits

**Location/quote:** article:1253–1256/report:2524–2527 and report table caption:2558–2562, “give the scan's word at all 16 cells where at least one phase is clean”.

**Evidence:** Replaying original `sd/verify-a/verify.json`, `verify-b/verify.json`, and the summary yields exactly 23 selected cells: 16 matching real-multiplier words, two complex-pair C cases, and five with no clean phase. All stored Newton convergence flags are true, with largest reported terminal residual 5.844194761985021e−12; all 21 real cases' stored parity checks are true. Thirteen real cases fail the original all-phase agreement gate. The later `verify-summary` rule keeps only separable phases with every |cos|>=0.5 and accepts if at least one remains and all retained words agree. One of the 16 confirmed cells has only two clean phases, so this is not the scan's at-least-three-votes rule. This is a reasonable disclosed exploratory diagnostic, but “18 confirmed” in summary JSON includes two complex cases with no defined word. “Unconfirmed” is correct for the five, and must not become “contradicted”. Selection is at large-patch centers and does not validate all 23 distinct W or every grid cell.

**Proposed text:** “Under a subsequent clean-phase scoring rule, 16 of the 23 selected cells reproduce the scan label; two other selected C cells have complex leading pairs, and five are unscorable because no phase passes. We retain the original all-phase failures. These checks support selected labels, not a universal validation of the map.”

**Independent-verification limit:** the original Newton JSON files do not retain orbit-state arrays, full Jacobians/monodromy matrices or eigenvectors. Therefore this audit cannot recompute full-state closure, primitive period, eigenpair residuals, Jacobian precision or the advertised exact-Jacobian result. The residuals, convergence and parity booleans above are explicitly stored producer assertions; only the arithmetic/counting/clean-phase rescoring was replayed. Future independent admission requires those arrays and their original source/runtime inventory. No model was loaded to fill this gap.


### S9. Narrow the random-start independence claim

Locations: article main533–534; assigned report5216–5217; blog2027–2031. The heading/sentence says “The start does not matter”. The agreement statistic in producer pv_summary.py is equality of the Boolean indicator `lyapunov>0.001`; it does not compare periods, phases, or orbits. Independent counts from all three original random-start maps give chaos-label agreement99.0478515625%,98.944091796875%,98.974609375%. Their detected period-label agreement is only88.92822265625%,86.474609375%,93.1396484375%, respectively. These finite period differences are not themselves proof of distinct asymptotic attractors, but they contradict any inference of full-map or orbit independence from the Boolean statistic.

Replacement: “For these three random starts, the chaos/non-chaos threshold labels agree with the base run in98.9–99.0% of cells. This does not establish independence of the reached orbit: the detected period labels agree in86.5–93.1% of cells.” Replace the blog heading with “The coarse chaos labels are mostly unchanged by these starts.”


### S10. Scope the reproducibility caption to the observed exceptions

**Where/quote:** article 621 and p6-only 577, Fig. `fig:repro`: “the periodic ones stay at the rounding level”.

**Evidence:** The PNG shows individual periodic/chaotic curves, with medians and quartiles; its source loops over all 192 cells of each label, adding only the median of the 64 other cells. Thus the requested “one line per cell” redraw is present. The blanket caption sentence conflicts with article 632–633, which already reports that only 182 of 192 periodic cells stay within 1e-2. The image has exceptional blue traces above the rounding band. No old REPRO numerical audit was repeated.

**Replacement:** “The chaotic runs mostly separate to order one; the median periodic distance stays at the rounding level, with exceptions.” Keep the exact 190/192 and 182/192 results in the adjacent prose. Panel (c)'s retrospective minimum-nonzero-distance definition matches implementation; the source module docstring's “first nonzero” description is stale, whereas the figure caption correctly says “smallest nonzero”.


## Notes and clarifications

### N1. Exact steering normalization

Locations: article main/p6-only184–194; report steering paragraph; blog“How we steer”. The16 texts and last-token difference recipe agree with the source. The normalization is not just a mean of16 last-token norms: source layer_states averages residual norms over non-padding tokens within each text; build_vectors then equally averages the16 per-text values. Suggested clarification: “rescaled to the mean, over the16 texts, of each text's mean non-padding-token residual norm at that layer.”

Original recorded raw difference norm8.095698612656449; target norm188.99483498183588; independently recomputed saved vector norm188.99483498183585. Equivalent alpha0.042835554809921225. For alpha0.0625–0.0775, push/raw-difference ratios1.459068296823466–1.809244688061098, consistent with an explicitly approximate1.5–1.8 description. Do not frame those rounded ends as certified enclosing bounds. No raw activation replay was performed.

### N2. HP5 counts and classifications

**Where:** 08:58 brief item2 and research-log 05:43 summary; these percentages were not found in the assigned article/report geometry prose.

**Evidence:** The nearest-FA-cell mask contains 29005 HP5 cells and reproduces the quoted 6501 period-4 cells. Under the stated phase-map criterion (positive exponent **and no detected period**), its counts are family 17083 (58.8967%), other periodic 7053 (24.3165%), chaotic 3492 (12.0393%), neither 1377 (4.7475%): **58.9%,24.3%,12.0%,4.7%**. The quoted 58.8%,24.3%,12.2%,4.7% follows when positive exponent overrides period: 33 periodic cells, including 32 family cells, move to chaos. The 89 rather than 90 period-16 cells in the log also follows from this override. Global HP5's stored `chaotic` flag correctly follows the no-period criterion.

**Replacement:** Use the no-period counts above and name the nearest-cell FA mask, or describe the larger 12.2% as positive-exponent share rather than chaotic share. HR12's `scan.json` similarly reports 2526 positive-exponent cells as chaotic_share, whereas only 2519 lack a detected period. This does not alter HR12's stated resolution.

The visual/source addendum extends this issue to the HP5 image itself; see M5. It is therefore not solely a research-log discrepancy.


### N3. Preserve visible unstable-branch encoding

**Where/quote:** article 1213, report 2484, kite caption: “solid where the orbit is attracting and dashed where not”.

**Evidence:** In the actual PNG, the middle unstable S branch between the two folds is a visually continuous gray line; the upper unstable tail is visibly dashed. `tools/plot_p6_kite.py:145–148` requests dashed styling independently for every adjacent pair. Restarting the dash phase on each very short segment can make the dense section appear solid. Color still distinguishes it from the blue stable portions, so this does not reverse the coexistence inference. The green interval starts at the flip as intended, and its bounds, folds, A/B/R intersections and located C marker match the already-audited JSON.

**Replacement/action:** Draw each contiguous unstable segment as a single polyline so the dash pattern remains visible, or caption/legend the stable branch as blue and unstable branch as gray. This is a visual consistency fix, not a new dynamical finding.

### N4. New citations and the scope of their use

The four expressly requested placements—Strogatz/Westervelt1989, Strogatz2018, Gallas1993 at the first shrimp attribution, and Kuznetsov at Neimark–Sacker—are appropriate. `citations-review.md` records primary publisher/author links and metadata. No bibliographic must-fix was found. The original note's narrow Gallas ending-page uncertainty is resolved by the existing preserved APS-deposited Crossref record: pages2714–2717, SHA256 `ea2f4fae403c7802eebae88457d1a8d6d4b7d1bcb7f26f4ef7ec7120bbb90f15`; see `citations-gallas-ending-page-resolution.json`. This used an existing primary-source receipt, not a repeat of the prior whole bibliography audit. Singer and Jones require the interpretive qualifications in M4/S7.

### N5. Reruns, precision of retained derived fields, and selection

The six merged SD bands reproduce their original-checkpoint plus rerun rows exactly. Their rerun rows are not bitwise continuations of HR10, and the observed differences do not isolate batch composition as the cause. Of ten unsplit bands, all eligible periods and chaos labels match; c13 has34 exponent differences of at most1.1102230246251565e-16. Thus a blanket bit-for-bit statement needs that negligible exception. Original projection/cosine fields are saved in float32 despite the underlying model being described as float64; the audit replays those retained values. Two unusable phases differ only in null-versus-zero representations for undefined errors, and neither contributes to a passing result. Full details and the original discrepancies are preserved in `symbolic-review.md` and `symbolic-results.json`.

### N6. Numerical values retain their scope

M5’s PV “gates pass” refers only to finite/grid/configuration/prefix identity gates. It is not trajectory convergence, effective-precision or chaos certification. The independent static reviewer noticed a possible missing-key blind spot in the original mismatch dictionary; a separate actual22-record key-set check found no missing or extra keys (`pv-argument-key-closure.json`). The period-aware classification issue remains explicitly open for producer correction. Rounded ranges should be read as approximate unless an enclosing bound is expressly stated; no rounded figure or summary was used as input for a raw bound.

## Evidence gaps requiring a specific source or narrower wording

1. **P6 body boundary counts:**294/419 and384/742 are not reproduced by the supplied whole-domain mask/curve recipe. Whole-domain eight-neighbor counts are388/916 flip and1720/1888 fold; four-neighbor counts are178/406 and718/798. Supply the exact body-coordinate mask, neighbor rule and original curve inventory, or leave the subset claim provisional. This does not show that the curves themselves are wrong.
2. **Stipple basin pair2:** the named `llm-soft-loop/kite/basin-check-stipple/basin-check.json` is absent. Supply its actual original path/receipt. Pair1 supports the two checked adjacent cells; it does not certify the entire edge or pair2.
3. **SD Newton admission:** only original JSON summaries were available for the23 checks. Supply saved orbit phases, full matrices/eigenvectors and original execution/source inventory to permit independent closure, primitivity, eigenpair residual and derivative-precision checks. Stored convergence flags alone do not provide this independent admission.
4. **Figure provenance:** repro and HP5 lack adjacent complete figure JSON, and kite JSON does not bind its background scan/resolution. All inspected PNGs match assigned-commit blobs and were viewed, but that visual match is not an independent data-lineage proof. Retain exact CLI/input hashes when redrawing. The brief's figure numbers are stale; file/label identities are authoritative.

These are documented verification limits, not failed scientific claims filled in by inference. No new numerical job was launched to replace absent evidence.

## Coverage and checked-correct results

| Request | Outcome |
|---|---|
|1 HR10|640x640,16 bands and66.52734375% reproduce; all409600 exponents finite.|
|2 HP5/FA|512x512,16 bands,60.396194458% reproduce.276/292 and717/754 boundary counts reproduce. Inside-domain mask naming/classification requires M5/N2.|
|3 Six cusps|All six coordinates/c/detJ rows and table bounds reproduce; cusp C prose needs S4.|
|4 R pair|Saved phases give6.379961949743435% aligned gap; both saved residual gates pass; multipliers−.9952248298/−1.0005445296. Nearest points1.7399/.2588 Fig1 steps. Separate independent manual run still active.|
|5 Kite branches|All124 LA and66 UT branch residual/convergence rows pass stored gates; fold estimates/intersections and attracting overlap reproduce. Rounded R and branch crossing estimates are not bisected roots.|
|6 DSS6|529/961 solved, minimumrho.00631905794 and third−.00509207 reproduce; determinant/sign-endpoint interpretation needs M1/S1.|
|7 PV|All22 arrays/prefixes/token IDs and original archive joins checked. Counts/ranges/Jaccard/frequency reproduce for exponent-threshold masks; see M5/S9.|
|8 Steering|16 source texts, layer/position recipe, saved norm188.99483498/raw8.09569861 and ratios reproduce; exact averaging specified in N1.|
|9 Figures|Four requested PNGs and extra fold PNG inspected and assigned-byte matched; one-line-per-cell repro redraw present. M5/S10/N3 and provenance gaps remain.|
|10 Return maps|Raw projection/cosine replay gives58 six-band cells,48 pooled passes,23 zero-error;7 of10 failures have multiple resolved turns. M3/M4/S7 qualify inference.|
|11 Words|All409600 statuses and44577 labels replay exactly;1469 failures decomposed,23 retained MSS words, independent quadratic counts1,1,1,2,3,5,9,16,28 reproduce. M2/S6/N5 qualify scope.|
|12 Newton word checks|Stored23 records reproduce16 clean matches+2 complex C+5 unscorable; independent full matrix/phase verification unavailable. See S8/gap3.|
|13 HR11/HR12/basins|512x512/640x640 stitch and four partial-row merges reproduce; pair1 saved gates pass, pair2 missing. No whole-edge proof.|
|14 Exponent offsets|+.00348215456/−.00725260785 reproduce from saved rows; causal attribution needs S5.|
|15 Blog/cross-version use|Blog prompt count/limitations and kite fields reviewed; p6-only remains older. New SD text is absent from the frozen blog/p6-only, so its findings are not attributed to absent sections.|

Detailed checked-correct values, exact source URLs, all raw-row gates and exceptions are in `geometry-review.md`, `symbolic-review.md`, `pv-steering-review.md`, `figure-pv-review.md` and `citations-review.md`. The finite numerical geometry does not by itself establish a global one-dimensional or two-dimensional reduction, a whole-box basin, fixed-start reachability, or an interval-certified full cascade.

## Execution and preservation

The independent geometry and symbolic agents used small bounded, single-threaded saved-data reductions; every original failure and later explicit correction remains. Geometry’s first HR12 comparison omitted a checkpoint reshape, corrected in the separately retained details output. Three curve-extractor failures were retained before the schema-correct successful attempt. The symbolic five-stage outputs all closed with exit0; the slower first source version and two excluded-phase representation mismatches remain. No completed stage was silently rerun.

Root’s PV saved audit was committed before its single0.492357207-second execution. Its first private push raced producer main and was rejected; an orchestration sequencing error let this small saved audit run before successful remote delivery. This is disclosed in `pv-steering-operational.json`, along with a draft timestamp error. Source and original output were subsequently privately delivered; no repeat. This limitation does not apply to the separate Task37 target, whose exact source/private checkpoint/admission were verified before launch. Original22-archive joins were separate metadata-only work, not model execution. No source or raw output of an active target was edited.

The raw snapshot/evidence retention receipts preserve the uncommitted blog and every audited source version; compact results and hashes are committed here. The live Task37 R-pair retains original limits and requires independent saved-output/host-closure verification before any scientific conclusion. No automatic body-point dispatch or retry is authorized by this review.
