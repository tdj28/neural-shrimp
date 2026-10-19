# Task38 geometry audit: original saved evidence

The saved orbits support two distinct period-6 loops near R and an attracting overlap on the continued S-shaped branch. The grid counts, six located cusp coordinates and most scalar summaries reproduce. The report/blog's DSS6 determinant and endpoint classifications do not reproduce as stated. No model evaluations, GPU calls, process controls, producer edits, or changes to Task37 were made.

## Snapshots and scope

- Article main: `b8413881e7ce97a9e58adcf1bf7d78a8cebb0514`, `main.tex`, SHA-256 `d18cb53e2b11117bb8b5494f8b2ff53537a0f8da7497972880767da5311c518d`.
- Article p6-only: `d49718d3294f180e67f94de8194536458655e918`, `main.tex`, SHA-256 `9df885c80e7cf3fa96edcf13ec00d3b7dc643c4cf0c577eee5f7e012291cb139`.
- Report: `651a50059a145e2bb0526f9675bcc41e9cb895a8`, `paper/llm/main.tex`, SHA-256 `a903c8a0373333f01a2bdbc3e4f66cfcdafc2101005077a076a75611661cc912`.
- Blog: root's read-only snapshot `/tmp/task38-root/blog-index.md`, SHA-256 `4c6646cae55b08d5d3d85dd19ef9d0ad11348a82bef504fc20f3be92ecd21926`; uncommitted source `.../chaos-bifurcations-language-model-feedback/index.md`.
- Brief read in full: origin/main `11895a13cd565907070c0182a2c1f992229a28da`, `comms/claude-to-codex.md`, 2026-10-19T08:58Z entry. Producer code was inspected as source only, never imported. Exact raw files/hashes are in the evidence outputs and manifest.

Owned scope: items 1–6 and 13–14. These are independent reductions of saved records, not independent model/derivative recomputation. Raw classifications are checked against saved fields; unavailable closure trajectories/eigenvectors cannot be recreated by metadata checks. Old Task37 Part1 is not repeated.

## Ranked findings

### G1 — must-fix: the claimed DSS6 determinant is complex at 240 solved cells

**Where/quote:** report 2446–2447, blog 4960–4961: “The product delta = mu1 mu2 of the two leading multipliers is negative in 151 of the 529 cells”. The following text treats delta as a real two-dimensional block determinant and discusses its zero curve.

**Evidence:** Original `pod64/dss/p6-body/grid.npz`, saved `multipliers` in descending modulus. At 240/529 solved cells, the first multiplier is real and the second is only one member of a complex subleading conjugate pair: their product has a nonzero imaginary part (maximum absolute imaginary part 0.00696132006799271). The count 151 is reproduced by **Re(mu1 mu2) < 0**, not by a negative real determinant. Eight of those 151 products are nonreal. The top two form a real/conjugate-closed pair in 289 cells; 143 of those products are negative. Example [row 0, col 12]: multipliers -0.3030628280 and 0.0016020178 + 0.0030695243i, product -0.0004855121 - 0.0009302587i. The explicit conditional-model paragraph does not cure this preceding empirical claim.

**Replacement:** “At 289 solved cells the two largest-modulus multipliers form a real or complex-conjugate pair; their real product is negative at 143. At the other 240 cells the leading real multiplier is followed by a complex conjugate pair, so selecting only the first two does not define a real two-dimensional determinant. We do not infer a determinant-zero curve from that selection.” Retain the separately qualified conditional model. Exact 240 rows: `geometry-details-001.json`, `DSS_nonreal_top2_product`.

### G2 — should-fix: the stated 7/38 endpoint rule gives 2/43

**Where/quote:** report 2447–2449, blog 4962–4967: “45 times; by an endpoint rule, 7 ... (|mu1| within a factor 3 of |mu2| on both sides) and 38 ... exchanges of dominance”.

**Evidence:** Original DSS6 grid contains exactly 45 horizontally/vertically neighboring solved pairs whose leading multipliers are real and change sign. Only **2** satisfy |mu1|/|mu2| <= 3 at **both** endpoints. They are [(11,17),(12,17)] with ratios [1.174574394,2.712749153], and [(14,18),(15,18)] with [1.396159214,1.470955436]. The other 43 do not satisfy the stated rule. Using “either endpoint” instead gives 22, not seven. Saved `mu`, `mu_imag`, `next_modulus` give the same answer as the saved multiplier triples. Endpoint values alone also do not establish zero crossings versus exchanges of identity.

**Replacement:** “The real leading multiplier changes sign across 45 neighboring solved pairs. Only two have leading and next moduli within a factor three at both endpoints; this endpoint test does not determine whether a continuously tracked multiplier crosses zero or exchanges dominance.” All 45 ratios retained in `geometry-results-001.json`, `dss.endpoint_rows`.

### G3 — should-fix: four accepted drawn P6 continuation points exceed the stated 1e-7 multiplier gate

**Where/quote:** article main 1144–1145, p6-only 1006–1007, report 2400–2402, blog 4834–4835: “with every point gated ... critical multiplier real and within 10^-7 of +/-1”.

**Evidence:** All 18 relevant P6/kite curve files were checked row by row (1564 stored rows). Four explicitly rejected rows remain excluded. Of 1560 nonrejected rows, four exceed 1e-7: `p6s-tu-fold-wide-minus/curve.json` indexes 106 and 109, deviations 2.55664474257955e-7 and 2.8779428795644435e-7; `p6s-tu-fold-wide-plus/curve.json` indexes 100 and 104, deviations 1.0302396213912601e-7 and 1.2035009988586864e-7. Their residuals are below 1e-10; all critical multipliers are real. They pass the figure's looser 1e-3 filter and are drawn. The eight main body curves and all four kite curves pass the stated strict gate after rejected rows are excluded.

**Replacement:** Either restrict the sentence to the body/kite curves actually satisfying 1e-7 and disclose the extended curves' four exceptions, or state the observed bound “all accepted drawn points have residual below 1e-10 and real critical multiplier within 2.9e-7 of +/-1; four extended-curve points exceed 1e-7”. Do not describe a changed bound as the original admission criterion. Per-file, per-row gates: `geometry-curves-004.json`, `curve_gates`.

### G4 — should-fix if p6-only is a submission candidate: it is an older geometry version

**Where/quote:** p6-only 1016–1022 gives “4.1 and 0.7 grid steps ... Fig. teaser” and “How far the two orbits coexist and where the kite's cusp lies are not established here”; 1046–1051 still gives three cusps and unresolved coexistence/cusp/symbolic sequences. Main has six cusps and the new kite continuation.

**Evidence:** The original flip curves' nearest saved points to the requested rounded R are 1.7398972838 and 0.2587736504 steps of Fig.1 (alpha spacing .015/511, temperature spacing .045/511), rounding to **1.7 and 0.3**. The old numbers used HR9 spacing. The six original cusp records and new S branch exist and reproduce. The alternative branch is not main with only the P5 discussion removed.

**Replacement:** Synchronize the new geometry, corrected Fig.1 distances and current figures into the alternative version, or label the branch explicitly as superseded. Preserve its intended P6-focused scope. This finding does not require a new numerical run.

### G5 — should-fix: round cusp C's upper bound upward

**Where/quote:** article main 1162, report 2430–2431, blog 4910: critical multiplier “within 7.6 x 10^-9 of +1”.

**Evidence:** Original `pod64/kite/locate-la-plus-14/locate.json`, final multiplier 0.9999999923760712, gives **7.623928821587356e-9**. The six-cusp table's overall 8.2e-9 bound is valid.

**Replacement:** “within 7.7 x 10^-9 of +1”. The coordinates, c, determinant, next modulus and table rows otherwise reproduce at their displayed precision.

### G6 — should-fix scope: the two measured exponent offsets do not isolate their cause

**Where/quote:** report 809–813: the random-start twin alignment “shifts the exponent ... offsets of +0.0035 and -0.0073”.

**Evidence:** `pod64/hr11/scan.npz` cells [317,298] and [318,298] and `llm-soft-loop/kite/basin-check/basin-check.json` reproduce offsets **+0.0034821545633140216** and **-0.007252607850311968**. Scan source initializes the random twin **after** the 300-step base transient, then immediately starts its 300-step average, so alignment is indeed included. The two comparisons do not separately measure alignment, finite perturbation bias, roundoff, or residual base-trajectory transient effects; no saved per-step alignment ablation is supplied.

**Replacement:** “The averaging window includes the twin's alignment from a random direction. At these two periodic cells the finite-window estimates differ from ln|mu1|/6 by +0.0035 and -0.0073; these offsets are consistent with that transient, whose contribution has not been isolated.” Period-based labels remain unchanged by this comparison.

### G7 — note/summary correction: HP5's inside-FA percentages use a different chaos rule

**Where:** 08:58 brief item2 and research-log 05:43 summary; these percentages were not found in the assigned article/report geometry prose.

**Evidence:** The nearest-FA-cell mask contains 29005 HP5 cells and reproduces the quoted 6501 period-4 cells. Under the stated phase-map criterion (positive exponent **and no detected period**), its counts are family 17083 (58.8967%), other periodic 7053 (24.3165%), chaotic 3492 (12.0393%), neither 1377 (4.7475%): **58.9%,24.3%,12.0%,4.7%**. The quoted 58.8%,24.3%,12.2%,4.7% follows when positive exponent overrides period: 33 periodic cells, including 32 family cells, move to chaos. The 89 rather than 90 period-16 cells in the log also follows from this override. Global HP5's stored `chaotic` flag correctly follows the no-period criterion.

**Replacement:** Use the no-period counts above and name the nearest-cell FA mask, or describe the larger 12.2% as positive-exponent share rather than chaotic share. HR12's `scan.json` similarly reports 2526 positive-exponent cells as chaotic_share, whereas only 2519 lack a detected period. This does not alter HR12's stated resolution.

## Checked and consistent

- **HR10:** 640x640; all 409600 exponents finite; 272496 chaotic = 66.52734375%, hence 66.5%. Exact window alpha [0.0718,0.0746], log10T [-0.157,-0.149]; spacings 4.38184663536e-6 and 1.25195618153e-5. All 16 constituent bands match stitched coordinates, periods and exponents bitwise.
- **HP5:** 512x512, all exponents finite; 158325/262144 = 60.396194458%, hence60.4%. All16 bands match. Boundary gate recomputed independently from original FA array and curve segments:276/292 flip pairs and717/754 fold pairs within1step;754/754 within2steps of fold **or flip** (fold-only736/754). Retained maxima and every curve's raw gate status.
- **Six cusps:** all six saved located flags/checks pass; each initial/final refinement bracket changes sign. All coordinates, c and detJ reproduce; detJ independently recalculated from saved2x2J. Maximum |final f|9.477318099e-5, maximum critical gap8.161603171e-9, maximum next modulus.7390799322: table bounds9.5e-5,8.2e-9,.74 hold. Initial maximum bracket magnitude spans2.459650168..39.108239996. Literal “values at the bracketing points” should mean each bracket's maximum magnitude: individual endpoints go down to.1704771. Final state closure residuals are absent in locate.json; the separate finite-difference unfolding-b estimate differs from final f (different step choices), and no extra derivative accuracy claim is made here.
- **R:** original saved arrays give best cyclic shift0, max point distance.01707902010461918, reference mean state norm.26769783643154804, relative gap.06379961949743435 (6.4%). Both saved residuals pass (2.8789e-14,2.0314e-13), multipliers -.9952248298 and -1.0005445296. Proper-divisor distances from saved phases are far from0. At(.0730,-.1539), the ad-seed attempt failed (residual8.9256e-4), so its apparent spectrum/gap is not an admitted orbit; retained. At(.0731,-.1537), both seeds converge to the same orbit (gap6.13e-14 relative); retained. Actual closure/model derivatives are not recomputed here.
- **Kite:** all124 LA and66 UT branch rows have residual<1e-10 and saved multiplier-convergence flags true. LA fold estimates round to alpha.072992 and.072793; flip seed.072906992852 gives the stated approximate coexistence interval; second-sheet flip bracket.0730069082..0730161158. UT fold estimate rounds to log10T-.153657. Crossing tags in branch.json are bracketing classifications, not bisected unit-circle roots: e.g. saved LA endpoint multiplier1.049385, UT1.014080. Fold-curve continuation starts and all kite curve points pass the strict unit-multiplier gate. Independently intersected saved polylines reproduce A(.07299132159,-.15398409397), B(.07289989526,-.15370652555), R(.07291177454,-.15410388096). Exact R remains an interpolation approximation; attributing its multiplier offsets “only” to interpolation is stronger than a measured error bound.
- **Gallas reading:** original PhysicaA202(1994) PDF journal pp.217–218, Fig.7 names C–A–R–B as a kite, identifies different orbit sheets at pseudo-intersections and discusses intertwined basins. Describing the observed two-sheet F^6 geometry as the analogous kite is fair at numerical-evidence level. Gallas specifically names tail T a pseudo-intersection of superstable curves; the paper's application to flip crossing R is its descriptive extension, not an exact quote from Gallas.
- **DSS6 basics:**529/961 solved;432 unsolved retained. Every solved row has same_tokens=true; maximum saved Newton residual9.572879009e-13. Minimum modulus.006319057939216359 at(.0730533333333,-.1512), leading pair.0003990227004 +/- .0063064470285i; third-.0050920721215. Nine complex-leading cells, moduli.006319..037996, arguments43.011..134.457degrees. The article's geometric mechanism remains expressly conditional and notes three comparable modes/no measured saddle or coupling. No crossing is inferred from unsolved cells.
- **HR11/HR12:**512x512 and640x640, all exponents finite; each stitched array matches all16band products. HR12's four merges (c02,c05,c08,c14) match flattened-checkpoint first38rows plus rerun last2rows after reshaping originals. Initial audit comparison omitted that reshape and falsely returned four prefix mismatches; corrected explicitly in geometry-details-001.json; both outputs retained. Kite caption's resolution ratios7.5 and4.6 reproduce.
- **Basin pair1:** both fixed-start solves and both cross-cell continuations pass saved residual gates; continued leading multipliers.1833483079 and.6011826819 stay attracting and gaps to the other orbit exceed.0215. This supports a switch in which basin contains the fixed start at the checked adjacent cells. A categorical statement about the entire extended edge exceeds these two cells; no whole-boundary continuation was supplied.

## Partial, missing or deliberately skipped

- Brief path `pod64/kite/newton-at-R` is incorrect: actual originals are `llm-soft-loop/kite/newton-at-R`. Correct path used and hashed.
- `llm-soft-loop/kite/basin-check-stipple/basin-check.json` is absent at audit time. No pair2 convergence or basin classification could be checked. HR12's stippled groups can be described visually/empirically, but a verified pair2 basin conclusion is pending. No unbounded search was performed to guess another location.
- P6 “over the body” boundary counts294/419 and384/742 are not reproduced by the supplied full-domain figure recipe: eight-neighbor full-domain counts are388/916 flip and1720/1888 fold; four-neighbor counts178/406 and718/798. These differences may reflect an unstated body mask and an older curve set. Keep the quoted subset result provisional until its exact coordinate mask, neighbor rule and curve inventory are supplied. The exact whole-domain figure JSON counts do reproduce; this is not evidence that the depicted curves are wrong.
- No extra run to isolate exponent alignment, no tangent/model replay, no interval certification, no curve continuation or root refinement, and no new basin trajectories. HR12's future random-start comparison is outside this supplied evidence.
- No pixel rendering audit here; root owns item9. The symbolic-dynamics checks are assigned to the companion agent.

## Execution and preserved failures

All saved-array reductions used existing `/data2/neural-shrimp/.venv/bin/python -B`, nice10, HF_HUB_OFFLINE=1 and OMP/OPENBLAS/MKL/NUMEXPR threads1. Runs are serial. Main audit/curves caps:2GiB address space,100CPU seconds,120s alarm; details cap:1GiB,20CPU seconds,30s alarm. Curve/details scripts refuse to start below16GiB available RAM. Main audit completed before that added shared-floor check and under2GiB hard cap. Metadata reductions do not claim any numerical admission for a model run.

`geometry-curves-001` failed on list/scalar division;002 on wrong P6 coordinate key `alpha`;003 on wrong P6 coordinate key `alpha2`;004 passed after using the recorded `alpha1` and `log10_temperature` axes. All errors and empty partial stdout files are retained, never overwritten. The first main audit's HR12 flat-array prefix mismatch is separately corrected, rather than suppressed. No target evidence was modified by these extractor errors.

Authoritative successful outputs: `geometry-results-001.json` (except corrected HR12 prefix comparison), `geometry-curves-004.json`, `geometry-details-001.json`. `geometry-manifest.json` pins the scripts, all retained output attempts, source snapshots and original files.
