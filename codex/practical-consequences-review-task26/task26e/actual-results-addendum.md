# Task26: actual-results wording addendum for Claude

This supplements, and does not replace, the preserved prospective wording file `a24b5ce306fdc0e2f3c44cd4cbd696871f80d2b2e1a191b76b65fdcb2d46550c`. Line numbers refer to `paper/llm/main.tex` at `666e9b72e9b384755bf52469dbf1ed6c1d4f89f3`, SHA256 `803d9a53c66732d787a12773b6736ac0cd08079fc7001f0ad7b6a16a9ae9c959`. Task27 must map these corrections onto its later frozen paper. No producer paper is edited here.

## Must fix: report the original crossover failures

At lines 398–403, after describing the four tangent arms, use:

> Arm 11 agrees with the saved twin exponent to within 0.012 at 23 of 24 cells. The remaining periodic cell has an absolute discrepancy of 0.9150, so the original all-cell validation criterion (every discrepancy at most 0.02) fails. In the 14 held-out chaotic cells, the readout effect is +0.8053 nats per step (95% cell-bootstrap interval [0.2933, 1.3453]), the network effect is −0.4895 [−0.6254, −0.3530], and the interaction is +0.2609 [0.0971, 0.4494]. The absolute signed-mean difference between the readout-only and full arms is 0.2286, exceeding the frozen 0.1 tolerance. Thus the original joint sufficiency criterion fails despite the positive average readout effect.

The failing validation row is cell 16: arm 11 = −12.120354287048665 versus twin = −11.205323828986392. The maximum discrepancy among the other 23 cells is 0.011804325387121395. The closeness gate is `abs(mean(arm10 − arm11))`, not the different descriptive statistic `mean(abs(arm10 − arm11)) = 0.3606454454663198`. Neither substitute a secondary gate nor drop the periodic failure.

At lines 396–397 replace “A crossover at fixed states rules this out” with:

> A crossover separates the derivative contributions along the recorded steered trajectories. These finite-time crossed tangent cocycles do not by themselves establish attractors of autonomous crossed systems or exclude changes in the states visited under steering.

At lines 401–403, explicitly describe the low-temperature values as averages over six held-out cells. The positive arm-10 average is not a universal sign: cell 3 has arm-10 growth −0.53453. The high-temperature findings concern the eight held-out cells in that subgroup. Keep the fixed development exclusions; do not reselect cells after inspecting signs.

## Must fix: three confidence-interval endpoints

The independent implementation exactly reproduces the producer's stored statistics, but the frozen prose/table contain these rounded-endpoint errors. With ordinary rounding to two decimals:

| Location | Quantity | Exact saved interval | Correct display |
|---|---|---|---|
| Lines 382 and 443 | R1 readout, chaotic minus near-zero steering | [1.7855411624624198, 3.5869886701923246] | [1.79, 3.59], replacing [1.79, 3.58] |
| Lines 383 and 444 | R1 network, chaotic minus near-zero steering | [−0.2209727719852761, 0.11857395671885369] | [−0.22, 0.12], replacing [−0.21, 0.11] |
| Line 386 | R2 readout, chaotic minus periodic | [0.9153638750006351, 1.9260422237523345] | [0.92, 1.93], replacing [0.92, 1.92] |

R1's network mean is −0.06837074136771622. Its row-bootstrap interval crosses zero although the literal permutation statistic gives p = 0.029397060293970604. Report the two procedures under their actual definitions; a small point estimate and a crossing interval do not establish equivalence or absence of a network effect. Do not silently force the two procedures into a single calibration.

## Finite gain and intrinsic-dimension scope

For lines 378–380 and 428–429, suggested clarification:

> The recorded normalized finite-twin gains obey a telescoping identity, with the normalization boundary term, on rows where the quantities are finite. The saved step arrays cover 1,000 steps whereas the original map summaries cover 1,500; these windows are retained separately. The contrasts use the original reference labels, field-specific finite eligibility, chaotic-count row weights, row bootstrap and within-row shuffles.

The stored identity residuals are at most 1.75×10^−12 in magnitude where finite. Do not extend that statement to the nonfinite rows: the two low-temperature map archives contain numerous nonfinite twin summaries and some nonfinite step gains. No reference-chaotic row has a nonfinite saved twin summary. All eight contrast families admit the same 18 temperature rows, from log10 T = −0.2 through 0.225. Near-zero and chaotic groups overlap in one observation at 0.15 and five at 0.2; retain the original definition and disclose the overlap.

For lines 433–434 and other definitions, use “estimated intrinsic dimension,” `trace(A)/estimated leading eigenvalue`, for the PSD readout covariance operator. This is not `stable rank(A)`. The producer command used four power iterations; it is neither an exact leading eigenvalue nor a guaranteed bound. Extreme estimates and nonfinite power-iteration fields remain in the record. The particular enormous alpha-map maximum and sub-unit minimum occur in temperature rows excluded by the frozen finite-count criterion; the bias-map extrema are outside alpha-map R2. They were not removed by a newly chosen value threshold.

## Pure-bias and precision statements

The original R4 count/share gate passes: 241/6,837 pure-bias cells (3.52494%) versus 331/5,353 residual-steering cells (6.18345%) have the saved chaotic classification. Both the absolute count and share exceed their frozen thresholds. Same-temperature factor comparisons remain descriptive, including rows with undefined comparisons and changing factor signs. A pure output bias preserves the network function but changes the trajectory; it does not preserve the evaluated Jacobian along all later states. Use “output-projected direction Wv” at lines 389–390.

The component source audits distinguish the float64-first global normalization patch in REPRO1 from the bfloat16-first REPRO-T load order. Do not transfer one precision description to the other. This review reproduces saved evidence; it does not rerun model generation or retrospectively establish unsaved token IDs, trajectories or remote model bytes.

## Actual evidence

- Task26(b) actual report: `3b95ea55025c8a6bd8d8a2ba7523a7036781999362d443bedeecd189ec5710aa`; accepted scientific review: `aceb846b039e4a1a7fe5495acf1df01d132c56e42c6392089dff509030f49725`.
- Task26(c/d) actual report: `601299cbee88f34a30458d6d7b80897ad0015238289b12923fb5db0e1d8dc6fa`; all 105 producer-summary comparison leaves agree exactly.
- Task26(c/d) final source/output/closure authority: `c9b65c9d094d757a90800c16fb1d910281e0c6d1953256c7b86fb69f0c254697`; manifest: `f7f19a9495aebe8a36262c34a2979d336cc13f115b38dfa2a9d7a70b0ca0cc4e`.

The full component review supplies the complete membership, cell, window, gate and comparison dispositions. The new 7B/depth/two-block experiments are outside this original-input review and must be reconciled separately in Task27.
