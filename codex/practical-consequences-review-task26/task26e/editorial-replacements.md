# Task26: exact-line wording proposals for Claude

These proposals use `paper/llm/main.tex` at `666e9b72e9b384755bf52469dbf1ed6c1d4f89f3`, SHA256 `803d9a53c66732d787a12773b6736ac0cd08079fc7001f0ad7b6a16a9ae9c959`. The exact-byte line map is `70409576faa113ade3249aebcdec0cef145cfd46ea72602a3b0fe396c9eef5d6`. They supplement the source-bound findings; none edits the producer paper. Task26(a) numbers below come from actual report `7dbdaad85343eece750927ec59cf699f8b95a38b7785690a31bc762862e50726`. Other numerical claims await the corresponding actual saved-data results. Task27 must map accepted corrections to its later frozen paper.

## Must fix

**Lines 666–668 and 705–706: batch padding description.** Replace “the same rows followed by the same cells in reverse order” with “the same target rows followed by additional rows whose steering vectors are reversed while temperatures retain their original order.” Add that prefix caches are recomputed at the two batch shapes. The paired first 448 target rows are unchanged; this corrects what the padding does.

**Lines 669–676, 708–709: threshold and forecast.** Separate crossing relative distance 0.01 from order-one separation. The historical plot uses the minimum positive distance over the whole trajectory, so label that curve “retrospective minimum-distance scaling.” Report a first-positive-distance calculation separately, without silently replacing the historical result or its censored cases. The exact counts, ratios and maximum/final-distance statements will come from Task26(b). The logarithmic expression is a duration from the seed observation, so retain the first-positive step and distinguish that duration from an absolute crossing-step prediction whenever the seed step exceeds one.

**Lines 670, 679, 700, 716, 735–738: readouts versus generated text.** Use “most-probable-token readouts disagree at least once during the final 500 steps.” The expected-embedding loop does not emit those argmax tokens into its next input. Replace the table heading with “any tail readout disagreement.” Keep final-step disagreement distinct. The caption already gives the tail-window definition, but the heading and surrounding prose should use it consistently.

**Lines 691–698: short answer test and its comparator.** Suggested replacement:

> In the fixed 32-step growing-context pipeline, canonical first-integer answers differed between batch layouts in 54 of 96 indexed problems at T=1 and 8 of 96 at T=10^0.2. The forced-length greedy comparator had no canonical disagreement. The mean excess disagreement over those two temperatures was 32.3 percentage points (paired problem-bootstrap 95% interval [26.6, 38.0]). In 10,000 paired sign flips none was as extreme; the plus-one estimate is 1/10,001, approximately 10^-4. No canonical answers differed between layouts in the corresponding float64 runs.

Then retain the existing warning about the greedy comparator's zero held-out accuracy; do not call it ordinary greedy decoding or a useful accuracy baseline. The parsed-answer counts differ from raw-text disagreement (68/96 and 26/96 at the two bfloat16 temperatures). The original inline producer resampling implementation was unavailable: the independent audit freezes an explicit convention, reproduces the rounded interval, and does not claim bit-exact recovery of an absent script.

**Lines 694–701: finite perturbation interpretation.** Suggested replacement:

> On six fixed float64 problems, the median finite perturbation-growth rate at T=1 was 0.033 nats per step over the tested kicks; individual rates ranged from -0.127 to 0.241. No canonical answer changed for kicks through 10^-4. This finite check is consistent with batch-dependent low-precision errors propagating through continuous feedback in this setup; it does not identify an asymptotic Lyapunov exponent or exclude nonlinear amplification in other contexts. This short growing-context experiment is distinct from the long autonomous-loop divergence experiment.

The phrase “grows by only 0.03” converts a median into a bound and should go. “The discrepancies are therefore not the float64 map's chaos” is stronger than the sampled finite experiment establishes.

**Lines 396–403 and 410–417: crossed derivatives and causal scope.** Replace “A crossover at fixed states rules this out” with “A crossover at fixed states separates the two derivative contributions along the recorded steered orbits.” Name these finite-time crossed tangent cocycles. They do not independently establish attractors of autonomous crossed systems or exclude state relocation; the later high-temperature discussion itself invokes relocation. Keep the low-temperature conclusion next to the high-temperature exception. Report the original all-cell validation and closeness-based sufficient criterion under their actual passed/failed dispositions once Task26(d) is read.

## Should fix

**Lines 689–693: held-out unit.** A short Methods clarification should say “96 indexed instances,” retain the fixed list, and disclose its 95 distinct ordered additions / 94 distinct unordered additions, one reversed overlap with a development problem, one exact distractor overlap, and the earlier unidentified held-out smoke problem. These are scope limitations, not authorization to select a cleaner subset after results.

**Lines 389–393: projected vector.** Replace “along the direct logit effect of the steering vector” with “along the output-projected direction Wv.” For a vector injected at an interior transformer layer, Wv is not its end-to-end logit derivative. “Network function unchanged” is accurate for a pure logit bias; “network Jacobians unchanged” along the resulting trajectories would not be.

**Lines 433–434 and matching definitions: intrinsic dimension.** The stored trace(A)/estimated-leading-eigenvalue quantity is an estimated intrinsic dimension of the positive-semidefinite readout covariance operator, not stable rank(A). Preserve the finite power-iteration qualification. This is a naming correction, not a request for another model run.

**Lines 380–387 and 428–429: finite gain and inference.** The normalized finite-twin gains telescope with their normalization boundary term. Distinguish this finite identity from an exact infinitesimal largest Lyapunov exponent. State that the map contrasts weight temperature rows by chaotic-cell count, bootstrap rows and perform literal within-row shuffles. They are observational group contrasts; overlapping groups, field-specific missingness and exchangeability assumptions must remain visible.

## Delivery scope

The actual Task26(b/c/d) reports supply the remaining numbers and gate dispositions. The newly added 7B, layer-depth, two-block and behaviour material is not automatically covered by these original REPRO1/MECH/XO inputs. Task27 should reuse accepted matching evidence and inspect new complete sources where needed, without rerunning completed analyses or treating an unpublished/incomplete replication as admitted.
