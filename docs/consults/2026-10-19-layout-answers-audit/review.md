# Verdict

The corrected headline rates agree with the supplied compact summaries. Their magnitude alone is not evidence of an artifact: feedback, stopping decisions, extraction, and sampling can turn small numerical differences into different recorded answers. The cited literature, however, cannot validate these particular percentages.

The defensible result is **layout sensitivity of specified finite-budget pipelines**, not a general failure of accurate reasoning, a uniquely continuous-feedback effect, or evidence that float64 chaos explains the bfloat16 answer changes.

Four blocking findings concern an unidentified comparison, mechanism wording, omitted constraints on the headline interpretation, and unsupported extensions. No raw-data submission is needed. I have not opened the named receipts or checked external sources. The responsible human remains the publication gate.

NOT READY FOR READER REVIEW

# Accuracy and claim findings

**B01 — Blocking: the 7B excess-disagreement statistic has an unidentified estimand.**  
**Affected:** Full report, arithmetic results. At the stated \(T=10^{0.2}\), soft changes 53/96 answers and greedy changes 10/96: their difference is **44.8 percentage points**, not 17.7. The latter could reflect averaging other conditions, but those control cells and weights are not supplied. In contrast, the 0.5B average of 32.3 points is identifiable from the stated counts.

**Minimum fix:** Give a compact temperature-by-arm table and define the averaging behind 17.7, its interval, and its test. Label the zero-accuracy forced-greedy baseline as a computational comparator, not a useful-answer baseline. Separately confirm that the workstation “native pipeline” rate of 74% uses parsed integers and is genuinely distinct from the corrected free-arm rate of 69%, rather than a surviving text-comparison result.

**Claim affected:** The size of the excess attributable to the soft pipeline.

**B02 — Blocking: intervention and dynamics language exceeds what the controls isolate.**  
**Affected:** Corrected paragraph and reset/replay interpretations.

- Shape changes can alter floating-point execution; absence of padding does not establish that reduction order is the *only* implementation difference.
- Reset overwrites the target’s cache and logits before each step. It prevents those differences from accumulating across reset boundaries; it does not eliminate the current step’s rounding or establish an intrinsic instability.
- Replaying reference feedback tokens fixes the feedback inputs, **not necessarily the layout-dependent cache or other downstream state**. “No rounding difference was fed back” is too broad.
- “A chaotic loop amplifies any other [difference] in the same way” is false without qualifications about perturbation direction, magnitude, trajectory, and finite-time regime. Periodicity likewise does not guarantee global contraction across basin boundaries.

**Minimum fix:** Describe the actual overwritten/replayed quantities. Say the controls support a contribution from accumulated feedback-state divergence, without uniquely identifying chaos or exponential amplification. Replace “the regime decides” with a bounded statement about the tested float64 trajectories and local perturbations. Retain “retrospective” for the Lyapunov schedule comparison.

**Claim affected:** The causal mechanism connecting layout, rounding, recurrence, and answers.

**B03 — Blocking: the headline needs the controls and qualifications that constrain its interpretation.**  
**Affected:** Corrected paragraph; “answers are fragile”; larger-budget claim.

The paragraph omits two particularly informative comparisons: ordinary greedy chain of thought changes **2.3%** of GSM8K answers at **89% accuracy**, while top-1 feedback changes **17.6%**, with no resolved difference from top-15 feedback. These constrain both practical relevance and soft-specificity.

The audit also discloses that sampling draws align by global step, not answer-token position. Different switch steps therefore alter answer-phase draws. That is a possible mediator of the pipeline’s total layout effect, but it prevents attributing all sampled-answer changes directly to changed answer distributions.

Budget status is measured after intervention. The capped/uncapped split is descriptive, not a randomized demonstration that truncation causes the flips. Moreover, the 4096-step rerun also introduces Cold Stop and, for the cited Qwen GSM8K cell, a different H100 variant.

**Minimum fix:** Include the cheap baselines and sampling caveat; describe the rerun as a modified exploratory configuration, not an isolated budget intervention. Replace “answers the model holds weakly” with the measured properties—low accuracy or budget attainment. Accuracy is not a confidence measurement.

**Claim affected:** Whether the effect is soft-specific, generalizes to successful completed reasoning, or persists because budget alone was increased.

**B04 — Blocking: several extensions cannot be assessed from the supplied definitions and summaries.**  
**Affected:** “The same holds” across larger models; fixed-shape control; final behavioral sentence.

The draft supplies no corresponding model-specific counts or operational details for “the same holds” in three larger models. The 448-cell control also exceeds the 384 classified cells, without identifying the remaining cells. The behavioral claim about information beyond first-token uncertainty and the steering estimate lacks its design, estimand, interval definition, and realized regime-transition summary.

**Minimum fix:** Supply a short claim-to-evidence table—not raw records—identifying denominators, thresholds, horizons, model-specific results, the other 64 cells, and the conditions underlying the injection-ratio range. For behavior, define the conditioning analysis and steering comparison, distinguish requested steering from realized regime changes, and identify the uncertainty interval. Otherwise remove or explicitly defer these claims.

**Claim affected:** Cross-model generalization, control coverage, and the bounded behavioral null.

# Statistical and scope findings

**S01 — Important: inferential labels and pooling need clarification.**  
**Affected:** GSM8K secondary tests and reasoning-model comparisons.

“Clear null” overstates the top-1 result: the interval \([-2.9,10.3]\) points does not establish equivalence. Use “no resolved difference.” Explain why the Llama accuracy interval excludes zero while the reported McNemar \(p=0.096\) does not reject; different procedures can disagree, but their estimands and units must be explicit.

Define the resampling unit and treatment of dependence from shared prompts and overlapping batches across layouts. Ordinary exact McNemar inference assumes independent problem-pairs; batch-clustered intervals do not automatically make those \(p\)-values cluster-valid. Identify the comparison family before describing the isolated \(p=0.029\) as significant.

The five-cell 27.1% pool mixes tasks and layout constructions. Label it a weighted summary of these five cells, not a general model susceptibility estimate. Accuracy ranges and ranges over kick conditions remain descriptive ranges, not uncertainty intervals.

**S02 — Important: the intervention-factor inventory is incomplete.**  
**Affected:** Methods and scope statements.

Add one compact table marking factors **matched, varied, fixed, or untested**, including:

- model/checkpoint, precision, GPU variant, and execution settings;
- prompt selection, companion identity, target row, exact lengths, actual batch sizes, fillers, and prior context;
- temperature, top-\(k\), feedback rule, cue, extractor, INVALID handling, and thinking/answer budgets;
- RNG indexing and switch timing;
- reset/replay quantities, timing, initialization, and decoder;
- perturbation size/direction/time and steering direction/strength/application schedule.

This matters because “batches of 8 and 16” includes short and identical batches, and “larger budget” includes other changes. Bind conclusions to the actual levels; do not imply invariance across untested temperatures, positions, companions, budgets, or hardware.

The frozen \(>10\%\) and \(>5\%\) thresholds are study predictions, not evidence of a community prior or surprise. Keep that distinction.

# Reference audit

**R01 — Important: bibliography and attribution require completion and source checks.**

No bibliography is supplied. The following inventories every identifiable external source in the draft; unseen entries cannot be audited.

| Identifiable source | Exact claim it is being asked to warrant | Minimum action |
|---|---|---|
| **He et al. 2025 / Thinking Machines**, `he2025nondeterminism` | “GPU kernels choose different reduction orders for different matrix shapes”; “Batch-invariant kernels … remove this one source of difference.” | Required source check: identify the work, authors, version/date, publication type, and scope of its invariance guarantee. It cannot warrant elimination of all nondeterminism. |
| **Yuan et al. 2025**, `yuan2025nondeterminism` | The attribution of batch-sensitive results to bfloat16 reasoning reproducibility, including the corrected paragraph’s opening sentence. | Required source check: exact identity and whether it demonstrates the claimed layout mechanism. Retain for its distinct reasoning/precision evidence; remove if it merely duplicates He without supporting a separate claim. |
| **Goldberg 1991**, `goldberg1991floating` | “Non-associative floating-point sums.” | Appropriate, identifiable mathematical role. It does not establish contraction of these periodic loops or amplification rates in these models. Complete the bibliographic entry. |
| **Soft Thinking paper**, unnamed | “With its published settings”: top 15, \(T=0.6\), 64 steps; use of reasoning models; the stated Cold Stop rule. | Add an explicit citation. Required source check: version, model-specific settings, stopping rule, and whether these are defaults or particular experimental settings. Clearly identify deviations. |

There is no identifiable orphan among these four, although He/Yuan may overlap. Do not add references merely for prestige. Add dataset/model and steering-method provenance where those sources are needed to identify the experimental materials. The strongest adjacent counterevidence is already internal—the discrete and ordinary-CoT controls—and should not be displaced by a longer bibliography.

For any preprint or technical report, identify its provisional status rather than treating it as authority. Publication status does not validate the present rates.

# Figure-to-text consistency

**F01 — Important: figure review is unavailable, not passed.**

Figure `repro`, its caption, and alt text were not supplied. No figure-to-text verification is possible.

Before release, check that the figure distinguishes:

- 1500-step distance separation from token disagreement during the last 500 steps;
- chaotic and periodic denominators from the 448-cell control;
- retrospective Lyapunov agreement from prospective prediction;
- rounding-scale layout differences from injected displacements;
- periodic-cell separation at larger kicks from confirmed attractor or basin changes.

A compact figure/caption/alt-text packet suffices; no underlying arrays are needed.

# What should remain unchanged

- The corrected integer-based reset counts, correction date, and preservation of the original summaries.
- The explicit low-accuracy, INVALID, tokenizer-amendment, filler, and budget disclosures.
- Zero observed changes at the stated high-temperature arithmetic settings—without extrapolating to other settings.
- The finding that sampled discrete thinking also changes answers.
- The narrow statements that nonsignificance does not establish equality and that 128 problems do not establish unchanged accuracy.
- The periodic-displacement control’s failure to meet its design limit. Keep basin crossing explicitly hypothetical.
- The distinction between frozen tests and exploratory reruns.

The supplied fractions support the rounded headline percentages; nothing here justifies discarding those bounded descriptive findings.

# Required revisions

**Before posting**

1. Resolve the arithmetic contrast and potentially stale native-pipeline percentage (**B01**).
2. Correct reset/replay and dynamical mechanism descriptions (**B02**).
3. Put the cheap baselines, draw-alignment qualification, and modified-rerun scope beside the answer claims (**B03**).
4. Supply compact support for the cross-model and behavioral extensions, or remove them (**B04**).
5. Complete the inference/factor descriptions and source checks, and obtain human figure sign-off (**S01–S02, R01, F01**).

A defensible one-sentence reasoning-model summary is:

> Under the tested 1536-step thinking and 256-token answer budgets, changing layout changed 14.5–28.9% of soft-thinking answers, almost entirely on problems reaching a budget; a modified exploratory 4096-step rerun still showed changes, including 8.6% soft and 17.2% sampled on 128 R1-Qwen GSM8K problems, without establishing the rate for successfully completed reasoning.

**Cheapest discriminating checks if stronger mechanism claims are retained**

| Candidate explanation | Compact check and interpretation |
|---|---|
| Extraction/INVALID inflation | Tabulate number–number, number–INVALID, and completed-answer changes by arm and budget status. Collapse under valid completed parsing implicates readout dependence; persistent number–number changes exclude a purely textual explanation. The existing 29/96 arithmetic cases already exclude that explanation there. |
| Answer-phase RNG remapping | On a small matched subset, index answer draws by answer-token position independently of thinking. Reduced flips identify the remapping contribution; residual flips show it is not sufficient. |
| Budget-forced readout | Compare budgets with GPU variant and stopping rule fixed; report completed-answer changes separately. Concentration at forced termination supports budget dependence; completed-answer changes establish a broader effect. |
| Top-\(k\) discontinuities | Use the existing full-vocabulary result: 22.3% versus 21.1% shows top-15 truncation is not necessary in that Qwen pipeline. Do not transfer this conclusion to untested reasoning-model cells. |

These diagnose pathways, not necessarily “fake” effects: RNG remapping and forced readout can be genuine consequences of layout within the implemented pipeline. Additional experiments can wait for the journal version **if the current claims are narrowed accordingly**.

# Publication checklist

- [ ] Every percentage has an identifiable denominator and readout.
- [ ] Every comparative statistic names its arms, conditions, and weights.
- [ ] Reset, replay, perturbation, and realized steering are distinguished.
- [ ] Completed answers are separated from budget-capped readouts.
- [ ] Controls accompany the claims they constrain.
- [ ] Nulls, descriptive ranges, and uncertainty intervals are labeled correctly.
- [ ] Bibliography and required source checks are complete.
- [ ] Figures, captions, and alt text receive human sign-off.
- [ ] Responsible author approves the bounded claims and disclosed amendments.
