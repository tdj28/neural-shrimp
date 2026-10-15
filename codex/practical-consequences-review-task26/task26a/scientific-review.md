# Task26(a): final REPRO-T saved-evidence scientific review

2026-10-15. The main finite operational result is confirmed: across the frozen 96 indexed held-out problems, the mean excess canonical-answer disagreement at the two primary bfloat16 temperatures is **32.291666666666664 percentage points**. The independently specified paired-bootstrap interval excludes zero. This is independent, unblinded arithmetic on retained original records, not another model run or proof of a chaos mechanism. The numerical result and original execution/source/output closure are complete and accepted; no lifecycle condition remains pending.

**Evidence and scope.** The completed [scientific report](/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26a/run-001/report.json) has SHA256 `7dbdaad85343eece750927ec59cf699f8b95a38b7785690a31bc762862e50726`; its output manifest is `701c37889db7578822a5b1378375161f8aea8602f4ba8e77cafbe298ceab26f1`. The [final authority](/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26a/run-001-authority/report.json) has SHA256 `7cb369ef209daec091133741db095c17f59f8c96c4d1b90eeaa2b0dac1124f1e`, with manifest `0316bbf313a2a0cfe57c6c6067ab7cae760d45a6db780a8075531e78fffbed8f`. It authenticates 2 output files, 9 supervision entries and 4 captured Git source blobs; session 99356 was reaped with exit 0, and the original supervisor 2887643/start 200065998 and worker 2887655/start 200066110, their groups and sessions are absent. Outer stderr is empty. These original execution and closure observations belong to **Inference**, relayed to and accepted by Root; this note is a separate scientific reading of the saved results. Own execution source is `883ee479329618992775a0ea30fe498fda87b5c3`; original evaluator source commit is `a9c411ff1bf992a20f595039000c55be15c6d12c`. The accepted local archive/retrieval/source basis does not reconstruct historical remote model bytes.

The admitted coverage is **832 pipeline records, 54 mechanism records and 208 progress records**, including 8 development and 96 held-out indices. All original saved-field checks pass with zero differences. This review reads the per-row dispositions and all eight endpoint, development/held-out condition, and mechanism summaries; it does not rerun parsing, statistics, resampling or the model. The full original report retains every distance, margin, row and resampling draw. The companion findings JSON transcribes selected exact fields and all 56 producer comparison leaves. Of those comparisons, 34 are nonzero and none undefined; these are primarily printed rounding or explicit convention differences, not 34 scientific reproduction failures. The producer summaries were exposed before the independent replay.

**Endpoint results.** The primary unit is the frozen indexed problem. Its paired contrast is `(d_soft 0 + d_soft 0.2)/2 - d_forced_greedy`; the two temperatures are averaged within each problem. Each endpoint uses 10,000 problem bootstrap draws and 10,000 paired sign flips, PCG64(0) reset independently for each calculation, linear quantiles, and inclusive absolute half-unit integer sums. The raw sign-flip fraction and plus-one `(b+1)/10001` are both retained. There were no undefined draws. The original inline producer RNG code was unavailable, so this reproduces the reported rounded result under an explicit independent convention, rather than asserting bit-for-bit recovery of absent code. Sampling and sign-symmetry assumptions remain necessary; this is not random treatment assignment.

| Endpoint (all n=96) | Difference, pp | Saved 95% interval, pp | Outward 1dp interval | Exceedances /10000 | Plus-one p |
| --- | --- | --- | --- | --- | --- |
| bf16 primary mean | 32.291666666666664 | [26.562500000000004, 38.020833333333336] | [26.5, 38.1] | 0 | 9.999000099990002e-05 |
| bf16 soft log10(T)=0.0 minus forced greedy | 56.25 | [46.875, 65.625] | [46.8, 65.7] | 0 | 9.999000099990002e-05 |
| bf16 soft log10(T)=0.2 minus forced greedy | 8.333333333333334 | [3.125, 14.583333333333334] | [3.1, 14.6] | 71 | 0.0071992800719928 |
| bf16 soft log10(T)=0.4 minus forced greedy | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 10000 | 1.0 |
| float64 primary mean | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 10000 | 1.0 |
| float64 soft log10(T)=0.0 minus forced greedy | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 10000 | 1.0 |
| float64 soft log10(T)=0.2 minus forced greedy | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 10000 | 1.0 |
| float64 soft log10(T)=0.4 minus forced greedy | 0.0 | [0.0, 0.0] | [0.0, 0.0] | 10000 | 1.0 |

Thus the primary raw Monte Carlo fraction is 0/10000, and the plus-one value is 1/10001, approximately 10⁻⁴. These are distinct conventions. The printed primary 32.3 pp and nearest-rounded interval [26.6, 38.0] agree approximately; when the endpoints are intended as enclosing decimal bounds, use the exact saved interval or outward [26.5, 38.1]. Ordinary approximate rounding is not a scientific discrepancy. Zero intervals for the other displayed conditions describe this empirical resampling distribution, not population invariance.

**All held-out conditions.** Each row contains 96 indices. “Canonical” is disagreement in the parsed first integer; “text” is any saved answer-string disagreement. Correct/INVALID counts are single-layout versus batch4. Two INVALID answers count as agreement under the frozen definition.

| Condition | Canonical /96 | Text /96 | Correct single / batch4 | INVALID single / batch4 | Pair INVALID |
| --- | --- | --- | --- | --- | --- |
| bf16 soft log10(T)=0.0 | 54 | 68 | 23 / 22 | 1 / 0 | 0 |
| bf16 soft log10(T)=0.2 | 8 | 26 | 48 / 51 | 0 / 0 | 0 |
| bf16 soft log10(T)=0.4 | 0 | 5 | 90 / 90 | 0 / 0 | 0 |
| bf16 forced greedy | 0 | 0 | 0 / 0 | 8 / 8 | 8 |
| float64 soft log10(T)=0.0 | 0 | 0 | 30 / 30 | 0 / 0 | 0 |
| float64 soft log10(T)=0.2 | 0 | 0 | 49 / 49 | 0 / 0 | 0 |
| float64 soft log10(T)=0.4 | 0 | 0 | 92 / 92 | 0 / 0 | 0 |
| float64 forced greedy | 0 | 0 | 0 / 0 | 0 / 0 | 0 |

All 96 saved rerun texts agree with the single-layout text in every condition; the corresponding original trajectory-identical flags are also true. These flags inherit the reviewed producer implementation: full saved state trajectories and complete answer-token IDs are unavailable for an independent equality reconstruction. The bf16 soft 0 invalid single answer is index 89. The bf16 forced-greedy pair-INVALID indices are 9, 13, 21, 27, 29, 49, 66, 73. Raw-text differences exceed canonical differences by 14, 18 and 5 at bf16 log10(T)=0,.2,.4 respectively; zero canonical disagreement at .4 must not become a claim of identical answer text.

The comparator forces 32 greedy tokens, appends the fixed answer cue, then generates 8 answer tokens without EOS stopping. Its held-out accuracy is 0/96 in both layouts and both precisions. It is a specified reproducibility comparator, not ordinary early-stopping greedy accuracy or evidence of a useful accuracy baseline. Gold sums are used after decoding for scoring; they are not supplied as answers in the input prompts.

**Development conditions, excluded from held-out endpoints.** Each row contains 8 indices. Rerun text and original rerun-identical flags agree for all 8 in each condition; the bf16 greedy pair-INVALID is index 4.

| Condition | Canonical /8 | Text /8 | Correct single / batch4 | Pair INVALID |
| --- | --- | --- | --- | --- |
| bf16 soft log10(T)=0.0 | 6 | 7 | 3 / 2 | 0 |
| bf16 soft log10(T)=0.2 | 0 | 0 | 4 / 4 | 0 |
| bf16 soft log10(T)=0.4 | 0 | 0 | 8 / 8 | 0 |
| bf16 forced greedy | 0 | 0 | 1 / 1 | 1 |
| float64 soft log10(T)=0.0 | 0 | 0 | 1 / 1 | 0 |
| float64 soft log10(T)=0.2 | 0 | 0 | 4 / 4 | 0 |
| float64 soft log10(T)=0.4 | 0 | 0 | 8 / 8 | 0 |
| float64 forced greedy | 0 | 0 | 1 / 1 | 0 |

**Distance and mechanism interpretation.** Pipeline distances are relative norms of expected embeddings, normalized by the single-layout embedding. In bf16 runs these embeddings are calculated in float32 before conversion to model input dtype; they are not independently reconstructed post-cast model-state distances. Prefix computations and batch-layout effects remain part of the tested implementation. The first/last saved clock is step 1/step 32; the displayed finite growth is `(log(d32)-log(d1))/31`. It is not an asymptotic Lyapunov exponent.

| Held-out soft condition | d1 median | d32 median | d32 actual range | Finite-growth median |
| --- | --- | --- | --- | --- |
| bf16 soft log10(T)=0.0 | 6.134470740212301e-07 | 0.2237169631880111 | [3.1325647231407897e-07, 2.687289246078446] | 0.41272287731500235 |
| bf16 soft log10(T)=0.2 | 2.4876866452628344e-06 | 0.012577752819178948 | [5.598423225373406e-07, 0.05999844936122562] | 0.276053416428921 |
| bf16 soft log10(T)=0.4 | 7.909331308336545e-07 | 0.003980488623475259 | [7.556340164968868e-07, 0.015332890221945618] | 0.2764170711623818 |
| float64 soft log10(T)=0.0 | 4.746268131185358e-15 | 1.3061946196516364e-13 | [4.650528589881559e-15, 1.7276517581926647e-09] | 0.10638747277175581 |
| float64 soft log10(T)=0.2 | 6.3658721522963945e-15 | 4.9843273299608914e-15 | [1.939463807073841e-15, 1.728995204985393e-14] | -0.007968189605729516 |
| float64 soft log10(T)=0.4 | 3.48983865817866e-15 | 1.749919140535584e-15 | [9.441106397519014e-16, 7.22087030701354e-15] | -0.0179208666728594 |

The deliberate-kick experiment is separate from these layout distances. It contains six fixed float64 problems and three deltas (10⁻¹², 10⁻⁸, 10⁻⁴) per temperature, hence 18 dependent perturbations per temperature rather than 18 independent problems. log10(T)=0 and .2 are the frozen primary temperatures; .4 is the retained secondary extension. All 54 growth values are defined and all 54 canonical answers are unchanged.

| log10(T) | Perturbation rows | Growth median | Actual growth range | Canonical changes | Text changes |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 18 | 0.033011474309378244 | [-0.12687039525407393, 0.24040185753056742] | 0 | 1 |
| 0.2 | 18 | -0.14121655361463398 | [-0.14849181446609633, -0.11937509982475608] | 0 | 0 |
| 0.4 | 18 | -0.10230577703895946 | [-0.12089785263635447, -0.08584962827750224] | 0 | 0 |

| log10(T) | Delta (n=6 each) | Growth median | Actual growth range | Undefined |
| --- | --- | --- | --- | --- |
| 0.0 | 1e-12 | 0.03629844626325246 | [-0.11440635395593363, 0.24040185753056742] | 0 |
| 0.0 | 1e-08 | 0.03174112320288842 | [-0.12686988458217982, 0.20781410845780232] | 0 |
| 0.0 | 0.0001 | 0.029962706087141938 | [-0.12687039525407393, 0.1708289389539753] | 0 |
| 0.2 | 1e-12 | -0.13738235080712735 | [-0.14849181446609633, -0.12212605750645587] | 0 |
| 0.2 | 1e-08 | -0.14121722852912766 | [-0.14482311351782742, -0.1193768337164635] | 0 |
| 0.2 | 0.0001 | -0.14121366253749354 | [-0.1448260126819668, -0.11937509982475608] | 0 |
| 0.4 | 1e-12 | -0.10238902274490938 | [-0.12089785263635447, -0.08829850698939475] | 0 |
| 0.4 | 1e-08 | -0.10231041623143439 | [-0.1206362033556405, -0.08584962827750224] | 0 |
| 0.4 | 0.0001 | -0.1023051378701216 | [-0.12062972629263378, -0.08585718558168576] | 0 |

At T=1, “growth only 0.03” would wrongly turn the median 0.033011474309378244 into a universal bound: the actual range is [-0.12687039525407393, 0.24040185753056742], safely enclosed by [-0.127, 0.241]. At log10(T)=.2 and .4 all retained first-to-last rates are negative. The one mechanism raw-text change is input line 862, problem 3, T=1, delta 10⁻⁴: both parsed answers are 63, but the trailing fragment changes from “Human: What” to “Human: Write.” Canonical-answer invariance is confirmed; complete text or token invariance is not. The finite check is consistent with low-precision layout errors propagating through continuous feedback in this setup. It does not prove or rule out chaos, exclude nonlinear amplification elsewhere, or identify the mechanism of the distinct long fixed-window autonomous loop. No source-order effect established for REPRO1 is transferred to REPRO-T: the latter runs bf16 before float64.

**Cohort and inference limits.** The 96 held-out indices are 95 distinct ordered additions and 94 distinct unordered additions. The ordered duplicate (77,96) occurs at indices 23/67; (28,93) is reversed across 14/24. Development index 4 overlaps held-out 50 under operand reversal, and distractor 0 exactly overlaps held-out 22. An earlier four-step local smoke test included one held-out instance whose index was not recorded, with unresolved chronology relative to the frozen push. Preserve the fixed index-based analysis and disclose this exposure; do not post-select or reweight a cleaner cohort. The 8 development instances and 3 distractors are separate from the 96 endpoints. The records support a finite reproducibility comparison on this source, device/protocol and cohort. They do not establish fresh unique-task validation, population-wide precision invariance, attribution to an autonomous chaotic attractor, or mathematical certification.

**Disposition and wording.** Confirm the operational parsed-answer result, the paired contrast under the disclosed resampling convention, all coverage and saved-field consistency, and the mechanism canonical-answer invariance. Correct claims that conflate raw text with parsed answers, treat the growth median as a bound, describe the baseline as ordinary greedy decoding, or portray the 96 indices as distinct unseen tasks. Exact historical full trajectories/token equality, remote model-byte loading and causal chaos attribution cannot be independently established from these retained files. Root’s [exact-line editorial proposals](/tmp/task26e-editorial-replacements.md), anchored to paper source 666e9b72, already locate the corresponding sentences. A compact replacement is: “In the fixed 32-step pipeline, bf16 canonical answers differed across batch layouts on 54/96 and 8/96 indexed problems at the two primary temperatures, versus 0/96 for the forced-length greedy comparator. The mean excess was 32.3 percentage points; the independently specified paired-bootstrap interval is enclosed by [26.5, 38.1], with plus-one sign-flip p=1/10001. This finite reproducibility result does not establish a chaos mechanism.”

**Resource attribution.** The arithmetic/report-construction timer is 0.11576672806404531 s, sampled before final input rehash, JSON serialization and writing. The process high-water RSS observed then is 60,860 KiB, not a separately measured final worker peak. Inference’s original supervisor window is 0.551870169 s. The verified hard address-space limit was 1 GiB; it is a limit, not an observed memory use. No new model, resampling, input audit or numerical run was performed to write this review. All numerical evidence, failed/undefined dispositions and original exposed comparisons remain in the immutable report.
