# NEED_TO_AUDIT items 3–5: closed saved-evidence review, 16 October 2026

The independent saved-data calculation completed. All **367 comparable publication leaves agree exactly**: 224 damping, 59 MoE, and 84 chaos-text leaves, with no missing or different comparison leaf. This confirms their literal arithmetic, not every interpretation of those numbers. The main required corrections are the depth endpoint/partition, the scope of the high-entropy and knockout claims, and the text-distance bounds. The routing and arbitrary-precision statements also need finite-evidence qualification.

This review reads the completed own outputs only. It does not rerun a reducer, model, bootstrap, control, producer script, or original target audit. All 432 damping rows, 90 knockout configurations, 6,658 scan-cell dispositions, 980 text-step probability/distance dispositions, 60 thread-comparison rows, 24 vocabulary records, and 367 comparison leaves were traversed in saved-field projections; the scientific summaries and adverse cases below were read directly. The companion coverage retains every damping identity residual and knockout array, all scan rows, every text-step gate, vocabulary records, and comparisons. This is a saved-evidence review, not a second independently executed numerical experiment.

## Authority and completed scope

The immutable output directory is `/data2/neural-shrimp-data/codex/need-to-audit-20261016/run-001/damping`. Its report is SHA256 `14bb480a4c8d7a3bd7e76ef81fe1dc35dfc66b26835cc07d8cb0a8b00f88a571`; the sibling `damping-supervision/receipt.json` is `2d379f1ccc8d7d70e1c7c470b866f6e6699f79f5ecb08ac58c8f244a6c30c924`; the parent `original-closure.json` is `b2d9e5a3051fe5d7c042dac18630031a27e8e67060efc827c07050dda66e19e9`. All six output byte counts and hashes match the receipt. The closure binds source commit `5be03229167e9c1a27009b6e1b588c297df027cb`, the reviewed source/card bytes, tool session 17422 reaped with exit 0, and the original supervisor/worker identities and groups absent. **Inference directly observed execution, reaping, and closure; this reviewer read those receipts.** The closure also records other lanes; this note does not accept their scientific outcomes.

The actual card is `f6e906c4c57331d9cf15bc8075104fb586ac3330c26ee36c313a966b7aec96e6`; driver `18f8606dbdd9344180aa3a4324d497eff3fb27d0988fb2cd0a0d1c01ff8604bc`, reducer `54b3c146babba280be0c7f3f7d5dede987a100ccffb183bccbb369771418a9d8`, controls `a4f612f956ab17f2d48cd851c0bb46d80546a1137d91758b1a7fc95b8a5244c2`. All nine integrated manufactured controls passed. External supervisor elapsed time was 1.149433068 s; 0.816951823 s and 49,436 KiB are the worker's observations before final rehash/report, not a separately measured whole-worker peak. Six outputs total 4,398,766 B.

The original invalid keyword in prepared driver `ad3e60a0` was caught by AST parsing before any control or target execution. Its source/card/release and the one-line syntax-only correction are preserved. This was a preparation refusal, not a scientific retry. The original DAMP/MOE failed histories, local-source attestation limits, and PT128 receipt remain bound by the input packet; no historical loaded-weight or process authority is invented.

All manuscript line references below refer to the frozen source at commit `2b2172946cb303b2d4462084df75adee027d11e0` in `/tmp/need-to-audit-20261016-source-001`, manifest `3fcbde19749f1cfe53e5309283e0b7a551dff616354e85502ff094abf9c82963`, not a later edited paper.

## Item 3: damping

The 14-model primary medians over **0.3 ≤ H < 6** reproduce the paper. Exact values and counts are:

| Model | Median log network factor | Temperatures |
|---|---:|---:|
| Qwen2.5-0.5B | −0.12103078333031841 | 8 |
| Qwen2.5-1.5B | −0.5671858176372644 | 11 |
| Qwen2.5-3B | −1.800198349391478 | 6 |
| Qwen2.5-7B | −0.8490603576953695 | 11 |
| Qwen2.5-14B | −1.45809468344966 | 7 |
| Qwen2.5-32B | −1.7276207162472041 | 14 |
| Qwen2.5-72B | −1.9931924364068705 | 9 |
| Llama3.1-8B | −0.1375896600862497 | 9 |
| Llama3.1-70B | −1.5038066410469892 | 10 |
| Llama3.3-70B | −1.2993761764705871 | 11 |
| OLMo2-1B | +0.10807140126766843 | 9 |
| OLMo2-7B | −0.5218893206237994 | 5 |
| OLMo2-13B | −0.788641666513205 | 10 |
| OLMo2-32B | −0.36464227155575357 | 9 |

All 18 × 24 classifications match the source labels and all recorded array elements are finite. The scalar network decomposition has signed residuals between −1.265688942542198e−13 and +1.1857181902996672e−13; the finite-twin identity residuals lie between −9.059419880941277e−14 and +2.3092638912203256e−14. None is silently replaced by zero. These are saved scalar identities, not a new derivative-precision or model verification.

**Required: correct the depth endpoint and partition.** `tools/damping_locate.py:57,250–251` saves block inputs, indexed 0 through L−1. `tools/plot_damping.py:73–85` uses the last such sample as `final_relative_change`. Thus `paper/llm/main.tex:2080–2082,2106–2108` and blog `3699–3703,3725–3727` mislabel a before-last-block value as after the last block. The actual final residual change is available separately from saved `mean_stack − mean_dilution`:

| Quoted model | Before final block, currently plotted | True after-final-block residual change |
|---|---:|---:|
| Qwen2.5-0.5B | −0.32411694968806337 | **+0.05420333414327061** |
| Llama3.1-8B | −0.5131136341508618 | −0.4016404024924416 |
| Qwen2.5-72B | −2.510810291874944 | −2.1062711374458605 |
| Llama3.1-70B | −2.081338141530522 | −2.0773543617954524 |

All 18 pairs remain in the coverage. Either relabel the existing profile and endpoint as ending before the last block, or extend the figure using the true final scalar and update the quoted numbers. The 0.5B sign changes, so this is material. Also, the stated “two parts” are not an exhaustive telescoping partition: `(A1−N1) + [(A_L−N_L)−(A2−N2)]` omits `(A2−N2)−(A1−N1)`. The implementation additionally uses L−1. Relabel them as two selected contributions, or include the missing second-block increment and the true final endpoint.

**Required: scope “only model” to Qwen2.5.** Main `2076–2077`, blog `3693–3695`: all three Llama models also have negative factors at every saved H > 10 point, with medians −0.9831536734198911, −1.3386842733467128, and −0.6561633089114903. Use “Of the tested Qwen2.5 models, only 72B remains negative throughout the saved H > 10 region,” and explicitly call −1.1174026110250526 and the other quoted numbers medians. Qwen7B's positive median does not imply every individual high-entropy factor is positive.

**The 0.44 knockout bound is correct for the stated selected rows, but the selection must be explicit.** All 90 knockout configurations qualify; the 14 primary nearest-to-one selections among the **five stored knockout temperatures per model** have entropies 0.12392092334945488 to 1.6092577781473463, not 14 measurements at H = 1. The largest factor rise on removal is 0.4394282843645603 (Qwen14B, H = 0.2780351171132282). This is a valid upper bound rounded outward to 0.44. It is not an all-temperature bound: a different Qwen32B knockout at H ≈ 0.342 has a 0.60428 rise, and a high-entropy Qwen7B knockout has a 1.29116 rise. Main `2085–2097`, blog `3705–3719` should say “At the stored knockout temperature closest to one nat for each model (H = 0.124–1.610), the largest 20-step mean rise from removing one sublayer was below 0.44.” The removal sign is free minus knockout; positive deltas are losses on removal. First-attention losses of 1.665815268 to 3.743778554 in Qwen1.5B–32B and the first-MLP loss 2.402783376 in 0.5B reproduce the claims as approximate values. “No component does the damping” is too general: these measurements constrain individual removals at selected states/directions and finite averages, not every component contribution or interaction.

**Keep the OLMo2 exclusion.** All 96 OLMo2 sublayer-share/gain/parallel rows are explicitly unqualified because post-sublayer normalization breaks the assumed raw-additive accounting. Maximum closure errors are 0.8794994072, 3.4142699443, 8.1877287442, and 151.6302933599. All other models' maxima are below 3.75e−7. OLMo2's network split, residual-state profiles and literal knockout measurements are distinct fields; they must not be used to rehabilitate the invalid sublayer decomposition. All knockout plain-twin diagnostics are retained, including the Qwen32B magnitude 5.098037592132787e−7; no new blanket numerical cutoff was imposed.

## Item 4: mixtures of experts

All 6,658 scan cells have finite exponents and valid period values. Literal `period == 0 and exponent > .001` counts equal finite-qualified counts. The source keeps no-period but nonpositive/low-exponent cells separate:

| Model / map | Count / cells | Maximum saved exponent | No period, exponent ≤ .001 |
|---|---:|---:|---:|
| Qwen3.5-4B, alpha0 / philosophy / hostility | 19/128; 153/1024; 323/1024 | .79126045; .84059850; .89070046 | 0; 5; 22 |
| Qwen3.5-9B, alpha0 / philosophy / hostility | 10/128; 73/1024; 161/1024 | .48284027; .67696131; .74570525 | 1; 9; 14 |
| Qwen3.5-35B-A3B, alpha0 / philosophy / hostility | 0/128; 0/1024; 52/1024 | −.26526369; −.11187397; .21607719 | 15; 184; 493 |
| OLMoE-1B-7B, alpha0 wide | 1/130 | .02259021897597471 | 7 |

Qwen3.5-4B hostility flat index 936 has period 5 despite exponent +0.0207654094596929; it is retained and excluded from the no-period count. The 52 MoE hostility cells and one OLMoE cell remain routing-unqualified irregular candidates, as the paper already labels them; unsteered routing diagnostics do not qualify steered maps. The finite grid and differing 300/300 map versus 1000/2000 alpha0 horizons do not establish absence of chaos outside their sampled scope.

The four extra 0.3–6-nat medians are Qwen3.5-4B +0.053560758932960896, 9B −1.4138375949500133, 35B-A3B −1.9019830869305911, and OLMoE −0.36189684832670455. At the three Qwen3.5 knockout rows with H in [2,4], first-attention losses are 1.2674519504485133, 1.3333124312576077, .7646662092332545; first MLP/expert losses .91749670797191, .7716404346808551, .24494235991027277. The first attention component is the largest saved mean amplifier at each of all five knockout temperatures for these three models.

All 48 unsteered routing temperatures have exactly zero stored switch counts and exactly zero frozen-minus-free mean network-factor differences. OLMoE's per-temperature median margin range is [.00011200551800918945,.01747425359703489], its minimum-margin range [3.381035749150385e−7,.006172921499050549]. Qwen35B-A3B's corresponding ranges are [.00012359231970027196,.004015889397064765] and [3.596459743349101e−6,.001343675822581325]. The stated median bounds 1.1e−4 to 1.75e−2 are conservative; “minimum approximately 3.38e−7” is preferable to treating 3.4e−7 as a strict lower bound.

**Qualify main `2117–2123` and blog `3762–3771`.** Use “No routing difference was observed between the measured float64 base/twin pairs, and freezing routing left their measured network factors unchanged.” This supports a smooth-branch interpretation along these finite trajectories; it does not establish one smooth neighborhood around the whole orbit. An absolute router-logit gap cannot be quantitatively compared directly to bfloat16's relative unit roundoff without the router-logit scale/error. Low-precision switching is a plausible possibility, not demonstrated here. The trained-checkpoint comparisons support these observed family members; they do not isolate total parameter count from architecture/training.

## Item 5: chaos-zone text

All eight stored cell labels agree with the source predicate, with positive exponents at both offsets and no detected period. All 980 step probability-order/finite-distance gates qualify, and all base/twin concatenations match. The earlier 60-step weak-chaos recording has no split and remains included; the fixed seven-row figure uses its later 500-step recording.

| Selected cell | First split | Rounded-log distance transformed | Top probability | Lead over runner-up | Twin matches runner-up text |
|---|---:|---:|---:|---:|---|
| 0.5B philosophy, .0652 / −.156 | 151 | .10232929922807542 | .19873696898992943 | .03136449492227891 | yes |
| 0.5B philosophy, .07484 / −.1226 | 44 | .9120108393559098 | .13927312545418047 | .0041528246982011074 | no |
| 0.5B philosophy, .07277 / −.041 | 51 | 1.7378008287493754 | .0637868478241812 | .019671553222852182 | no |
| 0.5B hostility, .08946 / −.05566 | 45 | .033884415613920256 | .3026260754153675 | .004664313794905783 | yes |
| 1.5B philosophy, .07376 / .0165 | 22 | .4168693834703354 | .13606642753879478 | .077913717210715 | yes |
| 1.5B philosophy, .10553 / .0322 | 10 | .03715352290971726 | .09132589946544893 | .0017879088108376795 | yes |
| 1.5B unsteered, 0 / .657 | 43 | .012882495516931342 | .00037560996249054397 | .0000007002047314390129 | yes |

The maximum probability .3026260754 < .31 and maximum lead .07791371721 < .08 confirm the bound-up wording; the five-of-seven decoded runner-up matches are correct. The original source stores only the base top-three token IDs/probabilities and the twin's decoded text, so these text matches do not independently reconstruct the original ID-based split. All seven stored split indices equal the first saved text difference. The 0–16-step delay after the displayed 1% distance crossing is exact for these rounded-log records, not an independently recovered unrounded crossing time.

**Required bound correction:** main `622,725`, Chaos `479`, blog `477,536` say 1.3%–170%. Transformed rounded-log extrema are approximately **1.288249552%–173.780082875%**. Because `tools/chaos_texts.py:94` rounds log10 distance to two decimals, the conservative ±.005-log envelope is **1.273503081%–175.792361396%**. Suggested wording: “The displayed split distances range approximately from 1.29% to 173.78%; allowing for stored-log rounding gives a conservative 1.27%–176% envelope.” “Between 1.3% and 170%” is not a valid bound. These are finite saved-distance statements, not interval-certified native trajectories.

**Required finite-sensitivity wording:** main `617–618`, blog `469,4193` should replace dependence “to arbitrary precision” with “the recorded texts are sensitive to initial changes of relative size 1e−10.” This finite experiment does not establish arbitrary precision or infinite-time chaos. The unsteered cell's maximum recorded leading probability is .0009246724516719298, so its <.001 claim is supported.

The six-thread versus four-thread comparison has 60 common steps: the first 27 leading token IDs and decoded texts agree; the first difference is step 28 (token 3405 versus 13885). Subsequent token matches recur, while the accumulated strings have diverged. Say “first differ at step 28” at main `648`, Chaos `481–482`, blog `544–545`; do not imply every later token differs. Probabilities already differ from step 1. Source/receipt association supports the stated thread setups, but the saved records do not reconstruct the separate 1000-step transient states or prove thread count was the sole historical runtime difference.

The 70B vocabulary packet retains all 24 rows: 12 positive-at-three-offset/no-period cases, six periodic cases, and six irregular negative-exponent cases; the tangent signs agree with these dispositions. Row 0 matches the paper's eight strings and IDs exactly, with exponents [.16733214345,.16785418248,.16462565001] and tangent .16249679588. The **17 distinct tokens in 300 steps is an inherited writer summary**, consistent with the eight saved unique IDs but not independently recounted: PT128 used `trace=false, save_sequences=false`, and the full leading-token sequence is unavailable. Main `650–651` and blog `548–550` can retain it as “the producer record reports 17 distinct leading tokens”; do not label the full vocabulary count independently reproduced. No tokenizer or native trajectory was replayed.

## Coverage limits and disposition

Items 3–5 are complete as the authorized finite saved-data audit. The literal arithmetic is accepted, subject to the specific wording/indexing corrections above. The original published values and all failures remain immutable. The counterpart Chaos version has the identified text-bound correction; its absence of the full damping/MoE sections is not a separate verification gap. Growth-slope prose not among the saved publication comparison fields was not refitted in this run. No new causal, asymptotic, native-precision, global-size, or routing-boundary theorem follows from these outputs.

Companion `/tmp/need-to-audit-20261016-damping/saved-review-coverage.json` SHA256 `5d6dd001bb4af2c074dbc77603a629a1a63597d8e9abe845e5a76c38f36ad8b1` (1,496,866 B) preserves all projected dispositions and original output pins. It can remain outside Git; the compact report and its hash-bound pointer are sufficient for delivery. The full original six outputs remain authoritative.
