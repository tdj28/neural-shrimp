# Task25 — independent saved-data size, gain, prefix and plateau review

Final accepted report, 15 October 2026. Root accepted this synthesis, both complete scientific notes and the original operational closure. Private evidence and delivery references are recorded separately. No further computation is proposed.

This is a report-only reconstruction of six assigned saved-data components plus a wording review of the frozen size section. It is not a new model experiment, blinded analysis or fresh confirmation. Prior Task27, manuscript, research-log and producer-summary exposure is disclosed. The original all-source gain result remains intact; the separately named wide and narrow reference views were added by a dated, source-bound amendment before the sole reconstruction. No thresholds, underlying medians, family labels or rows were changed to obtain agreement.

The reviewed manuscript is commit `0267f91e44a66ec352cbb4b09dd3879a74d4b24c`, sec:size lines 2434–2727, excerpt SHA prefix `a22f71c4`. Later OLMo prompt, longer 1.5B, clamp, base-model and growing-context claims outside these six components are covered only where exact earlier accepted coverage applies. This is not a new universal paper or native raw-data audit.

## What the saved evidence supports

Larger models do not uniformly lose the sampled sensitive regimes. A higher-temperature shift appears in specific tests, but it does not establish a universal explanation for size effects. The frozen core comparison holds for the tested larger Qwen models, but OLMo32B reverses the corresponding family trend. Wider scans also recover many chaotic Qwen14B samples outside that core. These are finite, descriptive results from saved soft-token dynamics; they do not establish a causal parameter-count law, verified shrimp geometry or a text-generation benefit.

The sole reconstruction completed successfully while retaining contrary scientific outcomes. Computational completion is not a claim that every prediction passed.

## Scale: retain the frozen domain and the counterexample

All 663 shared leaves of the bound published scale projection reproduce exactly. The primary rule is saved LE > 0.001 with no detected period. The source applies the core's upper bound, log10(T) ≤ 0.3 + 1e−9; the separately stored lower-bound check found no point below −1, so these actual inputs match the intended [−1, 0.3] core.

| Family | Saved core chaotic shares | Frozen result |
|---|---|---|
| Qwen2.5, 7B → 14B → 32B → 72B | 65/512 → 0/130 → 0/130 → 0/128 | The specified P1 comparisons and nonincreasing P2 sequence hold. |
| OLMo-2, 7B → 13B → 32B | 32/512 → 0/130 → 21/130 | The 13B comparison holds; 32B violates P1 and the sequence violates P2. |

Thus the all-family P1/P2 claims fail. Larger-model absence within the core cannot be restated as global absence or a monotone full-axis decline. Qwen14B has **53/224** chaotic samples on its expanded axis, all above the question's 0.36 threshold. Their exact sampled log10(T) extrema are **0.5194968553459121** and **1.1438095238095238**. An outward-enclosing four-decimal display is **[0.5194, 1.1439]**. These extrema do not define a continuously chaotic interval. Its maximum saved LE is **1.1394301767064778** at log10(T) = **0.7973015873015873**. The expanded weighted chaotic breadth is **0.5001242887091943**, compared with Qwen7B's **0.22623786138906316**; grids and domains differ.

The frozen adjacent-endpoint rule discards the later copy when coordinates agree within 1e−9 and chaos labels agree. It does not assert equality of the other measurements. All four full-view duplicate pairs retain nonzero scalar differences, including OLMo7B's discarded-minus-retained ΔLE **+0.27283018098625433**, and Qwen72B's ΔLE **−0.013093993101830126** with Δperiod **+2**. No averaging, alternate copy selection or new tolerance was introduced.

Across the 15 listed models, the saved descriptive correlations are **0.5500631995505209** between intermediate-entropy breadth and chaotic breadth and **−0.44275578849066516** between log parameter count and chaotic breadth. Architecture, training, temperature support, sampling and inherited precision differ. Unresolved no-period rows remain unresolved, including 28 Llama3.1-70B, 19 Llama3.3-70B and 15 MoE core rows. They are not silently classified as periodic, and the two 70B model families are not pooled.

## Gain: default ambiguity, two declared references, and sparse bins

All 15 source files and all 360 saved cells remain. All-cell medians use the original five half-open entropy bins, not a chaos-only selection. The middle-bin P3 comparisons use Qwen14B/32B and OLMo13B/32B against their own family 7B. The OLMo13B interpretation is explicit because the frozen wording names “14B and 32B” while its roster contains OLMo13B.

| Reference view | Qwen P3 dispositions | OLMo P3 dispositions | Aggregate |
|---|---|---|---|
| Original all-15-source default | 12 undefined because two Qwen7B sources make the reference ambiguous | 6 true, 6 false | Undefined |
| Declared manuscript wide7B | 12 true | 6 true, 6 false | False |
| Historical narrow Qwen7B | 8 true; 4 undefined because its 6–10 bin is empty | 6 true, 6 false | Undefined |

The selected views were a disclosed source/manuscript-based amendment before this reconstruction. They are not new blinded confirmation, and neither replaces the default result. The wide Qwen7B reference has only **one cell in 6–10**; that is enough to define the source median, not evidence of robustness. Other target/reference bins also contain only one or two cells. Empty medians remain null rather than zero. The narrow reference cannot verify the four 6–10 Qwen inequalities.

The six OLMo failures are net gain for 13B in 1–3 and 6–10; readout gain for 32B in 1–3 and 3–6; and both factors for 32B in 6–10. All signed differences and bin memberships remain in the component note and saved output. These results support family- and source-specific descriptive observations, not a universal mechanism by which size removes chaos.

The current phrase “gain split ... shows which factor removes or keeps the chaos” should be qualified: this is a finite-twin/secant decomposition of observed amplification, not a causal intervention or native-Jacobian proof. Likewise, “along the evolving tangent direction” should say **along the evolving finite perturbation direction**, or explicitly identify it as a finite-twin approximation. Entropy grouping does not match every within-bin distribution or isolate model size. No historical gain CLI/full-precision summary equality is claimed where that original publication is unavailable.

## Prefix and initialization: keep the frozen tests separate

The PS test contains all 20 two-model/prefix endpoints, each with 48 coarse and 24 adaptive fine cells. Prefixes 0/1 remain development; 2–9 remain held out. All 1,440 saved rows and every fine-grid association are qualified under the source contract. That coverage does not turn finite classifications into asymptotic or primitive-orbit certificates.

**Both frozen PS predictions fail.** PS1 has **0/8** predicted declines: 14B has more chaotic coarse-grid samples on every held-out prompt. PS2 has **3/8** lower maxima, at prefixes 7/8/9, below its frozen **6/8** requirement. The adaptive stage finds chaos for 7/8 held-out 7B prefixes and 8/8 14B prefixes. Adaptive samples are targeted observations, not an unbiased second estimate of coarse-grid fractions. All **496 shared published PS values are exact**; the extra empty missing-pair witness is not a published-value discrepancy.

The source-classified PS 14B chaotic coordinates run from **0.36484736355226643** to **0.9**. The 7B range extends from **−0.06808510638297871** (development prefix 0) to **0.9** (held-out prefix 6). This is a different sample from the 53-point expanded scale scan above. The paper's “mostly” qualifier for the 7B band matters. A higher-temperature displacement of the main observed band is supported in these particular two-model tests, and a fixed window misses 14B activity. The evidence does **not** establish that general size effects are “largely a shift,” a pure translation of the dynamics, or a causal size law. Keep the two failed predictions explicit and describe the shift locally.

All 14 CP models and all 140 ten-prefix profiles are present and finite, with the exact 601-point entropy and 81-point gain grids. The rounded-profile compatibility checks pass for all breadths, crossing widths, T=1 entropies, network ranges, maxima and their locations. All 140 breadth reductions are exactly equal to their saved scalars. All 140 maximum-gain reductions have nonzero differences inside their original five-decimal compatibility cells; the largest absolute saved difference is **4.966588267318883e−6**. This checks compatibility with rounded profiles, not unavailable native logits, hidden states or directional gains.

| CP quantity | Saved own rho | Saved approximate p | Disposition |
|---|---:|---:|---|
| Primary paper-prompt breadth (CP1) | 0.29535687366198754 | 0.3052709262839073 | Fails |
| Median of ten prompts (CP2) | 0.5171349644228008 | 0.05826423877056072 | Passes frozen rho ≥ 0.5 criterion |
| Network proxy (CP3) | −0.06850933289361891 | 0.8159856968536429 | Descriptive |
| Maximum one-step proxy (CP3) | −0.32486683662458 | 0.2570904292454167 | Descriptive |
| Log parameter count (CP3) | −0.36062299930265146 | 0.20527030103491833 | Descriptive |
| Expanded sensitivity: median breadth | 0.4437129337903899 | 0.11199999038564985 | Separate sensitivity analysis |

**CP2's cutoff pass is not a p < 0.05 significance claim.** The source's approximate p-value was not its gate. The frozen three-decimal outcome vector remains primary; current full-precision expanded sweep lengths are a separate sensitivity analysis and do not retrospectively replace it. The other expanded sensitivity values are retained in the full component note.

The one-step proxy uses `log(median directional g_net) + log(median directional g_read)`, not the median of paired products, a leading singular value or an orbit Lyapunov exponent. Its paper-prompt range is **−3.376931105474913** to **−0.5440567183766042**; an outward two-decimal display is **[−3.38, −0.54]**. “One forward pass” means one batched invocation containing the base state and eight perturbations. Replace an implication that predictability necessarily “has to be measured on the loop” with the narrower statement that these particular start-state summaries did not establish a reliable predictor. This saved-scalar replay does not independently measure the claimed approximately 1e−7 effect of the later normalization fix.

## Plateau and precision history: selected confirmations and preserved failures

All 251 rows from the ten selected records have the expected saved coverage and source/class association. The contradictory and precision-limited histories remain in the result.

| Record | Retained finite outcome |
|---|---|
| PTa | All 12 selected chaotic cells positive at all three offsets; all 12 controls negative. This does not preserve every period label. |
| PTb | Two of three weak chaotic selections confirm; the third fails. All three unresolved controls are negative. |
| B7a/B7b | Original contradictory offset histories remain. B7b has 56 irregular and 8 periodic labels; none positive at all three offsets, all 64 negative at 1e−5, but 60 positive at 1e−7, including four periodic rows. |
| PTc | All 16 negative at all three offsets, but only 11/16 meet the spread < 0.1 or ≤ 0.1 plateau rule. Five fail. |
| QFa | 6/29 qualify under both boundary variants: five visible and one hidden; the other 23 are periodic. |
| QFb, prefixes 2/3/4/5 | 0/12, 4/12, 0/12, 0/12 qualify; prefix 3 has two visible and two hidden. |

PTb's failed selection at logT **0.11574803149606305** has twins **[−0.07595480687544003, −0.1016744197041453, +0.020149432590770254]**. Its two confirming temperatures are **0.13622047244094482** and **0.19763779527559056**. These are selected finite confirmations, not qualification of all original 70B map cells.

All 77 QF probes retain all 924 saved finite-difference ladder rows and the unique h = 1e−4 reference. All 260 specified offset-resolution comparisons pass both < 0.1 and ≤ 0.1, with no equality or undefined case; the unavailable historical strictness choice therefore does not change these actual dispositions. None of the 77 has positive tangent and negative twin. The six positive QFa records have maximum absolute saved tangent/twin discrepancy **0.006926280931835116**; all 29 QFa records have maximum **0.014964320327290004**, at the negative unsteered logT = 0.37 row. These are saved local-probe/trajectory results, not a fresh native derivative replay, global precision bound or asymptotic proof.

Three concrete precision-history corrections are necessary:

- B7b's middle-offset range is **−1.1202160963330632 to +0.35189647737335106**. Replace an unqualified “near zero at 1e−6” with “mixed negative, near-zero and positive estimates at the middle offset.”
- A positive finite-time exponent together with a detected period is not, by itself, a mathematical impossibility for a stable asymptotic orbit. Keep finite-time, non-normal and numerical-detection qualifications. The inconsistent offset history and corrected finite results support a precision warning, not that stronger impossibility claim.
- PTa did not literally keep every period label: originally periodic selections 15/17 now have period 0 and negative exponents, and originally unresolved selection 21 now has period 14. Say all 12 selected chaotic cells remained positive at all three offsets and all 12 controls remained negative; retain these label changes.

At the common 1e−5 offset, only four PTc/B7b coordinates match exactly; all four exponent differences are retained and 12 PTc points are unmatched. No interpolation or coordinate tolerance was introduced. QFa's rounded selected coordinates and changed precision path are separate evidence, not replacements for this history. The selected positive QFa results support the stated finite examples, not a complete boundary, universal 70B rarity rate or certification of every map label.

## Published values versus additional witnesses

All 34 publication-comparison nodes remain in `comparisons.json`. Shared-field accounting reports **663/663 exact scale leaves**, **496/496 exact PS leaves**, and **158 exact plus nine nonzero CP differences among 167 shared CP leaves**. CP's 218 additional own keys and PS's extra missing-pair witness are diagnostic/representation fields, not failures of a published value. Repeated nodes and views are not independent observations. No gain, plateau or dedicated L70D machine summary was invented where unavailable.

All nine CP differences (own minus published) are retained without an agreement tolerance:

| Field | Signed difference |
|---|---:|
| CP1/rho | +5.551115123125783e−17 |
| CP1/p | −1.1102230246251565e−16 |
| CP2/rho | +1.1102230246251565e−16 |
| CP2/p | +1.3877787807814457e−17 |
| CP3/log_g_net_p0/rho | −1.3877787807814457e−17 |
| CP3/log_params/rho | −5.551115123125783e−17 |
| CP3/log_params/p | −1.1102230246251565e−16 |
| Expanded sensitivity/breadth_p0/rho | +2.7755575615628914e−17 |
| Expanded sensitivity/breadth_median/rho | −5.551115123125783e−17 |

These tiny arithmetic differences do not alter the saved dispositions, but they are not called exact equality. The full witnesses and all differences remain in the sealed original artifact rather than being regenerated to simplify its representation.

## Depth: finite differences, distinct denominators

All four Llama3.3-70B maps retain finite no-return positive-exponent cells. The writer's “positive” count uses LE > 0.001 alone; the separately named own qualification also requires no detected period and finite validity.

| Layer | Grid | Writer-positive count | Qualified no-return count | Qualified share | Unresolved no-period cells |
|---|---|---:|---:|---:|---:|
| 20 | 24×24 | 47/576 | 46/576 | 0.0798611111111111 | 12 |
| 35 | 24×24 | 25/576 | 25/576 | 0.043402777777777776 | 9 |
| 50 | 32×32 | 44/1024 | 44/1024 | 0.04296875 | 468 |
| 65 | 24×24 | 4/576 | 4/576 | 0.006944444444444444 | 1 |

All per-scan writer counts, period histograms and LE extrema reproduce their saved summaries. These grids differ in resolution and separately constructed steering vectors, so equal alpha does not establish equal state perturbations. No regridding or new uncertainty model was applied. Layer65 still has four qualifying cells: disappearance of chaos at deep layers, global absence in 70B, and verified shrimp geometry do not follow. Finite detected returns are not certified primitive orbits.

Baseline50 has separate historical `analyze_controls.py` authority, SHA **6d72d4cbd233815f80c7e486f4030c1752906ded8c2a6ba2fa70cb5f18a60ead**, within authority **cf516769371fbaf8e6349cf5a5777c78332c7ea9109afff3a0562293acc9514e**. Its rule is LE > 0.001, period ≤ 0 and non-NaN, agreeing with the own period-zero qualification on these finite valid cells. The generic saved field saying a historical stricter scorer is unavailable is overbroad for that baseline; it remains unchanged and is clarified here. It must not be expanded into an invented dedicated scorer for the other L70D layers. Original status1 tokenizer/plotting failures and requested float64/bf16/auto-device versus effective historical precision limits remain explicit.


## Execution, evidence and review attribution

The scientific capsule was privately checkpointed at `2497d55f529fb0d9b1673c452792a696ee50db19` before the sole actual run. Final card `7f05a5a630205c65d4abe2ed267beb5dc37ed87a2ceeadcbb265b06fb3e106e8` and launcher `d1fcdad1cecc0ad87d740cf211838488c8dcda8033b73236b647a3a58272bed2` bind six unchanged scientific source files. Input manifest `be61223ae317ce665bb3a5b4ef2cfa99d360a2e8d66fd11451320b1ef032ec39` binds 329 nested files, 112 roles and 34 publication nodes; including the outer manifest, there are 330 files and 9,382,083 bytes. The worker authenticated inputs before and after scoring.

One manufactured preparation passed all 38 groups and measured the full synthetic adapter; it was not a target timing trial. The separately admitted actual run completed once, with no retry or extension, under CPU1 plus supervisor, nice10, 55 CPU/60 wall seconds plus fixed 5+5 cleanup, 1GiB address space, 128MiB sampled outputs/streams, 20/16GiB available-memory floors and 16GiB free disk. No producer scorer or model ran.

Original actual report `8a2896947f80e7e9208790da8eabbe6a38fb0837041984236287d3997d3e2a41`; receipt `3e9a3bef9835941bdb23a81471c2464d24be02be0346fe87d56bfadbc0f666fa`. The supervisor measured **7.683673758991063 seconds external elapsed wall**. The worker observed **7.507569101871923 seconds** through serialization before the final rehash/report and **236656KiB RSS** at that point; no independently measured whole-worker peak is available. The supervisor sampled 62,843,855 output bytes; exact final sizes remain in the nine output pins.

Tool session 21323 was reaped with exit0. Original supervisor 3258801/start203401653 and worker 3258833/start203401664, and their groups/sessions, were absent at owner closure. Eight captured Git blobs and all nine outputs were authenticated. Owner closure `3714d60a82543181f185fb4b321b99e7d9627f7a47c9d840a13687e057ce1645` and seal `d39e8b975d7e30ab8d894716be04278f78c3e72b7104dde5d68934736024c4ff` bind those observations. Original data are in `/data2/neural-shrimp-data/codex/size-review-task25/run-001/`, with the adjacent supervision and dispatch directories.

Inference performed execution, reaping and the finite owner checks. Boundary read scale/gain/depth; Theory read PS/CP/plateau. Root fully read both scientific notes and selected actual summaries, flags and discrepancies, and accepted their stated complete stored-disposition coverage. Root and Inference do not claim to have manually read every byte of the larger companion JSONs. No reviewer replayed model or component arithmetic. The final component notes are:

- `independent-scale-gain-depth-review.md`, SHA `4c393de37121f096c7b1c4eb01f8f654385e7ebc013824694e2ca87e72e76b60`; complete coverage `0bd57a5174dad435256a2bbdd0fd85f7c337055b21534d0bc4611c9b7672a464`.
- `independent-prefix-start-plateau-review.md`, SHA `20c5ac76e6bd7824ef056ac1c221ba9108b92e5e38f4eefa714c8eba2876f8a7`; compact coverage `3292e1316eed4d9d8a4bf28605d06fe380ea2d91664ce0976289ec5743211a99`, larger projection `2e96b7cdb35b8b97ae1ef2e34a9e6b9f3cf7e243698efcb40c799a9e67381beb`.

The original report retains a preparation sentence saying selected arrays remain unread until separately admitted scoring. Run-001 performed that separately admitted scoring; this corrects the temporal attribution without altering the original bytes. The baseline50 generic-unavailability field is similarly preserved and qualified above. Every original failed/incomplete attempt, post-save error, partial/vector artifact and source amendment remains in its finite authority. Historical remote loaded bytes, model weights, process births and effective native precision are not reconstructed.

Our reports, code and run cards are in the owned `codex/size-review-task25` tree. An initially appended shared-log card was removed by an ordinary disclosed correction commit, verifying exact equality with the integrated producer-parent log bytes. Published history was preserved. No producer paper or tools were edited.

## Final acceptance, delivery and next queue

Root fully read and accepted the complete synthesis and standalone delivery draft. The exact dated acceptance is recorded at 2026-10-15T18:23:34.555367+00:00 in `root-final-acceptance.md`; `final-acceptance.json` and `final-evidence-manifest.json` bind this final report, both accepted component notes, their complete stored-disposition coverage and the original actual report/closure. Private evidence and standalone delivery commits are recorded in `delivery-receipt.json` after publication. No scientific value or historical pending-science flag is changed.

Task23 is already complete and delivered. Task15 and optional crossover remain conditional, with no automatic 200-temperature sweep after Task25. The quiet 15-minute watch stays active without a cutoff until an explicit human stop; new assignments change the work, not the watch stop condition.
