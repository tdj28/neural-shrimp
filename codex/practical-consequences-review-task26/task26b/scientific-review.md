# Task26(b): final REPRO1 saved-evidence scientific review

2026-10-15. The original REPRO1 crossing and tail-readout summaries are reproduced by the completed independent saved-array reduction: **all 12,544 original summary fields agree**, with no saved summary or progress differences. The scientific corrections concern definitions and scope: the historical forecast uses a future minimum rather than the first nonzero distance; threshold crossing is not order-one separation; a tail readout event is not generated text or final-step disagreement; and the historical bfloat16 implementation is specifically mixed and load-order dependent. Both the historical and separately corrected forecasts remain in the evidence.

**Authority and review scope.** The [completed report](/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26b/run-001/report.json) has SHA256 `3b95ea55025c8a6bd8d8a2ba7523a7036781999362d443bedeecd189ec5710aa`; [all cell results](/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26b/run-001/cell-results.json) have SHA256 `83a2ddd461a20b5be153f029848578d308386fb997d2819f58678fae82738c5f`. Output manifest: `4704c355bac789137b3f988ca6e657c27c4e67855dac0956ee74318c59c64e6b`. The [final metadata authority](/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26b/run-001-authority/report.json) has SHA256 `5e88e844938f8226fd554f00276f75983421d685af23728b7136ce637765fa9a`, manifest `300681d53e9bc102e6eb2406ace9c4984b058c32085fca98a31ff22dbc48e365`. It authenticates all 3 outputs, 10 supervision entries and 5 source blobs, reaped session 29522/exit 0, absent original supervisor 2914948/start 200189684 and worker 2914976/start 200189818, and absent associated groups/sessions. Outer stderr is empty. Direct execution and closure observations are Inference’s, not this reviewer’s. The own execution commit is `d7db3f64bac4379dcd00e70dce9b398a6bcd5e32`; original producer source is `dbd125a70dd1b1a1f54a12c1ff2fde275b4c7a49`. No numerical or model replay was performed to write this note.

This reading covers the saved dispositions of all **4×448=1,792 cell/comparison rows**, all 12 comparison-by-label groups, all 24 source subgroups, both complete 192-row forecast ledgers, censoring/late-prediction cases, and original summary/progress equality. The sole admitted reducer already checked finite numeric payloads and source/input bindings; this review does not repeat that audit. All original files and exact cell identities remain authoritative. The paper/log outcome summaries were known before the independent reduction, so this is unblinded reproduction, not fresh confirmation. The [accepted source-contract note](/tmp/task26b-source-contract.md) supplies the precise original source lines; later producer edits are outside the frozen cutoff.

**Cohorts, clocks and comparison definitions.** The 448 cells comprise 192 labelled chaotic, 192 periodic and 64 other cells, selected equally from two source maps: map-stripes-256 and cross-512 each contribute 96/96/32. These are the inherited source-map labels in the H2 selection file: chaotic requires map exponent>10⁻³ and no detected period; periodic has a detected period; other is neither. The audit does not load a later long-H2 relabelling, and float64 labels need not describe a lower-precision map. All runs retain 1,500 steps and a final-500-readout tail. The source compares batches of 448 and 896 rows for batch64/batch32/bf16; precision compares float64 and float32 at the paired 448 cells.

Distance is the dimensionless asymmetric norm `||x_A−x_B||₂/||x_A||₂` of the post-update one-token soft state. Agreement compares argmax readouts computed from the corresponding pre-update states. These argmax tokens are not fed into the next expected-embedding update. The arrays retain float32 distances and boolean agreement, not token IDs, text, logits, full network states or prefix caches. Strict `distance>0.01` is the crossing event. First crossing/disagreement use one-based steps; the original `disagreeing_for_good_from` is a zero-based observed suffix index, with a separately named one-based counterpart. It cannot assert permanent future disagreement. Every saved first-positive-distance step is 1 in this run; no late-seed clock correction has been silently introduced.

**All comparison/label results.** Crossing times below are conditional on crossing. “Any tail” counts cells with at least one disagreeing readout in the last 500 steps; “final” counts disagreement only at the last saved readout. Neither counts emitted text.

| Comparison | Source label | n | Cross >.01 | Median step | Actual crossing range | Any tail | Final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float64 batch layout | chaotic | 192 | 190 | 102.0 | [36.0, 1092.0] | 173 | 110 |
| float64 batch layout | periodic | 192 | 10 | 92.0 | [38.0, 296.0] | 7 | 6 |
| float64 batch layout | other | 64 | 14 | 70.5 | [36.0, 173.0] | 10 | 9 |
| float32 batch layout | chaotic | 192 | 192 | 20.5 | [6.0, 143.0] | 177 | 124 |
| float32 batch layout | periodic | 192 | 72 | 16.5 | [5.0, 164.0] | 44 | 33 |
| float32 batch layout | other | 64 | 43 | 20.0 | [8.0, 342.0] | 26 | 18 |
| float64 vs float32 | chaotic | 192 | 192 | 20.0 | [5.0, 152.0] | 178 | 118 |
| float64 vs float32 | periodic | 192 | 75 | 16.0 | [5.0, 146.0] | 37 | 31 |
| float64 vs float32 | other | 64 | 43 | 19.0 | [7.0, 886.0] | 29 | 22 |
| historical bf16 batch layout | chaotic | 192 | 192 | 2.0 | [2.0, 4.0] | 191 | 150 |
| historical bf16 batch layout | periodic | 192 | 192 | 2.0 | [2.0, 6.0] | 183 | 127 |
| historical bf16 batch layout | other | 64 | 64 | 2.0 | [2.0, 5.0] | 63 | 45 |

The table’s 0.5B headline counts and nearest-rounded tail shares are confirmed. For float64 chaotic cells, the tail count is 173/192 (share 0.9010416666666666), whereas the final-readout count is only 110/192 (0.5729166666666666). For periodic cells the respective counts are 7/192 and 6/192. These are different observables. The other group is retained throughout; it cannot be silently merged with periodic or omitted when claiming “all regimes.”

**Actual distance extrema.** These are saved extrema, not percentile endpoints. “Largest distance” means the maximum over all 1,500 steps for a cell. The final-column upper bound rounds that maximum upward; precise attaining cell IDs remain shown. Median crossing at 0.01 is not a measurement of the time to order-one separation.

| Comparison | Label | Final median | Actual final range | Median cell maximum | Largest observed distance @cell | Safe rounded upper bound |
| --- | --- | --- | --- | --- | --- | --- |
| batch64 | chaotic | 0.9062333405017853 | [4.484712162180701e-14, 3.1130950450897217] | 3.5520007610321045 | 4.52254056930542 @258 | 4.53 |
| batch64 | periodic | 2.3985441284762442e-14 | [6.0065596566667365e-15, 1.8165196180343628] | 2.861203353236863e-12 | 2.700376272201538 @173 | 2.71 |
| batch64 | other | 3.5876739133775676e-13 | [7.161877868370765e-15, 1.737779974937439] | 9.667169842408896e-11 | 4.093139171600342 @215 | 4.10 |
| batch32 | chaotic | 0.9772135615348816 | [2.0353743821033277e-05, 3.1979167461395264] | 3.5563597679138184 | 4.359681606292725 @53 | 4.36 |
| batch32 | periodic | 2.2521890059579164e-05 | [9.628199677536031e-07, 2.048598051071167] | 0.003188223927281797 | 3.7904348373413086 @404 | 3.80 |
| batch32 | other | 0.00032972787448670715 | [1.0029129953181837e-05, 1.778314471244812] | 0.12284921482205391 | 4.032870292663574 @421 | 4.04 |
| precision | chaotic | 0.9363516867160797 | [5.675319334841333e-05, 3.364471673965454] | 3.521681785583496 | 4.370391845703125 @65 | 4.38 |
| precision | periodic | 2.489636517566396e-05 | [1.0847738849406596e-06, 3.7081713676452637] | 0.0034352359361946583 | 3.7296249866485596 @126 | 3.73 |
| precision | other | 0.0007817761797923595 | [1.1832335985673126e-05, 2.334113836288452] | 0.14296270906925201 | 4.108911037445068 @194 | 4.11 |
| bf16 | chaotic | 1.0675935745239258 | [0.009596509858965874, 3.225166082382202] | 3.8184436559677124 | 4.490548610687256 @10 | 4.50 |
| bf16 | periodic | 1.011465311050415 | [0.00016399951709900051, 3.3057639598846436] | 3.4853944778442383 | 4.349340438842773 @126 | 4.35 |
| bf16 | other | 0.9926064312458038 | [0.002385484054684639, 2.734858751296997] | 3.6839743852615356 | 4.330322742462158 @195 | 4.34 |

For the 182 float64 periodic noncrossers, the median of their maximum distances is **2.302951666172115×10⁻¹²**, but their actual maxima range from **5.602559838583862×10⁻¹⁴** (cell 375) to **0.005087868310511112** (cell 333). Thus a median near 2×10⁻¹² is not an all-cell bound. Across all 192 periodic cells, the final-distance NumPy-linear 95th percentile is **2.9555333603669197×10⁻¹²**, while the number with final distance≤3×10⁻¹² is **182/192=0.9479166666666666**. The original approximately “95%” is numerically close, but literal “at least 95%” is not supported. Report the percentile or the exact count, without conflating conventions. The largest final periodic distance is **1.8165196180343628** at cell 403; periodic exceptions remain visible.

A median of 20.5 (batch32) or 20 (precision) is not “every chaotic cell within 20 steps”: the actual largest crossing times are 143 and 152. In the historical bf16 implementation every cell crosses, but its crossing range is 2–4 for chaotic,2–6 for periodic and 2–5 for other cells; the median 2 is not a universal two-step bound. Final states can again be close after an earlier crossing. The records neither measure order-one crossing times nor identify cycle-phase shifts versus switches between attractors.

**All source-map subgroups.** Every denominator is retained; summaries differ between maps. This is a descriptive split of the frozen selection, not a second independent experiment. The companion JSON retains each stratum’s exact final-distance and all-step maximum-distance extrema with attaining cell IDs; they are not replaced by a pooled range. For example, float64 chaotic final distances range from 2.795920504665428e-13 to 3.1130950450897217 in map-stripes-256 and from 4.484712162180701e-14 to 2.7050702571868896 in cross-512. The corresponding largest all-step distances are 4.294046878814697 and 4.52254056930542. All remaining stratum extrema are transcribed in the same companion structure.

| Comparison | Source map | Label | n | Cross | Median step | Actual range | Any tail | Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch64 | cross-512 | chaotic | 96 | 94 | 136.0 | [38.0, 1092.0] | 80 | 39 |
| batch64 | cross-512 | periodic | 96 | 3 | 113.0 | [77.0, 167.0] | 2 | 1 |
| batch64 | cross-512 | other | 32 | 4 | 53.5 | [41.0, 121.0] | 3 | 3 |
| batch64 | map-stripes-256 | chaotic | 96 | 96 | 76.0 | [36.0, 583.0] | 93 | 71 |
| batch64 | map-stripes-256 | periodic | 96 | 7 | 86.0 | [38.0, 296.0] | 5 | 5 |
| batch64 | map-stripes-256 | other | 32 | 10 | 79.0 | [36.0, 173.0] | 7 | 6 |
| batch32 | cross-512 | chaotic | 96 | 96 | 28.5 | [8.0, 143.0] | 84 | 49 |
| batch32 | cross-512 | periodic | 96 | 31 | 22.0 | [5.0, 164.0] | 17 | 11 |
| batch32 | cross-512 | other | 32 | 17 | 37.0 | [10.0, 342.0] | 9 | 4 |
| batch32 | map-stripes-256 | chaotic | 96 | 96 | 15.0 | [6.0, 39.0] | 93 | 75 |
| batch32 | map-stripes-256 | periodic | 96 | 41 | 14.0 | [5.0, 129.0] | 27 | 22 |
| batch32 | map-stripes-256 | other | 32 | 26 | 14.0 | [8.0, 67.0] | 17 | 14 |
| precision | cross-512 | chaotic | 96 | 96 | 26.5 | [7.0, 152.0] | 85 | 46 |
| precision | cross-512 | periodic | 96 | 32 | 18.5 | [5.0, 146.0] | 14 | 11 |
| precision | cross-512 | other | 32 | 17 | 31.0 | [8.0, 886.0] | 10 | 5 |
| precision | map-stripes-256 | chaotic | 96 | 96 | 14.0 | [5.0, 57.0] | 93 | 72 |
| precision | map-stripes-256 | periodic | 96 | 43 | 14.0 | [5.0, 83.0] | 23 | 20 |
| precision | map-stripes-256 | other | 32 | 26 | 14.0 | [7.0, 37.0] | 19 | 17 |
| bf16 | cross-512 | chaotic | 96 | 96 | 2.0 | [2.0, 4.0] | 96 | 70 |
| bf16 | cross-512 | periodic | 96 | 96 | 2.0 | [2.0, 4.0] | 92 | 65 |
| bf16 | cross-512 | other | 32 | 32 | 2.0 | [2.0, 5.0] | 32 | 23 |
| bf16 | map-stripes-256 | chaotic | 96 | 96 | 2.0 | [2.0, 3.0] | 95 | 80 |
| bf16 | map-stripes-256 | periodic | 96 | 96 | 2.0 | [2.0, 6.0] | 91 | 62 |
| bf16 | map-stripes-256 | other | 32 | 32 | 2.0 | [2.0, 3.0] | 31 | 22 |

**Historical forecast and the one-definition correction.** Both use the same source-labelled chaotic cohort and the same fixed rule: original map exponent>0 and an observed float64-batch crossing. There are 192 rows, 190 eligible crossings, and two censored noncrossers (302,313). No prediction is invalid/nonpositive. The historical plot uses `log(.01/max(min_positive_over_entire_run,1e-17))/lambda`; that minimum can occur later in the run and is not an initial-observation forecast. The separately labelled correction uses the first positive distance in the same formula, with no re-selection or fitting. Seeds differ in 46 of the 192 rows. Both computations retain the original floor; it is not a measured physical noise floor. The quantity is a duration from the seed observation, not automatically an absolute clock. The retained ratio is explicitly the original one-based observed crossing step divided by that duration; no clock-offset correction is silently added. All first-positive observations happen at step 1 here, so the proposed late-seed issue does not arise in these actual inputs.

| Definition | Crossing cohort | Median observed/predicted | Pearson r of log times | Actual ratio range | 5–95% ratio quantiles | Min cell | Max cell |
| --- | --- | --- | --- | --- | --- | --- | --- |
| literal_whole_run_minimum | 190 | 0.9124571415153397 | 0.732582409174588 | [0.03815999588143439, 1.747633024625164] | [0.38030314540969445, 1.4214086241231858] | [245] | [6] |
| first_positive_correction | 190 | 0.9127118865267128 | 0.7318043517938886 | [0.03865482702485321, 1.7566343755011984] | [0.38030314540969445, 1.4214086241231858] | [245] | [6] |

Both conventions therefore support the printed approximate median ratio 0.91 and log-log correlation 0.73, with the first-positive definition explicitly corrected. This is descriptive association conditional on observing a crossing; it is not a calibrated forecast, universal equality, or independent validation of an asymptotic Lyapunov exponent. Source-map exponents and finite-trajectory labels remain inherited. The ratio spans roughly 0.038–1.75/1.76, with raw extrema above; a median close to 1 does not imply uniformly accurate predictions.

| Definition | Cell | Map exponent | Observed step | Predicted duration | Ratio | Exclusion |
| --- | --- | --- | --- | --- | --- | --- |
| literal_whole_run_minimum | 245 | 0.005623693218084397 | 188 | 4926.6252696705815 | 0.03815999588143439 | [] |
| literal_whole_run_minimum | 294 | 0.015768347466399407 | 322 | 1783.9348602866337 | 0.18049986418689226 | [] |
| literal_whole_run_minimum | 302 | 0.013234012089892404 | None | 2120.071943330525 | None | ['observed_noncrosser'] |
| literal_whole_run_minimum | 313 | 0.0012698008130492785 | None | 22164.35256762962 | None | ['observed_noncrosser'] |
| first_positive_correction | 245 | 0.005623693218084397 | 188 | 4863.5581755190615 | 0.03865482702485321 | [] |
| first_positive_correction | 294 | 0.015768347466399407 | 322 | 1783.9348602866337 | 0.18049986418689226 | [] |
| first_positive_correction | 302 | 0.013234012089892404 | None | 2099.7843695103056 | None | ['observed_noncrosser'] |
| first_positive_correction | 313 | 0.0012698008130492785 | None | 21840.499026320038 | None | ['observed_noncrosser'] |

The two noncrossers’ predictions lie beyond 1,500 under both definitions; their actual maximum distances are 9.581850690665306×10⁻⁷ (302) and 5.139828829214821×10⁻⁸ (313). Retain their censoring rather than assigning step 1,500 or claiming they never diverge. The two observed crossers 245/294 also have predictions beyond the run but actually cross at 188/322. Those adverse forecast cases are retained in the 190-cell summaries and prevent a claim that a beyond-horizon forecast entails noncrossing.

**Source corrections that the numbers cannot remove.** The original padding reverses steering vectors but keeps temperatures in the original order; the extra rows are not the same complete cells reversed. The first 448 target rows remain mathematically paired, so the source mismatch does not invalidate the finite batch-layout contrast. Each layout independently prepares its steered prefix cache, meaning prefix-cache rounding is also changed, not just subsequent matrix shapes. In REPRO1’s original load order, the first float64 model installs a class-wide RMSNorm implementation; the later bf16 model inherits its input-dtype path. The loop also retains its float32 embedding reference when the model is subsequently cast to bf16. Report this exact historical bf16-model/mixed-state/readout implementation rather than ordinary serving precision in general. REPRO-T has a different load order and is separately assessed; this issue is not transferred to it.

The existing source-contract note locates these facts at original `repro_noise.py`32–38,50,83,113,126–142 and original dependency lines described there; `plot_repro.py`70–80 fixes the historical forecast selection. The current-paper cutoff is 666e9b72, not later edits. Source labels in the H2 selection file are not independently updated to long-H2 outcomes by this tool. Most-probable-token booleans cannot establish different generated text, exact token cycles or the cause of a discrepancy. No new model-state reconstruction, QR, eigenanalysis, statistical inference or producer execution is claimed. The separate 7B rows of `tab:repro` and newer replications require their own input/source authority and remain outside this 0.5B audit.

**Suggested finite wording.** “Across 192 source-labelled chaotic cells of the 0.5B maps, changing the batch layout in float64 led 190 to exceed relative soft-state distance 0.01 within 1,500 steps; the median first crossing among those 190 was 102. Most-probable-token readouts disagreed at least once in the final 500 steps for 173/192 cells, compared with 7/192 periodic cells. Among the 192 periodic cells, 182 never exceeded 0.01. The historical minimum-distance scaling gives median observed/predicted ratio 0.912457; using the first positive observation gives 0.912712, on the same 190-crossing cohort. These finite observations characterize the recorded implementation and inherited source-map labels; they do not identify emitted text or establish a universal causal or asymptotic prediction.” Keep the exact periodic quantile/count and bf16 scope qualifications nearby. Root’s [Task26 editorial proposals](/tmp/task26e-editorial-replacements.md) supply the corresponding original sentence locations.

**Resource attribution and release.** Calculation before scientific serialization is 0.3593444658908993s. The saved worker timer through scientific writes but before status is 0.4193043739069253s, with observed high-water RSS 68,100 KiB at that point. Inference’s separate supervisor window is 0.523765225s. The hard AS bound is 1 GiB; no separately measured final worker lifetime/peak is inferred from it. All inputs remain hash-unchanged. Final status has `computational_completed=true` and original summary consistency true; its `scientific_claims_accepted=false` correctly leaves interpretation to this review and Root. No output or original source has been rewritten. The companion findings retain exact selected group statistics and an exhaustive cell-ID grouping of the saved dispositions; the immutable full reports retain all per-cell values and both forecast ledgers.
