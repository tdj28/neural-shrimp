# Task16 — independent saved steering-control tabulation

The 16 requested datasets are complete under the frozen selection: all **112,640 selected cells** have finite exponents and valid integer period fields. The separately authored saved-field check passed **4,446,143 exact checks with zero failures and zero nonzero differences**, including all selected cells, all 124,928 allocated source rows and all 4,096 nonselected completed overlap observations. Root has read and accepted both the arithmetic result and the final checker source/output/outer-closure authority. Both the original tabulation and the independent saved check are closed and authenticated.

This is **independent, unblinded arithmetic reproduction of saved scans**. The provisional and final producer C1 summaries were seen during Task17 review before this work. This report reads completed JSON and the saved checker result; it does not rerun a model, decode target NPZ arrays, refit, retune a threshold, or repeat the checker. The scientific facts below are finite-grid findings, not a new confirmation on unseen data.

The requested random and SAE controls also contain cells meeting the finite chaos rule. These named directions therefore do not support a claim that semantic steering is necessary for that rule to occur. They do not establish equality of directions or a population law: the grid shares and their steering-strength profiles differ materially, and the Qwen exact-coordinate subset has a different temperature coverage and ordering from the complete nominal maps. Unsteered long sweeps also contain qualifying cells in all four models.

**Definitions and denominators.** “Chaotic” below is the frozen operational label: finite saved exponent **strictly greater than 0.001 AND detected period equal to zero**. Period zero in a completed cell means no return detected by the finite writer test, not proof of aperiodicity. Positive periods are finite detections, not independently verified primitive periodic orbits. “Unresolved” means completed, valid period zero and exponent at or below 0.001. Positive-period cells with an exponent above 0.001 are retained separately rather than silently counted as chaos. Medians condition on positive detected periods; half-integer medians are preserved. All selected denominators equal completed and valid denominators. No selected NaN/Inf, malformed-period or unfinished cell was excluded to obtain these shares.

All recorded maps use the open “Hello!” prompt, window 1, perturbation epsilon 1e-6 and return tolerance 1e-8. The four alpha-zero sweeps cover 512 log10-temperature points from −1 to 0.3, with 1,000 transient plus 2,000 sampled steps and period-search cap 64. Qwen directional maps use 128×128 nominal grids, log10-temperature −1 to 0.3, signed alpha −0.12 to 0.12, 500 transient plus 300 sampled steps and period-search cap 48. Llama directional maps use 64×64 grids, log10-temperature −0.3 to 0.3, alpha = 0 to 0.12, 300 transient plus 300 sampled steps and period-search cap 32. These differing horizons and grids prevent equating the long unsteered percentages with the alpha-zero columns of the shorter directional maps.

**All 16 overall results.** Shares are approximate displayed percentages; the exact count and denominator are authoritative. P>0/LE>0.001 is the retained discordant positive-period count.

| Dataset | Valid cells | Chaotic | Share % | Detected P>0 | Unresolved | Median P>0 | P>0 / LE>0.001 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 0.5B, alpha = 0 | 512 | 7 | 1.367188 | 504 | 1 | 5 | 0 |
| Llama 3.1-8B, alpha = 0 | 512 | 96 | 18.750000 | 414 | 2 | 6 | 1 |
| Qwen 7B, alpha = 0 | 512 | 65 | 12.695312 | 441 | 6 | 15 | 2 |
| OLMo 2-7B, alpha = 0 | 512 | 32 | 6.250000 | 477 | 3 | 2 | 1 |
| Qwen 0.5B R1 | 16384 | 1251 | 7.635498 | 15012 | 121 | 3 | 11 |
| Qwen 0.5B R2 | 16384 | 1359 | 8.294678 | 14832 | 193 | 5 | 20 |
| Qwen 0.5B R3 | 16384 | 2054 | 12.536621 | 14181 | 149 | 5 | 8 |
| Qwen 0.5B philosophy | 16384 | 1119 | 6.829834 | 15155 | 110 | 4 | 5 |
| Qwen 0.5B gratitude | 16384 | 1229 | 7.501221 | 15059 | 96 | 4 | 2 |
| Llama L16 concept | 4096 | 1155 | 28.198242 | 2856 | 85 | 2 | 7 |
| Llama L16 R1 | 4096 | 1060 | 25.878906 | 2902 | 134 | 2 | 13 |
| Llama L16 R2 | 4096 | 709 | 17.309570 | 3253 | 134 | 2 | 3 |
| Llama L19 concept | 4096 | 1140 | 27.832031 | 2859 | 97 | 3 | 18 |
| Llama L19 SAE hostility | 4096 | 1188 | 29.003906 | 2769 | 139 | 4 | 10 |
| Llama L19 SAE random-live | 4096 | 1086 | 26.513672 | 2879 | 131 | 7 | 9 |
| Llama L19 Gaussian | 4096 | 1593 | 38.891602 | 2355 | 148 | 3 | 26 |

Qwen R1/R2/R3 whole-map shares are approximately 7.635%, 8.295%, 12.537%, versus 6.830% philosophy and 7.501% gratitude. For Llama L16 the concept direction is approximately 28.198%, compared with 25.879% and 17.310% for its two Gaussian controls. At L19 the concept, hostility SAE, random-live SAE and Gaussian shares are approximately 27.832%, 29.004%, 26.514% and 38.892%. Thus neither a universal concept advantage nor a universal random advantage follows. The four long alpha-zero results are 7/512,96/512,65/512 and 32/512; this is not a controlled model-size comparison.

**Every fixed absolute-alpha bin.** Entries are exact chaotic/valid counts. The bins are [0,0.01), [0.01,0.03), [0.03,0.06), [0.06,0.09), [0.09,0.12], with only the final upper endpoint included. `—` is an empty bin with null share, not zero probability. Each bin’s positive/negative/zero split and full period histogram are transcribed unchanged into the accompanying JSON. All outside-bin counts are zero.

| Dataset | [0,.01) | [.01,.03) | [.03,.06) | [.06,.09) | [.09,.12] |
|---|---:|---:|---:|---:|---:|
| Qwen 0.5B, alpha = 0 | 7/512 | — | — | — | — |
| Llama 3.1-8B, alpha = 0 | 96/512 | — | — | — | — |
| Qwen 7B, alpha = 0 | 65/512 | — | — | — | — |
| OLMo 2-7B, alpha = 0 | 32/512 | — | — | — | — |
| Qwen 0.5B R1 | 13/1280 | 170/2816 | 305/4096 | 530/4096 | 233/4096 |
| Qwen 0.5B R2 | 15/1280 | 81/2816 | 258/4096 | 388/4096 | 617/4096 |
| Qwen 0.5B R3 | 15/1280 | 45/2816 | 390/4096 | 709/4096 | 895/4096 |
| Qwen 0.5B philosophy | 13/1280 | 42/2816 | 360/4096 | 401/4096 | 303/4096 |
| Qwen 0.5B gratitude | 13/1280 | 107/2816 | 520/4096 | 374/4096 | 215/4096 |
| Llama L16 concept | 131/384 | 234/640 | 184/1024 | 298/1024 | 308/1024 |
| Llama L16 R1 | 146/384 | 264/640 | 337/1024 | 178/1024 | 135/1024 |
| Llama L16 R2 | 168/384 | 186/640 | 119/1024 | 111/1024 | 125/1024 |
| Llama L19 concept | 160/384 | 228/640 | 293/1024 | 267/1024 | 192/1024 |
| Llama L19 SAE hostility | 163/384 | 265/640 | 311/1024 | 236/1024 | 213/1024 |
| Llama L19 SAE random-live | 157/384 | 235/640 | 211/1024 | 173/1024 | 310/1024 |
| Llama L19 Gaussian | 164/384 | 274/640 | 436/1024 | 380/1024 | 339/1024 |

The bin profiles are heterogeneous rather than a monotone law in steering magnitude. For example, Qwen R1 rises through [0.06,0.09) then falls in the final bin; R3 rises across these bins; the Llama L16 random controls decline markedly after small alpha. Combining alpha signs in the Qwen bins does not imply sign symmetry. The full signed counts and histograms remain in the JSON, including Qwen R3’s final-bin negative 713/2048 and positive 182/2048.

**Exact zero and signed nearest columns.** Each entry is chaotic/valid; the parenthesized value is the positive-period median. Qwen directional grids have **no exact alpha-zero column**. Their nearest columns are −0.0009448818897637712 and +0.0009448818897637851, with 128 temperatures each. Taking the first argmin of absolute alpha would select the negative column only and must not be called an alpha-zero estimate. Llama directional maps have exact zero and nearest positive alpha = 0.0019047619047619048, with 64 temperatures each; there is no negative column.

| Dataset | Exact zero | Nearest negative | Nearest positive |
|---|---:|---:|---:|
| Qwen 0.5B, alpha = 0 | 7/512 (P=5) | — | — |
| Llama 3.1-8B, alpha = 0 | 96/512 (P=6) | — | — |
| Qwen 7B, alpha = 0 | 65/512 (P=15) | — | — |
| OLMo 2-7B, alpha = 0 | 32/512 (P=2) | — | — |
| Qwen 0.5B R1 | — | 3/128 (P=5) | 1/128 (P=5) |
| Qwen 0.5B R2 | — | 1/128 (P=5) | 1/128 (P=5) |
| Qwen 0.5B R3 | — | 1/128 (P=5) | 1/128 (P=5) |
| Qwen 0.5B philosophy | — | 1/128 (P=5) | 0/128 (P=5) |
| Qwen 0.5B gratitude | — | 1/128 (P=5) | 1/128 (P=5) |
| Llama L16 concept | 26/64 (P=6) | — | 21/64 (P=3) |
| Llama L16 R1 | 27/64 (P=6.5) | — | 25/64 (P=3) |
| Llama L16 R2 | 27/64 (P=6.5) | — | 26/64 (P=5) |
| Llama L19 concept | 26/64 (P=6) | — | 26/64 (P=8) |
| Llama L19 SAE hostility | 27/64 (P=6.5) | — | 27/64 (P=3) |
| Llama L19 SAE random-live | 27/64 (P=6.5) | — | 29/64 (P=3) |
| Llama L19 Gaussian | 27/64 (P=6.5) | — | 28/64 (P=5) |

The Llama exact-zero columns are not numerically identical across historical directional runs: the two concept maps have 26/64 and median 6, while the other five maps have 27/64 and median 6.5. The complete zero-column histograms are retained. This discrepancy must remain visible rather than being merged into a single common baseline or attributed to a nonzero steering effect. The Qwen philosophy positive-nearest column has 0/128, while its negative-nearest column has 1/128; zero successes here is a finite count, not exclusion of chaos.

**Literal common-coordinate comparison.** The independent report uses the actual selected floating-point coordinates, not substituted canonical temperatures. The Qwen literal intersection contains 13,184 coordinates (103 temperature rows × 128 alpha columns), rather than the full 16,384. Twenty-five top-tile temperature coordinates differ from the nominal full-grid values by nonzero floating-point amounts; the largest recorded absolute discrepancy is 1.1102230246251565e-16 (a rounded-up bound is 1.12e-16). No tolerance, interpolation or outcome-dependent match was used. Excluding these rows changes temperature coverage and therefore the estimand. This is a separate descriptive subset, not a correction that leaves the whole-map result unchanged.

| Qwen direction | Exact-common chaotic / valid | Share % |
|---|---:|---:|
| Qwen 0.5B R1 | 495/13184 | 3.754551 |
| Qwen 0.5B R2 | 893/13184 | 6.773362 |
| Qwen 0.5B R3 | 959/13184 | 7.273968 |
| Qwen 0.5B philosophy | 548/13184 | 4.156553 |
| Qwen 0.5B gratitude | 315/13184 | 2.389260 |

This subset puts R1 below philosophy and gratitude below both, unlike the full-map ordering. It does not isolate a causal direction effect. Llama L16 and L19 common-coordinate groups each contain all 4,096 coordinates and reproduce their whole-map counts. Exact common-coordinate summaries and their full period histograms are retained unchanged in the companion JSON.

**Source selection, incomplete attempts and all overlaps.** The 19 original NPZ inputs contain 124,928 allocated source cells: 116,736 completed and 8,192 unwritten. The frozen nominal selection contains 112,640 cells; the remaining 4,096 completed observations are the extra 16 lower-tile rows of Qwen R2 and R3. The three lower random scans all ended with scientific-job status 124 even though their enclosing runner reported 0. Each top scan ended 0. The selection was frozen prospectively from original commands and metadata, before this independent arithmetic: use lower rows 0–95 and top rows 96–127, preserving actual source coordinates. R2/R3 completed lower rows 96–111 are kept as nonselected repeated observations rather than overwriting the top tile or silently disappearing.

| Segment | Allocated | Completed prefix | Selected | Nonselected complete | Unwritten | Full-prefix chaotic / valid |
|---|---:|---:|---:|---:|---:|---:|
| scan64/q05-rand1/scan-partial.npz | 16384 | 12288 | 12288 | 0 | 4096 | 305/12288 |
| scan64/q05-rand1-top/scan.npz | 4096 | 4096 | 4096 | 0 | 0 | 946/4096 |
| scan64/q05-rand2/scan-partial.npz | 16384 | 14336 | 12288 | 2048 | 2048 | 1252/14336 |
| scan64/q05-rand2-top/scan.npz | 4096 | 4096 | 4096 | 0 | 0 | 556/4096 |
| scan64/q05-rand3/scan-partial.npz | 16384 | 14336 | 12288 | 2048 | 2048 | 1397/14336 |
| scan64/q05-rand3-top/scan.npz | 4096 | 4096 | 4096 | 0 | 0 | 1356/4096 |

All 4,096 overlap records are preserved individually in the original report’s `nonselected_completed_observations`, with both source temperatures, exponents, periods, dispositions and signed differences. The separate saved-field check reproduced every record; zero audit differences means it reproduced the recorded overlap differences, **not that the repeated observations agree with each other**. For example, R2 row 96/column 0 has selected-minus-nonselected exponent +0.007423950581979755 and R3 at that same ordinal cell has −0.015451931369460006. These are examples, not global bounds. The complete rows remain linked by the exact report hash; no new aggregate of those scientific differences is introduced here.

The saved ledger preserves initialized zeros in unwritten partial suffixes as unfinished, not period-zero scientific observations. The lower partial NPZ files lack embedded scan_args; their parameters are admitted from the separate original-command authority. An earlier `llama-sae-validate` stage in the L16 concept batch ended 1, while the relevant map stage ended 0; both statuses are retained. No enclosing exit0 erases an inner timeout or earlier stage failure.

**Norm and precision scope.** Original vector metadata says the three Qwen Gaussian directions were rescaled to the philosophy reference norm 188.9948284931374. Gratitude is separate, with recorded mean-residual norm 200.5861730285094; this is not an all-five equal-norm comparison. The L16 Gaussian reference is 61.8173370088694. The four L19 builds record reference norms approximately 66.319 with small per-run numerical differences; their selected SAE features are hostility 37512 and random-live 55786. These are source metadata, not newly computed vector norms.

The independent work verifies saved count/selection arithmetic and source-file associations. It does not reconstruct historical model, tokenizer or vector bytes, establish effective float64 execution on historical split-device runs, or revalidate their finite exponents against single-device wander. The updated effective-precision policy remains a limitation on stronger model-level claims. The reported finite-return rule is not an interval certificate of chaos, a proof of a shrimp, or a result about ordinary sampled text generation. Named controls and fixed grids also do not justify claims about arbitrary directions or model populations.

**Complete detected-period distributions.** Each row lists every stored period:count entry, including 0 for no detected return. The corresponding bin, sign, zero/nearest and common-coordinate distributions are preserved in the companion JSON; all 16 overall histograms are reproduced below without truncating medians or dropping unresolved cells.

**Qwen 0.5B, alpha = 0** — 0:8, 1:32, 2:6, 3:24, 5:437, 6:4, 9:1.

**Llama 3.1-8B, alpha = 0** — 0:98, 1:16, 2:31, 3:22, 4:83, 5:3, 6:128, 7:1, 8:2, 9:7, 10:84, 11:8, 12:4, 13:12, 15:1, 16:1, 18:1, 21:1, 22:1, 23:3, 24:1, 30:1, 35:2, 52:1.

**Qwen 7B, alpha = 0** — 0:71, 1:5, 2:110, 3:5, 4:45, 5:4, 6:8, 7:9, 8:2, 10:26, 11:1, 12:3, 14:2, 15:6, 16:67, 18:2, 19:30, 20:41, 21:3, 22:8, 23:10, 24:9, 25:6, 26:9, 27:2, 28:4, 29:3, 30:4, 31:3, 32:1, 33:3, 34:1, 35:3, 38:2, 42:2, 43:1, 45:1.

**OLMo 2-7B, alpha = 0** — 0:35, 1:54, 2:415, 4:3, 6:1, 9:1, 11:1, 15:1, 26:1.

**Qwen 0.5B R1** — 0:1372, 1:1160, 2:6096, 3:1437, 4:512, 5:3588, 6:386, 7:360, 8:210, 9:92, 10:254, 11:177, 12:183, 13:169, 14:65, 15:29, 16:39, 17:32, 18:26, 19:18, 20:24, 21:23, 22:21, 23:10, 24:14, 25:10, 26:8, 27:5, 28:7, 29:11, 30:2, 31:1, 32:2, 33:5, 34:6, 35:2, 36:7, 37:1, 38:3, 39:2, 40:2, 41:2, 42:3, 43:3, 44:1, 45:1, 46:1, 48:2.

**Qwen 0.5B R2** — 0:1552, 1:2087, 2:1892, 3:2050, 4:665, 5:5079, 6:869, 7:719, 8:312, 9:242, 10:169, 11:145, 12:87, 13:91, 14:53, 15:59, 16:56, 17:24, 18:28, 19:21, 20:22, 21:13, 22:16, 23:8, 24:19, 25:11, 26:11, 27:4, 28:6, 29:9, 30:9, 31:3, 32:9, 33:5, 34:2, 35:5, 36:5, 37:1, 38:4, 39:1, 40:3, 41:1, 42:2, 44:5, 45:1, 46:2, 47:3, 48:4.

**Qwen 0.5B R3** — 0:2203, 1:532, 2:3196, 3:554, 4:1994, 5:6315, 6:496, 7:232, 8:308, 9:104, 10:117, 11:65, 12:64, 13:35, 14:19, 15:15, 16:26, 17:3, 18:15, 19:9, 20:20, 21:3, 22:12, 23:2, 24:7, 25:5, 26:4, 27:4, 28:2, 29:2, 30:3, 32:2, 33:1, 34:2, 35:1, 36:1, 38:1, 39:1, 40:1, 41:2, 45:2, 46:2, 47:1, 48:1.

**Qwen 0.5B philosophy** — 0:1229, 1:1075, 2:4277, 3:1098, 4:1788, 5:4719, 6:883, 7:157, 8:423, 9:120, 10:130, 11:100, 12:80, 13:48, 14:56, 15:14, 16:40, 17:9, 18:17, 19:15, 20:16, 21:18, 22:8, 23:2, 24:12, 25:2, 26:5, 27:4, 28:9, 30:6, 31:3, 32:1, 33:2, 34:2, 35:1, 36:1, 37:3, 38:3, 41:1, 42:1, 43:1, 45:1, 46:2, 48:2.

**Qwen 0.5B gratitude** — 0:1325, 1:809, 2:5059, 3:1088, 4:2454, 5:4262, 6:88, 7:608, 8:141, 9:116, 10:154, 11:43, 12:59, 13:16, 14:29, 15:13, 16:32, 17:15, 18:8, 19:4, 20:17, 21:1, 22:3, 23:1, 24:8, 25:2, 27:1, 28:6, 29:1, 30:3, 32:5, 36:4, 37:1, 39:1, 40:4, 42:2, 48:1.

**Llama L16 concept** — 0:1240, 1:486, 2:1217, 3:168, 4:157, 5:263, 6:126, 7:37, 8:48, 9:31, 10:48, 11:26, 12:89, 13:20, 14:25, 15:13, 16:29, 17:9, 18:11, 19:11, 20:8, 21:5, 22:2, 23:3, 24:7, 25:3, 26:7, 29:2, 30:2, 31:1, 32:2.

**Llama L16 R1** — 0:1194, 1:444, 2:1467, 3:335, 4:118, 5:54, 6:34, 7:54, 8:44, 9:29, 10:77, 11:38, 12:26, 13:11, 14:20, 15:21, 16:19, 17:17, 18:17, 19:9, 20:15, 21:10, 22:2, 23:2, 24:13, 25:1, 26:9, 28:2, 29:1, 30:7, 31:2, 32:4.

**Llama L16 R2** — 0:843, 1:293, 2:1471, 3:326, 4:704, 5:90, 6:82, 7:59, 8:61, 9:10, 10:47, 11:10, 12:30, 13:2, 14:15, 15:2, 16:10, 17:1, 18:4, 19:4, 20:7, 21:5, 22:4, 23:1, 24:11, 27:1, 28:1, 29:1, 31:1.

**Llama L19 concept** — 0:1237, 1:741, 2:622, 3:277, 4:50, 5:40, 6:178, 7:203, 8:76, 9:147, 10:189, 11:75, 12:82, 13:23, 14:28, 15:9, 16:20, 17:16, 18:18, 19:5, 20:11, 21:7, 22:10, 23:7, 24:13, 25:2, 26:1, 28:2, 30:4, 32:3.

**Llama L19 SAE hostility** — 0:1327, 1:401, 2:260, 3:320, 4:424, 5:83, 6:280, 7:26, 8:264, 9:192, 10:147, 11:34, 12:84, 13:31, 14:40, 15:11, 16:42, 17:2, 18:22, 19:8, 20:19, 21:8, 22:14, 23:6, 24:12, 25:1, 26:8, 27:3, 28:7, 29:3, 30:4, 31:3, 32:10.

**Llama L19 SAE random-live** — 0:1217, 1:219, 2:444, 3:208, 4:131, 5:176, 6:147, 7:254, 8:545, 9:42, 10:164, 11:48, 12:55, 13:25, 14:128, 15:59, 16:33, 17:20, 18:24, 19:15, 20:15, 21:22, 22:13, 23:12, 24:19, 25:11, 26:13, 27:8, 28:13, 29:2, 30:6, 31:1, 32:7.

**Llama L19 Gaussian** — 0:1741, 1:421, 2:338, 3:455, 4:189, 5:66, 6:210, 7:34, 8:110, 9:50, 10:130, 11:52, 12:36, 13:18, 14:54, 15:15, 16:22, 17:18, 18:22, 19:19, 20:16, 21:11, 22:6, 23:8, 24:13, 25:6, 26:6, 27:1, 28:7, 29:2, 30:8, 31:2, 32:10.

**Evidence and lifecycle.** Original worker runtime 10.691473881946877 s, peak 161,504 KiB; external supervisor 11.600534365978092 s. Original session 25265 returned/reaped 0 and the original supervisor/worker/groups/sessions are absent according to Inference’s authenticated closure. This report inherits that original execution observation; it does not claim a new process-closure audit. The saved checker took 6.584500964963809 s and peak 153,796 KiB and reported exact equality across 4,446,143 checks. Its final external authority is accepted: the two audit outputs, eight supervision entries, three captured Git blobs, persistent outer logs and original supervisor/worker/group/session absence were authenticated. Original audit session 50359 returned/reaped 0; this is Inference’s execution and closure observation, inherited here without a second lifecycle check.

The actual 19 NPZ input files total **1,956,924 bytes**. The earlier 3,248,045-byte planning inventory comprised these NPZ bytes plus 1,291,121 bytes from 16 scan.json files; it is not the scientific NPZ total. Inference’s separate metadata clarification preserves that distinction and the original cards. The original final manifest authenticates 51 outputs totaling 91,126,083 bytes; the closure also authenticates 16 supervision entries and 11 captured Git blobs. All output, input, per-dataset summary and authority hashes are retained in the companion JSON.

- Scientific source: `dc6cd5c6f3e04c6f17302509035288dcad2e4e77`; private admission checkpoint: `9b97f9f6e0d55bc5f02e3dc8676b398ab5812541`.
- Original plan: `aa49c0c6d508f2e605bc53ebdb4b0f5fdb4099cbfb5c4e5d1b1a4a126cc54d76`; card: `67cbbc0b9a554b79becaba983375c26b5dcd50f2c69ed136f253738919a7df5f`.
- Original report: `/data2/neural-shrimp-data/codex/steering-controls-crosscheck/run-001/run/report.json`, SHA256 `23e936a8f62c6b925dd1ea247f11a99890a975b66be534b343ed765903577ad5`.
- Original status: same directory `status.json`, SHA256 `21579f825b201b443ebbbbb56f6b491703868ea75370828464baaeb70ec0ceba`.
- Original final manifest: same directory `final-manifest.json`, SHA256 `c6242ae732d7959b82bf17e9c703d968e3216ca62b66cc38abfc5d86c02cdcbf`.
- Original supervisor closure: `/data2/neural-shrimp-data/codex/steering-controls-crosscheck/run-001-supervision/closure.json`, SHA256 `2c210be28006b0dd251e60ea3e019b0761c42852f0965949fe73a0a5c462fcb9`.
- Original finite authority: `/data2/neural-shrimp-data/codex/steering-controls-crosscheck/run-001-closure-001/report.json`, SHA256 `bb921c5fc1962c510daafc64c88021622632c9deed7bc70df6275ef07eea6cdc`; manifest `bdc86e253ab438053a16f3b5b8848c0ed9945c581170d402769df4d08b511ed2`.
- Independent checker: source `3d4436cf44fdf473c867846d5431161dc1583ac19f11eaca2b49c6f7d9b69ad3`, card `e835ab44ce3ceef3511dc16b15e8a534284edd3d274fe3b182d8a4f736832499`.
- Independent checker result: `/data2/neural-shrimp-data/codex/steering-controls-crosscheck/run-001-review-001/report.json`, SHA256 `f0544574ddd02f4f3329d52cc528559bc6b1a5c6aa69e75df6f1a6530de5e297`.

- Final checker metadata authority: `/data2/neural-shrimp-data/codex/steering-controls-crosscheck/run-001-review-001-authority/report.json`, SHA256 `acbe5368eaadc908e352a392931e2f33349b5d4f4e5ce88d84597e35d5154440`; authority manifest `da37f15353acbce478670aa0f6348b2794ee72d189ac620432cf4fb403b381d3`. It binds audit source commit `7a607602e98de099c9c610171035b2e34f7f8fcf`, result `f0544574…`, and original audit closure `92027136add3be8e065575f1ffab07c89ca1ff38a35e655fe206dd0a7c430de8`.

The companion JSON is a selected-field transcription of these saved results, not a second tabulation. It preserves exact floats, all 80 bins with signed splits and histograms, all zero/nearest summaries, all 16 overall histograms, all common-coordinate summaries, all 19 source-prefix summaries and source/input hashes. It points to the unchanged complete overlap and per-row/per-column evidence in the authenticated original report.

**Final verdict.** Task16’s independent saved arithmetic and its separately authored exact checker pass for the entire frozen selection and retained source history. No scientific or lifecycle acceptance remains pending for this report. The earlier draft is preserved as `/tmp/task16-scientific-report.md` (SHA256 `8258c9153b7b3520ed3632757fa2afc4d5bda1ba3044c8bf9823723cf9d84954`). This final revision changes provenance/acceptance wording and adds this verdict; it does not change a count, threshold, selection or scientific conclusion.

Suggested paper wording: “An independent unblinded tabulation of the saved scans reproduced all requested counts. The finite-horizon criterion (exponent greater than 0.001 and no detected return) occurred in unsteered sweeps and in named Gaussian and SAE controls as well as semantic directions. Its frequency varied with direction, steering magnitude and the sampled grid; these controls do not establish semantic necessity or equivalence of directions. Exact-zero, near-zero and common-coordinate results are reported separately, with original partial-run history and finite-precision limitations retained.”
