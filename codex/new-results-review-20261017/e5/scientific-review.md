# E5: independent saved-data result review

17 October 2026. This review accepts the completed audit as a finite reproduction of the declared saved-data calculations. It does **not** establish absence of an association, an asymptotic chaos classification, or a causal connection between the autonomous loop and sampled text. The proposed positive association was not detected in the frozen main analysis. The original period-32 sensitivity includes a negative system-cluster interval excluding zero and must remain visible.

Only the completed own JSON outputs and their source/receipt basis were read for this review. No period, entropy, bootstrap, permutation, model or producer calculation was rerun. Inference directly observed the sole execution, reaping and process closure; my role was source and saved-evidence review. Root separately read both statistical summaries, all matched-temperature rows, all 95 numerical differences and the six inherited metadata entries.

## Exact evidence and coverage

Base: `/data2/neural-shrimp-data/codex/new-results-review-20261017/run-001/`.

| Record | SHA256 |
|---|---|
| `e5/report.json` | `6dab96a4371f29fdf676ee3152c0a7508ad4d0f1e4051e77d9708fed5cb2d146` |
| `e5/analysis.json` | `647cd52aeacb45193e4f766222772835c88b67f8a1a2bfa9abb7296c655c3153` |
| `e5/evidence.json` | `ed31d60bc60af8bb0e439c709f6c489d1762f8ced6c9ae6b29faaa3bacab5c2f` |
| `e5/draws.json` | `ddf96c332f185534b3a46a89d351b84beb19735aa22c1ff6a55fc253aa9b9976` |
| `e5/comparisons.json` | `7574a332ff2db2a7bc1343935c35f64c30a0b7c3de7d98a801284d87c09d1a1f` |
| `e5/controls.json` | `6d976e3784b4405786af337a0edbc9423152c27a294a93c37533966d8473ed7f` |
| `e5-supervision/receipt.json` | `e8cb9995e3c7d655d2d76cb3d98ed5feec1e039ca4d45710add71aa7c6088c3b` |
| `e5-original-closure.json` | `f3b6ceb016c70944f916d72dd03dd9a573eb95d3351f55a88e385554600d026d` |

Executed source commit: `a2a63b3ec7ccf2c27377725d9f22cde37b69ec55`; actual card SHA `93ad74b9fa3f072b0259b7b491813e38dd947fe71e5a3ead5fde82ed84901670`. Arithmetic/run/controls hashes respectively: `cb68f60da0ca33cf77863facc2f5ffc2c7649068fe34131519cb2c7cddb251f4`, `22d5c1cfdf4e92391e054d5e4f5dfcbf39343476321149f719e99afe6c8cddce`, `0362e6f95ec05fc6ac97176a32ae9a266fd1d23acbb5db6212e3ce064cc3b0ea`.

The owner closure records exit 0, session 44510 reaped, five source blobs and six outputs authenticated, and absence of the original worker/supervisor identities and groups. The six outputs total 13,741,089 bytes. Worker time through five output serializations, before final rehash/report, was 27.602834587916732 s; observed RSS then was 164,660 KiB. The external window was 28.001981965266168 s. Ten manufactured controls passed in the separately admitted profile and were inherited through its exact receipt; they were not repeated here.

Companion [coverage.json](/tmp/new-results-20261017-e5-review/coverage.json), SHA `2d3c79501137dac149e9e73fe4159a82dba6ea2ae22871ed07ab040eb2217edf` (1,102,235 bytes), copies every saved analysis/evidence field, all 95 differences and six inherited entries, plus compact stored-field and draw dispositions. It covers all 768 loop rows, 32 entropy-vector records, 33 sample files, 48 prompt-panel records and their token-period/tail witnesses, all 48 matched-temperature rows, development decisions and all precision ladders. I read complete compact row/gate dispositions, prompt onsets, entropy roots, statistics and exceptions; the companion preserves the full numerical arrays without claiming human inspection of each individual resampling draw. All 160,000 saved endpoint/permutation values and the resampling indices remain in the pinned `draws.json`; all 39,696 exact leaves remain in the pinned `comparisons.json`.

## Main result and original-card sensitivity

The main rule is the frozen development-selected last-96-token suffix with primitive period at most 48, H target 2 nats, 24 held-out prompts, 24 temperatures and 12 sampled replies of 256 tokens per prompt-temperature. The original-card sensitivity uses period at most 32 on the same saved cohort. Neither panel was selected using these held-out results.

| Saved endpoint | Main, period ≤48 | Original-card sensitivity, period ≤32 |
|---|---:|---:|
| Spearman onset correlation | −0.2286975239953017 | −0.3346261835305691 |
| Prompt-bootstrap 95% interval | [−0.5907022095036059, 0.2409444445820201] | [−0.6975236351219404, 0.0941947662231081] |
| Opener-cluster 95% interval | [−0.5829906234404642, 0.46505844884389147] | [−0.7112509620157283, 0.1760819252796758] |
| System-cluster 95% interval | [−0.5659210352682955, 0.07742215691809227] | **[−0.617973090199024, −0.1850706887712575]** |
| Two-sided permutation, plus-one p | 0.27687231276872315 | 0.10878912108789121 |
| Partial rank correlation given log T_H | −0.22584568131008825 | −0.3313450333520206 |
| Partial 95% interval | [−0.588823999662614, 0.2747627608100432] | [−0.6926101716184138, 0.12910589238877135] |
| Matched chaotic-minus-periodic repetition fraction | 0.003083901908434707 | −0.0019776525264511 |
| Matched 95% interval | [−0.012372363226487045, 0.022758641407582783] | [−0.01849082934287163, 0.015570344188514516] |

Each of seven bootstrap streams per panel has 10,000 draws and zero recorded nonfinite draws. Each permutation stream likewise has 10,000 finite draws: 2,768 main exceedances (raw fraction 0.2768) and 1,087 sensitivity exceedances (0.1087). The plus-one p values, raw fractions and inclusive source comparison convention are distinct saved fields. The separate 8-opener and 6-system cluster sensitivities do not constitute a joint two-way clustering guarantee or random sampling of a prompt population.

Only **8 of 24 temperatures** contain both chaotic and periodic cells. All 24 rows, including 16 undefined group contrasts, are retained. The overlapping grid indices are 10, 11, 12 and 15–19; the first three have only one chaotic cell, and indices 17–19 have zero repetition in both groups. The declared weight is nC·nP/(nC+nP); irregular cells remain outside those two groups. Thus 0.003 is a weighted overlap contrast with the displayed uncertainty, not a general bound on every regime difference or an equivalence result. All matched resamples happened to remain defined; no undefined draw was filled with zero.

Other frozen sensitivities remain: uncensored-pair Spearman is −0.3894079718939161 (15 main pairs) and −0.42191942529152593 (11 period-32 pairs); edge-censor substitution gives −0.2286975239953017 and −0.33574440742873024; replacing chaos onset by first nonperiodic onset gives −0.36264058330307974 and −0.49162841678085845. The H–repetition correlations are −0.06676895160447167 and −0.095808771975052; H–chaos is 0.05545337678554953. Their saved prompt intervals include zero. No numerical primary-success cutoff was frozen, so the audit does not invent a binary hypothesis gate.

The main analysis has six no-chaos/right-censored prompts (16,25,32,41,47,48), four left-censored repetition onsets (16,27,59,63), no right-censored repetition onset and ten threshold recrossings. Among the 15 pairs observed within the grid, repetition onset is below chaos onset for 13, not all 15; prompts 29 and 36 retain the opposite order. The median log10(T_rep)−log10(T_chaos) across **all 15 observed pairs** is −0.4239130434782608, with quartiles [−0.4923913043478261,−0.14565217391304347]. It is not a median computed only over the 13 lower pairs. Period-32 yields twelve left-censored repetition onsets, eighteen recrossings and all 11 observed pairs below chaos, median −0.6521739130434782. Censored analysis values one grid step outside the range are conventions, not observed onsets. The stored `monotone` flag tests recrossing of the half-repeat threshold, not global monotonicity of the entire fraction curve.

## Gates, choices and numerical scope

All four loop-batch source gates and all 33 sample-file gates pass, without excluded failures. The development ledger has 177 periodic and 15 chaotic cells; the held-out ledger has 529 periodic, 43 chaotic and four irregular cells. All saved final states are finite, classification/label/period associations match, twin collapses are zero, and available leading-token-period comparisons match. The four irregular rows remain explicit: prompt29 at temperature indices12/18, prompt36 at22 and prompt47 at15. No supplementary cells occur in this admitted set.

This is not a claim that all fields are finite or all native periods were independently established. There are 704 rows with both Lyapunov fields stored as NaN because those estimates were not taken for those periodic rows. The 175 development retirement records are virtual (`retired=false`); the 521 held-out retirements are actual. The source residual gate applies to 177 development and eight held-out periodic rows, and exempts retired periodic rows. The saved final state cannot reconstruct the original ring or 64 consecutive retirement closures. Development virtual-retirement mismatches are zero both under the source's repeated per-prompt total and the unique-event presentation. These are source-defined finite-history checks, not stability spectra or asymptotic chaos proofs.

The declared development choices reproduce: the initial period-32 rule censors 4/8 prompts; period-48 censors 1/8 and is the first admitted alternative; window64/period32 censors 4/8. All eight development prompts reach H=2 in the grid. Zero virtual-retirement mismatches and zero gates justify the frozen retirement choice. The source timing projection is 1.3451171776926256 h loop plus 6.464271104323755 h sampling = 7.80938828201638 h, selecting 12 replies under its >7 h rule. This is reproduction of the historical projection formula, not a measured performance guarantee. The rubric was recorded before development outcomes, while choices commit `3a667908d9f871b24d51f37778153f28ebf8d7c4` preceded held-out outcomes. Both held-out sample invocations, including later resume/skip metadata, remain part of the historical record.

All 64 full/masked entropy roots from the 32 saved float64 first-step logit vectors are defined; largest absolute saved own residual is 2.6645352591003757e−15. The finite bisection bracket is not an interval-arithmetic enclosure. Original statistics deliberately use the saved H scalar after the independent root check, avoiding a change of rank protocol from last-bit root differences. Float32 sampling H values and sampled-path entropy means are inherited because the corresponding logits are not stored. All sample token IDs are in range, with zero masked EOS tokens and zero nonfinite saved entropy means. Buffered-versus-plain development prompt0 has exactly equal token arrays (384 samples); its saved entropy means differ by at most 9.470202400052585e−14.

Three original finite-difference ladders are retained as scalar evidence only. At log10 T≈0.1087 and0.3696, the recorded changes show approximately quadratic convergence over an intermediate h range, with minima 1.0757234987137312e−8 and6.84482365571683e−9 before rounding dominates. At −0.6087 the derivative norms collapse to zero; relative changes include infinity/NaN and do **not** establish quadratic convergence. No derivative vectors are available for an independent native precision replay.

## Every comparison difference retained

The result is not bit-exact: 39,696 exact leaves, **95 finite numeric differences**, and six inherited execution/path metadata leaves. There are no missing-own, structural or unexplained nonfinite comparison differences.

- 62 differences are entropy T_H/T_H_masked roots. The largest is +7.66053886991358e−15 at held-out prompt18: own0.38287516239502556 versus saved0.3828751623950179, for both roots. A decimal upper bound is 7.67e−15.
- Eleven differences occur in each of the main held-out analysis, its published copy and the period-32 summary, for 33 statistical differences. The largest absolute statistical difference is 2.220446049250313e−16 (upper bound2.23e−16). Main matched point difference is +1.3010426069826053e−18; sensitivity matched point difference is −8.673617379884035e−19.
- All signed values and exact paths are in the companion and original comparisons. Repeated publication comparisons are counted as separate leaves, not separate experiments. Source git/choice-path strings are explicitly inherited, not numerically reproduced.

The original analyzer at `29c399330879f3035dd8933213e5e4b50859c221` and frozen current tool share SHA `90f7d17a7894849c1527e55243b37097711efca6fa2e99f67bd70000900221f3`. An initial review misread the H–repetition bootstrap as unpaired; exact source checking showed paired `z[idx], y[idx]` in both versions. That allegation was withdrawn before execution, and no unpaired or purportedly repaired endpoint was calculated. Original prepared history remains preserved.

## Concrete wording corrections

Frozen paper `/tmp/claude-task26-27-20261017-source-001/tree/paper/llm/main.tex`, SHA `910c8e37a201c769937b3e524417d3adcbdb653bfd8bc1bbde808ad6c7c929e7`, lines2927–2934:

1. Replace “found no association” and “found no bridge” with “did not detect the proposed positive association in this finite main analysis.” No equivalence margin, causal test or sufficient-power claim was frozen.
2. Replace “differed by only0.003” with “the weighted chaotic-minus-periodic repetition difference was0.00308 (prompt-bootstrap95% interval[−0.01238,0.02276]), using the8/24 temperatures containing both regimes.” The displayed interval is rounded outward.
3. State “Among15 prompts with both onsets observed,13 had repetition onset below chaos onset; the median gap across all15 was0.424 decades.” Preserve censoring/recrossings and the two opposite-order cases.
4. Report the period-32 sensitivity, including its system-cluster interval [−0.6180,−0.1850] (outward rounding), rather than implying every sensitivity is null. This negative direction does not support the proposed positive relation; it also prevents a universal no-association conclusion.

Frozen `docs/pi/e5-STATUS.md`, SHA `022e0d66c7452dffdf3c03e30a9ec3f547d21631a9484b0b18ec540a1dea10f3`, lines9–10: “batch-position repeats bitwise identical” is too strong. The source definition at tool lines957–996 compares F(x) at different batch positions; saved maximum differences are0,2.914335439641036e−16 and2.7755575615628914e−17. Use “agreement within2.92e−16” and distinguish it from the truly identical buffered token sequences. Keep the saturated derivative case unqualified rather than calling all three ladders successful h² checks.

Suggested compact paper wording: “In this exploratory test of24 held-out prompts, the frozen main analysis did not detect the proposed positive relation between loop-chaos onset and repetition cessation in sampled text (Spearmanρ=−0.23,95% prompt-bootstrap interval[−0.60,0.25], permutationp=0.28). The matched-temperature repetition contrast was0.00308 [−0.01238,0.02276], using8/24 temperatures with both regimes. Of15 prompts with both onsets observed,13 stopped repeating at a lower temperature; the median log-temperature gap across all15 was−0.424. Results depend on the repetition rule and resampling unit; the original period-32 system-cluster sensitivity gives a negative interval excluding zero. These finite results do not establish absence of a relation or a causal bridge between the float64 fixed-slot loop and float32 growing-context sampling.”

The development outcomes, a negative held-out log heading and the published negative paragraph were exposed before this independent audit; no retrospective claim of reviewer blindness is made. The exact source protocol and frozen choices were retained, with all failures, undefined fields, uncertainty and rounding differences visible. No additional numerical work is required for this assigned saved-data review.
