# Task 28 — sampled basins of coexisting soft-token outcomes

Completed 2026-10-18T01:42:58.873540+00:00. Human-requested exploratory CPU experiment; Root reviewed the source and separately implemented saved-state arithmetic. No peer review or independent model trajectory is claimed.

The donor 2-cycle receives most starts under both tested random-start distributions, although the fixed newline start follows the crescent side. All three tested segments have multiple observed label changes, ruling out a single monotone threshold description on those sampled segments. Finite sampling does not establish fractality. Euclidean proximity is consistent with the chosen fixed start but fails as a general explanatory rule on the random starts.

At alpha 0.0644, the fixed-start census and the reference do **not** have identical finite period labels. That discrepancy is retained, so this cell is not evidence for a newly verified period-6 attractor.

## Method and scope

Qwen2.5-0.5B-Instruct, philosophy vector at layer 12, one soft slot, full 151936 vocabulary and 896 hidden coordinates; log10T=-0.156. The previously reviewed ManualQwen float64 implementation computes RMS normalization and rotary functions directly in float64, avoiding Hugging Face patch/dispatch hooks. A measured AD-versus-central-difference profile at h=.01/.003/.001 gave relative discrepancies .0011280/.00010162/.000011292, consistent with quadratic convergence. Exact model/vector/config and 37 prefix IDs are pinned; last prefix ID198 is the newline embedding. The period 3 reference at alpha.0642 emits ID271 (two newlines) with probability 0.7427487726738978, reproducing the stated approximate 0.74. This does not measure a causal proximity effect.

There are 323 starts per cell:128 uniform vocabulary embeddings,128 independent Dirichlet(1) convex mixtures of 8 distinct IDs from that cell’s fixed-start top64 distribution, 65 equally spaced segment points, and fixed/donor controls. Seed 280017 is reused across cells; cross-cell results are not independent. Additional cell .0644 was selected from the exposed cross512 map using its correct temperature-row/alpha-column orientation, before basin counts were observed. Original producer coexistence labels were already known.

The exploratory design was amended before target basin outcomes: a profile showed that uniformly running all starts for 1200 steps would take roughly 9 hours. Early stopping instead requires the same positive period at two successive 32-step checks, using maximum relative full-state return over 64 comparisons, lags1..32 and strict 1e-8 threshold. Earliest stop 192; maximum 1200. Phase agreement with the saved reference requires 1e-6 relative distance. Both confirming 129-state windows are retained. Variable active batching and early stopping can affect finite labels, particularly near tolerance. Other periods and failed reference matches stay separate. A no-return label is not a chaos diagnosis.

## Random-start fractions

Each row has n=128. Intervals are marginal Wilson 95% for the specified IID arm; no interval is assigned to the deterministic segment or controls. P2/P3 below mean reference-matched finite outcomes. Other labels do not establish separate asymptotic attractors.

| alpha | arm | outcome | count | fraction % | Wilson 95% |
|---|---|---|---:|---:|---|
| 0.0642 | single | crescent_period3 | 20 | 15.625 | [10.348, 22.905] |
| 0.0642 | single | donor_period2 | 107 | 83.594 | [76.218, 89.012] |
| 0.0642 | single | other_period2 | 1 | 0.781 | [0.138, 4.293] |
| 0.0642 | mixture | crescent_period3 | 6 | 4.688 | [2.166, 9.850] |
| 0.0642 | mixture | donor_period2 | 115 | 89.844 | [83.398, 93.968] |
| 0.0642 | mixture | other_period4 | 7 | 5.469 | [2.674, 10.858] |
| 0.0652 | single | donor_period2 | 105 | 82.031 | [74.478, 87.718] |
| 0.0652 | single | no_return_up_to32 | 22 | 17.188 | [11.633, 24.654] |
| 0.0652 | single | other_period2 | 1 | 0.781 | [0.138, 4.293] |
| 0.0652 | mixture | donor_period2 | 115 | 89.844 | [83.398, 93.968] |
| 0.0652 | mixture | no_return_up_to32 | 7 | 5.469 | [2.674, 10.858] |
| 0.0652 | mixture | other_period4 | 4 | 3.125 | [1.222, 7.760] |
| 0.0652 | mixture | other_period8 | 2 | 1.562 | [0.430, 5.518] |
| 0.0644 | single | donor_period2 | 106 | 82.812 | [75.346, 88.367] |
| 0.0644 | single | other_period2 | 1 | 0.781 | [0.138, 4.293] |
| 0.0644 | single | other_period6 | 21 | 16.406 | [10.988, 23.782] |
| 0.0644 | mixture | donor_period2 | 112 | 87.500 | [80.657, 92.157] |
| 0.0644 | mixture | other_period4 | 7 | 5.469 | [2.674, 10.858] |
| 0.0644 | mixture | other_period6 | 9 | 7.031 | [3.743, 12.823] |

At .0644 there were zero reference-matched P3 outcomes in either 128-start random arm (Wilson 95% [0, 2.914%] for each); this is not proof that its basin has zero measure. The referenceP3 endpoint remains in the segment. At .0652,29 random starts and 12 deterministic/control starts reached the1200-step ceiling with no detected return.

## Segment evidence

The segment is x(u)=(1-u) x_donor + u x_fixed-reference, u in[0,1], using the saved last state of each600-step reference. At .0652 the right endpoint is a finite irregular state, not a period 3 phase. Each tested orientation is phase-specific. All 65 coarse points per cell and all 152 refinement queries are preserved. Donor-P2/complement brackets are refined 8 times; all 19 final brackets have exact width1/16384. Non-donor labels in these records are original coarse-endpoint labels, not a promise that every refined complement endpoint has the same period.

### alpha 0.0642

Coarse segment counts: {'donor_period2': 52, 'crescent_period3': 13}. Observed label changes: 5; refined donor/complement brackets: 5.

| lower u | upper u | original coarse orientation |
|---:|---:|---|
| 0.0936279296875 | 0.09368896484375 | donor_period2 → crescent_period3 |
| 0.0941162109375 | 0.09417724609375 | crescent_period3 → donor_period2 |
| 0.57904052734375 | 0.5791015625 | donor_period2 → crescent_period3 |
| 0.61199951171875 | 0.612060546875 | crescent_period3 → donor_period2 |
| 0.8515625 | 0.85162353515625 | donor_period2 → crescent_period3 |

### alpha 0.0652

Coarse segment counts: {'donor_period2': 54, 'no_return_up_to32': 11}. Observed label changes: 7; refined donor/complement brackets: 7.

| lower u | upper u | original coarse orientation |
|---:|---:|---|
| 0.4354248046875 | 0.43548583984375 | donor_period2 → no_return_up_to32 |
| 0.437744140625 | 0.43780517578125 | no_return_up_to32 → donor_period2 |
| 0.63720703125 | 0.63726806640625 | donor_period2 → no_return_up_to32 |
| 0.685546875 | 0.68560791015625 | no_return_up_to32 → donor_period2 |
| 0.82806396484375 | 0.828125 | donor_period2 → no_return_up_to32 |
| 0.828125 | 0.82818603515625 | no_return_up_to32 → donor_period2 |
| 0.91522216796875 | 0.915283203125 | donor_period2 → no_return_up_to32 |

### alpha 0.0644

Coarse segment counts: {'donor_period2': 52, 'other_period6': 12, 'crescent_period3': 1}. Observed label changes: 8; refined donor/complement brackets: 7.

| lower u | upper u | original coarse orientation |
|---:|---:|---|
| 0.34344482421875 | 0.343505859375 | donor_period2 → other_period6 |
| 0.3519287109375 | 0.35198974609375 | other_period6 → donor_period2 |
| 0.43743896484375 | 0.4375 | donor_period2 → other_period6 |
| 0.4375 | 0.43756103515625 | other_period6 → donor_period2 |
| 0.58245849609375 | 0.58251953125 | donor_period2 → other_period6 |
| 0.61669921875 | 0.61676025390625 | other_period6 → donor_period2 |
| 0.87152099609375 | 0.87158203125 | donor_period2 → other_period6 |

The .0644 other-period 6/P3 coarse transition[.984375,1] is retained but not binary-refined because both endpoints are outside donorP2. Adjacent opposite brackets sharing a coarse sample (e.g..828125 at.0652) demonstrate small sampled alternations, not a resolved fractal dimension or a complete boundary enumeration.

## Fixed start and proximity

All 17 direct-source observations (initial state through iterate 16) are closer to the crescent reference than to donor phases in each cell . Initial absolute Euclidean distances (crescent,donor) are(.3234236345,.5729251224),(.2991247056,.5736944396),(.3195342765,.5730678621) for.0642/.0652/.0644. At .0652 the reference is a129-state irregular sample, making nearest-set distances especially incomparable to the two-point donor set. Only step 0 is independently reconstructible from saved initial states; first16 iterate distances are trusted execution summaries, without saved raw intermediate states.

The predeclared explanatory diagnostic performs poorly: at.0642 all 248 binary-eligible random starts were nearer P3, yet 222 reached donorP2. Accuracy 26/248=10.484%, versus 222/248=89.516% for always predicting donor. Eight other outcomes were excluded explicitly. At .0644 all 218 binary-eligible starts were nearer referenceP3 but reached donorP2 (0/218; majority baseline218/218);38 other outcomes were excluded. At .0652 no binaryP2/P3 score is appropriate. These within-sample comparisons do not establish causality or out-of-sample prediction. Proximity describes the fixed trajectory but does not explain basin membership generally.

## Finite period discrepancy and preserved limits

The .0644 reference passes lag3 after 600 steps (residual 7.1830837249722e-10). Its fixed-start census stops at 512 with lag3 residual 1.348052190171368e-8, just above1e-8, and lag6 residual 1.3840229072217864e-9. The saved token sequence repeats the three tokens twice. Thus the operational label is other_period6, but convergence/batching/tolerance are possible explanations for this finite discrepancy; their individual contributions were not tested; no new period 6 attractor or bifurcation is certified. Other_period4/8 and unmatched period 2 are likewise retained, not reclassified favorably. No post-hoc longer run or altered threshold was used.

At .0642 the fixed control matchesP3 and donor matchesP2; at.0652 fixed is no-return and donorP2; at.0644 fixed is finiteP6 and donorP2. There are zero disagreements between the two stored confirming-window period labels, which is a finite convergence check, not proof of asymptotic stability. No full spectrum, chaos, global basin volume, native-generation benefit, or geometry certificate follows.

## Evidence and closure

One actual run, no retry: source e8bc3a1708cc00dbd2aaab7cee4d263726ea1e37; external 13681.598448751029 seconds,64 output files/1,949,874,745 bytes, child peak7,807,160KiB as observed at reap. Actual session 40021 and saved-audit session 26933 exited0; original workers, supervisors and groups are absent. Saved audit took18.94434991432354 seconds externally. It hashes every actual output and checks all 969 census rows, confirming windows, reference gates, Wilson intervals, token IDs, all 152 queries and all 19 reconstructed brackets. Pure saved NumPy arithmetic does not replay the model.

Raw: `/data2/neural-shrimp-data/explore/basins/task28/run-001`; audit: sibling `saved-audit-001`. Exact summary: `docs/basins/basins-summary.json`. Original profiles, timing amendment, source/cards and receipts are preserved.

Actual report SHA256 `2627121fb9ca01ad11e3f442955083f65aa4c53a418a625383e502f3eb8231f7`; receipt `e9e2cc6a06d696b8e22c0530741bec41048a6392d7aaa6ebfe9168f89d797177`; saved-audit report `1038f80d7ff4b77dd213204117623c48be782deaea68843e45713e5837da1b3a`. Full manifest is in `codex/basins-task28/final-manifest.json`.
