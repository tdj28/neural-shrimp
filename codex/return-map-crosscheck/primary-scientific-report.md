# Task19 primary trajectory: independent finite-record scientific report

14 October 2026. The mandatory single-cell trajectory and its saved-evidence review are complete and accepted by Root. The 65-cell addendum and optional stress work remain pending; this report does not declare all of Task19 complete.

The independent full-state float64 run reproduces the main finite observation: the leading-token output repeats with primitive symbolic period seven throughout 2,000 recorded steps, while both fixed-offset twin-growth estimates are positive and the finite full-state return detector finds no period through lag 64. A phase-six scalar return fit has one quadratic maximum and an exponent close to the independently measured twin mean. These are numerical observations on a finite trajectory, not an asymptotic chaos or one-dimensional conjugacy proof.

## Experiment and accepted evidence

The fixed cell is Qwen2.5-0.5B-Instruct with the hostility-versus-kindness vector at layer 12, alpha `0.0655`, log10 temperature `0.1899492494639028`, dimension 896 and the complete 151,936-token readout. The independent ManualQwen implementation used a fixed start, 1,000 transient steps and 2,000 recorded steps. The two absolute Euclidean twin offsets were `1e-6` and `1e-9`, with seed zero and renormalization after every recorded return. Each row binds the readout of its input to the full-state image of that same input. There were 3,000 native calls and 7,000 row forwards.

Scientific source commit `be709c95e93e1e075d9ced10f27e8453a301ba75` and plan SHA `402d4b107a8a9a7b5832d0824fa40208e28a13caeb1f7c0f8d1e127b9cafe8c4` are retained in the original receipt. The worker completed in 1,221.013 seconds, with peak RSS 5,339,984 KiB; the original supervisor recorded 1,223.286 seconds, exit zero and no timeout. The coordinator authenticated all 43 original artifacts, 146,104,898 bytes and the ten captured source files, and verified that the original waiter, launcher, worker, groups and sessions were absent. This report relies on that closed evidence, not a fresh process observation.

The separate saved review completed 7,390 checks with zero failed checks in 0.446 seconds, peak RSS 381,740 KiB. It authenticated the complete original inventory and input authority; checked all 8,001 event positions, 3,000 timing associations, complete input/return/renormalization chains, every growth row, both means, all four disjoint and twenty running windows, all 64 finite lags, all token counts and cycle IDs, all seven phase dispositions, selected fit/rank/fold associations, all 21 band-pair witnesses and the saved producer comparisons. It did not rerun the model, refit polynomials, resolve roots or repeat the exhaustive nearest-pair search. Those numerical operations inherit the reviewed implementation and original captured execution.

The saved arithmetic review retained 41 nonzero comparison records, with maximum absolute difference `5.551115123125783e-17`, within the prospective `atol=5e-15`, `rtol=5e-13` reconciliation rule. Five concern nearest-witness arithmetic; 36 concern equal-fold MSE summaries, including the repeated checks when the fixed seven-phase and actual-period fits coincide. They are not rounded away or described as bit-exact agreement. These rounding differences are distinct from the larger, legitimate own-versus-producer discrepancies below.

## Full-state growth and repeating output

| Absolute twin offset | Independent finite exponent per step | Producer finite exponent | Own minus producer |
|---|---:|---:|---:|
| `1e-6` | 0.08887170325051022 | 0.09109385979024176 | -0.00222215653973154 |
| `1e-9` | 0.08826492488488497 | 0.09065116593171463 | -0.0023862410468296624 |

The independent mean used for scalar comparisons is `0.08856831406769759`. Neither offset used the distance floor. All four disjoint 500-step means are positive:

| Recorded interval | Offset `1e-6` | Offset `1e-9` |
|---|---:|---:|
| 0–500 | 0.0806667043348698 | 0.0782215997639610 |
| 500–1,000 | 0.0899803222746576 | 0.0899916213155740 |
| 1,000–1,500 | 0.0936972242604203 | 0.0936978379639670 |
| 1,500–2,000 | 0.0911425621320922 | 0.0911486404960376 |

The twenty saved running means also remain positive. These windows expose finite-horizon variation; they are not confidence intervals or an extrapolation to infinite time.

All 2,000 leading-token IDs agree exactly with the original producer trace, in the same phase. The primitive symbolic cycle is `[40, 0, 9454, 33360, 271, 61498, 271]`; only cyclic shift zero matches. There are six distinct IDs because token 271 appears twice in the seven-step cycle. Original decoded labels remain quoted producer metadata; this report did not load a tokenizer. The record establishes repetition for the observed 2,000 steps, not repetition without end.

The full-state detector returns zero because none of its 64 last-state lag distances is below `1e-8`. Its scale is `max(1, norm(last_state))=1`; this is distinct from the mean-state normalization used for band distances. The retained lag-28 distance is `2.7029573844386632e-5`, and the last-400-state radius-based diameter is `0.8617232373070048`. Failure to detect a finite return is not a proof of nonperiodicity at every period. Combined with the two positive finite twin estimates, it supports the declared finite repeating-text/irregular-state classification.

## Seven finite phase clouds

The minimum distance between actually sampled states in different phases, divided by the mean state norm, is `0.09832051568429243`. The minimum centroid distance is separately `0.1005363903625136`; substituting the latter for the former would overstate what the producer's centroid statistic measured. The common mean-state norm is `0.20672021613542882`. The minimum bounding-ball gap is `0.09287249153113472` in the same normalization.

All 21 phase-pair rows and their original nearest-witness indices were retained. The smallest sampled separation is between phases three and five, at original indices 808 and 803, with absolute distance `0.020324838252803748`. The saved reviewer checked every witness distance, centroid, radius, bounding-ball gap and aggregate. Completeness of the exhaustive search inherits the accepted own runtime. Positive separation here describes the recorded finite clouds; it does not prove separated invariant sets or a global band organization.

## Scalar return fits and every phase disposition

The scalar coordinate is leading-token probability sampled every seven steps. Phase indices are zero-based. Phases 0–4 have 286 observations and 285 consecutive return pairs each; phases 5–6 have 285 observations and 284 pairs each. All seven rows are fitted, all final fits have full numerical rank, and every retained fit has zero exact derivative zeros. Thus no fitted exponent is undefined in this run. The unqualified rows remain present rather than being converted to missing or favorable outcomes.

| Phase | Original degree-8 relative residual | Original exponent per repetition | Amended random CV residual (selected degree) | Purged contiguous CV residual (selected degree) | Value-fit disposition |
|---|---:|---:|---:|---:|---|
| 0 | 0.164524659 | -0.029013405 | 0.166994977 (20) | 0.169689109 (8) | Unqualified |
| 1 | 0.250112011 | 0.801659360 | 0.236781139 (22) | 0.236274005 (18) | Unqualified |
| 2 | 0.246842169 | 0.816408592 | 0.237614347 (24) | 0.238673014 (16) | Unqualified |
| 3 | 0.247503911 | 0.830033086 | 0.235795472 (24) | 0.236622061 (18) | Unqualified |
| 4 | 0.049621003 | 0.411055212 | 0.047446373 (22) | 0.048805710 (12) | Unqualified |
| 5 | 0.003303550 | 0.625207593 | 0.003344874 (10) | 0.003285811 (10) | Qualified; one minimum |
| 6 | 0.000444560 | 0.634297286 | 0.000457988 (10) | 0.000450360 (8) | Qualified; one maximum |

The original lowest-residual selection is phase six, also the fixed disclosed primary. Its precise residual is `0.0004445603435762713`; its single interior critical probability coordinate is `0.24316809393046046`, curvature `-403.34799604888605`, with derivative signs positive on the left and negative on the right. It is a fitted quadratic maximum. Its exponent is `0.6342972864974116` per seven-step repetition, or `0.0906138980710588` per map step, `+2.30961154%` relative to the independent twin mean.

The producer phase-six values were residual `0.00043156560189787504`, critical coordinate `0.24316576129366224`, curvature `-403.1347902799822` and exponent `0.6470239572654982` per repetition. All six field comparisons, including these nonzero differences, remain in the completed result. The independent record is not presented as numerical identity with the producer trajectory.

Phase five must not be described as another fitted maximum: its original critical coordinate is `0.005895302730639718`, curvature `+34457.2409158088`, and derivative signs negative on the left and positive on the right. Both degree-ten CV-selected fits also have one quadratic minimum. A count of one stationary point alone does not distinguish its shape from phase six.

The amended and purged methods both retain phases five and six. The historical amended value qualification uses CV relative error below 0.01. The own purged qualification additionally requires full fold and final-fit rank; it is explicitly an extra qualification. All selected folds qualify here. The saved original floor-based and unfloored logarithmic exponents agree for the selected fits because no derivative is zero; neither branch is silently substituted for the other.

| Saved scalar comparison | Included phases | Per-step estimate | Relative difference from own twin mean |
|---|---|---:|---:|
| Original minimum-residual degree-8 fit | 6 | 0.0906138980710588 | +2.30961154% |
| Amended random-CV median | 5, 6 | 0.08993033517144144 | +1.53781984% |
| Own purged-contiguous median | 5, 6 | 0.09007761433773956 | +1.70410861% |

The corresponding selected-phase spreads are zero, `0.0007780091340101325` and `0.0010725674666063473` per step. The phases 0–4 fail the value threshold even where a fitted logarithmic derivative happens to be positive. They supply no qualified scalar-exponent reproduction and are not included in these medians.

## Scope, exposure and remaining work

This is an exploratory independent computation after exposure to the producer's claims, including the original 49/62 and outcome-informed amended 51/62 cross-cell summaries. The amended degree/CV procedure arose after producer outcome inspection; using it on this independent primary trajectory does not retroactively make the producer amendment a fresh confirmatory design. The own scientific source and fixed primary were committed and reviewed before the own run. The own runtime read producer numerical summaries only after its own trajectory and fits completed; prior source/summary exposure remains disclosed.

A small scalar value-fit error and agreement of fitted logarithmic derivatives with a full-state twin estimate do not establish an injective observable, an invariant curve or smooth conjugacy. Even an exact scalar factor and equal largest exponent can coexist with additional expanding state dimensions. This report supports a useful finite scalar description at the qualified phases, not a proof that the entire recurrent dynamics are one-dimensional. Random-fold value prediction is also not an independent bound on derivative or logarithmic-derivative error; the purged result is a separately retained finite sensitivity check.

Full-state trajectories in an irregular regime need not remain pointwise equal between two implementations, and such equality was not required. The exact observed token agreement, positive finite twin-growth estimates, finite band distances and qualified return-fit comparisons are the reproduced observations. No claim is made about asymptotic chaos, perpetual text repetition, global basin structure, native autoregressive generation, behavioral usefulness or compute savings.

The original 65-cell saved-data addendum and any optional longer-horizon/stress work are not covered by this primary report and remain pending. No model call, fit, root solve, NPZ load or statistical rerun was performed while writing this report; only completed JSON records and authenticated receipts were read.

## Exact report inputs

All paths below are under `/data2/neural-shrimp-data/codex/llm-return-map-crosscheck/`.

| Original record | SHA-256 |
|---|---|
| `trajectory-001/receipt.json` | `b0a72180d59cb626ccaec4a4c58ba5077f014e37c951c9994a78c9f5978fb44e` |
| `trajectory-001/run/result.json` | `e7deb8295705d44a41c333602ba0a3a6751e6f72db5e293d82e09b62ee47938d` |
| `trajectory-001/run/status.json` | `86ad53edb240c47132f25f870103cf8481e672d020329c2489e8f84873084b55` |
| `trajectory-001/run/fixed-seven-phase-fits.json` | `0b9096790ad59e7f04df52451ac67356c4035d0ce5b95817322b135c724ccfb1` |
| `trajectory-001-closure-001/report.json` | `f30cae9c4ed82b6ed4c581995a46a1cf5b2825a3ee3549948f5853caca91a3b9` |
| `trajectory-001-review-001/report.json` | `e94af3ef46c33bdf2f8078ffbde551528f26438fcc02e27df3c2930b5cad76c4` |
| `trajectory-001-review-001-authority/report.json` | `7ba6b0d5519efff2c882ba5ab688afe6d56b968c50620918f65abf44031def11` |

The audit ran under source commit `81471174cb94094bc03a40e3c2c6efb9f85bef30`. Its sealed two-file output manifest is `257ea7984ff62ee7303fcd0bedafc1429925eb8a062924c458f466f2c84a42cf`; its original supervisor and worker were reaped with exit zero and separately authenticated as absent. Root's actual-evidence acceptance is separate from all historical preparation or pending-review fields.
