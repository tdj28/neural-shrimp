# Task19 scientific synthesis — draft awaiting optional stress evidence

14 October 2026. The independent primary trajectory and the complete 65-cell saved-data reduction are accepted. The fixed optional cells 04 and 13 have deliberately unfilled result sections below. This draft does not declare Task19 complete. It combines previously accepted reports and source/theory reviews; no new model, fit, statistical calculation, audit, array read or partial stress-output inspection was performed for this synthesis.

The accepted evidence supports a finite scalar description of repeated-token trajectories in many sampled cells. In the primary independent run, a primitive seven-token pattern persists throughout 2,000 recorded steps while two full-state twin estimates are positive and no full-state return is detected through lag 64. A fitted scalar return map reproduces the finite exponent approximately. Across the producer's 65 saved records, independent reduction reproduces the original **49/62** within-10% count and the disclosed outcome-informed amended **51/62** count; the separate purged-contiguous sensitivity method also gives **51/62**. Persistent outliers, rank deficiencies and interpretation limits remain part of those results.

## What was independently reproduced

| Evidence component | Independent work actually completed | Scope limit |
| --- | --- | --- |
| Primary cell | Own full-state float64 trajectory, two offsets, token record, finite return detector, sampled phase distances and all scalar phase fits | One fixed finite trajectory, after disclosure of producer summaries |
| All 65 cells | Own scalar reduction, symbolic periods, complete candidate/fold/rank/shape records and producer field comparisons on the original saved sequences | Full-state twins and state-period labels are producer inputs, not 65 own model replays |
| Optional cells 04 and 13 | Results unfilled in this draft | Accept only separately closed runs and their accepted saved-evidence reviews |

The primary setting is Qwen2.5-0.5B-Instruct, layer-12 hostility-versus-kindness steering, alpha 0.0655, log10 temperature 0.1899492494639028, full 896-state float64 ManualQwen and 151,936-token readout. The own source fixes 1,000 transient steps and 2,000 recorded steps, with absolute Euclidean twin offsets 1e-6 and 1e-9. Readout and state update share the same input clock. Original source, input and output identities remain in the component reports.

## Primary finite trajectory

The two own per-step twin estimates are **0.08887170325051022** and **0.08826492488488497**, compared with producer **0.09109385979024176** and **0.09065116593171463**. Their signed differences are -0.00222215653973154 and -0.0023862410468296624. The own mean used for scalar comparisons is **0.08856831406769759**. Both offsets avoid the distance floor; all four disjoint 500-step means and twenty running means are positive. These finite windows expose variation and are not confidence intervals for an infinite-time exponent.

All 2,000 own leading-token IDs agree exactly with the original producer record at shift zero. The primitive symbolic cycle is `[40, 0, 9454, 33360, 271, 61498, 271]`; it contains six distinct IDs. Full-state detection returns zero because none of its 64 tested lags passes the fixed tolerance. The saved lag-28 distance is 2.7029573844386632e-5. This establishes the recorded symbolic repetition and absence of a detected state return within that bound, not repetition forever or exclusion of every state period.

The minimum actual sampled cross-phase distance is **0.09832051568429243** after division by mean state norm. The centroid minimum is a different statistic, **0.1005363903625136**; the original plotting calculation cannot substitute one for the other. The minimum bounding-ball gap is **0.09287249153113472** in the same normalization. All 21 phase-pair witnesses are retained. These are separated finite sample clouds, not proved disjoint invariant bands.

All seven scalar phase rows remain visible. Phases 0–4 fail the value-fit qualification; phases 5 and 6 qualify. Phase 5 has one fitted quadratic **minimum**, whereas phase 6 has one fitted quadratic **maximum**. Neither the original stationary-point count nor its maximum interpretation may hide that distinction. All selected primary fits and folds have full numerical rank and no exact derivative zeros; other stages have rank warnings, as described below.

The original fixed phase-6 degree-8 fit has residual **0.0004445603435762713**, critical probability coordinate **0.24316809393046046**, curvature **-403.34799604888605**, and derivative signs positive then negative. Its exponent is **0.6342972864974116 per seven-step repetition**, or **0.0906138980710588 per map step**. The producer's corresponding residual, critical coordinate, curvature and repetition exponent are 0.00043156560189787504, 0.24316576129366224, -403.1347902799822 and 0.6470239572654982. These differences are retained; the independent irregular trajectory was not required to stay pointwise equal to the producer trajectory.

| Primary scalar procedure | Qualified phases entering estimate | Per-step estimate | Difference relative to own twin mean |
| --- | --- | ---: | ---: |
| Original lowest-residual power fit | 6 | 0.0906138980710588 | +2.30961154% |
| Amended random-CV phase median | 5, 6 | 0.08993033517144144 | +1.53781984% |
| Own purged-contiguous phase median | 5, 6 | 0.09007761433773956 | +1.70410861% |

The primary saved review completed 7,390 checks with no failures. It retained 41 nonzero arithmetic reconciliation differences, maximum 5.551115123125783e-17, within the prospective arithmetic tolerance. Those tiny differences concern saved witness/fold arithmetic and are distinct from own-versus-producer trajectory discrepancies. The completed source, state/twin chains, windows, finite lag ledger, token clock, all phase fits and band witnesses were authenticated; the saved review did not rerun the model, refit or resolve polynomial roots.

## Complete 65-cell reduction and the amendment

All 65 ordered records remain present. The 62 fitted records carry the declared finite `hidden chaos` label: no producer-detected state period, both saved twin estimates above 1e-3, and repeating recorded tokens. The other three are retained: cell 1 is irregular with symbolic period 1, no detected state period and two negative twins; cell 5 has symbolic/state period 1; cell 60 has symbolic period 7 and state period 56. The label is a finite operational classification, not an asymptotic theorem.

The historical procedure selects the smallest in-sample residual among the declared power-polynomial fits and uses residual <0.01 as its value qualification. Full rank is recorded, not silently imposed as an additional historical selection rule. The amended method selects a Chebyshev degree by fivefold random-CV value error, qualifies phases with CV relative error <0.01 and takes the median per-step exponent. Its literal floor and physical-coordinate imaginary-root convention are preserved for producer comparison. The separate own method uses purged contiguous folds, the unfloored logarithmic derivative and explicit full-rank qualification in selected folds and final fits. It is a sensitivity analysis and never replaces the historical two methods.

| Procedure | Within 10% of producer twin mean | Median absolute relative discrepancy | Cells with one stationary point in a selected fit | Cells with a fitted quadratic maximum |
| --- | ---: | ---: | ---: | ---: |
| Original power fit | **49/62** | **4.196231146%** | 59 | 36 |
| Outcome-informed amended Chebyshev | **51/62** | **3.658777310%** | 59 | 59 |
| Own purged-contiguous Chebyshev | **51/62** | **3.414727798%** | 59 | 58 |

The 10% comparison uses the inclusive absolute relative discrepancy against the saved producer twin mean. Shape counts mean at least one retained selected fit has the property, not that every phase is unimodal. In particular, the original 59 stationary-point count does not establish 59 quadratic maxima.

The amended method was designed after producer inspection of the bad fits/outlier and degree trials, including a rejected spline attempt. Declaring a common all-cell rule before the amended aggregate limits later discretion but does not remove this prior outcome dependence. The repository's adaptive protocol permits the adapted result when disclosed. The original 49/62, the amended 51/62 and the independent sensitivity 51/62 therefore remain separate exploratory descriptions on the same records, not a fresh confirmatory test.

The original thirteen outside-band cells are **4, 8, 13, 14, 16, 20, 24, 30, 34, 37, 38, 39 and 63**. Only cells 4 and 30 move inside under both alternatives. All eleven remaining outliers are retained. Cell 4 changes from -44.620763% to -7.054267% and -5.108400%; this substantial correction demonstrates method sensitivity. Cell 13 remains about +66% under every method: original scalar 0.007672242249660565 and amended/purged 0.007658074163085069 versus twin mean 0.004608228375251684. Its original value residual is only 9.303896945585786e-7. Small value error alone is therefore not a reliable bound on exponent error in these records. The full signed thirteen-cell table is in the accepted addendum.

All 1,128 phase rows, 12,400 candidate fits and 62,000 fold rows are retained, including **60 fold-rank warnings**. Cell 48 also has selected per-phase degree-24 rank-23 fits in amended phase 0 and purged phases 0 and 5; these are excluded from its aggregate-qualified phases 1, 2, 3 and 4. A cell-level empty aggregate-warning list must not be reported as full rank throughout the candidate or phase inventory.

The all-cell comparison retains all 4,140 original/amended producer leaves, with 2,070 nonzero numerical differences and no structural differences. The largest absolute field difference is 2.4693505525086312e-8 in cell 48's amended relative discrepancy: own -0.03685354704156941 versus producer -0.03685352234806388. The associated scalar difference is -7.053984063531971e-10. Counts and selections reproduce; arithmetic is not bit-exact. No rounding-to-zero or new post-outcome tolerance is used to erase these differences.

## What the scalar evidence does and does not establish

A fitted return relation with low value residual and an exponent near a full-state twin estimate is useful empirical evidence for a scalar approximation. It does not establish an injective observable, a one-dimensional invariant curve or smooth conjugacy of the full state dynamics. The simplest counterexample is the product map F(x,y)=(4x(1-x),4y(1-y)) with observable x. Its scalar return is exactly single-valued and unimodal, and the scalar exponent ln(2) equals the largest full-system exponent under the usual product invariant measure. Yet both full dimensions expand and x discards y. This analytic counterexample is not a target experiment.

For a genuine one-dimensional smooth coordinate change, finite-time exponent differences include a boundary term involving the derivative of the coordinate map. Its sublinear growth, and the appropriate nonsingularity conditions, are additional hypotheses. They are not supplied by a polynomial value-fit residual. Therefore low-residual phases need not be asserted to have identical finite exponents by theorem. The observed phase agreement and spread are measured evidence.

Random-fold value CV on adjacent trajectory pairs is not independent future-data testing, and degree selection using the same CV errors can be optimistic. Small value error does not control derivatives or log derivatives near a critical point. Purging addresses one dependence mechanism but does not prove derivative accuracy or create independent samples. A fitted quadratic maximum applies to the fitted curve on the sampled range; it does not establish exact global unimodality, universality, an invariant manifold or a behavioral mechanism. The accepted conclusion remains finite: repeated observed text can coexist with irregular full-state records and useful scalar fits, with measured exceptions.

## Optional cell 04 — explicitly unfilled

Fixed identity: original index **4**, log10 temperature **0.1799135096497498**, producer symbolic period **10**. This identity was fixed before independent stress outcomes; the outcome-informed method and prior producer disclosures remain acknowledged. The observed own symbolic period, including zero or a mismatch, must determine its fitting clock. The producer period is comparison metadata, not a replacement gate.

**No stress outcome, numerical value or pass/fail disposition is admitted into this draft.** Insert only after Root accepts the closed saved-evidence review: exact source/card/receipt/closure/review hashes; actual state/twin/token records and observed periods; all running/disjoint windows; qualified, unqualified and undefined scalar dispositions with every candidate/fold/rank warning; original and amended comparisons; all nonzero review differences and material producer disagreements. If the own record is a negative or mismatched result, retain it under that description without retuning.

## Optional cell 13 — explicitly unfilled

Fixed identity: original index **13**, log10 temperature **0.18478127233738384**, producer symbolic period **28**. Selection was fixed before independent stress outcomes. No shorter, longer or neighboring producer cell substitutes for this record.

**No stress outcome, numerical value or pass/fail disposition is admitted into this draft.** Fill the same accepted evidence fields as cell 04. The actual observed clock controls all phase fits; a zero detected symbolic period or an unqualified fit is a legitimate completed outcome, not permission to infer missing rows or retry. Distinguish sample-size limitations: a 2,000-step period-28 record has 71 or 72 observations per phase and 70 or 71 return pairs, not 72 pairs in every phase.

## Provenance, ownership and disposition

Both accepted components were independently computed after disclosure of producer summaries. The own primary source and fixed primary were reviewed and committed before execution, and the runtime read producer numerical result files only after own trajectory and fits were complete. This does not erase prior source/summary exposure. The all-cell analysis reduces the same original producer sequences; it is independent arithmetic, not fresh-data confirmation.

The coordinator authenticated the primary's 43 artifacts and captured source/closure, the saved review's actual authority, and the all-cell run's 148 artifacts/55,940,271 bytes/ten captured source blobs. I rely on those actual ownership records, not a fresh observation of execution. The all-cell outer waiter's birth was too short-lived to observe; original launcher/worker identities and tool reaping supply the disclosed finite closure basis, and the observer refusal remains preserved. Producer files were byte-stable when copied but original producer-writer termination was not independently attested. Current hash binding does not manufacture historical source, model-byte or lifecycle evidence.

The two component reports below are final for their assigned finite stages. This synthesis is **a draft with two pending optional sections**. It introduces no additional validation gates and requests no repeated primary or all-cell audit. Root handles scientific acceptance of future complete stress evidence; Inference alone owns process, repository adoption, private checkpoint and delivery actions. Task19 is not declared complete here.

## Source reports for this draft

- Primary report: `/tmp/task19-primary-scientific-report.md`, SHA256 `1536bebc078c702764f591d457c1af20567d82d721923938ad0967f09673983b`. Original result `trajectory-001/run/result.json` SHA256 `e7deb8295705d44a41c333602ba0a3a6751e6f72db5e293d82e09b62ee47938d`; accepted saved-review authority `7ba6b0d5519efff2c882ba5ab688afe6d56b968c50620918f65abf44031def11`.
- All-65 report: `/tmp/task19-addendum-scientific-report.md`, SHA256 `1b5b002387e37e51c63f7c12f26f64de0272ab3049c4126189866985928a8298`. Original result `saved-001/run/result.json` SHA256 `4c3f404b4514cac407e132a136b3408a892396de62081386729ec094035f03a3`; accepted closure `33daf8cdb1fbff919c520b07dcb7a4e7bd201318e11578c47c59bc6a03a9d011`.
- Original source/claim review: `/tmp/task18-task19-source-review-93dfb761.md`, SHA256 `5eafb53ad4c1638e00bc3c09bdd40ca13893de1c3410cbfbb0ed38a378911211`.
- Own reducer source review: `/tmp/task19-independent-source-review-001.md`, SHA256 `74e6d190331d744b6012615968b0889cc9d0d40d0deebc987f8d6cdf526c55ab`.
- Analytic counterexample: `/tmp/task19-scalar-factor-counterexample.md`, SHA256 `71585c4965e56578f0e0fe04a027aad87075ecf3218736cd8eb1c783bc6b25f0`.

The original results and authorities named above are under `/data2/neural-shrimp-data/codex/llm-return-map-crosscheck/`. The detailed component reports retain their complete individual input hashes, finite gates, signed disagreement tables, failures and source attribution. Their historical statements that later stages were pending are preserved as contemporaneous statements, not current reassignment or authorization.
