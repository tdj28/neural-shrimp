# Verification ledger

**Current EXP-008 release status:** local verification passed, strict Linux
push and PR verification failed, and PR #7 remains unmerged. This is a private
working diagnostic draft, not a fully reproduced release. Historical checks
below retain their original scope. Two later Linux saved-feature refits agree
on all penalties and learned actions while retaining strict failures; see the
dated failure and diagnostic entries at the end.

| Component | Verification at initial freeze | Limit |
|---|---|---|
| Neural gradient, Hessian, training map | Local analytic-control and independent finite-difference tests | Float64 CPU only |
| Recurrence / cycle / multipliers | Exact logistic and Hénon controls; false-period and transient tests | Numerical verification, not interval proof or shrimp geometry |
| Utility intervention | Analytic quadratic-GD paired-path fixture | No initial empirical benefit claim |
| Manifest and raw receipts | Deterministic plan/control matching; actual CLI archive fixtures | Not a registry preregistration; no resume |
| RunPod archive and process watchdog | Offline integration and failure tests | Never called provider live; provisioning/teardown controller not implemented |
| Dependency environment | Locked dedicated local environment | GPU/image parity untested |
| Paper | LaTeX/bibliography compile plus six-page PDF visual inspection | Working draft; human publication review pending |
| GitHub | Private repository creation/push verified live via gh | Remains private |

At the initial freeze and after evidence packaging, all 52 local tests passed.
The implementation also passed both push and PR CI runs. The complete pilot
trial census and receipt hashes passed `tools/summarize_pilots.py`; see the
[results update](initial-results.md). No “verified-live RunPod” or GPU speedup
claim is implied.

## 2026-09-11 visual revision

- All 59 local tests pass, including seven direct algebra checks for the tied
  scalar two-cycle, its multiplier, prediction errors, and stability boundary.
- Both figure indexes and all ten per-figure receipts pass source/output hash
  checks. All 323 original trials retain their original receipt hashes and
  classifications; no new target experiment ran.
- Every finite neural/policy terminal test MSE is reconstructed from frozen
  inputs and saved final weights. All six empirical plates' plotted values
  and selections are re-derived under declared numeric tolerances.
- Same-environment regeneration is byte-identical for the empirical and
  conceptual figure families. Temporary image and derived-value corruptions
  are rejected; original artifacts are restored. See
  [the figure audit](empirical-figure-notes.md).
- Independent mathematical and design review verified the exact example and
  corrected three future-design issues: generic period information now belongs
  to the baseline; final scoring data are separate from online validation;
  finite null readouts do not establish exact invisibility.
- The revised PDF has 17 pages and ten figures. All 17 rendered pages were
  visually inspected; compilation has no LaTeX warnings or overfull/underfull
  boxes. Root review also inspected every empirical plot and representative
  final pages. Final PDF/source hashes are recorded in `paper/provenance.json`.

This extends artifact verification and the prospective research plan. It does
not establish neural shrimp geometry, effective recurrence control, net
compute savings, a GPU result, or paid-review approval.


## 2026-09-11 numerical correction: actual neural shrimp

The historical entries above describe earlier revisions. The current manuscript
is rebuilt around EXP-004 and EXP-005; its final page/figure inventory and file
hashes are in `paper/provenance.json`.

- All 85 local tests pass. The 26 Hopfield tests cover independent state and
  Jacobian formulas, integration order and event timing, variational dynamics,
  analytical Floquet controls, recurrence rejection, failure preservation,
  and the eight independently initialized control calls.
- EXP-004 retains all 475 receipt-bound artifacts, including 39,523 grid/transect
  settings and every selected refinement. All artifacts, coordinates, event
  sequences, recurrence classifications, and census were verified after copying.
- Three EXP-004 selected orbits pass independent full-state shooting, primitive
  peak counts, neutral-flow alignment, and transverse Floquet checks. The
  originally selected exterior fails its intended interpretation: 0/9/9 peaks.
  It is explicitly retained in the manuscript and original waveform plate.
- EXP-005 retains all 21 receipt-bound artifacts. Eight fresh common-initialization
  runs use two steps and a 10,000-unit settling horizon. Two controls certify
  nine-peak cycles; both distant controls have no short recurrence and positive
  finite-time exponents at both steps. Those are bounded irregularity checks,
  not proofs excluding arbitrarily long transients or high periods.
- Independent saved-data audits reconstruct neuron/activation moments and
  perturbation separations and check the monodromy eigenvalues and neutral mode.
  Very small transverse multiplier magnitudes/signs are numerically unresolved;
  the reported dominant modes are separated from unity. The original dense-output
  cycle-RMS scalar is receipt-bound, but its unsaved dense residuals cannot be
  reconstructed at 1e-9 accuracy from the coarser saved sample grid alone.
- Original EXP-001--003 raw data and all ten historical figure sets remain
  unchanged. Their verification still passes. The training/policy pilot remains
  null; it is not reinterpreted using the recurrent-network results.
- Exact pre-execution source revisions are preserved under
  `archive/EXP-004-source-v1` and `archive/EXP-005-source-v1`. Those tags were
  created after execution to retain history; the original pushed commits are
  the prospective records.
- No RunPod resource, paid review, or other paid compute was used. The new
  experiments' measured local wall times are 140.54 and 4.17 seconds.

See [EXP-004 results audit](EXP-004-results-audit.md),
[EXP-005 results audit](EXP-005-results-audit.md), and the explicitly limited
[claim ledger](claim-ledger.md). Full boundary continuation, basin-volume
estimation, phase-aware robustness, and task utility remain future experiments.

## 2026-09-11 EXP-006 utility and branch audit

This entry updates the research state after the historical entries above.
EXP-006 executes the initial waveform-utility and local branch-continuation
experiments. It does not complete the full boundary or coupled-control agenda.

- The complete source, design, and machine plan were pushed at
  `ddb568e6cd67ef2cfbdb45e05710cd86025ef548` before outcomes. All 136 tests and
  source-freeze CI passed. Integrated result/manuscript checks are defined in
  `.github/workflows/tests.yml`; exact run status is retained in GitHub.
  No public-release approval is asserted here.
- All 4,472 receipt-listed artifact byte counts and SHA-256 values match.
  The census contains exactly 4,320 complete trial-step records and 12/12
  eligible attempted anchors. No condition or failed anchor was replaced.
- The independent utility audit recomputed every saved activation waveform
  RMS at its recorded single global phase; maximum discrepancy was zero.
  It separately reconstructed the declared contrasts without importing the
  experiment aggregation helper. Numerical-step sensitivity and zero-dose
  floors remain reported rather than used to select preferred outcomes.
- The primary result is dose dependent: the largest-dose atlas comparison
  favors the atlas versus the declared symmetric local policy, both smallest
  doses favor local at all anchors, and the pooled local comparison is
  unresolved. The `tangent_plus` diagnostic has lower mean error than the atlas
  at every dose and remains an explicitly exploratory comparison.
- The immediate waveform window and eventual nine-peak assay have different
  horizons. The frozen example remains `anchor-08`, which favors the local
  comparator despite eventual nine-peak retention for all displayed arms.
  No high-error immediate trace is thereby certified an asymptotic waveform.
- A separate continuation audit reconstructed the two-dimensional Poincaré
  spectrum from saved matrices using a cross-product orthonormal section basis,
  independent of the implementation's SVD basis. It checked neutral-flow and
  phase residuals, spectra, stability flags, crossing candidates, reversals,
  and computational limits. No new target integration was performed.
- Four branches retain 196 node records, 122 marked unstable, and five failed
  correctors. These include overlapping branches and repeated seeds.
  Independent geometric checkpoint agreement is 27/28. The near-edge initial
  positive-gamma branch's node 40 fails the original closure gate:
  1.0590609546e-8 exceeds 1e-8. Its matching state and period do not convert it
  into a pass. No threshold was weakened.
- Compact continuation nodes do not store integrated endpoints. Their closure
  and primitive-event checks were audited against the recorded gates, not
  independently reintegrated. Crossing brackets are arclength-neighbor
  candidates, not exact gamma intervals at folds or a complete shrimp boundary.
- The receipt records 423.866 seconds for collection and analysis before final
  verification. Its 48 local-sensitivity solves total 26.627 seconds within that
  clock; the prior atlas cost 140.54 seconds separately. No RunPod, other paid
  compute, or paid Pro review was used.

See [EXP-006 utility results audit](EXP-006-utility-results-audit.md),
[EXP-006 continuation results audit](EXP-006-continuation-results-audit.md),
and the [EXP-006 receipt](../evidence/EXP-006/attempt-001/receipt.json).
The current final PDF inventory and artifact hashes belong in
`paper/provenance.json` after its final render-and-verify pass.

## 2026-09-11 EXP-007 prospective freeze

The new-start selector, simulator-preview policy comparison, separate scored
window, failure accounting and analysis are covered by local synthetic tests
and two local audits. The complete plan is compiled from existing atlas data
and the EXP006 manifest; no EXP007 target outcomes exist at this entry.
Runtime and scientific outcome verification follow in a separate dated entry.
The literature note also has independent analytical checks of printed
partial-refresh identities; these do not reproduce the preprint numerics.

## 2026-09-11 EXP-007 completed evidence and release checks

- The only attempt ran from the pushed source
  `ac098c3e3833daaf27c7dc2869d45e8b7085f948`, preserved under
  `archive/EXP-007-source-v1`. All 12 baseline attempts qualified; all 4,032
  previews and 2,784 target records completed. Frozen verification exited 0.
- The copied evidence passed exact checks for all 7,149 receipt-listed
  artifacts. A separate auditor with no experiment imports reconstructed
  504,320 RMS values in 9,600 metric blocks, with maximum discrepancy 4.44e-16,
  and independently reproduced all selections and bootstrap contrasts.
- A separate scientific release review reaggregated all 35 table means and
  30 displayed contrast/interval numbers directly from records. All primary
  pooled intervals span zero at both steps. Tangent and coordinate actions
  match exactly in 192/192 conditions and 384/384 target arrays.
- The new driver, selection and analytical preprint checks pass alongside the
  previous tests. No paid review, RunPod resource or other paid compute ran.
- Three new empirical figure families retain plotted arrays and exact source
  and image hashes. The display-only amendment enlarges the utility plate;
  its numerical arrays and sample selection do not change. Final PDF inventory,
  test count, render inspection and file hashes are in paper/provenance.json.
- GitHub's required tests check remains the final merge gate. The research
  repository and its complete raw evidence remain private.

- The initial source-freeze GitHub run (34657214248) failed because the existing
  PDF provenance still bound the preceding workflow file. This was a stale
  release-metadata hash, not a numerical assertion failure. Final provenance
  was refreshed from the actual revised files; the checker and thresholds
  remain unchanged. The original failed run remains in GitHub history.
- Final visual QA inspected all 27 pages at 125 dpi. After the utility-plate
  layout correction, page 23 was inspected again and exact PNG hashes confirmed
  the other 26 pages unchanged. The final LaTeX log has no warnings or box
  overflows. All 186 tests passed locally.

## 2026-09-13 EXP-008 completed evidence and local audits

- The complete source and plan were pushed at
  `3c5604497ec57b19d62fbb595da7b190102f4b33`, retained under
  `archive/EXP-008-source-v1`, before the exclusive attempt. The source-freeze
  local suite passed 249 tests. An isolated source-only CLI check used exactly the
  45-file bound source list, without host-checkout imports; see
  [preflight](EXP-008-preflight.md). Those are pre-execution checks, not a claim
  that a subsequent final release or CI job already passed.
- All 10,650 receipt-listed artifact sizes and hashes match, totaling
  728,224,123 bytes. All 186 old-input artifacts and 45 frozen source files
  match the manifest. All 24 baseline origins qualify, all 3,456 previews
  complete, and the exact 7,008-target census includes all 96 required shams.
  The frozen verifier completed successfully **locally**. No failed condition, origin or
  consumed attempt was replaced, reset or deleted.
- Independent local raw auditing reconstructs all 10,464 preview/target traces'
  inputs and 517,248 RMS values using a separate interpolation implementation.
  Maximum RMS discrepancy is 4.44e-16 under unchanged 1e-12/1e-14 tolerances.
  It independently checks all baseline and geometry feature rows, both shuffle
  matrices, local Gram matrices, sealed input/decision chronology, all policy
  choices, complete finite-action oracles and paired-block aggregation.
- Independent local weighted SVD verifies fold exclusion, training-only scaling,
  all 180 validation-loss entries and all chosen penalties. **A separate
  coefficient comparison fails and remains failed:** eight baseline and eight
  shuffled coefficients exceed rtol=1e-7/atol=1e-9; geometry coefficients pass.
  Maximum coefficient discrepancy is 7.52e-8. Ill-conditioned designs explain
  the sensitivity, but do not authorize a relaxed threshold. Maximum test
  prediction discrepancy is 1.46e-9, and none of the 288 learned action choices
  changes in that local audit. Local frozen same-algorithm replay and independent coefficient agreement
  are distinct checks; the former does not convert the latter into a pass.
- The secondary prediction-MSE mean improves by 17.17%; the primary action
  contrast is 0.000857628 [-0.007701467, 0.006545434], unresolved. Direct
  minimum-preview selection has 41.30% lower mean error than the geometric
  learner using the same eighteen forecasts. The finite oracle cancels from
  paired regret differences and is not independent supporting evidence.
- Across all 3,504 paired numerical-step outcomes, the median absolute RMSE
  change is 2.91e-7 and the maximum is 0.476851. Selected-policy pairs are
  steadier (maximum 0.001000), but this does not certify every candidate or
  training label. Negative regression predictions, outside-crop masks, all
  step-sensitive cases and the unfavorable final test block remain reported.
- Three new empirical figure families provide fourteen panels, lossless plotted
  arrays and complete input/output hash receipts. All three final figure PDFs
  were rendered with Poppler and visually inspected. The six-panel utility
  plate is landscape with standalone labels of at least 9.5 pt. Its full
  local hash/census/summary/array check passed; cross-platform image-byte equality is
  not required. The frozen first example is retained: every policy selects
  candidate 4, and the waveform plate explicitly labels the coincident curves.
- Collection and summary took 162.352 seconds, excluding final verification;
  23.770 sensitivity seconds and 0.079 fit/seal seconds are within that clock.
  The older map cost 140.54 seconds separately. No paid compute, RunPod resource
  or paid review was used. Final manuscript inventory and hashes belong in
  `paper/provenance.json`; GitHub's required tests remain the final merge gate.

Read the [raw/feature/outcome audit](EXP-008-results-audit.md),
[independent model audit](EXP-008-model-results-audit.md),
[receipt](../evidence/EXP-008/attempt-001/receipt.json), and
[figure provenance](../paper/figures/hopfield-information-figure-provenance.json).
Repeat saved-evidence checks without integrating a new trajectory:

```sh
python tools/run_hopfield_information.py verify --output evidence/EXP-008/attempt-001
python tools/audit_hopfield_information.py --out /tmp/EXP-008-independent-check.json
python tools/audit_information_models.py evidence/EXP-008/attempt-001 \
  --out /tmp/EXP-008-model-independent.json
python tools/plot_hopfield_information.py evidence/EXP-008/attempt-001 \
  --destination paper/figures --check
```

Use the locked project environment and keep new audit reports outside the
immutable attempt. Read the retained failed SVD coefficient comparisons in
the model report even when the reporting command completes successfully.

## 2026-09-13 local / 14 September UTC: Linux release checks fail

Release candidate `1d12ebeda0400c179e54dd36e647a835f7e0d91d` has two retained
strict Linux failures. These are additional observations after the local
checks above, not revisions to frozen experiment data or tolerances.

| Linux run | Stopping point | Absolute discrepancy | Unchanged combined check |
|---|---|---:|---|
| PR 34803148459 | `development fit.models.baseline.coefficients[1]` | 3.68269315e-8 | rtol=1e-7, atol=1e-9 |
| Push 34803133930 | Shared feature `[34]` | 1.39784018e-14 | rtol=1e-12, atol=1e-14 |

The [PR log](EXP-008-linux-pr-failure.txt) reaches model-fit comparison after
the raw metric loops. The [push log](EXP-008-linux-push-failure.txt) stops
earlier in feature reconstruction and therefore does not establish completion
of all its waveform/feature checks. At this first-failure stage neither run
measured a complete Linux decision comparison; the later diagnostic below
supplies a separate, bounded result. Original logs and stopping points remain
preserved.

## 2026-09-13 local / 14 September UTC: Linux saved-feature diagnostics complete

At diagnostic commit `bb4ecd25cf5c025064e7bd200b6acd898953d0e5`,
[PR run 34804593644](EXP-008-platform-linux-pr.json) and
[push run 34804589708](EXP-008-platform-linux-push.json) both pass exact raw
evidence bindings and enumerate 176,256 shared-feature values, 5,184 candidate
predictions and 288 learned action decisions. Their
[retrieval provenance](EXP-008-platform-linux-provenance.json) is retained.

| Measured quantity | PR 34804593644 | Push 34804589708 |
|---|---:|---:|
| Shared-feature failures at unchanged tolerance | 2 | 0 |
| Baseline coefficient failures | 8 | 10 |
| Shuffled-geometry coefficient failures | 9 | 10 |
| Geometry coefficient failures | 0 | 0 |
| Changed selected penalties | 0 / 3 | 0 / 3 |
| Changed learned actions in saved-feature refit | 0 / 288 | 0 / 288 |
| Maximum prediction difference | 1.3874261184e-9 | 1.9615957192e-9 |
| Saved-model score replay at rtol=1e-12, atol=1e-14 | Pass | Pass |

The minimum saved best-to-second score gap is 3.7010270022e-6. Every
condition's gap exceeds twice its observed maximum refitted-score discrepancy;
action agreement is also checked exactly. No new score-difference tolerance
is introduced. The PR shared-feature failures are `local_projected_rmse` for
`development-10`, phase 675, radius 0.016, candidates 16 and 17.

These refits use **saved feature rows**, including when independently
reconstructed feature values fail tolerance. Feature reconstruction uses saved
preview metrics and certificates; the diagnostic does not repeat waveform
metric checks or integrations. It does not establish action agreement from
a completely regenerated input pipeline. To reproduce this separate audit:

```sh
python tools/audit_information_platform.py evidence/EXP-008/attempt-001 \
  --out /tmp/EXP-008-platform-audit.json
```

The strict frozen gate has **not passed**. This diagnostic runs after failures
without replacing, relaxing or bypassing it. General portability remains
unresolved, PR #7 remains **unmerged**, and the project remains a private
working diagnostic draft. Numerical portability takes priority over the
proposed four-preview experiment. Original Linux logs, local results,
thresholds and raw evidence remain unchanged.

## 2026-09-14: executed server numerical follow-up

| Capability | Verification and scope |
|---|---|
| Shared-server policy and private checkpoints | Hardware queried live; external data root created; private SSH pushes verified by remote SHA. GitHub API authentication for the review PR is still pending. |
| Arithmetic diagnostic | Pushed source `b3cf51bb88df75eccec4b19e0f7a15976b3a9b21`; complete saved-input diagnostic, all 3,456 directions and 195 ridge objectives. Full output remains external. No original failure threshold changed. |
| Saved-trajectory regeneration | Pushed source `d276782c229aa33e2819ec89f15df4e23c7f7834`; all 48 contexts/3,456 previews/7,008 targets completed, exact before/after evidence bindings passed. Exit 1 intentionally retains 19 comparison failures; all choices and aggregate scores agree. |
| EXP-009 | Same pushed source; all 2,776 planned integrations complete; built-in saved-array/complete-score verification passed. Raw data and original inputs remain hash-bound. This verifies execution, not universal numerical convergence. |
| Combined code | 325 tests passed in 21.97 seconds before the integrated freeze; independent local review findings were fixed before outcomes. |

Reports: [arithmetic](EXP-008-arithmetic-results-audit.md),
[regeneration](EXP-008-regeneration-results-audit.md),
[EXP-009](EXP-009-results-audit.md). The original EXP-008 strict verifier
remains unchanged and failed; these separate checks do not replace it.

## 2026-09-14 local / 2026-09-15 UTC: complete preview qualification

EXP-010 source `98632899646f3eefa292b8a9f4c1b5123ae37ba0` was privately
pushed and verified before execution. All 348 project tests passed before
the freeze. The attempt completes all 6,944 required traces and its complete
saved-array verification; the producer exits zero. Independent checks cover
all 13,896 receipt artifacts, 3,463 old inputs, fourteen source files, every
trace/metric/window and 3,504 waveform comparisons. The separate ranking
audit passes 56,275 exact checks, with all 192 complete orders unchanged.
All 3,456 scalar/waveform cases and sixteen independent sentinels pass their
declared criteria. No raw evidence is added to Git and no paid resource is
launched. See the [EXP-010 audit](EXP-010-results-audit.md).

## 2026-09-14 local / 2026-09-15 UTC: complete shortlist feasibility

EXP-011 source `64712453153ade24d2b22cf811f7a83811304de4` was privately
pushed and its remote SHA verified before the first fit. All 385 repository
tests passed before the freeze. The new feature adapter, nested model and
runner passed separate independent pre-fit reviews; the full plan was
reconstructed without fitting. Original EXP-009/010 source bindings remain
intact.

The exclusive external attempt completes all 24 origin folds, 96 model
recipes and 192 contexts. Every SVD/QR score, selected-penalty and ordered-list
gate passes. Both full label-resolution recipes preserve their chosen
penalties, shortlists and actions. Built-in replay reproduces all 96 model
packets; a separate invocation of the frozen verifier exits zero, verifies
all 346 receipt-listed artifacts and repeats all 96 recipes, access/seal,
label and aggregate checks. Its output is
`verified_saved_shortlist_evidence`, with `all_complete: true`.

The receipt lists 94,844,269 artifact bytes excluding itself and records
114.067794487 seconds through execution, built-in replay and receipt sealing.
Preflight and the separate verification are outside that elapsed time.
Four bounded CPU workers and one numerical thread per worker were used;
no new preview integration or paid resource was required. Full raw evidence
is in `/data2/neural-shrimp-data/raw/EXP-011/attempt-001/`.

Independent evidence reconstruction passes 76,598 checks with zero failures,
including every raw/source/input binding, 2,208 inner-fold score recipes,
27,648 held predictions, 1,536 complete rankings, 1,920 policy scores and
all training/reveal events. It uses an independent elementwise prediction
reduction from saved coefficients, without producer imports or refitting.
Its largest held-prediction discrepancy is 8.9928065e-15. The last prediction
seal precedes the first held-label reveal by 1.364514597 seconds; every
four-label winner seal precedes full-eighteen oracle access. These conclusions
combine persisted logs, hashes and the reviewed source; they are not an
OS-level proof about unlogged file access.

The independent statistical audit passes 17,742 checks, exactly reconstructing
all 380 bootstrap intervals and the complete per-origin/aggregate trees.
It also reproduces all solver/resolution rankings, choices and cost summaries.
Its original sandbox-only process claim was preserved and explicitly corrected
without changing any statistical value. Root's actual host inventory at
01:44:26 UTC confirms every one of the 24 worker PIDs and producer 1716452
absent. A separate host inventory at 01:47:43 UTC likewise confirms all 6,944
EXP-010 workers and producer 1649165 absent, closing the previously uncertain
namespace scope of that earlier audit. Both dated host reports are hash-bound
in the corresponding compact summaries; no unrelated process was changed.

See the [results audit](EXP-011-results-audit.md) and
[compact summary](EXP-011-summary.json) for exact source, manifest, receipt
and audit bindings. Numerical completion does not establish general geometry
utility: signed axes have lower pooled preview error than either learner,
and the observed shared-minus-geometry interval spans zero. The original
EXP-008 strict Linux gate remains failed and unchanged.
