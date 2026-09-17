# EXP-020: nonlinear first-bit readout, retained incomplete attempt

**TLDR:** In one tested near-edge case, the network still carried the earlier input after a second arrived: changing the observer raised its calculated readout accuracy from 50.3% to 99.7%. Under this noise model, the linear observer had missed accessible memory. The complete-study result is still withheld because 25 core-cycle cases failed a numerical cross-check. The simpler one-mode observer also recovers this example, so it does not establish a shrimp-specific advantage or a result for trained networks.

Status: **incomplete, primary withheld**. Frozen replay, the standalone physical-binding audit and independent summary arithmetic pass. The incomplete status is unchanged. All thresholds remain unchanged. This is an outcome-motivated saved-evidence regression over the same 24 phases per setting, with no fresh holdout and no new neural integration.

## What ran and what failed

The frozen source is `69ccc6e2c43236950dec246cfd3d66874bc89278`, and the complete plan SHA-256 is `bf5c6a13973d6449ef3be4f992b66b8d14e940b4db5c453c08f69007f04b5712`. All 432 reference-only classifier definitions were saved and reloaded before any finite observation vector was read. The 144 jobs covered both settings, all phases, the identity arm, four signed input combinations and windows ending at 2, 4 and 8 cycle periods.

Every one of the 4,704 mixture probability slots returned a record. Of these, 4,400 completed and qualified; 304 remained unavailable after the declared precision ladder. All 1,728 auxiliary fixed-linear reference calculations completed and qualified. There were no job timeouts or missing computation suffixes. The result is 119 complete cells and 25 incomplete cells, all in the core setting: 12 at 4T and 13 at 8T.

Each unavailable mixture probability retained an Arb enclosure. The failing lane was the separate binary64 nominal quadrature: 774 precision attempts failed its warning/convergence rules and 138 failed its agreement rule. Increasing the precision of the certificate did not repair the separately rounded nominal integrand. Saved diagnostics point to cancellation when subtracting two nearly equal `log1p(exp(...))` values for very small signals, followed by division by a small perpendicular response. This is a numerical diagnosis, not a new estimate or a waiver of the cross-check.

The summaries exclude entire incomplete cells. Consequently, their accepted mixture count is 3,896; another 504 complete probability records remain retained in those incomplete cells. Raw evidence preserves all returned records, including all failed precision attempts. All 72 near-edge cells qualify locally, while 47 of the 72 core cells qualify. The predetermined complete-study primary requires both settings, so it remains null.

## Audited descriptive near-edge output

These are analytic first-bit accuracies under the declared independent Gaussian observation noise, with an unknown balanced second bit and known cycle phase. They are conditional on deterministic saved neural responses and a calibrated noiseless reference baseline. The figures below are not noisy-trial hit rates or population estimates.

| Window | Original linear observer | Full reference-mixture observer | Difference |
|---|---:|---:|---:|
| 2T | 62.6449% | 62.8904% | +0.2455 percentage points |
| 4T | 55.0547% | 58.9815% | +3.9268 percentage points |
| 8T | 55.0816% | 56.9753% | +1.8937 percentage points |

These descriptive means use all 24 reused near-edge phases. They do not replace the withheld complete-study primary. At the motivating phase 16 and 4T, the original linear accuracy is 50.3095% and the new nominal accuracy is 99.6905%, with certified interval [99.6904%, 99.6941%]. That example was known to motivate the analysis and supplies no fresh validation.

At 4T and 8T, the declared one-mode nonlinear observer reproduces the full-mixture mean to the displayed precision. At 2T it performs worse, at 59.0364%. Thus the delayed recovery is consistent with nonlinear decoding of a simple overlapping two-input amplitude signal; it is not evidence that the shrimp geometry is uniquely responsible. Failure of the linear readout also does not demonstrate information erasure.

The retained 24-phase near-edge gain interval is [3.9253, 3.9327] percentage points. It is a numerical enclosure conditional on the qualified setting, not a confidence interval or a qualified complete-study result.

## Audit scope

The frozen verifier reproduced 119 complete cells and all 808 returned calculation slots from incomplete cells, covering all 4,704 mixture records without filling any missing suffix. Its controller author ran this replay, so it is implementation reproduction rather than independent numerical validation. It retained `complete=false`, `qualified=false` and a null primary.

A separate audit passed 190,103 checks of classifier geometry, physical reference/input bindings, exact rational projections, old linear-score replay, every returned probability record and gate logic. Its independent alternate-reference reconstruction differed by at most 1.11e-16. This auditor authored the new evaluator and older acquisition code but did not import the new evaluator or probability core for the checks. It did not implement a second arbitrary-precision integral.

Root's independent score-to-summary audit passed 11,276 arithmetic/census checks with zero recorded numerical difference. It checks aggregation and conservative interval arithmetic, not raw trajectories. The compact summary binds the audit reports and states these limits.

## Resources, preservation and next correction

The local run took 135.75 seconds with four single-thread CPU workers. Sampled combined owned RSS peaked at 920,154,112 bytes, counting shared pages once per process. All owned workers were joined and their original process identities were absent after cleanup. No GPU, paid compute or additional Pro call was used.

The 1,170 immutable artifacts total 161,170,315 bytes under `/data2/neural-shrimp-data/raw/EXP-020/attempt-001/`. The receipt SHA-256 is `f54d3e16c528867bdca88131b53a650e0d6efbeca03b7d761bddf6949006e2e5`. The compact [summary](EXP-020-summary.json) retains every cell and method-group slot, while full arrays and calculation journals remain external. This local raw directory is not an independent backup.

The successor will correct the nominal arithmetic, retain the original classifiers, priors, observation windows, complete cohort and numerical gates, and test tiny-signal synthetic cases before another private source freeze. It will receive a new experiment ID and raw attempt. EXP-020 remains incomplete; it will not be relabeled successful after the correction.
