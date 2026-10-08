# Tiny-LM calibration outcome: independent audit

**TLDR.** Nonlinear plane-coherence features did not pass the declared
prediction gate in these six drifting tiny-transformer trajectories. The
challenge has sparse correlated events; it does not establish that useful
geometric predictors are impossible in other language models or settings.

**Verdict:** the saved evidence and failed gate are verified. No source or
outcome blocker remains for reporting this narrow negative calibration result.
No assessment, validation or policy stage was run by this reviewer.

Producer result commit: `9f98bb6df2fdf6fb08773715fd04839f91492fa3`.
Execution source: `b9e5b8c3973e8bd776402b0cb3d8d61609bb71f1`.
Independent source review: `39ba17219075c92cd33038b032501bb60da6fd25`.
Audit source: `a6503e1dbafe5eb0e8ab9a3941a8f045e23e44ed`.
Compact audit receipt: `tiny-lm-calibration-outcome-audit.json` in this folder.
Raw audit output: `/data2/neural-shrimp-data/audits/tiny-lm-calibration-outcome/`.

The CPU1/nice10 audit completed in0.618s under its external30s limit.
All221 collection files/307,259,570 bytes match the exact inventory and hashes.
Both launchers bind the execution source and source review, exit successfully,
and match their retained log hashes. Both imported analysis source hashes and
its exact source commit match. The six fixed seed/factor cases each completed
800 updates without a numerical guard or wall stop, retaining801 finite states
and800 feature/CE rows:4,806 states and4,800 updates in total.

Labels were independently regenerated from saved CE. Every case has753
uncensored labels, with31 initial unavailable rows and16 right-censored final
rows. Fit/tune/test counts are352/184/184 percase. Training-only imputation and
scaling, selected penalty/threshold, selected tuning score, saved predictions,
per-trajectory and seed-pooled Brier/AP, and episode gates replay correctly.
Saved leave-one-seed-out Brier/AP were also checked. No gradient/training run
or logistic optimization refit was performed; alternate-penalty fit coefficients
and leave-one-seed-out coefficients were not independently reconstructed.

| Test endpoint | B2 | B3 |
|---|---:|---:|
| Eventful trajectory28152/factor2.5 Brier | 0.1890333260 | 0.1913465058 |
| Same trajectory AP | 0.6934098942 | 0.6942446838 |
| Seed28152 mean Brier across three rates | 0.0630111341 | 0.0637822191 |
| Seed28152 AP pooled across three rates | 0.6851260690 | 0.6476791098 |

Thus B3 slightly improves AP **within the eventful trajectory**, while its
Brier and seed-pooled AP are worse. These quantities must not be conflated.
Only that trajectory contributes late positive labels:49 overlapping horizons,
grouped into3 operational episodes. Seed28151 has zero late positive labels
and undefined AP. Both fail the required5 episodes perseed, and the aggregate
gate is false. B0 also outperforms B2/B3 on the eventful trajectory (Brier
0.1735573508, AP0.9111334758). These are descriptive development comparisons,
not independent-event significance tests or a broad impossibility result.

The first audit stopped at an unnecessarily strict2e-12 reconstruction
comparison; its stdout/stderr are preserved. A dated, committed amendment
changed only the auditor's approximate B3-from-B2 reconstruction tolerance to
absolute1e-9, leaving producer data, scientific thresholds and gate rules
unchanged. The maximum difference across the completed audit is
`1.8051338201985345e-11`; all availability masks match exactly. Individual dot
products and matrix Gram products have different floating reductions, amplified
by nonlinear coherence and tiny history variation near collinearity. The result
is numerical agreement to the stated tolerance, not bit-identical reconstruction.

Collection manifest SHA256:
`a8a4166cb0b03c5f8b72b0a8039dfb99acc0e4ad1fc12a53d4d3371f0b0d4bff`.
Saved analysis SHA256:
`70cf98f6eeec6aab03c31cc319394e363c751553319763c324eab94cc3283d96`.
All original artifacts remain unchanged and read-only. Assessment text/seeds,
validation outcomes, reserved seeds and policy experiments remain untouched.
