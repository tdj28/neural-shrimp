# ARP1 research log

## 2026-10-21: initial design, before acquisition

The observed visual difference is development evidence. Ten settings were
selected using existing periodic-orbit and Lyapunov records, before computing
any new activation score. Two calibration settings and eight test settings
are fixed in `plan.json`. No target outcomes have been generated. The pilot
has one start per setting; all claims remain exploratory and model-specific.

Local inventory: about 50 GiB RAM available and CPU load below four. GPU
availability is unknown because the restricted driver query failed; no GPU
is requested. One CPU process, four numerical threads, fixed batch 30,
one-hour acquisition limit. Benchmark and precision qualification precede
acquisition. No changes to the parent thread's experiment or resources.

## 2026-10-21 23:28 UTC: benchmark-001 complete

Source `c9094d9a89fa609650198174e43e8625c262c3c1`. Five metric tests passed.
All 30 numerical rows passed effective-precision checks: the minimum
central-difference error reduction on halving the step was 3.9968; the
largest finest-step relative error was below 4.364e-7. Exact-JVP and ordinary
forward primals differed relatively by 3.214e-15 or less. The fixed batch
took 0.2764 seconds per step or less, predicting about 288 seconds for the
1040-step acquisition. No target activation scores were inspected.

Raw benchmark receipt:
`/data2/neural-shrimp-data/audits/activation-regime-probe/benchmark-001/manifest.json`.
Proceed with the original ten settings and unchanged plan. The analysis
implementation records every layer and window, learns only from the two
calibration settings, and retains unresolved reference rows explicitly.

## 2026-10-21: collection-001 complete; analysis preparation

Source `b9b96dc910de14b9e21a4be1cc5aeb66856330d9`. The acquisition completed
all 1040 model steps and its source-binding checks, taking 381.419 seconds
including loading, repeated precision qualification, benchmark and hashing.
Raw output: `/data2/neural-shrimp-data/audits/activation-regime-probe/collection-001`.
No target arrays or reference labels had been inspected when amendment 01
was committed as `e8cc018a`. Its additional two-cycle comparator is now
implemented; original scores and reference gates are unchanged.

## 2026-10-21: analysis and local audit complete

Scoring source `78baf332e51654552557cfa22c686626b20d42de`.
`analysis-001` retained all ten rows. The two calibration references resolved
as P8 and finite-time chaotic; the eight test references resolved as P1, P2,
P6, P20 and four finite-time chaotic trajectories. No row was unresolved.
At 128 observations both dimensionality classifiers scored 7/8, roughness
scored 6/8, and both recurrence closed-set comparators scored 8/8. Recurrence
already reached 8/8 at 64 observations. Non-recurrence remains operationally
unresolved until an independent growth check; it is not a chaos certificate.

The chaotic table cell had lower effective dimensionality than P6 and P20.
The fixed-point control looked rough in the learned layout. These are retained
counterexamples to a universal order-versus-chaos rule. See `RESULTS.md` for
all budgets, errors and limits; no incremental gain over recurrence was shown.

The local audit in `audit-001.json` passed every reference row, all twenty
comparison counts, both recurrence baselines and the independent SVD/edge
checks. Six synthetic tests passed. Figures were inspected. `figures-002`
separates two overlapping labels and moves the key clear of the curves;
`analysis-001` and its original figure remain immutable. No scores changed.

Metadata correction after acquisition: the model configuration's optional
`_commit_hash` was absent, so raw `run.json` reports null in that field. The
actual loader explicitly pins revision
`7ae557604adf67be50417f59c2c2f167def9a775`; the pin is recovered from the
hash-bound acquisition source and recorded in the compact receipt. The
runner now records the loader's revision directly. No model or numerical
computation changed and the original metadata is preserved.
