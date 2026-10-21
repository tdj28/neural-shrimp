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
