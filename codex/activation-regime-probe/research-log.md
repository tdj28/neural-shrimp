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
