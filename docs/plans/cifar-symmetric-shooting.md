# CIFAR symmetric two-phase Newton pilot

Exploratory amendment,10 October2026. Six dampedLSMR steps reduced closure
by3.49% and then stagnated; do not repeat that scheme. For scalar-rate GD,
the two-phase residual(F(x)-y,F(y)-x) has the exact symmetric Jacobian
[[Jx,-I],[-I,Jy]]. At y=F(x), eliminatingdy recovers the original
(JyJx-I)dx=-R equation. Symmetric MINRES avoids storing a long Arnoldi basis
and may converge differently; no speedup or better condition is assumed.

Prepare one fixed400-iteration local direction from the last accepted consumed
CIFAR state. Validate native symmetry with deterministic directions and an
independent finite-difference action before solving; retain all vectors.
The true eliminated forcing must be <=.5 and direction must descend the
original closure merit before at most13native backtracking trials. Preserve
all attempts. Closure1e-8, phase separation1e-7 and actual Armijo gates stay
unchanged; no orbit accepted or spectrum computed by this pilot.

No fresh dataset, holdout, rate or model selection. One local CPU thread,
nice10; no GPU or paid work. Exact counters, measured timing gate, immutable
raw receipts and owned process cleanup required before target launch.
This source-only kernel checkpoint does not launch a target.
