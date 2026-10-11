# Adapted Clark RNN local refinement, 9 October 2026 PDT

The first panel, source4285339d/execution3056d0f5, revealed an interior period5
recurrence component for N4 seed28171, four cells at delta.9525 and gains
8.32358–14.08452. The p3 and p4 components touch panel edges. Selection after
seeing the panel is disclosed: lowest period with a strictly interior component
of at least three cells and three finite-positive-growth neighbors. The p7
component is the next such component and a fixed comparison, not a generic
utility control. Old raw and source remain unchanged. This is exploratory.

One exact same J0/two-original-starts panel uses51×51 evenly spaced gains
6.5..17.5 and delta.90..1. A comparator uses original gain indices8..14 and
original delta indices14..17, a7×4 Cartesian patch. All starts are fresh at
every cell. N4 seed28171 only; no protected seeds/data, training, GPU or paid
resources. Bind the entire original panel inventory, byte-compare generated
J0/initial/tangent arrays, and pin the original profile run manifest.

Increase burn6000→24000, tangent alignment1000→2000 and measurement4096→8192;
retain the last513 full states and32 block-growth values per case. Maximum
recurrence period remains64, so longer periods and slow transients are unresolved.
Two4096-update halves must both exceed.002 per physical time to nominate finite
positive growth. This is not an asymptotic chaos proof. Original delta is the
Euler/leak parameter; no continuous-time interpretation is inferred.

Correct at most128 seed cases using unchanged core gates and full durable native
iterate/linear/eigen returns. In the p5 patch choose32 consensus-period5 cells,
then16 consensus-period10/20/40 cells; in p7 choose16 consensus-period>=3 cells.
For each group choose all cells if below its cap, otherwise integer positions
from evenly spaced linspace(0,count−1,cap) in sorted cell order. Correct both
starts. No stability ranking or result-dependent extra corrections. Save all
uncorrected proposals, disagreement and failures. Distinct cycle/window counts
require phase/sign matching; native full-state evidence is retained.

CPU/BLAS1, nice>=10. Internal300s, external330s+5cleanup. Worker448MiB plus64MiB
launcher/headroom; scientific raw300MiB plus1MiB launcher. Prewrite reservation
and1MiB final-metadata reserve inherited unchanged. Stop on any cap/failure;
retain partial, no automatic retry. Require at least1GiB fresh MemAvailable and
leave all foreign processes untouched. Owned process group must be absent.

Reuse the measured N16/81-cell profile, with no dimension discount. Source/core
and original profile/library identity must match; the adapted driver/card is
separately source-bound. Let a=measured batch seconds, b=period64 unit seconds.
Resource projection is2*(2629/81)*(34192/11096)*a+128*8*b+20<=300seconds.
Memory is profile_RSS+3*(513*5202*4*8)+64MiB<=512MiB. For the retained profile
these are243.111160seconds and385781440bytes. This is a disclosed extrapolation,
not a newly measured profile. The original larger dimension makes it conservative
in arithmetic work; fixed Python/storage overhead and shared load can still
cause a retained stop. Learning/dynamical favorability cannot affect this gate.

Exclusive output: `/data2/neural-shrimp-data/explore/clark-rnn/refine-p5-001`.
Command: `nice -n10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn_refinement.py --out /data2/neural-shrimp-data/explore/clark-rnn/refine-p5-001`.
The first answer is whether longer transients preserve the finite-state p5
component and whether it has resolved folds/flips or merely complicated basins.
A denser island alone is not a shrimp. Any continuation/certificate follows as
another disclosed bounded unit, with independent review before strong claims.
