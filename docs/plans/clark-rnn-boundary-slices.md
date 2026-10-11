# Native p5 RNN boundary slices (adapted exploratory)

After refine-p5-001, the p5 band is not a closed island at gain17.5. Instead of
more plotting, identify its first stability boundary in each delta direction
at exactly three gains: local grid indices8,16,35 (8.26,10.02,14.2). Use the
middle sorted consensus-p5 delta cell at each gain, original start0. Full raw
manifest and input/batch hashes bind every starting point; no new seeds/data.
The original two-start and odd-symmetry full-state evidence remains unchanged.

Correct the p5 orbit at delta increments.0005, at most64 steps each direction,
starting fresh from the central corrected orbit for each direction. At each
point retain complete native cycle/Jacobians/monodromy, corrections and spectrum.
Seven Newton evaluations, max correction.1cycle-scale, closure<=1e-11 and
proper-divisor1 separation>1e-6. Stop at first unstable closed orbit, correction
failure, delta domain(.85,1), or step limit. Do not select a later nicer boundary.

On correction failure try one +1 augmented root from the last closed state.
On first instability try the closest of real+1/-1 only if the saved eigenvalue
is real to1e-8 and within.7. No NS substitution. Unknowns(x4,delta,q4), equations
(F^5-x,(M-signI)q,(q^Tq-1)/2). Twelve Newton evaluations, delta within.005 of
that proposal and full-state shift within.2scale; every returned derivative and
linear solve retained. Analytic first/second derivatives directly through tanh;
independent centered-difference controls are tests only. No finite differences
in target derivatives. Native model/map and original source stay unchanged.

Qualification requires augmented residual<1e-9, multiplier error<1e-7, isolated
critical eigenvalue distance>1e-4, attracting complement radius<1-1e-5, left
residual<1e-7, primitive separation and nonzero fold(a_delta,c) or total along-
branch flip derivative (>1e-5 in absolute value). This is floating numerical
support, not a theorem. No cubic flip criticality or all-phase claim is made.
No 2D curve connection, full shrimp or continuous-flow conclusion follows.

Before target input loading, measure ten synthetic N4,p5 native returns and ten
augmented returns, including writes. Resource-only projection4000*max(native)
+96*max(augmented)+10<=60seconds. This covers the384*7+3*7 ordinary corrector
calls and<=6*13 augmented/qualification evaluations with headroom. It is an upper-cost screen,
not a favorable scientific-outcome gate. Any failed profile stops before target.

CPU/BLAS1,nice10,60s internal75+5external,448MiB worker+64MiBheadroom;64MiB raw
plus1MiB launcher,5000 attempted native/augmented call cap. Same prewrite reserve,
cooperative TERM, full ownership/immutable manifests. Fresh host check; foreign
jobs untouched. Stop and retain any failure, no retry in this unit.

Command: `nice -n10 /data2/neural-shrimp/.venv/bin/python tools/launch_clark_rnn_boundaries.py --out /data2/neural-shrimp-data/explore/clark-rnn/p5-boundary-slices-001`.
If these slices show a consistent fold/flip pair, the next unit can continue the
actual codimension-one curves in(g,delta), rather than infer their shape from
attractor basins. Any interval target will be separately source-reviewed.
