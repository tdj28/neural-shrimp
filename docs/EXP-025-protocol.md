# EXP-025: what changes across the periodic window?

Prospective exploratory pilot, 23 September 2026. No new target outcomes were
available when this design was written. This is a private Git freeze, not a
formal preregistration or a population-wide confirmatory study.

## Research-priority amendment

The user's central question is what shrimp-associated regimes mean for neural
behavior. EXP-024 established input interaction at selected periodic anchors;
it did not compare behavior across the periodic-window boundaries. This study
supersedes the proposed T/2-only follow-up as the immediate priority. Historical
protocols and negative results remain intact. The branch is based on the
existing research branch at 1a7eb64b because the required scientific tools and
evidence are not on protected main; it is a dependent research change.

The competing predictions are: (P) repeating regimes improve usable memory;
(I) irregular regimes improve usable memory; (H) effects are small or differ
across the window. Input mixing and repeatability provide additional behavioral
evidence. The protocol permits a clear rejection of a blanket recommendation
to seek or avoid the periodic region. It cannot establish creativity, brain
criticality, trained-network behavior, or a shrimp-specific computational
advantage over every other periodic window.

## Fixed population and autonomous geometry

The manifest contains 26 coordinates: 13 fixed gamma values on each of the
alpha=1.064 and alpha=1.0661333333333334 transects. Prior EXP-005 controls and
EXP-006 stability crossings informed these coordinates. Every coordinate is
run, even if its regime classification is surprising. Each starts separately
at (0.1,0.1,0.1) and (0.2,-0.1,0.3), with independently initialized RK4 steps
0.01 and 0.005, burn 10,000, measurement 2,000. The existing full-state
recurrence detector searches primitive peak periods through 32 using four
repeats. Both step sizes must agree on a positive candidate, followed by the
existing independent DOP853 full-state shooting and Floquet certificate, for
the start to be labeled periodic. No candidate at both steps and FTLE >.005
at both supports finite-horizon irregularity, not a chaos theorem.

A coordinate receives a common periodic/irregular label only when both starts
agree in regime class. Otherwise it remains mixed/unresolved. Different
certified periods between starts remain visible in the census; class agreement
does not prove a unique attractor or large basin. No further adaptive burn or
replacement is allowed. These transects cross observed operating-regime
changes; they do not localize exact bifurcations or certify a complete shrimp.
Unstable continued orbits are not substituted for actual operating states.

The entire geometry census is saved before functional acquisition. Eligible
comparisons are all adjacent fixed coordinates on the same transect whose
classes are periodic and irregular-supported, in either order. Noncrossing
neighbors and all unresolved points retain their functional results. Adjacent
coordinates can be separated by intervening regimes; claims refer to the
sampled endpoints, not an unobserved single transition between them.

## Equal access to inputs, time and observations

For each start, continue the fine-step terminal state for 2,000 units at
dt=.005 and save a tape every .05. A preseeded random sample of 96 distinct
tape indices defines 32 training and 64 held-out contexts; the same indices
are used at every coordinate, with separate seed streams for the two starts.
The resulting 64 training and 128 held-out contexts receive all four balanced
input-sign combinations. Training and held-out indices are disjoint; their
conditional time samples share the same finite autonomous tapes. They are not
independent networks, basins, or uniformly sampled cycle phases.

Both inputs add +/- amplitude to neuron 1's potential at external times
0 and 20. Amplitudes .01 and .1 are both retained. Every task clone begins at
its exact saved context, at both RK4 steps .0025 and .00125. No phase-null input,
period-normalized timing, phase decoder, or baseline subtraction is provided.
All three tanh activations at integer times 20..25 and 55..60 are observed.
The t=20 state is after the second pulse. Complete integer-sampled states
0..60 are saved. This tests isolated two-pulse computation, not a continually
driven stationary regime; inputs can perturb the autonomous attractor.

For each parameter/amplitude/window, a single shared readout is fit to all
training contexts and evaluated on all held-out contexts. Targets are first
sign, second sign and their product. An affine readout is primary. A secondary
quadratic feature readout includes all upper-triangular feature products.
Both use fixed mean-centered ridge regression, penalty 1e-4 in mean-squared
loss units, with an unpenalized intercept and no feature standardization or
hyperparameter selection. R2 is 1 minus held-out MSE because targets are exactly
balanced +/-1. Negative values are retained. Sign accuracy is descriptive.
Quadratic results test a second observation class; a bad affine decoder alone
does not establish erased information. No optimal-decoder claim is permitted.

Separately, every context has two unforced clones separated initially by
1e-6*(1,1,1)/sqrt(3), integrated to t=60 at both steps. Their activation
separation measures same-time repeatability, not phase-corrected contraction
or task performance under continuous noise. No stochastic measurement noise is
added in this pilot. Reliability and memory may change without either being
caused uniquely by shrimp geometry.

## Primary decision and secondary evidence

The primary endpoint is held-out first-sign R2 at amplitude .1, t=55..60,
affine readout. At every eligible adjacent crossing compute periodic R2 minus
irregular R2. Equal-weight the crossings. Bootstrap held-out context indices
jointly across coordinates with seed 2026092305 and 4,000 draws for a 95%
conditional interval. This interval conditions on the fitted training data,
two finite tapes and fixed parameter points; it is not uncertainty over neural
architectures or the true bifurcation location. It is descriptive exploratory
evidence; the practical .05 threshold is chosen before outcomes.

P is supported only if both transects contribute crossings, mean difference
>.05, interval lower bound >0, and at least 75% of pair differences are positive.
I uses the symmetric negative rules. Otherwise report inconclusive/heterogeneous.
Both step sizes must give the same decision. Secondary amplitudes, windows,
targets, quadratic readouts and repeatability are fully reported and cannot
rescue or redefine the primary endpoint. They may motivate a separately frozen
replication. A single counterexample can challenge a universal recommendation;
neither sign establishes that all shrimps should be sought or avoided.

## Numerical qualification, resources and retention

Maximum tanh-activation difference across both RK4 methods, all saved task
traces and both amplitudes must be <=1e-4 for every coordinate. All readout R2
differences must be <=.002. Additionally, for context 0 from each start, all
four signs at amplitude .1 are checked against DOP853 at rtol2e-12,
atol2e-14,max_step=.025; activation error must be <=1e-4. No failed coordinate
is silently removed: any numerical failure withholds the primary decision.
Independent checks are sampled, not a full second implementation of all tasks.
Existing integrators and new assay arithmetic are tested on artificial
alpha=gamma=0 and synthetic inputs before the target freeze.

Six CPU threads run point tasks with BLAS/OpenMP each capped at one thread.
Historical EXP-005 took 4.17 seconds for four control points; short-trace
throughput is checked on an artificial fixture before launch. Wall ceiling is
1,800 seconds with a 1,900-second owned-process external termination backstop.
Memory expectation <2 GiB; live resource checks precede launch. The GPU was
100% occupied with 373 MiB free at preparation; existing CPU kernels are
qualified and prior GPU replacement failed validation. No paid resources.

Exclusive raw directory: ../neural-shrimp-data/raw/EXP-025/attempt-001.
Preserve geometry cells, tapes, task arrays, fitted readouts, partial memmaps,
errors, exact source commit, source hashes, compiler/library metadata and a
SHA-256/byte inventory. Regular private pushes protect code and compact receipts;
external raw data are local evidence, not Git-backed-up raw data. Saved-data
hash verification and independent arithmetic review precede interpretation.
