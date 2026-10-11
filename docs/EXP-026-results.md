# EXP-026: input history survives the weak readout

**TLDR:** The earlier input is still distinguishable in the late neural
activity in both periodic and irregular regimes, given the exact starting
context, the forward responses and the specified measurement precision.
EXP-025's weak shared-readout scores did not establish erasure. Stable rhythms
can carry input history even when a shared phase-unaware decoder fails.

This is a post-run exploratory diagnostic of existing EXP-025 trajectories.
The new analysis was privately frozen before calculation at
`b4d6406f6e746bf3384c017013ab0aff61882262`; it is not an independent experiment
or learned held-out decoder. No new neural trajectories were generated.

The primary observation is the same 18 activation samples at time55..60,
input amplitude .1, with independent Gaussian measurement noise of SD .001
per observation coordinate. For each starting context, the observer knows the
four possible two-input responses. The second input is a balanced nuisance.
Both regimes receive identical privileged access.

At all 26 settings and all 128 evaluation contexts per setting, the
constructive lower bound on first-input recovery exceeds 95%. That includes
all 1,536 periodic contexts, all 1,664 irregular-supported contexts and the
128 contexts at the unresolved setting. The periodic mean lower bound is
greater than .99999999999; the irregular mean reaches 1 at displayed floating
precision. We report the conservative >=95% per-context result rather than
claiming literally zero mathematical error.

The lower bound is achieved by either a context-specific affine classifier,
a nearest-four-template classifier with a Gaussian error union bound, or the
chance baseline. A complementary upper bound uses convexity of total variation
between the two Gaussian mixtures. These bounds constrain optimal binary
accuracy under the specified model; they are not ODE interval proofs. Both
inherited numerical streams pass the .002 point-mean discrepancy gate across
all amplitudes, windows, targets and noise levels. The complete secondary
noise/strength results remain in the machine summary.

An independent scalar-erf implementation reconstructs all primary bounds and
fixed subsets of every secondary condition, and checks every reported bound
mean and high/low fraction. Six synthetic tests cover exact chance, exact
two-Gaussian recovery, nonlinear XOR, invariances and input validation.
The independent audit and parent hash verification passed.

This result distinguishes surviving input information from practical
accessibility. The observer has exact context and forward-response knowledge;
those are unavailable to the ordinary EXP-025 readout. We have not isolated
phase as the sole missing variable, shown a learned generalizing decoder, or
proved that either regime is useful under uncertain initial state. Those
claims would require a matched-access intervention. The immediate next test
is whether an observer calibrated to the exact context becomes more fragile
outside the periodic region when the true starting state is slightly perturbed.

- [Protocol and derivation](EXP-026-protocol.md).
- [Compact summary and audit receipt](EXP-026-summary.json).
- Raw: `/data2/neural-shrimp-data/raw/EXP-026/attempt-001`.
- Audit: `/data2/neural-shrimp-data/audits/EXP-026/validation-001`.

All artifacts are retained. No paid compute or review was used. Git protects
code and compact receipts; the sibling raw directory is separate local storage.
