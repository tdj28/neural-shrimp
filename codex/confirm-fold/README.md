# Task 5: fresh fold-passage predictions

This task tests whether a local quadratic fold model forecasts full-state passage
counts in the pinned Qwen2.5-0.5B-Instruct expected-embedding feedback map. Target
A is the P2 fold at log10(T)=-0.140 on the alpha~0.067 branch. Target B is the P8
fold at alpha=0.0717 near log10(T)=-0.1033. Claude assigned these points before
our outcome generation; they did not determine the original passage mechanism.
The checkpoint, prompt, steering vector and independent float64 map remain the
same as Task 2. This is fresh parameter/orbit validation within one artificial
inference map, not a new-model, new-prompt, discrete-token or behavioral test.

## Setup before the prospective freeze

Producer curve arrays are permitted only as starting guesses. A uses index 6 of
`explore/llm-soft-loop/pod64/curve-p2-fold-plus/states.npy`, coordinate
(-0.13904681164063706, 0.06693879185356373); B uses index 0 of
`explore/llm-soft-loop/curve-p8-up-plus/states.npy`, coordinate
(-0.10334624213935097, 0.0717). The source metadata/arrays are hash-bound by the
setup receipt. No producer coefficients or new-target passage outcomes are
used. No Task 5 passages may run during setup.

The independent implementation locates each fold with augmented Newton,
classifies a unique real +1 multiplier, verifies primitive full-state return,
and records the complementary spectral radius (required <0.95). Nested AD of
the composed map G=F^k supplies a=l^T partial_p G and
b=(1/2)l^T D_x^2G[r,r], with ||r||=1 and l^T r=1; finite differences check the
AD values. A's unfolding parameter p is alpha; B's is log10(T). A failed gate
stops that target and is reported. Solver/setup failures are retained, never
quietly replaced. These are numerical validations, not interval certificates.

## Prospective passage plan

After successful setup and before passages, `build_design.py` will write the
complete numeric plan, save entrances and predicted counts, and bind source,
model and setup hashes. It and the runtime/analysis are reviewed, tested and
pushed to private origin before any outcome. Use eps=3e-6,1e-5,3e-5,1e-4 in
this order, all three entrances at each epsilon. Orient r,l by sign(b) so the
quadratic coefficient is positive, and move p by sign(a*b)*eps onto the ghost
side. The oriented quadratic map is u'=u+|a|eps+|b|u^2. U=0.1/|b|. Entrances
are x_c-U r, x_c, and x_c-U r+U w, where w is a fixed-seed unit vector in ker(l).
The added transverse kick has norm U; total displacement is not normalized.
Seed A=10501, B=10502 (never reserved seeds). Save the actual entrance arrays.
No burn-in, fitting, coefficient replacement or predictor tuning is allowed.
The endpoint is first strict u>U after a complete G step (2 or 8 model calls).

Primary success is |N_full-N_quadratic|<=1 for EVERY row, without rounding or
pooling failures away. Separately test |N_full-N_arctan|<=1 using unrounded
arctan values; claim both only if both conjunctions pass. Record signed and
absolute errors, counts in G and F steps, every missing/censored/failed row,
and pre-exit transverse norms. Finite-horizon failures do not pass. There are
24 planned rows; gate-failed targets retain 12 not-run rows. Count-cap choice
is based only on scalar predictions, before outcomes; it will be in the freeze.
No statistical p-value or population/general mechanism claim is supported by
these deterministic, selected cases. Success supports the stated finite-range
local forecasts; failure narrows that accuracy claim. A transverse failure is
a robustness result, not permission to shrink the kick.

## Local resource envelope and evidence

CPU only, nice 10, at most two workers with four BLAS/Torch threads each
(eight total, below the handoff cap of 12). OpenBLAS/NumExpr one thread.
Each worker has an 18 GiB address-space and 6-hour wall ceiling; setup starts
conservatively with Jacobian chunks of 16. Check available RAM and CPU before
launch. Full float64 RMSNorm/rotary/monodromy; no precision fallback. All workers
are source-snapshotted and record owned PIDs. Only their process groups may be
terminated. No paid compute or calls. Raw evidence stays in exclusive attempts
under `/data2/neural-shrimp-data/codex/task5-fresh-prediction/`, with compact
hash receipts in Git. Private Git protects code/receipts, not the bulky data.

Post-outcome changes require a dated research-log amendment (shared log helper;
other policy files are Claude-owned), a new pushed design, preserved originals
and explicit downgrade of affected rows. No public release or submission.
