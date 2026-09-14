# EXP-008 numerical arithmetic diagnosis

Prepared 2026-09-14 before executing this new diagnostic on the sealed evidence.
This is post-run exploratory numerical development: the EXP-008 outcomes and
Linux failures are already known. It is neither a new experiment nor a repair
to the original gate. Commit and push this method, executable diagnostic and
synthetic tests before running the diagnostic on research data. Record the
exact diagnostic source commit, report hash and any incomplete attempt.

## Question and bounded scope

Two existing mechanisms may explain numerical replay fragility:

1. `phase_projected_gram` already forms projected residual columns. It does
   **not** subtract two large Gram matrices. However, `base_features` computes
   the local predicted error as `sqrt(max(0, delta @ Gram @ delta))`. Near a
   weak-sensitivity direction, this directional quadratic form can cancel
   large positive and negative terms after the two-column Gram is rounded.
2. Ridge fitting forms weighted normal equations. Correlated standardized
   columns and a small ridge penalty can make coefficient coordinates
   sensitive to small floating-point changes, even when fitted scores and
   selected actions scarcely change.

These are hypotheses from source inspection and existing diagnostics. The
new report will quantify them instead of assuming that a numerically different
method is more accurate just because it passes a historical comparison.

The executable is `tools/diagnose_information_arithmetic.py`. It reads only the
original completed EXP-008 attempt, validates its frozen receipt, manifest,
all artifact hashes, chronological seals and development-data hash, and
creates a new report outside the attempt. No trajectories, target outcomes,
feature definitions, regularization grid, original coefficients, saved
decisions or verifier thresholds are changed. Test target labels are checked
by the common binding audit but do not enter any fit or new endpoint analysis.

## Directional quadratic forms

For flattened sensitivity columns S, phase vector v, and parameter delta d,
define P = S - v (v^T S)/(v^T v), with P = S when v is zero. For m flattened
samples, the mathematical squared local error is

`q = d^T (P^T P / m) d = ||P d||^2 / m`.

Use every saved sensitivity context and all 3,456 saved feature rows, including
duplicate candidate directions. Retain each original saved delta, avoiding
confounding a changed eigenvector with arithmetic on a fixed direction.
Compare, in a fixed sequence:

- the unchanged frozen Gram and binary64 quadratic form;
- the direct squared norm of the frozen binary64 projected columns times d;
- extended-precision evaluation of the already-rounded binary64 Gram;
- extended-precision evaluation of the frozen binary64 projected columns;
- an extended-precision projection and directional norm from the saved inputs.

This separates quadratic-form evaluation, rounded Gram formation, and
projection differences. Record their absolute errors, all original feature
tolerance failures, negative unclamped quadratic forms, and the cancellation
ratio `sum(abs(d_i G_ij d_j)) / q_reference`. For an exactly zero reference,
the ratio is undefined and recorded as null. Keep frozen eigenvalues and
saved-versus-recomputed direction comparisons for each context.

The reference uses NumPy long double only on a host where its mantissa is
wider than binary64; otherwise the diagnostic fails explicitly. Record both
precisions. This reference gives additional arithmetic precision on exact
saved binary64 tensors, not ground truth for the underlying differential
equation, exact real arithmetic or a proof of cross-platform portability.

## Ridge arithmetic

For each learner, every original leave-one-development-anchor-out fold, the
final development fit, and every original penalty, use the **same frozen
standardizer and intercept** for three solvers:

- the exact existing weighted normal-equation arithmetic;
- weighted-design SVD with shrinkage `s / (s^2 + lambda)`;
- augmented least squares `[A; sqrt(lambda) I] beta = [b; 0]` with SciPy's
  pivoted QR (`gelsy`).

Here `A = sqrt(w) Z` and `b = sqrt(w) (y - intercept)`. All algorithms optimize
the same real-valued ridge objective. The square-root weight construction is
itself rounded; its differences from the original nonsquare-root reductions
remain part of the documented floating-point comparison. Reusing the
standardizer isolates solver changes; it does not evaluate alternative
centering or scaling rules. An earlier independent model audit already
compares a separate group-wise standardizer.

Record all coefficients, CV errors, selected penalties, singular values,
the estimated condition of the regularized normal matrix and augmented
design, pairwise coefficient and prediction differences, and extended-
precision objective/stationarity values on the common weighted design.
Evaluate stationarity as `A^T (A beta - b) + lambda beta` without forming a
normal matrix. The norm divided by lambda is a conservative coefficient-error
bound for an exact gradient of the regularized objective; this reported
floating-point estimate is diagnostic, not a certified interval. Compare
saved coefficients at the original selected penalty using the unchanged
original tolerance, and retain every failure. Report each solver's selected
penalty separately. No solver is selected according to held-out task utility.

This report does not re-evaluate selected-action utility. Existing reports
already record unchanged learned actions on saved feature rows. If altered
features or a solver are adopted later, that requires a separately frozen
amendment and fresh validation, preserving these development observations.

## Execution and interpretation

Use local CPU with one BLAS/OpenMP thread. The job processes one sensitivity
context at a time and only small ridge designs; no GPU or rented resource is
required. The command requires a clean working tree and an exact expected
source commit. The operator verifies that this commit was pushed before
execution and verifies the actual server-side ref. The CLI also requires a
local origin remote-tracking ref containing the source commit, which is an
additional guard rather than proof that the remote still exists. Existing
output paths and outputs anywhere inside the repository or attempt are rejected.
The report is reserved with launch metadata before computation so an
interruption leaves a visible incomplete attempt; the same invocation fills
its final report after completion or a caught error.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  .venv/bin/python tools/diagnose_information_arithmetic.py \
  evidence/EXP-008/attempt-001 --expected-source FULL_PUSHED_COMMIT \
  --out /data2/neural-shrimp-data/audits/EXP-008-arithmetic/attempt-001/report.json
```

Retain a failed attempt and report its error before changing the method. An
exit code of zero means only that this diagnostic finished, regardless of
frozen comparison failures or method disagreements. No old gate becomes
green by running this tool. Report measured limitations, changed penalties,
solver disagreements and negative quadratic forms without cherry-picking
the method or platform closest to the saved result.
