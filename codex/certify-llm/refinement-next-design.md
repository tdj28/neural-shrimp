# Resumable refinement after the first before/after pilot

Design only, 2026-10-13. No new implementation, model calls or runtime promise.
Choose the next resource cap only after the current 2048-bit attempt measures
its prefix and one-phase costs. The numerical seed audit establishes usable
proposals, not how many corrections will be needed.

## First priority: reuse complete evaluations without changing their inputs

Use an append-only sequence of attempts. Each new run card names the exact
center tuple, source/map/asset hashes, precision, prefix-cache binding, parent
receipts and the finite list of missing phase evaluations. Never resume by
writing into an old attempt or alter its failed/incomplete status.

A reusable phase record must bind its **exact dyadic input vector**, the exact
map, a reviewed evaluator/source and all its dependencies, precision, complete
finite primal output, and a completion marker written after atomic output.
Verify the source receipt, output hash and receiver/cache contract. Matching a
phase number or a rounded decimal center is insufficient. Preserve all 2048-bit
center mantissas/exponents; no binary64 conversion on export or import.

Reuse has two useful cases:

- A completed `after` pass at `c_new` is already the next correction's complete
  `before` pass. It need not be evaluated again. Each additional correction
  then needs eight new one-phase evaluations, rather than another duplicated
  before/after pair. This is an operation count, not a wall-time estimate.
- If a timeout leaves only some phase images, a new attempt can reference those
  completed outputs and evaluate only the missing phases at the **same frozen
  center tuple**. It must still obtain all eight images before constructing
  another whole-tuple correction.

The current driver does not implement this partial-attempt import. A new,
explicit per-phase admission path is required: a nonzero-exit parent attempt
may contain a valid completed phase, but that does not make the parent a
completed setup. Do not weaken the existing successful-prefix/seed admission
checks indiscriminately. Any source-version change needs review for the same
exact map/evaluator semantics before reusing results across it.

Store image `F(c_i)` separately from residual `F(c_i)-c_(i+1)`. On reuse,
recompute the residual using the current next center. A cached image can only
be reused when its own input is identical; an old residual is not reusable if
its next center changed. A reuse manifest records each producer attempt,
phase/input/output hash, new phase position and recomputed residual witness.
Count original and resumed work separately, and retain cumulative resource use.

## Baseline: fixed shooting preconditioner

Continue with the already reviewed implicit `R` from exact dyadic `A_i,B`:
`c_new=mid(c-R H(c))`. Retain correction balls, then evaluate the full new tuple.
This baseline has minimal extra machinery and provides the first measured
convergence factor and residual enclosure floor. If it already produces a
suitable center, there is no reason to add a more complicated accelerator.
A smaller residual upper bound alone does not establish a root; compare upper
and lower norm bounds when claiming an actual decrease.

## Option A: limited-memory inverse-Broyden proposals

If fixed-preconditioner progress is inadequate, retain a short sequence of
complete center/residual pairs. Let `s=c_new-c_old` and let `y` be the difference
of chosen exact midpoint residual vectors. An inverse secant proposal is

```text
R_new = R_old + (s-R_old*y) * y^T / (y^T*y).
```

This is the inverse-Broyden second update. Represent it as the existing implicit
base `R` plus a bounded number of dyadic outer products `U V^T`, never as a
7168-square matrix. Use high-precision proposal arithmetic, choose and save
exact dyadic midpoint factors, and acknowledge that rounding those factors
need not preserve an exact secant identity. Applying the operator adds vector
dot products and updates to the existing implicit solve; it requires no new
model evaluation by itself.

Reject or skip a secant update when the denominator cannot be safely resolved,
when the residual difference is dominated by its enclosure uncertainty, or
when the proposed coefficients/step exceed declared finite conditioning and
step-size guards. Near-collinear history can make updates poor even if every
number is finite. Freeze bounded memory, reset rules and guards in the next
exploratory run card after the first measured correction; retain rejected
updates and their diagnostics. A failed proposal must not erase the previous
best fully evaluated tuple.

Every proposed tuple still needs a fresh complete residual pass. Broyden may
reduce the number of such passes, but the current data do not demonstrate that.
It is a proposal generator only. The existing shooting proof gate accepts the
specific cyclic `A_i,B` preconditioner, **not** an arbitrary low-rank-updated
operator. Later certification can use the refined centers with the original
admitted `A_i,B`, or separately reviewed replacement proposals. Reusing the
updated inverse inside a proof would require additional algebra/enclosures;
that extension is outside this design.

## Option B: bounded-history Anderson proposals

Apply Anderson acceleration to `T(c)=c-R H(c)`. A short history combines saved
`T(c_j)` values using coefficients whose sum is one and whose chosen residual
combination is small. Solve the small coefficient problem at adequate precision
and save its exact dyadic proposed coefficients, conditioning and truncation/
damping choices. New combinations still require fresh residual evaluations.

This avoids building an approximate inverse update, but nearly dependent
residual histories can produce large extrapolations. Unweighted phase/state
scales can also dominate the coefficient solve. Any scaling, damping, restart
or history limit must be specified and recorded, not silently adjusted to
hide a bad step. A favorable fitted residual combination is not the actual
residual of the combined nonlinear state. Keep this as an alternative to test
if needed, rather than enabling two unmeasured accelerators at once.

## Decision after the first pilot

First measure complete phase costs, actual residual decrease and enclosure
widths. Resume/reuse should precede any acceleration experiment. Continue the
fixed-preconditioner baseline if useful; otherwise trial one bounded proposal
method with all history and failures retained. If evaluations are limited by
arithmetic enclosure width, extrapolation cannot manufacture more information;
address precision or evaluator dependency instead.

Select proof-box radii from the newly measured **preconditioned** rigorous
residuals together with valid whole-box derivative bounds. Do not choose a
radius smaller than the inclusion calculation can support. Successful center
refinement, whether plain or accelerated, still leaves separate whole-box
root, primitive-period and attraction obligations.
