# Low-rank midpoint acceleration: design only

2026-10-13. No implementation, model evaluation, or alteration of the current
fixed-R path accompanies this note. The assumed gain of roughly 13 decimal
digits per pass is a hypothesis to assess from the completed refinement
attempt, not an observed convergence rate established here.

## Operator and candidate methods

Flatten the eight phase states into c in R^7168 and write
H_i(c)=F(c_i)-c_(i+1). Let R_0 be the approximate inverse of DH represented
implicitly by the exact dyadic A_i and B in `shooting_correction.correct`.
That routine computes d=-R_0 h using its two recurrences and proposes c+d.
Applying R_0 to an arbitrary vector therefore means negating that routine's
linear correction recurrence; no 7168-square matrix is needed.

**First candidate: limited-memory inverse Broyden.** For two fully evaluated
centers, set s=c_new-c_old and y=mid(H_new)-mid(H_old). Use the inverse-secant
update

\[
R_{k+1}=R_k+u_k w_k^T,\qquad
u_k=s_k-R_k y_k,\quad w_k=y_k/(y_k^T y_k).
\]

This enforces R_(k+1)y_k=s_k in exact arithmetic, with the next correction
given by -R_(k+1)H(c). Apply the baseline R_0 and then a short list of
rank-one terms. Storage is O(m*7168), and each low-rank application costs
O(m*7168), beyond the existing baseline recurrence. Start with a small fixed
history, such as m=4, and restart to R_0 when full; silently dropping an old
term does not preserve the same secant construction.

**Alternative: Anderson on the fixed-R iteration.** Define
f(c)=-R_0 mid(H(c)), G(c)=c+f(c). With recent difference columns DeltaC and
DeltaF, solve a small least-squares problem
\(\min_\gamma\|f_k-\Delta F\gamma\|_2\), then propose
\[
c_{k+1}=c_k+f_k-(\Delta C+\Delta F)\gamma.
\]
Use high-precision, rank-revealing QR with scaled columns, rather than normal
equations. A nearly collinear history must reduce rank or restart. This has
similar vector storage but introduces a least-squares conditioning decision;
inverse Broyden is the simpler first engineering pilot.

## Potential gain and limitations

If the fixed-R error is locally e_(k+1) approximately E e_k with
E=I-R_0 DH(c_star), then secants can remove error along directions actually
visited. A few dominant modes can make acceleration useful; a high-rank,
strongly nonnormal E or changing error directions can defeat a short history.
Neither method guarantees quadratic convergence here. Multiprecision matters:
binary64 secant arithmetic would discard the very corrections being sought.

At a sustained 13 digits per new eight-phase pass, moving from a 14-digit
residual to a 200-digit residual would require about 15 further passes.
That is an extrapolation, not a runtime or accuracy prediction. Point-rounding
width, center conditioning, and the eventual shooting inclusion inequality
also matter; small residual alone does not specify a usable phase-box radius.

## Required safeguards for a later pilot

- Keep all centers as exact dyadic midpoint proposals. Residual *midpoints*
  may guide proposals, but preserve the original full residual balls and
  compare fresh rigorous residual norms afterward. No inverse update becomes
  a proof preconditioner without separately verifying the final proof gates.
- Admit secants only from complete, same-map, same-precision, hash-bound
  eight-phase evaluations. Require a nonzero, well-resolved denominator and
  secant magnitude comfortably above the residual-enclosure uncertainty.
  Normalize secant vectors before dot products; record their scale, numerical
  rank/conditioning, update size, and every rejected update.
- Set a fixed history cap and correction-size cap in the run card. If the
  accelerated step is nonfinite, ill-conditioned, or excessively larger than
  the fixed-R step, restart and select the fixed-R proposal before evaluation.
  Optional damping must be predetermined and recorded, not hidden trial runs.
- Evaluate every admitted candidate with all eight fresh rigorous point
  images. Retain worsening and failed candidates. A worsening complete result
  can restart the next proposal from the best already fully evaluated center
  and its intact images. An incomplete attempt supplies no accepted secant.
  Do not add an unbudgeted line-search sequence of whole-map passes.
- Compare progress per full map pass with the retained fixed-R baseline.
  First use synthetic high-precision nonlinear systems with known roots,
  low- and full-rank inverse errors, nonnormal matrices, collinear secants,
  and realistic interval noise. Such tests validate implementation and
  expose failure modes; they cannot predict acceleration on Qwen.

The existing fixed-R continuation remains the reproducible baseline. A
separate reviewed proposal driver and bounded run card would be needed to
try either acceleration method. All resulting points would still require
fresh residual enclosures, full-box derivative bounds, shooting inclusion,
primitive-period separation, and attraction before any cycle certificate.

## Subsequent authorized pure inverse-secant kernel

A separate `inverse_secant.py` now implements the first option, with no model
or driver implementation and no edits to the fixed-R or evaluator sources.
Its source SHA-256 is
`01065e31a70b7b7b0b394609e2afef78ea96669455a72fd4c17546b94d8d914b`;
new tests SHA-256 is
`19c6f9f518ebad94618c734d24f94496407e68bc7426b0c58bf59d5e0e4387f8`.

API: construct `InverseSecant(A, B, max_history=4, signal_ratio=128,
update_norm_limit=1024, step_ratio_limit=4)`, then `update(old_centers,
old_residuals, new_centers, new_residuals, evidence=...)` and
`propose(centers, residuals, evidence=...)`. Inputs retain every phase and
coordinate. Evidence labels are snapshots of caller declarations; this pure
kernel never authenticates a model, receipt, or evaluation history.

The baseline action is the same cyclic recurrence with the sign R versus
correction -R made explicit. Secants use exact dyadic center differences and
full residual-difference balls. A power-of-two scale normalizes residual
midpoints before the denominator; midpoint rank-one coefficients remain exact
dyadics. Their arithmetic enclosures, scaled secant defect, supplied residual
radii, and every decision are retained. Zero center steps, zero/noisy secants,
excessive update norms, and exhausted history reset to the baseline. Oversized
or nonfinite accelerated steps also fall back to the baseline. The numerical
history is bounded; the full audit log grows with calls and needs persistence
by a future driver. Collinear secants are admitted when the noise/denominator
and norm gates pass: inverse Broyden has no least-squares rank requirement.

Independent of any target data, **32 kernel-plus-existing-correction tests
passed in 0.21 s**, CPU1. Tests exercised the exact cyclic inverse sign,
rank-one secant identity in a dyadic example, collinear history, explicit
restart, noisy/zero secants, 2^-2000 differences, enclosure retention,
update/step fallback, and runtime instrumentation confirming only phase-sized
matrix/vector allocations rather than a global shooting matrix.

An eight-step synthetic experiment at 256-bit arithmetic gave these final
**midpoint residual diagnostics**, using an explicitly more permissive
step-size ratio cap of16 for this comparison:

| Two-dimensional shooting system | Fixed R | Inverse secant |
| --- | ---: | ---: |
| Low-rank inverse error | 0 | 0 |
| Full-rank linear error | 7.6294e-5 | 6.6821e-83 |
| Nonnormal linear error | 1.5259e-4 | 1.5387e-80 |
| Quadratic map with known zero root | 4.4433e-6 | 9.5087e-12 |

Acceleration helped the latter three chosen tiny examples; it offered no gain
in the first, where fixed R already reached zero. These are illustrative
implementation diagnostics, not interval convergence-rate theorems or an
estimate for the 7168-dimensional Qwen shooting problem. Explicit synthetic
failure cases triggered the designed baseline fallbacks. No target evaluation,
three-correction replay, or accelerated target run has been implemented or
performed. Independent review remains required before a future driver.
