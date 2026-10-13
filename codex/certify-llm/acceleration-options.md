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
