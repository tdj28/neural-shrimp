# Lower S1R gap: next bounded decision — 2026-10-08 UTC

**Decision: first audit six fixed states at the gap; do not launch another
continuation or pseudo-arclength run yet.** The smallest useful question is
whether the saved failed Newton correction solves the stated local equation,
and whether its local parameter geometry supports a turning-point diagnostic.
A larger iteration budget alone would not distinguish these explanations.
This is design only. Reading existing scalar/array values and checking their
identities introduced no model, derivative, norm, spectrum or correction run.

## What the saved failure says

The independently audited path reaches h=log10(eta_h)=0.992 at fixed output
log-rate0.6271428571428572. Macro54 needed subdivisions; macro55 retained no
accepted subpath. The five target controls are0.990,0.991,0.9915,0.99175 and
0.991875. Calls469–473 fail the first phase with `line_search_stalled` after
4,7,7,6 and3 Newton iterations. The source uses the exact dense Newton equation
for R(theta,h)=F_h(F_h(theta))−theta and strict decrease of the unscaled
infinity-norm residual over13 trial factors1 through2^-12. These are line-search
failures, not rejection for instability, remote-state distance, exhausted
Newton iterations, missing second-phase agreement or a resource deadline.

Recorded phase-zero cond(M−I) values rise from171.673 at h=.994 to247.618 at
.993,343.835 at.9924999999999999 and831.314 at.992 (opposite phase791.047 at the
last point). The last complete saved spectrum includes the distinct real
values1.0055189561104383 and1.0508888817716477. Mode identity was not tracked;
the spectral radius alone is not a fold diagnostic. Terminal relative Newton
corrections in the five failed calls are0.138402,0.171168,0.086779,0.048692 and
0.236645, while their saved accumulated seed distances remain small. This
supports investigating conditioning/curvature, but proves neither a singular
Jacobian nor loss of the periodic branch. Nonconvergence below the last orbit
does not bracket a fold.

Competing explanations remain: a genuine local turning region; secant error or
strong curvature outside the fixed-parameter corrector's basin; a poorly solved
linear system or derivative inconsistency at the failed iterate; and a valid
Newton direction needing smaller steps or different scaling/globalization than
the fixed13 trials. The stored monodromies are at qualified orbits, not the
failed iterate. No true linear residual or failed-iterate Jacobian was saved.

## Exact next diagnostic, subject to source review and separate authorization

Use only the six roles pinned in `s1r-lower-gap-inputs.json`: both native phases
of predecessor node0226 and last witnessed node0230, plus call473's original
seed and final stalled state at the closest failed target0.991875. Preserve the
saved control/rate bytes and native/carried phase conventions. Retain the final
stored Newton step/residual and its13 saved line-search records. No alternative
state, longer route, opposite-direction bootstrap or parameter search is in
this unit.

At each fixed state evaluate R, A=D_theta R=M−I and b=partial_h R analytically.
For the common log-hidden-rate coordinate,
partial_h F=−ln(10) P_hidden grad(L), and
b=DF(F(theta)) partial_h F(theta)+partial_h F(F(theta)); P_hidden contains the
hidden rate on the hidden block and zero elsewhere. This derivative refers to
the underlying smooth parameterized model, not a derivative of rounded `pow`.
Retain both maps/Jacobians, R,A,b and all decomposition inputs. At the stalled
state check A*p_saved+R, with absolute backward error and true relative forcing
reported separately; abstain from a relative verdict when ||R||_2 is at most
1e-12 max(1,||theta||_2). Compare the stalled R to its stored vector and qualified-point matrices to
their stored monodromies, retaining the existing normalized closure gates.
Use the previously independent analytic gradient/Hessian implementation for all
six roles, comparing qualified-point outputs to the saved shared-model values
without duplicate target evaluations; discrepancies stop interpretation.

Compute complete singular values/vectors of A and [A,b] (retain the full
73-column right basis for the augmented matrix), residuals of the
reported singular vectors, and the oriented null tangent of [A,b]. Report raw
coordinates and one fixed scale (state scales max(1,abs(theta_last)), h scale
0.000125; no scale tuning after outcomes). Cap at24 SVDs: both matrices, raw
and scaled, for six states; convert scaled tangents back to original units. Compare predecessor/last tangents
with carried secants; do not identify modes by nearest angle. Read and reduce
the13 saved trial residuals to distinguish an accurately solved Newton equation
from its finite-step nonlinear failures; do not reevaluate these trials or
apply any new step. Source review must freeze all agreement tolerances and
zero-denominator/finite-value handling. No new orbit is accepted by this unit.

The hard operation cap is six fixed R/A/b evaluations:12 map values and12
state Jacobians, plus derivative assembly from the same analytic gradients;
no extra target finite differences, Hessian-vector probes or corrective solves.
Reuse the independent analytic gradient/Hessian implementation already audited
for the lower outcome when possible. Pin its complete import/source inventory.

## Consequences and possible next branch question

A failed residual/Jacobian consistency or material true-linear-forcing check
first motivates an implementation/numerical-solve repair, not continuation.
Well-solved equations with finite-step merit failure leave predictor and
nonlinear globalization explanations open. A soft isolated direction of A,
full row rank of [A,b], and nonzero left-direction coupling to b would make a
small pseudo-arclength proposal reasonable, not prove a fold. If A remains
comfortably resolved, compare a frozen tangent predictor and the unchanged
state predictor at exactly0.991875 in a separately reviewed witnessed-link
unit; success would extend the local numerical association only that far.

If pseudo-arclength becomes justified, the next design should solve the73
unknowns (theta,h) using72 residual equations plus one arclength constraint.
Discrete cyclic phase needs explicit carrying, not an extra continuous-time
phase equation. Use at most a few fixed out-and-back steps, full primitive-two
qualification and direct/half/reverse witnesses, and stop at any gap. A fold
claim would additionally require an isolated simple +1 mode, nonzero
left-null coupling to b and quadratic coefficient along the right-null
vector, and a qualified connected tangent turn; rigorous certification would
require further enclosures. No present evidence proves these conditions.
Neither diagnostic outcome identifies the lower h=.95 pair or historical
ancestry, and neither implies a training benefit.

## Cost, durability and preparation

The existing lower audit took3.5129596840124577s while checking all234 qualified
pairs and ten independent analytic phase derivatives; the producer took
38.103352084057406s for473 phase calls. Use the former only as a conservative
historical cost screen (four times that time within a15s internal cap), not an
exact-workload timing guarantee. Proposed CPU/BLAS1, nice10,15s internal,
20s external plus5s cleanup,8MiB aggregate (7MiB data with terminal reserve,
1MiB launcher), one exclusive attempt and no retry. The six-state cap and
partial-result checkpoints make a resource stop informative. No GPU, paid
compute, holdout, subsetB, trajectory or new bootstrap is involved.

Before any new arithmetic: implement in a separate adapter without altering
old bound sources; commit/push code, synthetic controls and exact run card;
obtain independent source review and root launch authorization. Controls must
include a simple fold, an ill-conditioned nonsingular non-fold, corrupted
Jacobian/step, zero residual, competing near-unit modes, scaling sensitivity,
phase exchange, finite-value failures and interruption/cap durability. Preserve
all unsuccessful checks and raw matrices; a distinct saved-output audit must
precede any scientific interpretation or manuscript update.
