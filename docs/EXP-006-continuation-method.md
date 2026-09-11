# EXP-006 periodic-orbit continuation method

Prospective implementation, 2026-09-11. This supplement follows the existing
nine-peak periodic orbit while its weights change. It distinguishes a surviving
unstable orbit, a stability crossing bracket, and an unresolved numerical
corrector. None of these outcomes is equated with the basin behavior measured
by the separate forward weight-drift assay. No neural continuation targets were
executed while developing this method; the tests use analytic fixtures.

## Frozen targets and limits

The driver supplies the EXP-005 core and near-edge certified state and period.
Each seed has two oriented branches, initially increasing or decreasing gamma.
Alpha remains fixed. The scalar continuation parameter is `s`, with
`gamma = gamma_seed + 0.41*s`, and state scales `(1, 1, 20)`. Log-period and `s`
have scale one. Gamma must stay within `[9.9, 10.7]`. The period stays within
`[T_seed/4, 4*T_seed]`, where the seed is the original branch seed, not the most
recent corrected orbit.

Each branch permits at most 48 accepted successor nodes and 96 attempted
correctors. The initial arclength step is 0.005, the minimum is 0.00001, and the
maximum is 0.01. A correction uses at most ten Newton updates. Each update tries
the full Newton step followed by at most six halvings, accepting the first
trial satisfying an Armijo decrease with coefficient 0.0001. All trials and
their outcomes are retained. A failed corrector halves the arclength step. A
successful correction taking at most three Newton updates multiplies the next
step by 1.25, capped by the maximum; one taking more than six updates halves
the next step, floored by the minimum; otherwise the step remains unchanged.
Failure below the minimum stops the branch as unresolved. Loss of attraction
never stops continuation. A parameter reversal is retained, rather than forcing
the branch to remain a single-valued function of gamma.

## Equations and numerical gates

For unknown `z = (x0/state_scales, log(T), s)`, solve full-state periodic closure
and a local phase condition. The phase normal is the normalized vector field at
the previous accepted initial state. The phase condition is its inner product
with the difference from that state. It is divided by the norm of the phase
normal times the state scales; closure components are divided by their state
scales.

DOP853 jointly integrates the state, monodromy matrix `M`, and parameter
sensitivity `S`, with relative tolerance `1e-10`, absolute tolerance `1e-12`,
and maximum step `min(0.2, T/100)`. The sensitivity obeys

`S' = J S + (0, 0, 0.41*tanh(x1))`, `S(0) = 0`.

Before scaling, closure Jacobian columns are `M-I`, `T*f(endpoint)`, and `S`.
The phase row has the phase normal in the state columns and zeros elsewhere.
The normalized SVD null vector gives the tangent of this four-equation system
in five unknowns. Its initial orientation follows the requested sign of gamma;
subsequent tangents have positive inner product with their predecessors.
A predictor is followed by a bordered Newton correction with an additional
arclength hyperplane equation. The bordered residual norm must be at most
`1e-9`. The separately evaluated relative physical closure must be at most
`1e-8`, absolute physical phase residual at most `1e-8`, and relative neutral
flow residual `||M f - f||/||f||` at most `1e-6`. Nonstationary flow and section
transversality are required.

Every accepted neural node receives a separate adaptive integration of all
three states with downward zero crossings of `f3` detected as maxima. This
uses tolerances `1e-10/1e-12` and maximum step `min(0.05, T/900)`. There must be
exactly nine distinct maxima in one period. A maximum split numerically
between the two endpoints counts once. Every pair of maximum states must have
relative separation greater than `1e-7`; this rejects multiple covers with
repeated full-state maximum points. The first maximum supplies a rephasing
state for independent shooting checkpoints at nodes 0, 8, 16, 24, 32, 40,
48, and the last accepted node if it is not already a checkpoint.

Checkpoints invoke the existing certificate without changing its tolerances.
Their full diagnostic fields are retained, excluding dense trajectory arrays.
The checkpoint period and rephased state must separately agree with the
continued period and first maximum state to relative tolerance `1e-6` (the
state norm denominator is floored at one). A checkpoint failure is visible evidence and is
not erased; it does not retroactively reject a node that passed the geometric
continuation gates. In particular, the old certificate's eigenvector-based
neutral identification can become ambiguous when another multiplier approaches
one. Report checkpoint agreement separately from continuation evidence.

## Transverse stability and boundary reporting

Avoid deleting whichever monodromy eigenvalue happens to be nearest one. For
phase normal `n`, flow `f`, and an orthonormal basis `Q` of `ker(n^T)`, compute

`A = Q^T [I - f*n^T/(n^T*f)] M Q`.

The two eigenvalues of this Poincare map are the transverse multipliers. Full
`M`, `A`, flow residual, phase normal, closure residual, and both multipliers
are saved at every node. A node is attracting when its spectral radius is less
than `1-1e-5` and unstable when greater than `1+1e-5`; intermediate values are
left undecided. The direction of change of the physical parameter is recorded
to identify reversals.

Neighboring multipliers are paired by minimum total complex distance. A real
multiplier with imaginary magnitude below `1e-7` that straddles +1 or -1 gives
a corresponding *bracket*. A change in the spectral radius across one gives a
unit-circle bracket. These endpoint checks do not prove nondegeneracy, exact
bifurcation location, or that an interval contains only one crossing. An
unresolved corrector does not establish orbit destruction. Finite continuation
coverage does not establish a complete shrimp boundary, and the four specified
branches are local gamma sections rather than a two-dimensional boundary atlas.

## Evidence and analytic validation

`continue_ray(...)` returns compact accepted nodes and all corrector attempts,
including iterate coordinates, residuals, line-search trials, predictor, step,
failed validation gates, and stopping reason. Driver callbacks receive each
node and attempt as it becomes available so an interrupted run can retain
partial evidence. The neural adapter is
`continue_hopfield_branch(alpha, gamma, section_state, period, direction, ...)`.

The analytic fold fixture is

`r' = r*(mu-(r^2-1)^2)`, `theta' = 1`, `z' = -0.7*z`.

Its exact cycle branches satisfy `r^2 = 1 +/- sqrt(mu)` and `T = 2*pi`.
The radial multiplier is `exp(-8*pi*r^2*(r^2-1))`, while the other transverse
multiplier is `exp(-1.4*pi)`. Tests require continuation through the fold at
`mu=0`, reversal of the parameter direction, and retention of the unstable
inner branch. Other tests check the scaled shooting Jacobian against central
differences, invariance to initial phase, unambiguous geometric projection
at the double-unit-multiplier fold, synthetic +1/-1 bracket labels, and
preservation of deliberately failed and out-of-bounds correctors. No test
integrates a neural target.
