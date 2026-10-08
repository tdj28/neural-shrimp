# A low-rank curvature mechanism to test before larger-model claims

8 October 2026 UTC. Mathematical design note only: no empirical source,
profile, target run or launch authorization. This is elementary linear algebra,
not a novelty claim or evidence that a rank-one approximation fits our networks.

## Question and possible use

The two failed LM prediction pilots tested features of observed update motion.
They did not establish that low-dimensional curvature geometry predicts a
complex pair of the full two-step tangent map. A different question is whether
the dominant sharp direction rotates between phases sufficiently to explain
that pair. If a small curvature subspace accurately predicts the measured
complex pair in independently checked closed-orbit references, it would justify
testing a cheaper matrix-free estimator in larger models. If the approximation
fails against the full matrices, it should not become another LM feature sweep.

Neither outcome demonstrates a useful optimizer intervention. A subsequent
usefulness question needs a concrete decision and strong loss, gradient,
curvature-only and matched-information baselines. The previously deferred
frozen-curvature LM proposal remains deferred; this note does not reopen it.

## Exact algebra for the proposed approximation

For positive fixed block-rate matrix P, whiten the phase Jacobians as
S_j = I − P^(1/2) H_j P^(1/2). The full two-step monodromy is similar to
M = S_b S_a. H_j is the Hessian at phase j of an actual period-two orbit;
fixed-parameter full-state closure must be established before calling M a
monodromy. Along a drifting trajectory the same product is a finite-horizon
tangent map, with no automatic Floquet or bifurcation interpretation.

Consider symmetric approximations

    S_a^0 = a I − alpha u u^T,      S_b^0 = b I − beta v v^T,
    ||u|| = ||v|| = 1,             c = u^T v.

Here a,b describe isotropic bulk factors and alpha,beta are departures along
the two selected directions. They are not automatically the leading Hessian
eigenvalues: the bulk convention must be declared when fitting this model.
For linearly independent u,v, their span is invariant. The restricted product
has trace T and determinant D given by

    T = 2ab − b alpha − a beta + alpha beta c^2,
    D = ab(a − alpha)(b − beta).

All directions orthogonal to that span have multiplier ab. A complex conjugate
pair occurs exactly when T^2 < 4D; its modulus is sqrt(D) and its angle in
(0,pi) is atan2(sqrt(4D − T^2),T). The degenerate cases c=+1 or−1 have a
one-dimensional selected span and must be handled separately, not called a
rotational pair. In the real symmetric aligned case the product is diagonal
in a common basis. More generally, a product of two real symmetric factors
can have a complex pair only if both factors are indefinite; a positive or
negative semidefinite factor forces a real spectrum. Indefiniteness is necessary,
not sufficient.

For the simpler a=b=1 case, alpha=beta=2 makes each selected factor a reflection.
Their product rotates the u,v plane, with T=4c^2−2 and D=1. Parallel directions
give multiplier+1; orthogonal directions give the repeated real multiplier−1;
intermediate angles give a complex pair. Rotating sharp directions can therefore
produce complex two-step motion in arbitrarily large ambient dimension.
This reflection example has neutral complementary directions and proves neither
attraction nor an NS bifurcation in a gradient-descent training map.

For the two-dimensional approximation a unit-circle crossing needs D=1 and
|T|<2; attraction of the other modeled directions additionally needs |ab|<1.
Transversality would involve the total derivative of D along the orbit branch,
not merely a frozen-Hessian rate derivative: at D=1 the modulus slope is D'/2.
At fixed a,b,alpha,beta, orientation c changes T but leaves D unchanged. It can
change real-versus-complex character and angle, but cannot by itself change the
complex modulus or create a transverse unit-circle crossing while that pair
remains complex. Nonresonance, nonzero cubic
coefficient and all actual transverse modes remain separate requirements.

If S_j = S_j^0 + E_j, submultiplicativity gives

    ||S_b S_a − S_b^0 S_a^0||
      <= ||S_b^0|| ||E_a|| + ||E_b|| ||S_a^0|| + ||E_b|| ||E_a||.

This bound uses the whitened coordinates. Returning to the original Euclidean
coordinates can multiply it by the condition number of P^(1/2); eigenvalues
are similarity-invariant but norm and eigenvector-conditioning claims are not.
It is an operator-error bound, not an eigenvalue-error bound for a
nonnormal product. Spectral residuals, separation and the correctly paired
left/right eigenvector conditioning (or a suitable pseudospectral bound) must
be checked before drawing spectral conclusions. A small angle between vectors
or a nonzero commutator alone is not sufficient.

## Smallest useful empirical design, not yet implemented

Start from already saved, numerically qualified S2 and S1R closed-orbit states
with full phase Jacobians; use stable real-pair, stable complex-pair and
crossing-neighborhood references selected by an explicit rule. First inventory
which phase matrices are actually retained. Do not silently recompute missing
derivatives or choose only examples that fit. Preserve model/data/source hashes.

Before inspecting approximation errors, freeze the bulk-factor convention,
subspace fitting rule and comparison set. Compare the two-phase model with a
commuting/frozen-direction counterfactual retaining the same phase strengths.
Report operator errors, target-pair residuals/conditioning, angle and modulus
errors and failures. Separate orientation's incremental prediction of pair
character/angle from a phase-strength-only prediction of complex modulus. Do
not credit orientation for a quantity that the model already fixes through D. A bound covering the claimed prediction matters more than
a visually matching projection. It must be clear whether a fit uses information
from the target multiplier, in which case it is descriptive rather than predictive.

The decision is whether a limited matrix-free curvature representation deserves
a separately costed larger-model test. Success must improve a declared pair
prediction or reduce diagnostic cost at comparable accuracy; failure should
reject the chosen approximation, not trigger an undisclosed rank/point search.
No threshold or compute cap is proposed until the saved-input inventory and
actual workload measurement exist. No holdout, model trajectory or additional
LM training is authorized by this note.

Prior-work comparison must include Hofmann's two-mode complex multiplier
calculation and existing edge-of-stability/curvature analyses before claiming
an explanatory contribution. The product-of-reflections algebra itself is not new.

Independent static review: `docs/reviews/curvature-subspace-mechanism-review.md`
binds the initial note and motivates the qualifications added here. No empirical
fit or full prior-work novelty comparison has been performed.
