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

## 8 October 2026 UTC — concrete metadata inventory and first empirical decision

**Decision: prepare one small fixed-state acquisition/fit unit; do not launch
it yet.** The [input inventory](curvature-subspace-input-inventory.json) binds
13 input/receipt files and nine inspected source files at the stated base
commit. This amendment inspects schemas, saved scalar labels and identities
only. No new norm, eigendecomposition, SVD, derivative, map or approximation
error has been evaluated. The panel is exploratory and adapted after seeing
historical development metadata, not confirmatory. No protected split or
seed29000–29999 is used.

### Fixed six-reference panel

Use two losses already studied, S2-28011 (13D) and S1R-28002 (72D), with one
reference in each of three strata. For stable strata, the rule is the first
serialized verified attracting primitive-period-two certificate in that
family's **original base and zoom pool**, classified by its already saved
leading multiplier's imaginary part. Do not search other seeds or replace a
failed slot. This gives the following exact zero-based rows of
`raw/EXP-028/attempt-002/stage3/certificates.json`:

| Family | Stable real-dominant | Stable complex-dominant | Crossing-neighborhood reference |
|---|---:|---:|---|
| S2-28011 | row36, component0 | row125, component23, zoom | criticality curve row0 |
| S1R-28002 | row17, component0 | row98, component0, zoom | audited original anchor node-0000 |

“Real-dominant” labels the historical leading mode, not the whole spectrum
or an already-qualified two-mode pair. The
crossing rule first reuses the already independently audited original anchor
where one exists (S1R); otherwise it uses the first serialized NS-candidate
row of that family's original criticality curve (S2). These are one-sided
near-boundary references, not a two-sided crossing test or a common connected
branch. S1R node-0000 inherits criticality row2 and has both phases plus saved
monodromies/eigenvectors. S2 row0 has only one orbit point and scalar NS
metadata. Neither stable-certificate file retains phase derivatives. The
exact row pointers, rate literals, state JSON/array identities, qualification
labels, alternatives admitted by the rule and all missing fields are recorded
in the inventory before any curvature-fit errors.

Source inspection confirms three important gaps. `period2_branch.spectral_checks`
saves the two monodromies, not the phase Jacobians. `dense_two_step` in the old
criticality producer computes phase Jacobians transiently; its output retains
neither them nor the opposite phase. The new lower-gap six-role collection
retains phase derivatives, but represents the near-real lower-endpoint question
and cannot substitute for this NS reference panel. Historical complex
`dominant_eigenvalue_condition` fields are invalid and must not be reused.
The current source inventory documents inspected implementation semantics; it
does not establish a missing historical execution binding for the criticality
JSON. The new acquisition must bind its own complete transitive source and
verify input identities, without upgrading the old provenance.

### What the next unit must acquire

Load only consumed training x/y from the existing S2 exact-dyadic C1 input
file and S1R branch input archive, whose source, model and hashes are bound in
the inventory. Verify the original EXP-028 shape-plus-bytes hashes as well as
current raw-byte identities. Do not call `make_s2`, which also creates the
protected validation/test arrays. Preserve both bias groups, ridge and loss
normalization. At each of the six fixed references, retain the two gradients,
Hessians, phase Jacobians and map images: **12 primary phase Hessian/gradient
acquisitions and at most24 map values in total**, no trajectory or correction.
The12 primary images check stored-pair transitions. To check actual two-step
closure separately at both native phases, also acquire F(F(theta_j)) for each
j; reuse a primary image only when the intermediate input is byte-identical to
its recorded input, with an explicit cache record. For S2's single-phase
crossing candidate, F(theta_0) defines theta_1 and F(theta_1) checks phase0
closure, but phase1 still needs F(F(theta_1)); proximity to theta_0 is not a
cache identity. Unused map capacity does not authorize another reference.
Extra closure images use a map-only adapter. If the implementation always
returns a Hessian with a map, count every such analytic call as a Hessian
acquisition (up to24), and disclose this implementation choice and its cost
before source review; never report only12 while silently doing more.

Recheck transitions/closure with the original componentwise
`1e-9 + 1e-8 max(|state|,|image|)` scale and primitive separation1e-6. Require
finite positive fixed rates and finite arrays; preserve every rejected return.
Where saved monodromies or multipliers exist, require consistency within
absolute1e-10 plus relative1e-10. Stable slots must keep their historical
real/complex leading-pair class and attraction margin1e-7; crossing slots need
closed-orbit qualification but are not required to pass an attraction label at
that margin. A failed identity, state or label gate stops the positive panel
interpretation. Retain the slot and reason; do not correct, substitute or widen
tolerances. The six references are the units; two phases are not replicates.

### Frozen fit, counterfactual and estimand

For each phase set K=P^(1/2) H_sym P^(1/2), S=I-K. First retain the unsymmetrized
H and require relative Frobenius asymmetry <=1e-12; only then use
H_sym=(H+H^T)/2. Choose u as the eigenvector of the **largest algebraic**
eigenvalue lambda1 of K. Define bulk curvature kbar=(tr(K)-lambda1)/(n-1),
a=1-kbar and alpha=lambda1-kbar. Do this independently at the second phase to
obtain b,beta,v. This is one fixed top-curvature-plus-isotropic-bulk fit, not an
optimization against the target monodromy or an adaptive rank search. Require
top-eigenvalue gap >1e-6 max(1,||K||_2); otherwise record an ambiguous-direction
failure. Obtain ||K||_2=max_i|lambda_i(K)| from the same retained Hermitian
eigensystem, not an additional norm SVD. Treat 1-(u^Tv)^2<=1e-10 as the one-dimensional aligned case, not a
spurious rotational plane. Zero strength, ties and borderline discriminants
are retained abstentions. For discriminant sign use the fixed unresolved band
1e-10 max(1,T^2,4|D|).

Compute the rank-one product's **full modeled spectrum**, including all n-2
bulk multipliers ab, before taking its leading pair. The measured target is
the two largest-modulus eigenvalues of the full whitened product S_b S_a,
with deterministic real/imaginary tie order and explicit conjugate-pair checks.
The selected two modes must be two real eigenvalues or a conjugate pair; a
mixed/nonconjugate selection fails that pair check and remains unresolved,
without replacing the reference or matching a more favorable pair.
A modeled plane that does not actually dominate cannot be matched to a favorable
non-leading full-system pair. Compare pairs by the smaller of their two possible
assignments; this is the fixed unordered-pair metric, not point/mode selection.

The primary error is maximum complex eigenvalue error in that pair, normalized
by max(1,full spectral radius). Also report real/complex character, angular
error for complex pairs, modulus error and the relative full-product Frobenius
error. The matched-strength commuting baseline holds a,b,alpha,beta fixed and
sets v=u. An explicitly declared orthogonal commuting baseline (u^Tv=0) is a
second control; conservatively compare to the better of these two fixed errors
at each reference, labeling this an oracle comparator, not a deployable
predictor. Apply the same full-spectrum leading-pair selection to both controls:
the aligned case has (a-alpha)(b-beta) and n-1 copies of ab; the orthogonal case
has b(a-alpha), a(b-beta) and n-2 copies of ab. Keep multiplicities and repeated
or tied leading modes explicit; neither baseline may discard a dominant bulk
mode to improve the comparison. Both retain phase strengths. A separate strength-only report uses
sqrt(D) **conditional on a complex modeled pair**. Orientation may improve
character/angle; it gets no credit for complex-modulus prediction, which D
already fixes. No orientation-only NS-crossing claim is possible.

Success needs more than recovering the known fact that noncommuting symmetric
factors can have complex eigenvalues. Freeze the exploratory engineering gate:
all six slots qualified without replacements; no false complex prediction at
either real-dominant control; all four complex/crossing slots keep complex
leading pairs; pair error<=.02 at every slot; angular error<=.05radians at each
complex slot; and mean pair error at least25% and .001 absolutely below the
stronger commuting comparator. These tolerances are chosen before fit outcomes
as useful approximation targets, not claims of statistical significance.

Also require a floating-point error diagnostic: retain phase residuals E_a,E_b
and compute delta_op from the stated product bound using the rank-one factor
operator-norm formulas max(|a|,|a-alpha|), max(|b|,|b-beta|) and Frobenius
residual norms. These formulas are exact in real arithmetic; their floating
values do not enclose matrix-formation or rounding error. Retain the actual
product difference as a consistency diagnostic. Obtain correctly paired
left/right vectors from the single declared general eigensystem call per
reference, and retain their residuals and target separation.

For the computed complete right-vector matrix V and diagonal eigenvalue matrix
Lambda, retain R_V=M V-V Lambda. Use the one declared eigenbasis SVD to obtain
sigma_min(V), sigma_max(V), and kappa_2(V); reject a zero/nonfinite minimum or
failed residual/conditioning checks. The residual-aware diagnostic radius is

    delta_diag = ||R_V||_F / sigma_min(V),
    rho_num = delta_diag + kappa_2(V) * delta_op.

For exactly invertible V, the exact identity
V^(-1) M_0 V-Lambda = V^(-1)(M_0-M)V + V^(-1)R_V explains this expression;
the diagonal reference Lambda gives the usual eigenvalue-distance inequality.
Numerically estimated singular values, products and norms do not turn this
into a rigorous Bauer-Fike enclosure. Report rho_num as a **floating-point
spectral diagnostic**, never a certified enclosure or validated exact eigenbasis.
Any strict mathematical enclosure or theorem claim remains pending outward-
rounded interval validation, including matrix formation and invertibility.
No interval work is included in this unit.

Keep the existing numerical attribution thresholds unchanged: for a particular
complex pair rho_num must be<=.02 and less than half the relevant separations
to other computed eigenvalues and the real axis, and each modeled root must
lie in its assigned target disk under the fixed two-way matching rule.
Otherwise mark spectral attribution unresolved even if direct numerical
agreement looks good. A passing numerical screen is only a numerical diagnostic;
it does not establish disk eigenvalue counts or a rigorous pair enclosure. Ill-conditioned or
defective eigensystems, large bulk residuals or a failed bound are legitimate
negative results. No rank2 rescue, alternate bulk formula, parameter scan or
new reference follows within this unit.

### Cost, stopping and what a pass would change

Before acquisition, implement and independently review the exact adapter and
synthetic symmetric-product/commuting/degenerate controls. A single prospective
resource profile at the **first S1R panel phase** may acquire one of the12
required derivative returns and be reused only under exact source/input/device
identity; its eligibility depends on time/memory only, never fit quality.
It must also time one fixed synthetic72D symmetric eigensystem, general
eigensystem and condition SVD, so analysis cost is measured rather than
inferred from derivative time. Freeze those synthetic matrices in the source
card; they are engineering controls, not additional network references.
Proposed ceiling: CPU/BLAS1 nice10, profile5s internal/10+5s external, total
acquisition+analysis60s internal/75+5s external,16MiB immutable outputs and
2GiB monitored RSS. Profile must project the remaining fixed work within60s
using a2x margin on measured remaining derivative, closure-image and decomposition
costs, plus5seconds overhead. Charge the reused first phase against both the12
primary acquisitions and24-map cap. Time a separable map-only constituent of
that same phase acquisition if available; otherwise conservatively charge
all additional closure images at the measured full-phase acquisition cost,
without a second target timing call. If a fused helper necessarily acquires
extra Hessians, the projection must include up to24 full acquisitions.
Expected analysis inventory is12 symmetric phase eigensystems, six full general
eigensystem calls returning both left and right vectors, and at most six
complete right-eigenbasis SVDs. Reuse those singular values for all V norm,
inverse-norm and conditioning diagnostics; no hidden cond/norm SVD and no
new Newton solve. These are proposed limits, not measured feasibility or
launch authorization. The reviewed lower-gap unit's12 phase derivative pairs
and24 SVDs took1.148s on this host, a cost precedent only; it is not a substitute
for this source-bound profile. Any cap, missing slot or accuracy-gate failure
ends the unit with all evidence retained; no automatic retry or rank expansion.

A pass would justify a separately costed matrix-free estimator comparison on
these same references before a larger-model experiment: can a small number
of curvature actions recover the declared accuracy and its uncertainty more
cheaply than the full product? It would not establish training, inference or
LLM utility. A fail rejects this particular rank-one/bulk explanation as the
next bridge and avoids another weak LM feature sweep.

[Hofmann, §6.2 Proposition16 and AppendixE.2](https://arxiv.org/html/2607.04993v1#S6.SS2)
already derive physical off-diagonal misalignment through symmetric two-phase
products in a two-mode depth-two linear network, including complex rotational
instability. Our possible addition is a numerically assessed approximation with measured
error/control gain in the specified nonlinear networks, not a new
product-of-reflections result or discovery that rotations can occur.


### 8 October 2026 UTC — independent static-review amendment

The source-only review at `docs/reviews/curvature-empirical-design-review.md`
corrects closure-operation accounting and replaces an overstrong floating
Bauer-Fike enclosure claim with the residual-aware numerical diagnostic above.
It also makes complete-spectrum baseline multiplicities and decomposition
reuse explicit. The six selected references, fitting rule, comparison units,
all accuracy/qualification thresholds and proposed wall/raw caps are unchanged.
No fit outcomes were read or generated, and no numerical target operation ran.
Profile and acquisition remain unimplemented and unauthorized.
