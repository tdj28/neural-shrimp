# Task14: independent cubic-coefficient and source review

13 October 2026. Source-only review of the complete Task14 assignment and export
`/tmp/task14-source-assignment-20261013`, manifest SHA256
`4fe637389d1a45fc3fe7a95de0c08d6dbf08bc0603aa1c9b1b3819758d2c77e2`.
Producer implementation is `cb6a105c3c7c077e9db7baadaeb0fd943e726f4c`.
No model, producer code, target/profile experiment, raw Part A data or pod was
accessed or executed. The quoted endpoint numbers are producer claims in the
assignment, not independent results.

## Formula and normalization

The proposed formula is correct at an exact simple +1 fixed point of the
**full period map** G, including ordinary fold points where b is nonzero.
Let M=DxG, B=D²xG, C=D³xG at the fixed point; Mr=r, lᵀM=lᵀ,
||r||₂=1, lᵀr=1. Choose the center coordinate u=lᵀ(x−x*) and write

    x(u) = x* + r u + (1/2) h2 u² + (1/6) h3 u³ + O(u⁴),
    u'   = u + b u² + c u³ + O(u⁴),      lᵀh2=lᵀh3=0.

Expanding G(x(u))=x(u') gives

    (I−M)h2 = B(r,r)−2br,               b = lᵀB(r,r)/2,
    (I−M)h3/6 = B(r,h2)/2 + C(r,r,r)/6 − r c − b h2.

Left projection eliminates both (I−M)h3 and b h2, yielding exactly

    c = lᵀC(r,r,r)/6 + lᵀB(r,h2)/2.

The factor 1/2 in the center-manifold parameterization matters. One must not use
this equation for an h2 defined without that factor. Under simultaneous r→−r,
l→−l, a and b change sign, while h2 and c are unchanged. Thus b sign(a) is
orientation invariant whenever a≠0. A positive rescaling r→κr, l→l/κ instead
rescales b by κ and c by κ²; matching the unit-r normalization is necessary for
numerical magnitude comparisons.

For an exact eigenvalue µ merely *near* +1, the corresponding quadratic equation
is (µ²I−M)h2=B(r,r)−2br, rather than (I−M)h2. The cubic projection still has the
same form in the l-orthogonal gauge. Therefore the supplied formula evaluated
at an approximate fold is a numerical fold approximation, not an exact identity
at arbitrary µ. Retain closure, |µ−1|, both eigenpair residuals, the full border
residual, its extra scalar and |lᵀh2|. Because the complement can be nonnormal,
small |µ−1| alone is not an error estimate for h2 or c. A condition/sensitivity
diagnostic is useful; it does not replace the independently measured derivative
precision or create an interval-proof obligation for this numerical cross-check.

## What the requested endpoints establish

A reproduced sign change in b sign(a) along the same consistently oriented fold
branch, with a remaining nonzero, supports a cusp candidate between the points.
Nonzero c at both endpoints does not establish c≠0 at the intervening b=0 point,
nor does a parameter-curve turn alone establish cusp nondegeneracy. The endpoint
check may be reported as an independent numerical verification of fold cubic
coefficients and of a candidate cusp bracket.

A numerical nondegenerate-cusp claim additionally needs a located full-state
fixed point with simple +1, b≈0, c demonstrably nonzero at the achieved precision,
and a rank-two two-parameter unfolding. For clarity, with physical parameters
p=(p1,p2), one suitable local construction uses

    ai = lᵀG_pi,
    (I−M)yi = G_pi − ai r,   lᵀyi=0,
    di = lᵀ(DxG_pi r + B(r,yi)).

At the cusp the 2×2 matrix with rows (a1,a2) and (d1,d2) must have rank two.
This explains the additional mathematical condition; computing it is outside
the currently requested two-endpoint check. No interval proof is required to
label a properly qualified result as numerical/rung 2.

## Concrete source findings for the independent implementation

1. `tools/bifurcation_curves.py:201–244` (SHA256
   `f2e3faac2dc471a16cf7490a8a4c228e889c2f5e793f56c882713c8df2b68a8a`)
   implements the correct factors and border formula, but selects the eigenvalue
   nearest +1 and takes real eigenvectors without refusing a complex/non-+1,
   nonfinite, multiple or unqualified crossing. This is a concrete **tool guard
   gap**, not evidence that either actual endpoint fails. Independently check
   full-state closure and primitive period, a simple real +1, left/right
   eigenpairs and a hyperbolic complementary spectrum before coefficients.
   Record both the complementary maximum modulus and its nearest-unit-circle
   distance. Other unstable complementary directions do not invalidate a fold
   cubic coefficient; all complementary moduli below one is a separate
   attraction condition, not a prerequisite for this coefficient-only check.
2. `soft_family` in `tools/explore_llm_soft_loop.py:411–495` (SHA256
   `12bb0c6b445a15b0e9e9e75e8799d29357a5f7d0f3ebd5376b011f282416965c`)
   can use loop-precision dense monodromy with reference-float64 JVPs. The source
   alone does not tell us the actual V11 precision settings. The independent
   ManualQwen path must use its genuine CPU float64 map throughout and compare
   composed-map dense actions, JVPs, critical multiplier and finite differences.
   A one-step dtype or derivative check is insufficient.
3. Use nested AD for B(r,r), C(r,r,r) and B(r,h2), with full retained vectors and
   both cubic contributions. Compare every supplied finite-difference step row
   (relative steps 1e−2, 3e−3, 1e−3, 3e−4) to the independent values. The producer
   chooses its flattest adjacent c pair after evaluation; do not select the
   independent comparison pair to minimize disagreement. Individual fine-step
   agreement, finite values and cancellation diagnostics are needed; a stable
   total alone can conceal errors in large opposing contributions. Store actual
   absolute steps, using the stated scale ||x||₂ (or1 at the origin).
4. `soft_family` uses theta=(log10T, alpha×alpha_scale). The assignment's quoted
   a≈+.57/+.53 does not itself identify the derivative parameter or scaling.
   Declare physical units and bind producer parameter metadata before comparing
   a or b sign(a). The independent hostile steering vector, full vocabulary,
   fixed prefix and one-token window must match the Task7 authority. This is a
   full return-map coefficient, with clock one G iterate = one complete period.
5. The requested dense 896-JVP CPU timing gate precedes substantial target work.
   Preserve the profile and its source/card, then admit a bounded endpoint run
   only if the measured resource envelope fits alongside priority Task1. No
   optional period-28 run is included in this review.

The algebra and source guard findings have been sent to Root and the independent
implementation author. Review of that author's new source/profile card remains
pending its release; this note authorizes no execution.
