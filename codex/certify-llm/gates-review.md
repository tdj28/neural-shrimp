# Independent review of conditional proof gates

Date: 2026-10-13. Reviewer: Codex `literature_cascades`. Scope: mathematical
sufficiency, implementation, synthetic controls, and a separate source-only
review of the signed-direction pilot. No target model was evaluated.

Reviewed `proof_gates.py` SHA-256:
`36d8cd4dfceec3abd64b122db4183c33973ef4c127c643bf8f8515a4b2b0708d`.
Reviewed test SHA-256:
`0ee68e10074354173f188980b3b7a78e1749fba294b44104b3b913f29271af77`.
The inspected version uses `conditional_gates_passed`, not a certificate flag.

**Approved as a conditional algebra checker. No mathematical false-positive
path was found when its stated full-box enclosure contract is actually true.**
This review does not establish that any transformer evaluator, saved Jacobian,
divisor image or monodromy satisfies that contract. No target certificate is
approved by this document.

## Existence and uniqueness throughout the nominal box

Let `X=c+diag(r)[-1,1]^n`, with strictly positive exact dyadic radii. For
`H=F^p-I`, let the supplied `JH` enclose `DH(z)` for every `z` in `X`, and `Hc`
enclose `H(c)`. The exact dyadic preconditioner is `R`.

The code bounds the map `P(x)=x-R H(x)` by the centered enclosure

`P(X)-c subset -R Hc + (I-R JH)(X-c)`.

Its strict coordinate margin test proves `P(X)` lies in the interior of `X`.
The weighted infinity bound `q=||I-R JH||_r<1` proves that `P` is a contraction
on this **whole convex nominal box**, by the mean-value integral along each
segment. Banach's theorem gives a unique fixed point of `P` in `X`. Moreover,
`||I-R DH(c)||_r<1` implies `R DH(c)` is nonsingular, so square `R` is
nonsingular and a fixed point of `P` is a zero of `H`. Thus uniqueness is not
limited to the smaller reported Krawczyk image.

The implementation retains the primal defect `-R Hc`; it does not silently
assume the floating center is a root. It requires both strict inclusion and
contraction. A singular preconditioner, equality at the inclusion boundary,
or equality in the norm test cannot pass.

The Arb radius used to represent `c +/- r` or the displacement box may round
outward. This is conservative: inclusion is still compared against the original
exact nominal radii. An enclosure valid for the nominal `X` is sufficient for
the displayed mean-value argument; multiplying it by a slightly enlarged
zero-centered displacement is an overestimate, not an assertion that the
Jacobian contract also holds outside `X`.

## Primitive period and local attraction

The primitive gate requires exactly every proper positive divisor of `p`,
without silently dropping missing or malformed entries. For each divisor it
requires a strictly separated coordinate of `F^d(X)` and `X`. Consequently a
proved `F^p` fixed point in `X` cannot have any proper-divisor period. A touching
or overlapping image fails. Period one correctly has no proper divisors; it
would be a fixed-point result, not evidence of a nontrivial training cycle.

The monodromy enclosure `M` must contain `D(F^p)` at the root. Native interval
matrix multiplication and binary exponentiation enclose every actual power
of that same root matrix. Treating repeated interval factors independently
can widen the result, but cannot exclude the true power. If any declared power
has a rigorous induced weighted-infinity upper bound strictly below one, its
spectral radius is below one and the orbit is locally attracting. The code
retains signed matrix multiplication before taking norms, which legitimately
captures point-matrix cancellation in nonnormal powers.

This attraction result does not say that all of `X` lies in the dynamical basin.
The contraction used by the root gate is for the preconditioned zero-finding
map `P`, not the dynamics `F^p`. The aggregate result concerns a unique `F^p`
fixed point in `X` whose orbit is primitive and attracting; it does not place
all `p` distinct orbit phases inside `X`.

## Arithmetic and false-positive checks

All admitted numeric enclosures must be finite Arb balls; floating arrays are
not silently converted into proof bounds. Centers, radii, preconditioner and
stability weights must be exact dyadic points, with positive radii/weights.
Norm row sums use outward absolute upper bounds, interval arithmetic and an
outward upper endpoint; acceptance margins use lower endpoints and strict
comparisons to zero. No floating eigenvalue or approximate norm is a proof
gate. If insufficient precision erases a strict margin, the code refuses it.

The domain and enclosure hashes bind the supplied mathematical bounds and
period. `R`, the declared powers and stability weights are checked locally and
retained in the returned witness; they need not be asserted by the upstream
enclosure provider. Exact midpoint/radius records retain uncertainty without
midpoint projection. Tests cover nonfinite/malformed data, missing divisor
rows, singular `R`, unstable cycles, large center defects, exact equality,
nonnormal cancellation, interval powers and a precision-floor refusal.

**Metadata is not proof.** A caller can fabricate internally hash-consistent
matrices and write the prescribed declaration. Such inputs may pass all
conditional algebra gates even though they are not bounds for the named map.
The module explicitly returns
`enclosure_validity_checked_by_this_module=False`. The renamed
`conditional_gates_passed` accurately labels its limited responsibility.
A future certificate finalizer must independently establish the full-box
contract from the actual rigorous evaluator and bound artifacts; copying this
boolean or checking hash syntax is insufficient.

## Verification performed

The final gate suite passed independently: **42 tests in 0.13 seconds**, CPU1.
Additional read-only synthetic probes gave the expected results:

- A genuinely nonlinear two-dimensional system
  `H(x,y)=(x+x^3, y+x/4)` on `[-1/2,1/2]^2` passed with a valid interval
  Jacobian and exact triangular preconditioner.
- `H(x)=x^3-x` on `[-2,2]`, which has three roots, failed for each tested signed
  preconditioner `-1,-1/4,0,1/4,1` using its valid whole-box Jacobian range.
- An image outside a deliberately non-magnitude-representable nominal radius,
  but inside the outward-rounded stored radius, was correctly refused.

These checks support the code review; they do not formally verify Arb, Python
or the implementation.

## Separate signed-pilot source review

The inspected `signed_pilot.py` confines itself to phase zero and coordinate
directions `e_0,...,e_7`. It binds the retained full proposal by hash, writes
all signed Arb columns before the later floating comparison, and explicitly
returns `certificate_passed=False`. The float64 comparison is appropriately
only a consistency diagnostic: exact-real point balls need not contain values
from a distinct rounded floating map. It reports neither a full Jacobian nor
a full-box derivative remainder.

No additional mathematical blocker was found in this driver. Receiver-bound
prefix-cache provenance is a separate review responsibility of the cache
reviewer; this document does not assume it from the function name. Requiring
an explicit binary64 proposal dtype was suggested as minor admission hardening
alongside the existing shape/finite/hash checks. Any launch still requires the
pushed source, bound plan and owned resource envelope.
