# Next certificate: planar period-2 NS crossing and criticality

**Exploratory design only. No new certificate or timing run.** Prefer a direct
three-variable augmented certificate over isolated endpoint certificates.
This is a small polynomial problem; endpoints alone cannot establish an NS.

## Exact model, target and scope

Let `L=(uv-d)^2/2+lambda*(u^2+v^2)/2`,
`F_eta(u,v)=(u-eta*((uv-d)*v+lambda*u),
             v-r*eta*((uv-d)*u+lambda*v))`, and `G_eta=F_eta o F_eta`.
Use exact binary64 coefficients from existing scalar case 15:
`d=5404319552844595/4503599627370496`,
`lambda=5404319552844595/18014398509481984`, `r=4/1`.
These dyadics represent binary64 1.2 and .3, not exact decimals 6/5 and 3/10.
The bifurcation rate **eta_* is an unknown exact real**, not the dyadic seed.
All fixed coefficients, proposal coordinates, boxes, preconditioners and
endpoint rates are serialized as exact rationals. No float library function
may decide a proof gate. The model is exact-real GD, not machine arithmetic.

Selection is outcome-informed: the existing rung-0 proposal is
`(u,v,eta)=(.6919404589222307,.5180447042439237,.5483468723300842)`, with
trace `-.7958762525`, separation `.9446611888`, coefficient `-1.7002019651`
and modulus slope `3.3529863565`. These are not bounds. Ratio 4 is exact;
the proposed trace avoids strong resonances and the coefficient avoids zero.
No undisclosed target search is planned. This local certificate would not
prove connection to the preceding flip, a global branch, basin accessibility,
shrimp organization or utility.

Source bindings from the integrated checkout: `scalar_mechanism.py` SHA256
`e2104b9e4b961a2ba72a167d915e79200a09ed0527782cc34cd16e30f634a78b`;
`criticality.py` SHA256
`2433f44033c9d70808ec1de46690a99f8a50c154fa77815d2ca8221a06937d24`;
scalar receipt SHA256
`79cb579b0c6f0f07da16ede15bbae8546b5b537de19e04495e89d3ed6f9f8381`.
The seed is `/data2/neural-shrimp-data/explore/scalar-mechanism/attempt-002-pilot/case-15/result.json`,
SHA256 `cbef5ea46ee1a0253dccbe93fbc4e36523a47f9363533861c8bce09e4381215e`.

## Minimal validated system and gates

For `A=D_x G_eta(x)`, `T=tr(A)`, `D=det(A)`, solve three real equations
`H(x,eta)=(G_eta(x)-x,D-1)=0` in `(u,v,eta)`.

1. Prepare the center using exactly five 256-bit Newton proposal updates,
   taking exact dyadic midpoints after each solve and retaining every residual,
   correction and center. A failed solve stops the attempt; proposal convergence
   is never a proof gate and does not trigger a box/precision search. Then use
   Arb at 256 bits, fixed coordinate radii `2^-40`, exact-dyadic
   approximate inverse R, strict Krawczyk inclusion and
   `||I-R DH(Z)||_inf<1` on the 3D box Z. This establishes a locally unique
   augmented root; floating-point Newton residuals are proposals only.
2. Exclude zero in a coordinate of `F_eta(X)-X` over Z, proving primitive
   period two. Save the companion-phase box. Uniqueness is local to the
   selected phase, not a global exclusion of companion/sign-symmetric cycles.
3. The exact root satisfies D=1 by H. Prove `-2<T<2`: the complete planar
   spectrum is a simple nonreal unit pair. Exclude orders 1–4 using T disjoint
   from `{-2,-1,0,2}`; no acos/float angle is needed for this gate.
4. `A-I` is invertible since `det(A-I)=1-T+D>0`. The implicit-function
   theorem gives a local periodic branch. Enclose
   `x'=-(A-I)^(-1)*partial_eta G` and
   `D'=partial_eta D+(partial_x D)*x'`. Require D'>0; the modulus derivative
   at the crossing is D'/2. Do not substitute a finite-difference slope.
5. Prove a strictly negative first Lyapunov coefficient with the convention
   below. A root without transversality/nonresonance/sign is only an algebraic
   unit-pair point, not a generic NS certificate.

Two compositions should have little wrapping. If this system fails, preserve
that result before a disclosed five-variable multiple-shooting amendment:
`F(x)-y=0, F(y)-x=0, det(DF(y)DF(x))-1=0`. Endpoint-only fallback is narrower.

## Analytic derivatives and sign convention

Implement an independent ball-polynomial evaluator, not interval wrappers
around NumPy or finite differences. A three-variable Taylor jet through total
degree three has 20 coefficients: it supplies G, first–third x derivatives
and mixed eta derivatives for DH and D'. Account explicitly for multi-index
factorials. Cross-check the existing analytic composition identities for B/C.

With `G(x_*+z)=x_*+Az+B(z,z)/2+C(z,z,z)/6+...`, choose
`Aq=mu q`, `A^T p=conj(mu)p`, `q^*q=1`, `p^*q=1`, and enclose

`c1 = p^*[C(q,q,qbar)+2B(q,(I-A)^(-1)B(q,qbar))
                   +B(qbar,(mu^2 I-A)^(-1)B(q,q))]/2`,
`ell = Re(conj(mu)*c1)`.

Use complex balls and accept only an ell enclosure wholly below zero.
Construct `mu=(T+i*sqrt(4D-T^2))/2`, proving the radicand positive; never
normalize a float eigenvalue onto the unit circle. D=1 proves its exact
modulus separately. Use `q0=(A12,mu-A11)` after a nonzero-pivot check,
normalize its ball norm, and take `p^*` as row one of `[q,qbar]^(-1)`.
Enclose every denominator/determinant away from zero, including the two
resolvents. These identities hold at the exact root; evaluating their
expressions on Z encloses that root's values without asserting that arbitrary
off-root box points satisfy D=1.

The convention and local theorem match
[Kuznetsov and Sacker](https://www.scholarpedia.org/article/Neimark-Sacker_bifurcation)
and the integrated coefficient source. Positive modulus slope and negative
ell imply supercritical direction under that theorem. Any qualitative local
invariant-curve consequence is theorem-dependent, not an explicit enclosure
at a prescribed finite detuning or a rotation-number proof. For F there are
two small curve components exchanged by F; each is invariant under G.
No irrational/quasiperiodic dynamics or chaos follows from this certificate.
The theorem application/code need independent mathematical review; Arb/FLINT
and standard interval theorems remain trusted, not proof-assistant-formalized.

## Endpoint proof versus interval continuation

| Passing stage | Statement supported | Still missing |
|---|---|---|
| Two endpoint orbit inclusions, with attraction/repulsion tests | Two primitive cycles of opposite local stability | Same branch or any intervening NS |
| Augmented root, complex pair, primitive exclusion | Locally unique unit-pair point | Genericity and criticality |
| Add analytic crossing slope, nonresonance and ell sign | Generic local NS, with supercritical direction | Explicit finite parameter interval/global ancestry |
| Uniform parameterized inclusion on a tube | Same local branch for every eta in that stated interval | Continuation outside that tube |

Optional finite-interval stage: exact dyadic endpoints at seed eta +-2^-20,
after proving they straddle the certified eta box. For every eta in I, prove
a uniform parameterized Krawczyk contraction for `G_eta(x)-x` on a common
state tube containing the augmented root box. Pointwise uniqueness and the
implicit-function theorem link the endpoints to one continuous branch.
Require complex discriminant <0 and D'>0 throughout; check all Jury margins
positive at the lower endpoint and D>1 at the upper. Independent overlapping
point boxes or endpoint signs alone do not supply this uniform statement.

## Bounded execution and artifacts

No operation timing for the proposed evaluator has been measured. Expected
cost is seconds: 20-coefficient jets, 3x3 real and 2x2 complex solves, no
trajectory/eigensolver/GPU. This is an estimate, not evidence or a guarantee.
First commit code, controls and a run card. An optional synthetic non-outcome
unit can time 128/256/512-bit polynomial boxes, one CPU thread, nice 10, external 30-second
cap, without evaluating the selected target. Bound the first target attempt
by a 120-second process cap. No automatic box/precision search or paid fallback;
record each failure and amend before further attempts.

Controls: exact polynomial derivative/factorial identities; normal forms with
known positive/negative/zero coefficient signs and slope; resonant/real-unit
rejection; collapsed fixed point; tangent crossing; sign-straddling balls;
failed inclusion; corrupted input/source/summary rejection. Require an
independent coefficient implementation or algebraic cross-check before
reporting its sign. Save exact inputs, all boxes/intermediate enclosures,
per-gate verdicts, immutable attempts, source/library/precision/runtime and a
compact hash-bound receipt. One offline verifier recomputes all quoted bounds
and gates. Partial endpoint success cannot mask an inconclusive sign/root.
Existing seed evidence remains rung 0; only actual passing rigorous gates earn
rung 3. Review status, theorem dependence and failed attempts remain explicit.

## First selected-target result (2026-10-08 UTC)

Status: every rigorous gate passed on the first declared attempt; independent
mathematical/code review is pending. This is exploratory target selection, not
a fresh confirmatory test. The implementation applies trusted ball arithmetic
and standard theorems; it is not proof-assistant formalized.

Execution source: `0652d49529265565078b63360ec76bd9dd5f21a8`.
Receipt: `certificates/scalar-ns-receipt.json`.
Immutable raw directory:
`/data2/neural-shrimp-data/explore/certificates/scalar-ns-2026-10-07/attempt-001`.
Certificate SHA256:
`7e1897addaa9648a597c1cdd2b674504740d2becb942fe4428caf27ffac72e05`.
Input SHA256:
`fe92de4a2866756024aab649ddea13e6af0bc528ffde79863f64e7d8e6b93662`.
The receipt binds the full exact enclosures, all five proposal iterations,
source hashes, exact dyadic coefficients, runtime and library versions.

The unique augmented root in the fixed three-coordinate box has unknown
exact-real rate enclosed by `[0.54834687233 +/- 9.94e-13]`. The final exact
center is recorded in the certificate; its eta coordinate is
`63494229973848339256844513796926116069469096436775413280300460976013066680955 / 2^256`.
The requested radius was `2^-40`; Arb's outward-rounded stored radius is
`536870913 * 2^-69`, slightly larger. No alternative target, radius, precision,
iteration count or proof attempt was used.

| Rigorous gate | Enclosure or conclusion |
|---|---|
| Augmented root | Strict inclusion in all three coordinates; contraction bound below `6.063e-10` |
| Primitive period two | Both coordinates of `F(x)-x` exclude zero; first is `[0.12522884204 +/- 6.47e-12]` |
| Complete planar spectrum | At the proved root `det DG=1`; trace is `[-0.7958762525 +/- 7.57e-11]`, and `4 det DG - trace^2` is positive |
| Nonresonance | Trace enclosures exclude `-2,-1,0,2`, ruling out roots of unity of orders one through four |
| Branch and crossing | `DG-I` invertible; analytic modulus slope at the root is `[3.35298636 +/- 5.55e-9]` |
| Criticality | Resolvent formula `[-1.700202 +/- 3.69e-7]`; independent planar scalar-coefficient formula `[-1.7002020 +/- 9.28e-8]`; both strictly negative and their difference enclosure contains zero |

Subject to independent review of the implementation and theorem application,
these conditions prove a generic local supercritical NS bifurcation of the
exact-real primitive period-two orbit for the stated exact dyadic scalar
model. The exact crossing rate is enclosed, rather than treated as a dyadic
input. The implicit-function theorem supplies a local orbit branch through
that root and the positive slope gives the stability-change direction.
The standard NS theorem supplies a qualitative local invariant-curve
consequence. No explicit curve at a prescribed finite parameter, finite
continuation tube, connection to the original equilibrium/flip branch,
irrational rotation, chaos, basin statement, utility effect, or claim about
finite-precision training follows from this certificate.

The target run took 0.0783 seconds with maximum RSS 47,012 KiB, one CPU thread,
nice 10, python-flint 0.9.0 and FLINT 3.6.0. No GPU or paid resource was used.
Ten synthetic controls passed before target execution. After execution the
offline verifier reproduced the full result byte-for-byte. An untouched copy
with a relocated receipt passed. Copies with altered input bytes, omitted
inventory, changed quoted summary, or a modified certificate with an updated
digest each failed closed; the original evidence was never modified.

Replay from this source (the later documentation commit leaves these source
hashes unchanged):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/certify_scalar_ns.py verify certificates/scalar-ns-receipt.json
```

The preceding design's unmeasured-cost paragraph describes the pre-run plan;
the measured cost above supersedes that estimate for this single attempt.
No endpoint or continuation scan was run. Paper inclusion remains pending
independent mathematical review.
