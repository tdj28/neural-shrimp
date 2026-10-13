# Next method after the first Arb setup failure

Date: 2026-10-13. This is a method proposal, not new computation, a certificate,
or permission to exceed the next committed run card.

## What was measured

Pilot `arb-pilot-001`, source
`77fecc38ca077665cbb44c4e611d7b52442b9210`, failed while constructing the fixed
prefix at zero-based decoder layer 16: the interval softmax denominator could
not be proved positive. At 384 bits, the maximum prefix radius grew from
`1.7248793e-107` after layer 0 to `1.3195834e-8` after layer 14 and
`0.0536306896829` after layer 15. Driver elapsed time was 282.119 seconds;
launcher elapsed time was 283.577 seconds. See the immutable raw attempt and
`certificates/llm-p8-pilot-001.json`.

The separate numerical proposal did qualify: relative period-eight return
`1.602313717e-14`; divisor 1/2/4 separations approximately
`2.00332 / 1.43121 / 1.39166`; the assigned token loop matched cyclically.
These are numerical proposal facts. The rigorous point map and state-box JVP
were **not reached**. Consequently this attempt has measured neither a failed
state-box inclusion nor target dynamical instability. It measured loss of
sharpness while enclosing deterministic prefix constants.

## Immediate controlled retry

The proposed identical-arithmetic 1024-bit retry is justified. All uncertainty
in the fixed prefix originates from enclosing its exact real arithmetic;
there is no intentionally varying input state box there. Increasing precision
from 384 to 1024 adds approximately 192.66 decimal digits. If the observed
narrow-ball amplification remains similar, this should strongly reduce its
rounding-origin radii; that is a prediction to check, not a success guarantee.

Keep the same parameters, proposal gate, arithmetic, layer telemetry and
`2^-80` state-box/tangent-cube diagnostic. Preserve the entire first attempt.
A fresh owned output and pushed amendment must bind the changed precision.
The existing one-CPU, 18-GiB, 3600-second envelope is a bounded diagnostic;
reaching its limit is another retained setup outcome.

Extra precision cannot by itself remove the dependence overestimation caused
by an actual `2^-80` state box or `[-1,1]^896` tangent cube. If the higher-
precision prefix and point succeed but the box fails, that separates the next
obstacle from point-rounding failure. Do not repeatedly increase precision as
though those two obstacles were identical.

## Concrete next algorithm if state-box wrapping dominates

Use **eight-phase multiple shooting with signed first-order coefficients and
an enclosed nonlinear remainder**. The residual equations are

`H_i(x_0,...,x_7) = F(x_i) - x_(i+1 mod 8) = 0`.

This evaluates each of eight one-step maps on its own phase box, avoiding the
unnecessary propagation of point-rounding uncertainty through 192 successive
decoder blocks just to obtain the orbit residual.

For one phase, represent each activation as

`a(z) = a0 + A z + R(z),  z in [-1,1]^896`.

Here `a0` and the **signed** coefficient matrix `A` are Arb enclosures, the
state-box scaling is already included in `A`, and `R` is an outward-enclosed
nonlinear remainder. Fixed-weight affine operations multiply `A` before taking
absolute bounds. Thus cancellation in the point Jacobian is retained rather
than discarded at every layer by turning the tangent into an interval cube.

For a scalar nonlinearity, enclose its first derivative at the center and its
second derivative over the complete activation domain. Bound the discarded
term by the Taylor remainder, including propagated previous remainders.
Attention products require their explicit bilinear cross term; RMSNorm,
reciprocal and softmax require their proven positive denominator domains.
This must be implemented as validated Taylor/jet arithmetic, not by subtracting
a numerical linear prediction from an interval forward result. That naive
subtraction retains the original dependency loss.

After the signed point derivatives and derivative-remainder bounds exist,
form the cyclic block Jacobian of `H`. A floating inverse or adapted basis may
propose a preconditioner, but the inverse defect, full-box derivative remainder
and strict inclusion must be evaluated in Arb. Primitive divisors and attraction
remain separate gates. Small numerical spectral radius alone supplies none
of these bounds.

### Bounded implementation checkpoint before a full Jacobian

After the current retry is analyzed, prepare and test the centered algebra on
small exact/synthetic controls. Then propose one fresh target pilot limited to
one stored phase and eight signed point directions (for example `e_0,...,e_7`),
using the precision shown adequate for its constants. Retain each point-JVP
coefficient, radii, exact directions, peak memory and elapsed time. This checks
that cancellation is retained and measures the cost of coefficient blocks.
Use one CPU, at most 18 GiB and at most 3600 seconds unless a new measured plan
justifies another envelope. Do not launch an unbudgeted 896-column/eight-phase
job on the basis of the affine microbenchmark alone.

The complete proof still needs all 896 input directions, or an explicitly
proved replacement bound. Vocabulary operations must stream direction blocks:
materializing all 151,936 vocabulary rows by 896 arbitrary-precision columns
would defeat the memory envelope. Eight directions are an engineering pilot,
not coverage of the omitted 888 directions.

## Conditional low-rank option, with the missing obligation explicit

A numerical rank-one monodromy or concentration on a few decoded tokens can
suggest an adapted basis; neither proves a small complementary operator.
At each phase, retain a candidate rank-r factorization and prove a bound on
**the entire residual operator in all 896 coordinates**. A rigorous tail bound
may come from complete signed Jacobian columns, or from a full-vocabulary
softmax derivative decomposition whose discarded probability-weighted terms
and logit derivative norms are all bounded. Floating singular values, random
probes, a top-token cutoff or observed label periodicity do not bound that tail.

Cheap full-vocabulary probability-tail summaries at the successful point could
screen this route, but they are insufficient without derivative-tail bounds
valid on the full state box. Until such a bound is available, use low-rank
coordinates only as a numerical preconditioning proposal, retaining all
complementary directions in the validated computation.

## Decision rule

1. Constant-prefix/point widths fail at the new precision: retain the attempt
   and revise arithmetic precision or constant evaluation, using measured widths.
2. Points are sharp but the natural box/JVP is broad: proceed to centered signed
   coefficients and remainder bounds, rather than changing the dynamical label.
3. Centered bounds are sharp enough: refine all eight centers in high precision
   and attempt the full multiple-shooting inclusion with a radius informed by
   rigorous defects, then primitive-period and attraction gates.
4. A low-rank shortcut is used only after its full complementary-tail bound is
   explicit and independently checked.

No branch of this decision rule upgrades a numerical proposal or a failed
setup attempt into an orbit certificate.
