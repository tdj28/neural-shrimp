# Task14: independent numerical near-cusp endpoint check

Prepared 2026-10-13; exploratory reproduction of exposed producer findings.
Only the dense timing/derivative-cost profile is being proposed for admission.
No endpoint coefficient run, cusp location, optional P28 check, model run or
profile has yet been executed by this implementation.

## What this can establish

This asks whether two saved period7 fold candidates of the hostile soft-token
map reproduce under our separately implemented, full-vocabulary float64 CPU
ManualQwen evaluator. The eventual endpoint calculation may support a numerical
change in the oriented quadratic coefficient and nonzero endpoint cubic terms.
It does not prove that a cusp exists between them, locate a cusp, certify its
unfolding, prove an interval bound, or establish behavioral utility. Neither an
approximate multiplier at+1 nor endpoint signs alone are an exact cusp proof.

Read before preparation: Claude's full19:33 assignment, producer source at
cb6a105c3c7c077e9db7baadaeb0fd943e726f4c, the source-only export manifest
4fe637389d1a45fc3fe7a95de0c08d6dbf08bc0603aa1c9b1b3819758d2c77e2,
our previous Task7 sources and profile receipt, and curve metadata. The assignment
exposes b*sign(a)=+31.17 to-58.67, a about.57/.53 and c about-5.77e5/-5.30e5
with reported spread.007. These are producer reports, not our results. Initial
curve-header inspection also displayed producer rows0–1; no own candidate
arrays or cubic-record values have been decoded. This is not an unseen test.

## Fixed map and candidates

Use unchanged codex/crosscheck/qwen_map.py and the accepted Task7 asset loader.
Qwen2.5-0.5B-Instruct revision7ae557604adf67be50417f59c2c2f167def9a775,
full151936-token readout, hidden dimension896,24 layers, steering layer12,
window1, exact37 prefix IDs from the evaluator, hostility-vs-kindness vector
from vectors-l12-p1b. Model/config/vector content hashes are checked before
construction. All weights, RMS normalization, rotary construction, attention,
softmax and AD operations remain CPU float64. No Transformers or producer
analysis implementation is imported; no checkpoints are downloaded.

The candidate authority manifest is
`/data2/neural-shrimp-data/codex/llm-cusp-crosscheck/input-authority-001/manifest.json`,
SHA1bb981cc55cbf617069ab09c1dbc23a1d9b79e39394c4e6e5b2863d3ec5866e4.
Its read-only state/curve copies equal the corresponding authenticated returned
archive members. Only row13 is decoded by the profile. Rows13 and14 are the
only planned later numerical targets; no search or data-dependent row choice.

| Row | log10(T) | Physical alpha |
|---|---:|---:|
|13|.1893809804983369|.06661104169386665|
|14|.1893794755174379|.06660969640358283|

Curve theta[1] is10*alpha; all own parameter derivatives use **physical alpha**,
including its influence on the prefix cache. Quantitative comparison of a to a
producer value requires explicit confirmation that its derivative used the
same physical parameter. Our a_alpha is defined regardless of that comparison.
The prefix/branch ancestry is separately hash-bound at admission.

## Profile only

At fixed row13, build the complete896-column derivative of F^7 by56 batches of
16 coordinate directions. This is a direct JVP of the composed map, not a
product of rounded one-step matrices. Retain the exact state, parameters,
prefix cache, full request/return arrays for every batch, and assembled matrix.
The matrix can later be reused only with exact state/parameter/evaluator/source
identity; a state refinement requires new derivatives and separate admission.

Then retain one deterministic composed JVP and compare it with Mv, and one
fixed second, third, mixed second and physical-alpha AD cost probe. The vectors
are fixed coordinate formulas, not eigenvectors. Do not diagonalize M, select a
critical direction, compute a/b/c, or promote a fold during this profile. Full
returned arrays precede finiteness/resource refusal. Failure/timeout and partial
columns stay immutable; no automatic retry or missing-column extrapolation.

Earlier Task7 CPU2/period2 batch16 measured5.329830808s with maxRSS5,825,380KiB.
Thus56*(7/2)*5.329830808=1044.646838s is a first-order dense timing estimate,
not an observed period7 runtime. Twice that plus300s=2389.293677s fits the
proposed2400s external hard deadline. The extra allowance includes loading,
serialization and derivative probes; nested third-AD cost has not yet been
measured and may exhaust the cap. The full profile result, including that
uncertainty, is required before a subsequent target card.

Envelope: workerCPU2, one lightweight supervisor (three auxiliary slots),
nice10,12GiB hard address space and sampled RSS watchdog,28GiB startup
MemAvailable and16GiB runtime floor,64MiB total output, worker60MiB with4MiB
closing reserve,2400s deadline plus at most10s owned cleanup. Task1's seven slots
remain untouched. Sampled RAM/disk monitoring is a watchdog, not a hard kernel
RAM or filesystem quota. Source snapshots and birth-bound worker/session
cleanup reuse the accepted Task7 launcher machinery; Inference owns lifecycle.

## Subsequent endpoint design: prepared, not admitted

For each row separately, before coefficient interpretation:

- Record full-state normalized closure ||F^7(x)-x||/max(1,||x||), requiring<=1e-10;
  for primitive7 require the full-state divisor1 separation>1e-6. Retain all
  phases. No low-dimensional projection or logits-only residual substitutes.
- Use all896 eigenvalues of the composed float64 matrix. Require a real
  multiplier within1e-8 of+1, imaginary part<=1e-10, isolated by>1e-3, all other
  multipliers>1e-3 away from the unit circle, and dense eigenpair residuals<1e-8.
  Record critical modulus/angle, next-largest modulus and nearest-other-unit
  circle distance. Hyperbolicity is required; attraction is a separate label.
  Reject a flip/complex pair/nearby second unit mode rather than switching modes.
- Normalize ||r||2=1,l^Tr=1. Set the largest-absolute coordinate of r positive
  for a deterministic recorded orientation. Report a_alpha,b,a_alpha*b and
  b*sign(a_alpha); simultaneous r,l reversal flips a,b but leaves c unchanged.
  Report left/right angle and eigenpair conditioning. No division by a if its
  sign is unresolved.
- Compare dense Mv with fresh composed JVPs in r and one fixed probe; retain
  central-difference ladders to diagnose the effective float64 precision.
- Compute full vectors Brr=D²F^7[r,r], Crrr=D³F^7[r,r,r] and B(r,h2) using
  nested AD. Solve the bordered (I-M)h2=Brr-2br,l^Th2=0 and retain h2,border,
  full residual and orthogonality (candidate numerical threshold1e-8).
- Also solve the near-multiplier homological equation with mu²I-M using the
  same M/Brr. Report mu-1,h2 differences, residuals and c sensitivity. This is
  a sensitivity diagnostic; neither solve turns mu≈1 into exactmu=1. Evaluating
  B in both h2 directions will be explicitly counted on the later target card;
  neither requires a second dense896 Jacobian.
- Use c=l^TCrrr/6+l^TB(r,h2)/2, retain its direct and center-manifold components
  and cancellation ratio. In x(u)=x*+ru+h2*u²/2+..., the additional cubic
  term proportional to b*h2 has zero left projection because l^Th2=0; thus the
  formula at an exact fold also applies at nonzero b under this gauge.
- Compare AD vectors and projected contributions with **every** fixed central
  JVP-difference step. Proposed relative step ladder.01,.003,.001,.0003,.0001,
  .00003,.00001; no flattest-pair selection or silent step deletion. Use an
  explicit finest-two agreement/convergence rule in the reviewed target card,
  retaining failure/cancellation diagnostics. Outcome-dependent tolerance or
  refinement changes require a disclosed new plan, not a silent continuation.

If original row closure/crossing gates fail, preserve that result and stop the
coefficient qualification. No Newton correction or parameter search is
currently authorized. Any bounded refinement and comparisons will be separately
costed after the profile; raw successful profile M remains bound to originalx.
Cubic values at both endpoints do not replace a continuation/intermediate-zero
argument or two-parameter cusp nondegeneracy check.

## Preparation controls

Tiny polynomial controls cover nested mixed/third AD, a nonnormal conjugated
fold with analytic c, normalization/orientation, near-mu solve sensitivity,
real+1/hyperbolic classification, coordinate-column order and failed-return
retention. They load no model or target arrays. Initial source-only testing
caught a `profile.py`/stdlib `profile` name collision (three failures) and then
a misspelled C-vector variable (one failure). Renaming to`dense_profile.py` and
correcting that identifier changes no mathematics or tolerances; all attempts
are disclosed here. Final control status is recorded with the source release.
