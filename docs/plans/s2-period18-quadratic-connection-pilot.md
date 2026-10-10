# 10 October 2026 UTC — quadratic predictor for scalable period18 continuation

This exploratory pilot addresses a different bottleneck from the successful
root-localization unit: the distance from the original center h≈71.338 to the
certified fixed-output fold h≈75.764. A uniform width1/256 root branch and a few
tiny attractive subintervals do not establish that boundary connection.

## Fixed model and proof semantics

Use the unchanged original `certificates/s2-period18-receipt.json`, SHA256
f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277, its complete
raw inventory including `receipt.json`, and original model inputs SHA256
9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f.
The center actual hidden rate is1254985102288567/17592186044416; output rate is
exactly1848862009810323/2251799813685248. These are the same 13D objective,
consumed data and native18 phases. No exponential-ray redefinition, phase
permutation, new data, trajectory or fresh orbit selection.

New source is `rigorous_quadratic_connection.py`, runner
`tools/certify_s2_period18_quadratic_connection.py`. All historical modules stay
byte-identical. The explicit analytic loss gradient comes from the reviewed
`FoldTanhMap.jet_step`; the reviewed `FlipJet` third-order chain rules are used
only as scalar path jets (empty base-gradient fields). The actual hidden-rate
jet is (h+d,1,0,0), with the fixed output unchanged. No exponential-rate model
is called. Existing `MeanValueHiddenRateMap` and C1 Hessian bound the shooting
Jacobian on error segments.

## Exact quadratic-predictor argument

After two bounded Newton proposal steps, propose exact dyadic preconditioner C
and slope S≈−C H_h. The affine path's second derivative H2 proposes exact
T≈−C H2. Neither proposal is accepted as an exact cancellation equation.
For q(d)=U+S*d+T*d²/2, retain the full Arb residual coefficients H0,H1,H2 at
d=0, then enclose H3 on the whole quadratic predictor interval. Initialize
path-state derivatives as q'=S+T*d, q''=T, q'''=0; the third chain rule therefore
includes acceleration cross terms. Taylor's theorem gives

`H(q(d),h+d) in H0 + H1*d + H2*d²/2 + H3(D)*d³/6`.

Use this residual enclosure in the same uniform Krawczyk identity as earlier,
on q(D)+E with centered E=[−r,r]^234. Strict coordinate inclusion plus
`||I-C*A||inf<1` proves a continuous locally unique root at every parameter.
All proper divisors1,2,3,6,9 are excluded from full phase boxes. Identify the
original anchor, and later the first chart's exact endpoint root, only when
all234 coordinates lie within the new uniqueness domain at the same exact
parameter. Mere overlaps never suffice. Endpoint residual enclosures use the
same rigorous third-derivative bound with the endpoint's signed d.

No attraction gate is included in this scaling pilot. A successful result is
an enlarged connected primitive branch of the same orbit, pending independent
review. It is not an added attracting instance, a fold/flip connection or a
whole shrimp certificate. Every matrix and residual coefficient is retained.

## One disclosed settings choice

Two charts, halfwidth2^-8 (four times the last affine chart), radius2^-24
(64 times smaller), 256 bits, two proposal Newton steps each. Chart0 remains
centered on the original exact center; chart1 center is h0+2*delta and its
anchor is chart0's right endpoint at h0+delta. Only attempt chart1 after all
first-chart root/primitive/anchor gates pass. If both pass, total connected
width is1/64. This is one exploratory setting, not a width/radius grid.

The radius is a proposal motivated by replacing a quadratic-in-width residual
bound with a cubic remainder and retaining exact cancellation defects. It is
not inferred as a theorem from previous outcomes. If the image fails but the
defect contracts, residual/remainder control is the next obstacle. If the
defect does not contract, a better Jacobian enclosure/preconditioning strategy
is needed; blindly increasing predictor order alone cannot fix that bound.
Preserve both diagnostics, even when neither passes.

## Resource profile and target card

First one synthetic profile: the same24-sample zero-label ridge1 model as prior
engineering profiles, native18 repeated zero states. It is explicitly a fixed
point, not a primitive18 observation. Both quadratic charts run to exercise all
paths. Complete counts are144 wrapped C1 acquisitions (288 unit Hessian calls)
and108 one-step directional jets; both uniform-root gates must pass for the
profile to qualify. Expected synthetic primitive/anchor failures are harmless
only in this engineering mode.

CPU/BLAS1, nice>=10, precision256, no GPU/paid work. Profile20s internal,
25+5s external; target120s internal,135+5s external;300 ticks,2GiB RSS at
operation checkpoints. Worker128MiB with120MiB ordinary allowance plus1MiB
launcher; each compressed artifact retains a checksum and uncompressed256MiB
cap. The profile must have matching source/library/thread/settings, finite
bounded time/RSS, exit0 and no owned PIDs. Time projection4*t+2<100s. Storage
projection uses max(3*profile bytes,3*31,341,878 prior real-chart bytes)<120MiB,
explicitly correcting the earlier synthetic compression underestimate.

Freeze source/tests/card before the profile; root reviews the mathematical
adapter before acquisition. After profile completion, check launcher exit,
owned-PID absence, all hashes/decompression identities, immutable bits and
exact source/library settings; commit the measured receipt and exact profile
pin before the target. Recheck shared host RAM before either launch and avoid
overlap with other>1GiB work; do not disturb foreign processes. No automatic
retry or changed settings within an attempt. Any later adaptation gets a dated
card and preserves this attempt.

Command uses `/data2/neural-shrimp/.venv/bin/python` with `PYTHONPATH=src`:
`tools/certify_s2_period18_quadratic_connection.py --mode profile --out
/data2/neural-shrimp-data/explore/certificates/s2-period18-quadratic-connection-2026-10-10/attempt-001-profile`.
If the gate passes, target uses exclusive `attempt-001-target` plus
`--profile <profile/run> --profile-manifest <exact SHA>`. The same standalone
driver has a retained-output replay mode; no replay is launched by this card.

## Source controls and continuing decision

Controls cover an exact quadratic branch, two joined charts, nonlinear
primitive2, nonzero cancellation defects, actual-rate/model parity, independent
Arb-series third derivatives, acceleration terms, failed anchors, interruption,
source/profile identity and exact operation caps. Early synthetic failures
revealed fixture-only unsupported jet division/missing schema and a real
default-settings omission; all were fixed before freezing. A proposed generic
Arb `**` on a sign-changing interval returned NaN; explicit interval products
are used instead, with no target evaluation or hidden alternative attempt.

After the fixed target, use its retained error/contraction bounds to choose the
next scaling method. Do not spend on full stability coverage until root charts
are sufficiently large. Near the fold a nonsingular augmented chart and exact
root containment into the fold uniqueness box will still be required.
