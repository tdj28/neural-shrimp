# Independent refinement audit guide

Prepared 10 October2026; source/data map only, no independent outcome claim.
Producer currently88 closed slices on charts145–156; root independently audited
24 base-only slices on152–154. Chart146 is an informative next audit: all four
base norm gates fail while all four refined gates pass. No existing target
needs rerunning. Root owns the independently implemented arithmetic audit.

## Exact target and binding

Raw root: `/data2/neural-shrimp-data/explore/certificates/s2-period18-quarter-chart-stability-2026-10-10/attempt001-target/run`.
Execution `b8ab6ba630f87304488aef8cae17c886164e6ab1`;
worker manifest `4c1e73612a1b78d86dee337ffb6f861ffee7c6122ceb4d98f07c81893f54b3de`;
outer `b44b231b4a31a02581c7e438896f548d1a419a350a1619ef9e5872d1dad8fb91`.
Compact receipt `docs/reviews/s2-period18-quarter-chart-stability-outcome.json`.
Complete644 stages,648C1/1296unit/144mixed calls, four exact closed slices:
[139/512,35/128], [69/256,139/512], [137/512,69/256], [17/64,137/512].
Authenticate source files against historical Git and every worker/outer inventory;
`inputs.json.gz` contains the native complete chart146 and fixed physical map.
Do not replace its parent binding by merely finding an overlapping chart.

Every slice has161 stage records. For slice index i, use stage offset161*i.
All filenames below describe i=0; other slices use the same relative offsets.
Each `.json.gz` has a corresponding `-payload.json` uncompressed hash record.

## Base proof prerequisites, already supported by root's base auditor

- `0000-base_whole_slice_start`: fixed exact preconditioner R and recentered
  predictor. `0001`–`0003-base_whole_slice_intersection`: three conditional
  root intersections; last Jacobian is whole A=D_U H on its actual saved
  derivative domain. All true roots remain inside the final physical box.
- `0004-base_uniform_tangent`: weighted Neumann proof A U'=e235 and retained
  full235D tangent. The augmented shooting residual H has18 native phase
  blocks F(theta_j,h)-theta_next and one linear gauge equation minus s.
- `0005-base_center_slice_start`, `0006`–`0008-base_center_slice_intersection`:
  same center root and its final augmented Jacobian A_c.
- `0009-base_physical_inputs`: actual post-decoding whole and center state/rate
  boxes and full tangent passed to physical derivatives, exact slice center
  and halfwidth. These actual boxes, not rounded displayed values, are the
  domains for the first MixedJet pass.
- `0010`–`0027-base_phase_derivatives`:18 phases in native order; center_J,
  whole_J_mixed_diagnostic, whole_total_Jprime. First total derivative uses
  the original full tangent, including actual h'.
- `0028-base_frame_proposals`, `0029-base_frames`: exact dyadic proposals and
  certified cyclic frames. Original base norm stages0030–0084 remain failed.

## New second derivative arithmetic

Source: `src/neural_shrimp/rigorous_tangent_variation.py`; composition in
`rigorous_refined_stability_slice.py`; total physical derivatives in
`rigorous_correlated_cocycle.py:total_jacobian_derivative`.

`0085-tangent_refinement.json.gz` contains everything for this step:
center_derivative, native_forcing_rows, forcing, second_derivative,
centered_derivative_enclosure and refined_derivative_box.

Independently reconstruct the center solve A_c v_c=e235 using the same fixed
R and exact positive chart weights. For either solve with RHS b, set
D=I-R A, q=max_i sum_j |D_ij| w_j/w_i <1, v=R b, and
rho=||D v||_w/(1-q). The initial enclosure is v_i+[-w_i*rho,w_i*rho].
Apply three intersections with v+D*box. The source's v is an interval RHS,
not necessarily an exact center; preserve its entire enclosure. Recompute
bounds with directed arithmetic, and distinguish saved producer256-bit replay
from an independently widened384-bit comparison.

For each phase, write J=D_theta F at the actual whole state/rate domain and
J'=dJ/ds from the first MixedJet pass. With v_j=theta'_j and h'=U'_234,
let B_i,k=(delta_i,k-J_i,k)/h for hidden rows i<9, zero for output rows.
This recovers the hidden loss-Hessian rows because F_hidden=theta-h grad L.
Then b'_j=-B v_j and the curvature forcing is
c_j=J' v_j+b'_j h'. The last gauge row is exactly zero.
J' already includes one theta/rate cross term; b'_j h' supplies the SECOND
one. Omitting it is wrong; adding another copy after this formula is wrong.
There is no hidden-rate second derivative because F is affine in h.
Thus H''[U',U']=c and A U''=-c. The saved forcing is computed on the ORIGINAL
uniform tangent; no refined tangent is used circularly to obtain this bound.
Use the same whole augmented A and weights to enclose U'' by the above solve.

At every s in the slice, U'(s) lies in center_box+Usecond_box*[-delta,delta].
Intersect this with the original full tangent. Validate every retained
coordinate, including physical h'. Smoothness follows from the same true-root
branch and uniform nonsingular A; the intersection does not prove new roots.

## Actual refined derivatives and final metric

`0086-refined_inputs.json.gz` is the ACTUAL post-decoding physical whole box
and refined full235D tangent handed to the second MixedJet pass, with exact
halfwidth string. Whole state/rate domain must be the outward-restored
`0009` actual whole box; this additional restore can widen Arb radii, so do
not demand bytewise equality with the earlier serialized box. The tangent
must be the outward-restored `0085` refined derivative enclosure;
account for any serializer/restore widening rather than demanding a false
exact equality of internal Arb representations.

`0087`–`0104-refined_phase`:18 fresh native total derivatives. For phase j,
use theta indices13*j:13*(j+1), rate index234, tangent matching those indices,
rate-prime index234. MixedJet's physical gradient holds h fixed; its direction
includes h'. Each row saves actual whole_J_mixed_diagnostic and
whole_total_Jprime. center_J is the SAME original bound from0010+j.
Independently recompute these18 derivatives on the saved actual domains.

`0105-refined_frames` through `0160-refined_norm_metric`: reuse root's independent
norm audit with the refined total derivatives, unchanged center_J, same exact
halfwidth and exact cyclic frames. Verify actual frame identity, all product
cross/remainder terms and strict P>0/P-M^T P M>0; do not inherit a pass boolean.
For j=0..17 the repeated triple is0106+3*j norm_phase_start,0107+3*j norm_phase,
0108+3*j norm_product. Final0160 retains strict refined metric witnesses.
`slice-00-result.json.gz` duplicates the composed result for consistency checks.
A successful audit of all four covers chart146 only. It does not automatically
certify other producer-only slices, the neutral fold, or complete shrimp geometry.
