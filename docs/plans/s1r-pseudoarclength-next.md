# Small local pseudo-arclength decision — 8 October 2026 UTC

**Recommend one two-edge pilot after implementation and independent source review.**
Stopping now remains scientifically valid: the lower connection is unresolved.
The added value of this pilot is a small test of local continuation through the
region where fixed-hidden-rate corrections stalled. It has no training/inference
estimand and does not attempt to reach or identify the lower bootstrap family.
This is design only: no new model, derivative, norm, SVD, solve, trajectory or
profile was executed. The designer authored the gap producer and reviewed its
separately implemented auditor; root independently reviewed both. This is not
another independent outcome review.

The audited saved equation is accurate (q=3.1382112783e-9, backward error
1.6312269079e-16), but all13 finite trials fail, including ratio1.3377827666 at
2^-12. Qualified endpoint tangents still have negative, nonzero hidden components.
The opposite raw/scaled signs at the **nonclosed** failed state reflect different
orientation metrics; they do not bracket a branch turn. That state is excluded
from initialization and prediction. A pseudo-arclength success would assess the
combined new chart/predictor/globalization method, not isolate which change helped.

## Exact starting evidence and metric

The companion `s1r-pseudoarclength-inputs.json` binds committed summaries and the
full old collection manifest72f88bd7…, model/data record00000, predecessor
node0226/04820, sole starting orbit node0230/04918 and phase-carry edges05002/05267.
It also binds gap execution8143a886, producer source5cecd5f0, independent audit
80972cb9 and outcome7b29add8. No failed-state array is a seed. At execution the
full original inventory, selected arrays/descriptors, source bytes and native
phase join must match before model work. Output log-rate remains the exact saved
binary64 value0.6271428571428572; rates retain `10.**float(log_rate)`.

Use z=(theta,h), h=log10(eta_hidden), for the carried native phase, with the fixed
**raw Euclidean73-coordinate metric**: identity row/column scales throughout.
This is a declared coordinate choice, not intrinsic conditioning. At the last
qualified phase0, saved augmented condition estimates are884.743 raw and4412.073
under the previous scaling; phase1 gives850.194 and4372.164. Raw coordinates avoid
introducing the strongly weighted h_scale=.000125 metric into this small pilot.

Requalify the untouched starting pair; derive the initial tangent from its raw
[A,b] using the full73-column right basis. Require the old rank windows1e-12/1e-10,
SVD/null residual1e-10 and raw secant-orientation ambiguity threshold1e-8. Join the
predecessor and last native phases through the saved edge bits. Check the initial
h component against its pinned raw diagnostic summary within2e-11 absolute.
Its absolute value must be at least1e-4. Set
`s0=min(0.005,0.000125/abs(saved carried-phase t_h))` in binary64. The numerator is
one quarter of the nominal last accepted hidden increment and the closest failed
hidden offset. This is a planning scale, not an attempted fixed-h target. It is
bounded even near a turn and is computed only once; never divide by later t_h.

Advance at most two accepted edges. Keep the last accepted arc length without
growth. An edge may fail once and retry at exactly half that length, from the
same last fully witnessed orbit; a second failure ends the unit. Thus at most
four attempted edges and16 arclength correctors. Parameter progress need not be
monotone. Require |h−h_anchor|<=.001 and ||theta−theta_anchor||_2<=.02 in the carried
phase, including trials; leaving this local box is a retained edge failure.

## Corrector and qualification

For parent zP and consistently oriented unit tangent tP, use the fixed plane
`tP dot (z−zP)=s`. Solve G=(R(theta,h), plane residual)=0 with bordered derivative
`[[A,b],[tP^T]]`; no extra continuous-time phase equation is appropriate for a
period-two discrete orbit. Use the already independently written analytic
loss/gradient/Hessian primitive and its explicit hidden-log-rate derivative.
The smooth parameter derivative remains distinct from differentiating rounded pow.

At most eight Newton solves per corrector, and64 globally. Record complete inputs,
returned steps, G, bordered derivative, true linear forcing and backward error
before subsequent gates. For non-negligible ||G||_2 require forcing<=1e-6 and
Frobenius backward error<=1e-12. When ||G||_2<=1e-12 max(1,||z||_2), abstain from
a relative forcing claim and require absolute linear residual<=1e-12 plus the
same backward bound, with exact zero equation/step handled explicitly. Failure
of a linear gate stops that corrector; no regularization or least-squares fallback.

Use merit .5||G||_2^2 and its actual directional slope G^T DG delta. Require a
negative slope and Armijo coefficient1e-4 at factors1,1/2,…,2^-12; retain every
trial. This is explicitly different from the historical infinity-merit solver.
At convergence preserve the existing theta relative-correction gate1e-10,
require |delta_h|<=1e-10, and plane residual<=1e-10. Nearzero G with a small
qualified step may finish without a strict-descent trial. The final state must
pass the original physical closure gates; neither merit nor a small step replaces
them. For each direct/midpoint/half-endpoint/reverse leg define
ell=||z_predictor−z_seed||_2>0, and require ||z−z_predictor||_2<=ell/2.
Thus reverse travel has a positive locality radius despite targeting plane0.
Also preserve the old normalized theta distance from predictor to corrected
candidate<=.02; otherwise reject the local candidate. Eight iterations is a tighter
work limit, not a weaker qualification gate.

Construct the candidate pair as (theta,F_h(theta)), preserving explicit native
indices and carried phase. **This replaces independent fixed-h Newton correction
of both phases.** It avoids requiring a potentially singular fixed-h solve after
the arclength solve. Both native closures and cross-phase transitions still must
pass the existing componentwise1e-9+1e-8 scaling, primitive separation>=1e-6,
phase-distance/cost-gap gates, complete72-mode spectra at both phases, left/right
eigenpair residual<=1e-10 and spectral multiset agreement<=1e-7. Instability is
reported, not rejected. Preserve pair_replay_agreement1e-7. Check raw null tangents
at both native phases and their derivative transport. Use each actual native
phase derivative DF(F(theta_i))DF(theta_i), not an approximate phase swap,
and charge every required analytic phase call. A new auditable spectral adapter
must take the two actual monodromies separately. It must preserve the old
left/right residual and spectrum gates while exposing exactly one norm SVD and
one (M−I) condition SVD per phase, saving their inputs and returned singular
values. Do not call old spectral_checks/eigen_residuals, norm(matrix,2), or cond
in addition: those would add hidden decompositions. Keep those old helpers
unchanged. Require normalized transport
residual<=1e-10 and cosine disagreement<=1e-7. Orient every later
primary tangent using the previous carried tangent in this same metric, never
by resetting the sign of t_h. Ambiguous overlap<=1e-8 rejects the edge.

## Witness each accepted edge, with common controls

One edge contains four arclength correctors: direct target s; midpoint s/2;
endpoint from that midpoint targeting the **same parent plane s**; reverse from
the direct candidate targeting the **same parent plane0**. Use midpoint/child
tangents for the latter predictors by intersecting their tangent line with the
specified parent plane. Reject nonpositive or <=1e-8 tangent-plane denominators.
Reverse travel uses a negative predictor length without flipping the carried
forward tangent orientation. No independent hidden target is substituted. Every candidate retains its own
full pair, exact controls, native carry, tangent, trials and qualification.

Direct versus half-endpoint and parent versus reverse must first agree in h
within1e-10 and pass augmented/locality checks. Since their corrected h bytes may
differ, **do not call that a common-control pair match**. Re-evaluate the comparison
pair's maps/closure/primitive gates at byte-identical direct h (or parent h for
reverse), without a fixed-h solve; then apply the existing phase/replay gates at
those identical controls. These extra checks do not reclassify spectrum at the
slightly changed control: the direct/parent supplies the accepted node and its
full spectrum. If common-control state validation fails, reject the edge.
Require parent/direct, parent/midpoint and midpoint/endpoint phase matches and
consistent cyclic-bit composition. The accepted primary phase must map to the
previous carried primary phase. Preserve native records; store carry separately.

Only after all four nodes, common-control checks and durable edge-complete
record pass may the direct node become last_witnessed. Failed direct/half/reverse
attempts and qualified-but-unwitnessed nodes remain separate. No failed midpoint
or endpoint becomes a seed for a new main edge. These finite witnesses support
local numerical association, not uniqueness or an interval continuation proof.

## Bounded cost, retention and decisions

Proposed ceilings: CPU/BLAS1, nice>=10,60s internal,75s TERM+5s KILL, owned-group
supervisor80s; monitored2GiB child-group RSS, explicitly not a hard memory bound.
Use one exclusive attempt under `/data2/neural-shrimp-data/explore/s1r-pseudoarclength/`
and a sibling launcher. Total128MiB:127MiB Store with its25% finalization reserve,
1MiB launcher and128KiB log. No GPU, paid resource, protected data or new bootstrap.

Global attempted/returned caps include256 analytic phase calls (256 map values
and256 Jacobians),64 bordered solves,17 full orbit qualifiers (anchor plus16
corrector outputs),8 common-control state-only requalifications,34 eigensystems
and102 SVDs. The last number includes34 full augmented SVDs and **68 implicit
spectral norm/condition SVDs** (four per pair in existing spectral_checks).
Implementation must expose/reserve these operations explicitly and retain their
inputs/returns before checks, without changing the old helper. Reusing saved
values is allowed only for byte-identical inputs with an explicit cache record.
All trial evaluations and starting/common-control checks count against256;
no per-corrector allowance overrides that global cap. A cap stop is unresolved.

The historical planning screen is
`2*(1.1480005020275712*max(256/12,102/24)+38.103352084057406*(16/473))+5` seconds,
using the six-state diagnostic and original continuation summaries only. It fits
the60s envelope heuristically; this is not a representative benchmark or a
promise that every allowable operation combination finishes. For storage,
implementation-time synthetic records must show that worst-size phase, assembly,
SVD, eigensystem and solve payloads plus metadata fit the95.25MiB ordinary budget.
If not, amend the design before any target run; do not trim failed evidence.

Stop after two witnessed edges, a second failed attempt on either edge, any
source/phase/rank ambiguity or a resource ceiling. A qualified connected path
extends only this local numerical lineage. A carried h-component sign change
would be a numerical turning candidate, requiring further genericity/quadratic
checks for a fold claim. Continued negative components mean no observed turn
on this small path. A method failure leaves the existing gap unresolved; it
cannot prove disconnection. This unit performs no lower-endpoint association.

Before implementation or execution: root accepts this design; source/tests/card
are then committed/pushed and independently reviewed, with a prospective saved-data
auditor frozen before numerical outcomes. Synthetic controls must cover a simple
fold and nonsingular curved branch, native phase exchange, common-plane versus
common-h matching, strict merit/linear gates, orientation ambiguity, full hidden
operation accounting, and every before/after-return interruption boundary.
Only a separate root launch authorization permits the single attempt. No automatic
retry, scaling change, smaller-step ladder, tolerance relaxation or extra edge is
allowed; any later method amendment must be dated and separately scoped.
