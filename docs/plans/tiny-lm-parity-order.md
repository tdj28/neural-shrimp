# Proposed next LM contrast: reversal-even dependence on time parity

11 October 2026. Source/design only; no new native run is admitted.

The qualified ordered 16-step response is accurate, while the initial frozen
and temporal-mean approximations fail. The prior reversal experiment found a
small reversal-odd term relative to a very large frozen error; that does not
bound the reversal-even order dependence. The new saved 16D mean-Krylov model
abstains on operator leakage in all four matched cases. Neither failure justifies
a larger projected model without new qualification.

A direct full-15776-dimensional contrast can ask whether retaining the actual
alternation of even/odd-time Jacobians matters. Reversal preserves this
alternation; parity grouping destroys it without changing any individual
Jacobian, its weight, its spectrum, the initial vector, or the set of 16 states.
Thus the test adds a specific contrast absent from the prior reversal result.
It is not a generic search over permutations and does not test a new nonlinear
training trajectory.

## Fixed objects and four product paths

Use precisely the two seeds28161/28162 expanded-arm256 update600 cases and their
already qualified same-checkpoint eta/4 controls from the temporal-mean pilot.
All are consumed development. Preserve original training tokens/objective/ridge,
exact float64 states theta0..15, eta and the literal normalized observed gradient
direction u. Use J_t=I-eta H(theta_t) acting in the full state space.

Application lists (each listed index acts next, starting from u):
- P: [0,1,...,15], the chronological product.
- R: [15,14,...,0], so R=P^T under the symmetric Hessian contract.
- Q: [0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15].
- T: reverse of Q, [15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0], so T=Q^T.

Compute the predetermined reversal-even contrast
D_even = (Q u + T u - P u - R u)/2,
and the two reversal-odd vectors (P u-R u)/2 and (Q u-T u)/2. Report the full
vectors, norms, signed alignments, gains, and terminal original training-CE
linear readout. Also retain the exact identity
Q u-P u = D_even + [(Q u-T u)-(P u-R u)]/2,
including its squared norm cross term. No additive error percentages.

Pure commuting temporal variation gives Q=T=P=R even when a product differs
from a power of its mean. Therefore a resolved D_even demonstrates directional
order dependence beyond the existing reversal-odd quantity. It does not isolate
basis rotation from other manifestations of noncommutation, identify individual
commutators, explain the entire frozen/mean error, change an operator norm, or
establish forecast/training/geometry utility. This is an intervention on the
linear response cocycle along saved states, not on data order or training.

## Fixed qualifications and descriptive decision

The inherited 1% two-amplitude/ordered-transport/CE qualifications remain exact
prerequisites for all four cases. No reduced-space fidelity gate is needed because
there is no projection. A fresh chronological path must agree with the saved
ordered response at h1,4,16 to relative1e-10 (floor1e-12); failure abstains.
For each transpose pair require |u^T(Pu-Ru)| and |u^T(Qu-Tu)| normalized by the
relevant vector norm products <=1e-8. Retain a failed check and stop interpretation;
these finite arithmetic controls are not interval or full-operator proofs.
Require scaled vector/squared decomposition residuals <=1e-12. Preserve all
acquired returns even if a subsequent finite/identity gate fails.

Primary descriptive gate, fixed before target: ||D_even||/max(||P u||,1e-12)>=.20
in BOTH original-rate seeds. The eta/4 values are matched ordinary controls:
report every result and the original/control contrast; do not require a chosen
control separation as an unannounced second gate. Also report ||D_even|| relative
to the original frozen error, so material directional sensitivity cannot be
misrepresented as explaining that error. A negative result concerns this single
parity-grouping intervention, not invariance to all permutations.

## Fixed native work and admission proposal

Reuse existing exact original P and R as historical references, but acquire a
fresh P for all four cases to validate the native/runtime/state replay. Acquire
Q and T for all four cases, and acquire R only for the two eta/4 cases that lack
it. Counts: 4*16 fresh P + 4*16 Q + 4*16 T + 2*16 R = 224 HVP; zero gradients,
zero CE. All original R reuse must bind the independent ordering audit and exact
state/direction/eta identity against the mean experiment, never only case labels.
The terminal CE gradients already retained by the mean experiment are reused.
Store all intermediate vectors, raw HVP inputs and returns with exact-byte aliases;
no hidden renormalization or restart. Report h16 only as a same-multiset contrast;
intermediate prefixes are retention/debug data, not comparable multiset controls.

Before target: peer-reviewed helper/wrapper, tests, complete actual metadata
preflight, committed plan/card, and a fresh full-size two-HVP profile at the
previously consumed seed28160/update50/arm256. Retain cold costs and exact source/
libraries. Admission forecast 2*224*max_HVP_seconds+30 <=240s, and twice retained
native/input/vector bytes plus64MiB reserve <=192MiB. CPU1/BLAS1/nice10,4GiB RSS,
8GiB AS; proposed profile60s/80+5 external, target240s/260+5 external. Exact final
storage formula must count all 224 evolving inputs and returns, 14 paths*17
retained vector references, profile/source/manifest bytes. Do not assume arrays
are deduplicated unless the actual store guarantees it; a failed profile is
preserved, not repriced by dropping cold calls. No adaptive extra permutations,
new direction, state, rate, seed, holdout, or acquisition budget.

Synthetic controls: commuting time-varying symmetric matrices (mean failure but
all four products equal); alternating two noncommuting symmetric matrices (small
or zero reversal-odd yet large parity-even contrast); explicit dense products
verify lists/transposes and vector/squared identities; mutated states/order/
direction/eta reject; inherited qualification failure abstains; chronological
replay mismatch; each input/return retained before any failure; exact224 count
including the two unmatched historical R paths. These controls and source review
must complete before any native profile or target.

## Approved preparation and fixed cost amendment

Root accepted this full-state contrast on11October2026: source preparation,
peer-reviewed profile, then separately committed target only if its fixed budget
passes. The measured effective HVP cost is max(new two-return profile, the exact
maximum over ALL1088 completed same-shape temporal-mean HVP returns, including
retained cold profile returns). No discarded cold measurements.

Complete lossless-vector upper bound528 =64state +4initial +224subsequent path
+224HVP return +12derived vectors. The source store must use per-row content
references, never additional dense copies. Twice528 full NPY vectors plus64MiB
metadata/closing headroom and actual current profile/source bytes must fit192MiB.
This conservative bound is deliberately close to the fixed ceiling; a storage
failure is retained, never silently compressed away. The64MiB headroom includes
the4MiB finalization reserve.

## Complete prepared caller

Boundary reviewed ddf38a70 helper:6 controls PASS0.11s. Caller additionally requires
all four fresh replay/algebra qualifications before a primary descriptive result.
Original reverse paths are byte-bound to the independent160-action reversal audit;
each of their16 actual state/direction/HVP returns reproduces its stored trace at
the exact original rate. The two quarter controls acquire their missing reverse
paths. All four original1% gates and source-matched native libraries remain bound.

The full historical1308-event inventory and both mean-profile failures remain
mandatory. New maxima use every1088 actual HVP time plus both original cold HVP
profile times and both fresh times. Full path records use the already reviewed
lossless per-row NPY aliases, without copies of historical reference/frozen/CE
vectors. Only three derived vectors percase are newly stored. Actual card/source
bytes enter storage admission, and all closing failure records retain a4MiB
reserve. Actual --help under inherited512MiB soft/8GiB hard AS and isolated
archive import pass. Source/preflight controls inspect all historical metadata
without opening scientific arrays. Initial preparation exposed and corrected
a missing whitespace syntax error before any profile or target.

Card status remains source_prepared until complete caller peer review and
current host admission; the subsequent target requires literal successful
new-profile source/outer/worker pins and unchanged profiled file hashes.
