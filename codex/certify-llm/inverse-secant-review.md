# Independent inverse-secant review

Reviewed `inverse_secant.py` SHA-256
`01065e31a70b7b7b0b394609e2afef78ea96669455a72fd4c17546b94d8d914b`
and `tests/test_codex_llm_inverse_secant.py` SHA-256
`19c6f9f518ebad94618c734d24f94496407e68bc7426b0c58bf59d5e0e4387f8`.
Review scope is the pure proposal kernel. No checkpoint, Qwen evaluation, actual
proposal array, or retained target outcome was loaded for this review.

No mathematical blocker was found in that scope. Independently reran the secant
and existing shooting-correction modules: **32 synthetic tests passed**. A
separate check made **130 exact/enclosure comparisons**, including a displacement
of `2^-1800` from a centre near one, an independently assembled four-phase,
two-coordinate noncommuting cyclic inverse, and all coordinate actions of its
stored rank-one update. These checks do not establish target convergence.

## Mathematical and implementation scope

For `H_i(c)=F(c_i)-c_(i+1)`, the baseline action is `R0 h`; its negative agrees
with the existing shooting correction. A secant uses exact dyadic centre
difference `s` and the midpoint `y` of the full residual-difference enclosure.
The intended update is

`R_new = R + (s-R y) y^T/(y^T y)`.

Scaling both `s` and `y` by the same exact power of two cancels in this formula
and protects tiny secants from binary64 underflow; no binary64 conversion is
present. The actual update stores exact dyadic midpoints of enclosed `u` and
`w` factors. Its retained scaled secant defect documents the resulting rounding:
the implementation does not assert that the exact secant equation holds for
all nondyadic divisions. All source residual enclosures remain in the record;
midpoint use is restricted to forming the proposal inverse.

The signal test compares the residual-difference midpoint magnitude with the
actual outward radius of that difference. The norm cap uses
`||u||_infinity * ||w||_1`, with outward Arb arithmetic, on the stored midpoint
factors. The proposed step cap uses an accelerated upper bound and a baseline
lower bound. An uncertain cap comparison therefore does not admit the larger
step. Equal cap values are allowed; these are engineering caps, not strict
certificate inequalities.

Zero steps, zero/noisy secants, the history cap, oversized updates, and oversized
accelerated steps have explicit rejection/reset or baseline-fallback records.
The history cap clears the whole active history and rejects that incoming
secant; it does not evict one pair and silently keep the rest. Collinear secants
are admissible for this rank-one formula: there is no least-squares Gram inverse
to become singular. This does not guarantee that the updated inverse is
nonsingular or useful. There is no claimed convergence theorem.

Active inverse storage is `O(m*p*n)` beyond the existing phase matrices. No
`(p*n)`-square matrix is constructed by the kernel. Audit records retain full
input/enclosure histories and grow with the number of calls, so a future driver
must bound the total number of imported updates and new proposals as well as
`max_history`. The constructor deliberately retains references to the exact
`A_i` and `B` matrices; their immutability is a caller requirement. The evidence
label is snapshotted but is not an authentication mechanism.

## Requirements for a future authenticated driver

1. Use a new committed plan and exclusive attempt. Pin the original base plan,
   complete parent receipts, driver/evaluator/correction/importer/kernel source
   hashes, exact parameters, model/vector/config hashes, prefix IDs, steering
   layer, arithmetic versions, precision, and CPU/resource envelope. Keep all
   existing completed parents immutable. A source hash declaration alone is
   insufficient: match it against both the captured upstream file and the
   reviewed current source.
2. First admit the whole completed `center_refine`/`center_resume` lineage with
   the existing complete-parent checks, including successful receipts, every
   artifact hash/size, complete ordered phase inventories, source-bound plan,
   prefix receiver binding, and exact reuse manifests. Refuse failed, timed-out,
   partial, cyclic, cross-map, or changed-backend parents in the first version.
   `load_completed_parent` returns the latest tuple; it does not by itself expose
   every historical pair needed for acceleration. Historical tuple extraction
   needs its own checked enumeration after that full admission.
3. Select consecutive complete tuples by a rule fixed in the new plan. Preserve
   each tuple's attempt/receipt, iteration, phase index, exact input hash and
   image hash. Do not choose only pairs that improved residuals, discard noisy
   attempts without records, or count a reused tuple as a fresh evaluation.
   Deduplicate inherited aliases by their full input/image provenance; identical
   centres encountered as separate genuine tuples must still produce the
   kernel's explicit zero-step decision if selected.
4. Decode exact centre mantissa/exponent records without converting to floats.
   Decode every image ball outward, checking exact rational endpoint
   containment against its original saved encoding. Require that each image
   was evaluated at the exactly matching centre. Recompute every full residual
   `F(c_i)-c_(i+1)` from that tuple's own complete centre list and images. Preserve
   radii and source-image containment, including the existing one-ULP outward
   reconstruction allowance; do not compare against an unnecessarily decoded
   wider image as if that were the original producer's input.
5. Build fresh exact `A_i,B` from the already bound preconditioner. Replay the
   selected chronological secants deterministically with plan-pinned history,
   signal, update-norm and step caps. Do not trust serialized low-rank state or
   an `accepted` flag without replay. Keep matrices immutable. Retain every
   accepted/rejected/reset decision, the full residual enclosures, scaling,
   factors, cap witnesses and chosen proposal before evaluating any new point.
   Persist admitted inputs before calling the kernel so an exception before a
   decision record cannot erase the attempted calculation.
6. For every new proposal, save all exact centres before starting the complete
   fresh eight-phase evaluation, then save each input/image and duration as it
   completes. Recompute the residual only after all phases complete. Keep every
   comparison, including regressions and baseline fallbacks. The first version
   should not admit a later partial failure as a completed parent. All status
   fields must keep `certificate_passed=false`: improved residuals or accepted
   secants establish neither existence nor primitive period nor attraction.

A future plan should separately state whether history is reconstructed from
fixed-R ancestors only or may include a completed inverse-secant parent. The
latter requires a reviewed source-bound replay format; it is not covered merely
by accepting the existing resume schema.
