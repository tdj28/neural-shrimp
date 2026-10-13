# Independent review: exact-real Qwen map and JVP

Reviewer: Codex subagent `literature_cascades`. Scope: source review and tiny
synthetic CPU calculations only; no target checkpoint was opened and no target
model evaluation was performed. Date: 2026-10-13.

## Reviewed sources and conclusion

The inspected `ball_qwen.py` snapshot has SHA-256
`2c8ee93c0379dd3826f16d378301be8d764cc476b8524a438e83cbea4226b91e`.
The independent reference `codex/crosscheck/qwen_map.py` has SHA-256
`493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e`.
The reviewed `ball_linear.py` has SHA-256
`5d54200f5880d818eabc97e7eed2f1cf02b83b617b39380bef38d49a7ac45ff2`.
Line references below refer to this snapshot, which already includes the
constructor inventory and finite-value hardening. Cache immutability guards
and strict integer-prefix admission are being added by the root agent; those
must be checked in the final run source rather than inferred from this hash.

**No incorrect architecture formula or state-JVP formula was found.** The
source is suitable for a bounded, hash-pinned enclosure feasibility pilot,
subject to the explicit immutable-input conditions below and a pushed run
card. This is not approval of an orbit certificate: no inclusion, primitive
period, or attraction proof has yet been evaluated by this code review.

## Formula and enclosure checks

- `linear_sequence` (111-123) packs each token's primal/tangent into adjacent
  columns; bias is added only to the even primal columns. `ball_linear.affine`
  uses exact binary16/32/64 constants and rigorous Arb block products. It does
  not use a floating matrix product or an empirical rounding allowance.
- RMSNorm (105-108) computes `weight * x / sqrt(mean(x*x)+epsilon)` and
  differentiates the entire expression. The denominator must be proved
  strictly positive. General interval multiplication of a variable by itself
  may widen more than a dedicated square; it remains an enclosure and may
  conservatively refuse a wide box.
- RoPE (154-161, 195-203) encloses the real expression
  `exp((-2*j/head_dim)*log(base))`, followed by exact-integer-position sine
  and cosine. The first/second-half rotation signs match the independent
  reference. Frequencies and trigonometric tables are not imported from
  floating inference caches.
- Q/K/V projection shapes, heads, KV sharing and output projection agree with
  the reference. `kv = head // groups` (225) matches repeated KV heads.
  Attention uses real `1/sqrt(head_dim)`.
- Prefix attention admits only keys through the current prefix position
  (222); a subsequent one-token state admits the complete causal prefix and
  itself. The current token has fixed rotary position `len(prefix_ids)`.
- The residual additions, post-attention RMSNorm, SiLU-gate product and second
  residual match the independent computation (230-237). Steering is added
  after the selected zero-based decoder block, at every prefix/current
  position (238-239). Prefix K/V values are computed before that block's
  steering, and later layers see the steered residual, matching the reference.
- The final tied-embedding logits include every vocabulary row. Final
  softmax is over that full vector; multiplication by `embedding.T` forms the
  full expected embedding, with no selected-token approximation. The affine
  kernel supports this transposed noncontiguous view without changing the
  exact stored constants.
- Temperature is the real expression `exp(log(10)*log10_temperature)` (155).
  The binary64 input parameter is exact dyadic; the temperature itself is an
  Arb enclosure of a real transcendental expression. It is deliberately not
  the pre-rounded Python result of `10.**log10_temperature`.
- Softmax (85-102) uses an exact common midpoint shift only for conditioning.
  Its derivative is correctly evaluated as
  `p_i * (dz_i - sum_j p_j*dz_j)`. No derivative of the chosen shift is needed:
  softmax is invariant under a common shift. The floating comparison used to
  select that shift is not a proof gate; any finite common shift is valid.
- Reciprocal, exponential and inverse-square-root derivatives (61-78) have
  the correct signs/powers. Domain checks refuse a denominator containing
  zero or one whose strict positivity cannot be established.
- Prefix tangents are zero for a **state** derivative at fixed alpha and
  fixed temperature. Cached prefix balls still carry arithmetic enclosure
  uncertainty. This API does not implement parameter derivatives; it must
  not be used as though alpha/temperature tangents had been propagated.

## Admission and cache requirements

The first review identified that missing tensor inventory/shape checks could
allow the many internal `zip` calls to silently represent a different network.
The inspected snapshot now checks the exact key inventory and every expected
shape, binary16/32/64 dtype and finite tensor value (164-185). These are required
admission gates, not optional assertions. The scalar converter also now
rejects `np.longdouble` rather than silently narrowing it (23-24), and rejects
nonfinite Arb/scalar inputs.

The driver must load only hash-pinned assets, preserve their stored binary
values, and make the admitted arrays read-only for the entire calculation.
Configuration dimensions, prefix IDs and steering layer must be actual
integers, not nonintegral values silently truncated through `int`. The steering
vector must be admitted as finite stored binary values without narrowing a
wider numeric type. The weights, configuration, vector, alpha, temperature and
prefix must remain unchanged while a cached prefix is in use. The initial
snapshot checked precision only. The root agent is adding identity/parameter
cache guards and strict integer-prefix admission. Those guards complement the
bounded driver's immutable-object precondition; they are not a claim of
resistance to arbitrary mutation of private objects or their backing arrays.

The cache must be reconstructed after changing Arb precision. It must not be
reused for another parameter value. The bounded pilot does not need a generic
mutable-parameter cache API: one fresh, immutable map instance per admitted
parameter/precision is adequate.

## Synthetic evidence

The affine kernel's 59 tests passed (CPU1, no model load). They include exact
rational dot-product oracles, every corner of a small interval box, extreme
binary64 exponents and cancellation, primal-only bias, transposed/noncontiguous
float32 constants, cache mutation detection, bounds on retained entries, and
invalid/nonfinite input rejection.

The root's combined tiny-transformer/affine suite initially reported 65 passing
tests. The tiny map compares both state values and analytic JVPs with the
independent Torch implementation, checks contained interior state/direction
points, and tests precision-cache invalidation and softmax derivatives. The
Torch comparison is a formula regression, not a rigorous rounding oracle.

I additionally recomputed the complete tiny prefix plus current token in one
causal sequence, with no prefix cache, at 192-bit precision and one CPU thread.
Every final primal and tangent enclosure overlapped the separately cached
forward result. The largest cached radius was approximately
`3.183788584292172e-57`. This directly exercises the causal-cache equivalence
and zero prefix state-tangent convention. It is synthetic evidence, not target
model evidence.

## First-pilot interpretation and required observations

The proposed first pilot at 384-bit precision should be explicitly bounded to
prefix construction and one target phase, with a point evaluation and an
input state box of radius `2^-80`, plus a precisely recorded tangent set. Save
per-stage/decoder enclosure widths and elapsed time, including prefix widths,
so a first widening or positivity refusal is localized. Record whether the
tangent represents a single direction or a full box: they answer different
questions. Retain the attempted state/tangent balls and exact source hash even
if the enclosure fails. Every stopping condition must be reported directly.

The known floating center closure of order `1e-13` is vastly larger than
`2^-80` (about `8.27e-25`). That radius therefore cannot be presented as a
candidate successful inclusion around the unrefined center. This pilot
measures arithmetic enclosure width, directional/box wrapping and cost. A
failed inclusion at that center says nothing against the orbit's existence.
A subsequent proof needs high-precision center refinement, a radius informed
by the rigorous residual, and validated preconditioning/inclusion, primitive
period and attraction gates.

A returned box enclosing `DF(X)V` is an operator-action enclosure for that
particular state and tangent set. It is not automatically a dense Jacobian,
a contracting monodromy or a Krawczyk inclusion. Large interval wrapping is
possible despite the numerical orbit's small spectral radius; extra precision
reduces arithmetic rounding but does not eliminate dependence overestimation.
The observation log must keep these two causes distinct.

## Arithmetic throughput, not a target forecast

Synthetic CPU1 Arb128 timings for two matrix columns were:

| Matrix | Weight chunk | Elapsed |
|---|---:|---:|
| 896 by 896 | 32 by 256 | 0.103 s |
| 4096 by 896 | 64 by 896 | 0.472 s |
| 896 by 896, retained exact cache | 64 by 896 | 0.113 s cold / 0.0226 s warm |

A separate 896-square test with nonzero input radii took approximately 0.104 s
at 128 bits and 0.103 s at 256 bits. These results measure conversion and affine
ball arithmetic only. Prefix token count, attention, nonlinear operations,
vocabulary softmax, actual parameter storage and wrapping are not included in
an extrapolation. The parent should use the admitted first pilot to set any
later model-run resource plan.

## Final pre-pilot review addendum

The following final source versions were reread in full:

| File | SHA-256 |
|---|---|
| `ball_qwen.py` | `3250d8b262e4a940b83de84005ad6f3155bc70635bfc623b0a000804cfc28c3e` |
| `pilot.py` | `09ccc111a314f68a1b721c9c8bdb6de465ea874264d0e430007f4591308eb6ea` |
| `launch.py` | `86e83d17b5a5415734ddb42036cd181c3aa9462aaa666e7965bc6ef699ef270e` |

**Approved for the specified bounded setup pilot after its source and run card
are privately pushed. No remaining pre-pilot correctness blocker was found.**
This approval does not cover a certificate assertion or broader runs.

The previously outstanding pre-pilot items are resolved:

- Strict integer prefix IDs, exact tensor inventory/shapes, finite binary
  arrays, rejection of wider scalar floating types, precision guards, and
  common parameter/array-identity cache guards are implemented. The driver
  owns the admitted read-only weight views and performs no mutation after
  prefix construction. Arbitrary external/backing-array mutation remains
  explicitly outside this API contract. Pinned configuration and vector
  loading supply the documented exact-constant input types for this pilot.
- Every numerical recovery cycle checkpoints its latest full state and return
  history. Nonconvergence is explicitly failed; the final nine-state proposal
  and measured divisor/closure diagnostics are stored before their gate can
  refuse continuation. No failed proposal is silently replaced by a default.
- The exact state and tangent input balls are written before the box-JVP
  attempt. The tangent cube is explicitly `[-1,1]^896`; its enclosed image
  supplies a conservative one-step induced-infinity-norm upper bound.
- Input metadata records Python, python-flint, the exposed FLINT version,
  NumPy, Torch, Arb precision and thread count, in addition to exact parameter
  ratios, seed and asset hashes, prefix IDs and source commit.
- The launcher snapshots only committed source after verifying its ancestry
  on fetched private `origin/main`. Its isolated child has one CPU thread,
  an 18 GiB address-space cap, nice level 10 and a maximum 3600-second wait.
  Timeout signals target only the owned process group. The exit/signal race
  now tolerates `ProcessLookupError`, allowing the receipt to be written.
  Artifacts and source are hash-listed and made read-only after completion.
- Both success and failure statuses explicitly set `certificate_passed=False`.
  The float64 recovery and primitive-return diagnostics are labeled proposal
  qualification, not exact-real root or primitive-period certification.

I independently reran the final two synthetic suites: **67 passed in 1.94 s**,
with only an unrelated Torch deprecation warning. No checkpoint/model access
was involved in this review or those tests. The additional tests cover exact
ball serialization round trips, cache rebinding refusal and wider-float refusal.

The producer's summary artifact supplies no full target state; the disclosed
use of our own hash-bound P8 fold state is a numerical seed selection, not
producer proof evidence. Even a recovered P8 proposal at the assigned
parameters remains to be refined and certified in the exact-real map. The
pilot's point defect and interval widths should determine the next bounded
amendment, with failed attempts preserved.

The final pilot delta also checks the eight numerical argmax token IDs against
the assigned token sequence up to cyclic rotation, recording the tokenizer hash
and both ID lists. Full proposal states are saved before this label check.
This is a useful target-identification filter; tokenizer labels are not part
of the exact-real map or a substitute for a rigorous orbit certificate. The
delta was source-reviewed and does not change the arithmetic approval above.
