# Full-coverage derivative assembly for candidate shooting boxes

13 October 2026. Design only; no new target evaluation or proof claim. The
prototype phase-0 boxes used by `box_derivative_pilot` and `streamed_pilot` are
engineering inputs and must not enter this assembly as final orbit evidence.
The intended final boxes are still **candidate** boxes until the separate root,
primitive-period and attraction gates pass.

## Smallest useful contract

Use a new externally SHA256-pinned `candidate-final-shooting-boxes` design,
including all eight exact 896-coordinate centres, eight exact positive dyadic
scalar radii, precision, exact-real map manifest, checkpoint/config/steering
asset hashes, exact alpha/temperature semantics, full prefix IDs, prefix cache
lineage and permitted complete evaluator-source hashes. Require a new final
shard schema/status; current prototype schemas, drivers and statuses are refused.
A field saying “final” is not mathematical evidence: the design and producer
must be reviewed independently, and the design must identify its centre source.

A final shard represents a particular phase and a nonempty ordered subset of
positive coordinate basis directions. It retains the full 896-dimensional
actual state box and all 896 rows of each selected signed derivative column.
It binds the design, nominal-domain hashes, actual-input hash, map/assets,
precision/backend, source commit, complete source inventory, phase index and
exact basis-column indices. The derivative direction really used must be the
exact zero-radius vector e_j, not merely a declared integer label. The full
finite primal output is retained for evaluation audit, although it is a box
image rather than the separate point image required by the root gate.

An outer reader authenticates externally pinned receipt bytes, every inventoried
artifact and every captured source using the existing strict receipt helper.
It also checks successful final-shard completion, source inventory equality to
the permitted evaluator and the committed plan. A pure inner aggregator can
then check scope, bindings, domain coverage, directions, shapes and completeness.
It cannot prove that a dishonest or erroneous producer evaluated its declared
map; that remains the independently reviewed producer obligation.

## Exact domain cross-binding

`phase_domain` hashes canonical JSON containing `period`, `dimension`, `centers`
and `radii`. `shooting_gates.domain_sha256` hashes a labeled, newline-delimited
stream of Arb exact dyadics. These digests are intentionally different. Compute
both from the **same exact numeric data** and retain both; never equate the two
hex strings or accept a merely matching display/float representation. Canonical
packed dyadics avoid alternative mantissa/exponent spellings of the same value.
The shooting hash must not round high-precision centres to derivative precision:
construct the exact dyadic Arb midpoint from its integer mantissa/exponent, and
require exact equality to the packed value.

For every shard, first compare the original serialized actual state-box
endpoints to `c_i +/- r_i` with exact rational arithmetic. All lower and upper
margins must be nonnegative in all coordinates. **Do this before decoding**:
Arb's magnitude reconstruction can widen a serialized radius, so a decoded box
could contain the nominal domain even when the producer's actual original box
did not. A later outward decoder must never retroactively repair that omission.
The saved actual-input hash must identify these original producer-input balls.
Different shards may use different wider receiver boxes if each contains the
same nominal phase box; identical actual hashes are convenient but not required.

Derivative output reconstruction may widen safely. Check every original packed
record is finite/nonnegative-radius, reconstruct with its radius retained, and
verify exact endpoint non-narrowing. Persist/hash the actual reconstructed
matrices used by the root gate, in addition to each original shard artifact.
No midpoint-only reconstruction, dtype-derived exactness, or unverified
self-reported input hash is sufficient.

## Coverage and memory

Maintain one coverage bitmap per phase. Reject out-of-range indices, duplicates
within or across shards, missing phases, missing columns, wrong order/shape,
nonzero-radius directions, nonfinite derivatives, and mixed designs or sources.
All 8 x 896 coordinate directions must be present exactly once; full finite
primal/output rows must be retained. A repeated column is an error even if its
values happen to agree. Explicit expected shard pins also prevent silent
omission of a failed planned shard. Failed attempts stay in history and do not
become coverage.

Process one receipt-bound shard at a time and populate eight native 896-square
Arb matrices. This avoids a 7168-square shooting matrix and avoids holding every
shard JSON tree concurrently. Holding these eight derivative matrices is itself
nontrivial memory; measure it before the real assembly. The collector should
return a phase at a time or permit a caller-provided persistence sink if the
actual 8-matrix footprint exceeds the measured budget. No throughput claim is
made from synthetic fixtures.

## Connection to mathematical gates

Successful assembly means only full derivative coverage with checked domain and
provenance bindings, conditional on the reviewed evaluator's whole-box AD
validity. Its report must keep `certificate_passed=false` and must not use the
shooting gate's pass flag. Point images F(c_i) still need an authenticated
complete point pass at exactly these centres. Only then can the caller compute
`shooting_gates.enclosures_sha256(point_images, assembled_J)` and construct the
explicit enclosure contract with the actual assembled matrix hashes.

The shooting gate proves a unique root tuple in the product phase boxes under
its strict inclusion/contraction conditions. It does not automatically establish
all F^8 fixed points in X_0 whose intermediate phases leave the other boxes.
Primitive period needs phase-box separation at proper divisors 1, 2 and 4;
attraction needs a separate rigorous monodromy bound. Token labels likewise
require positive margins over all vocabulary competitors on the corresponding
proved boxes. Coverage or a successful prototype overlap test replaces none of
these obligations.

## First implementation: pure collector only

The first increment deliberately introduces **no receipt schema or reader**.
`derivative_coverage.collect_derivatives` accepts externally authenticated
in-memory shard records and explicit caller-supplied design/map/evaluator-bundle/
backend hashes. The future adapter owns all successful-receipt, source-inventory,
asset, prefix and committed-plan authentication described above. The collector
reports `external_authentication_performed=false`, even after every internal
binding and coverage check passes.

Source SHA256: `ccbed33cd14c0bd4d5ded040641a8fdb8816ffefa1011961a57750c4cc740240`.
Tests SHA256: `102910cd754be9b17ef21098051ada85f682ee5ec5883700404a51c3af2b5e2e`.
Fifty new synthetic checks plus domain and shooting-gate regressions:
**122 tests passed in 0.46 seconds**, CPU1, no target inputs or model calls.

The required shard role is `candidate_final_shooting_boxes`. Required shared
fields are `design_sha256`, `map_sha256`, `evaluator_sha256` (complete source
bundle), `backend_sha256`, producer `precision`, `phase_domain_sha256` and
`shooting_domain_sha256`. Each shard adds `phase_index`, increasing
`direction_indices`, full original packed `actual_inputs` and its canonical
hash, full row-major `directions` and `derivatives`, and full `primal` balls.
Extra provenance is retained through a complete original-shard-content hash.
Only canonical exact-dyadic encodings are admitted. Basis vectors must be exact
positive e_j, including every zero entry. Shard order may vary; column indices
within a shard are ordered, and each phase's assembled columns are canonical.

The returned eight native matrices are hashed after actual reconstruction and
assignment using the same matrix convention as `shooting_gates`. Producer and
assembly precision are separately recorded; e.g. 1024-bit derivative production
and 2048-bit gate assembly are compatible. Nominal centres are constructed from
exact integer mantissa/exponent pairs and remain exact even when their mantissas
exceed the assembly context precision. Tests specifically refuse a too-narrow
original producer input that would appear sufficient after outward radius
reconstruction, while accepting safe derivative-output radius widening.

Centre selection/history is provenance, not an additional mathematical
assumption for derivative validity at the specified exact candidate boxes.
Root/point-image/primitive/attraction obligations remain separate. No actual
8x896 collection has been measured, and this implementation does not authorize
such a calculation or a final-shard launch.

### Pre-release consistency addition

Root review requested exact original-primal consistency across same-phase shards.
The collector now intersects each coordinate's original serialized primal
intervals and refuses any empty intersection. Equality at an endpoint is allowed.
Because all receiver boxes contain the same nonempty nominal box, a correct
same-map producer must have a nonempty intersection. This test uses original
endpoints, so output-decoder widening cannot hide a contradiction. The exact
intersection endpoints and their hash are retained solely as a consistency
witness, with no point-image claim.

Final source SHA256:
`eb01dc001193b5e3982bcb41a582ac7c61cfe4e8fcd2a1ddf249126623fe63cf`.
Final tests SHA256:
`acd86b8b18406cebb815145d5d1884e591130c6355e09a34eeab5651e83615a9`.
**53 new / 125 combined tests passed in 0.47 seconds**, including disjoint-output
refusal, allowed touching intervals, and a disjoint-original/overlapping-decoded
counterexample. These hashes supersede the first-increment hashes above.
