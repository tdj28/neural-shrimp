# Independent pure derivative collector review

Reviewed `derivative_coverage.py` SHA-256
`eb01dc001193b5e3982bcb41a582ac7c61cfe4e8fcd2a1ddf249126623fe63cf`
and its tests SHA-256
`acd86b8b18406cebb815145d5d1884e591130c6355e09a34eeab5651e83615a9`.
No mathematical/coverage blocker was found within its explicitly conditional,
pure in-memory scope. No model, checkpoint, receipt or target derivative artifact
was loaded for this review.

Independently reran **125 collector/domain/shooting-gate synthetic tests**.
Additionally checked **400 original-output endpoint containments** using exact
`Fraction` comparisons for eight phases and five coordinates, shuffled shard
order, and 32-bit/256-bit assembly contexts around a separately declared 64-bit
producer. High-precision dyadic centre bits were retained. These are small
arithmetic/coverage checks, not evidence about the target Jacobians.

The collector checks a complete exact nominal domain under both hash conventions,
then admits one and only one exact positive basis direction `e_j` per coordinate
and phase. Derivative values remain signed. It rejects missing/duplicate columns,
wrong phase/index/direction dimensions, mixed map/source-bundle/backend/design
bindings and prototype roles. It allocates phase-sized matrices only; missing
columns cannot silently remain zero in a successful return.

Input containment uses the **original serialized producer box endpoints** before
any Arb decoding. Thus a too-narrow original state interval cannot be repaired
by an outward-widening constructor. Output decoding and the subsequent native
matrix assignment each receive a separate non-narrowing endpoint check. Matrix
hashes refer to the actual Arb matrices returned, under the downstream gate's
matrix-hash convention. Producer and assembly precision are recorded separately;
a lower assembly precision can make a valid enclosure less useful, but cannot
legitimately narrow it.

The cross-shard primal consistency refinement is sound: all receiver boxes for
one phase contain the same nonempty nominal box and use the same map, so their
coordinate output enclosures must have a common intersection. The code forms
that intersection from original exact rational endpoints; it refuses a disjoint
intersection even when decoded balls would overlap. Touching intervals are
correctly allowed. The retained intersection and its hash are consistency
witnesses, not separately authenticated point-image evidence.

External authentication remains a real caller obligation. Hash strings and the
final-domain role do not authenticate a receipt, demonstrate that the evaluator
ran on those inputs, or establish evaluator correctness. The caller must first
verify the original artifact bytes, complete successful receipt/source bundle,
map/assets/constants, full vocabulary, prefix-cache construction and arithmetic
backend. Here `expected_evaluator_sha256` denotes the complete source bundle,
not only a top-level Python file. All input records must remain immutable during
the call.

The collector correctly reports `external_authentication_performed=false`,
`point_images_supplied=false`, all proof gates unevaluated, and
`certificate_passed=false`. Its assembled Jacobians still require separately
bound point images at exactly the nominated centres, the full enclosure-contract
review, and successful root, primitive-period and attraction gates. The distinct
phase-domain JSON hash and shooting-domain streamed hash are derived from the
same canonical exact dyadics; their strings are not equated.
