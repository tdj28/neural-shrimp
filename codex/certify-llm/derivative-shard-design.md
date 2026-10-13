# Minimal final-domain derivative shard contract

Design only: no implementation, launch, batch size, resource reservation, or
certificate is authorized by this note. The measured streamed-64 pilot and its
independent audit must determine a future costed batch size and per-shard
CPU/RAM/time envelope. The intended coverage is **all eight ordered phases and
all 896 signed coordinate directions per phase**, with the unchanged full
151936-element vocabulary map.

## Domain definition independent of centre search

A new candidate-final-box artifact explicitly pins the exact `8 x 896` centre
coordinates and eight positive scalar nominal radii. Its versioned schema must
be distinct from all prototype-pilot/domain schemas. These data define
`X_i = c_i + [-r_i,r_i]^896`; the derivative evaluator does not need a bespoke
importer for the optimizer that proposed `c_i`. The selection source, receipt,
iteration and file hashes are recorded as provenance, without claiming that
search convergence proves anything about the boxes.

The initial implementation admits **packed exact dyadic rational centres and
radii**, with zero encoded radius on each nominal value, using the reviewed
`phase_domain` representation. Non-dyadic rational centres are mathematically
legal but are not supported by these existing helpers/gates; adding them would
require a separately reviewed representation and endpoint contract. Floats,
missing coordinates, reordering phases and midpoint-only substitutes fail.

The plan externally pins the complete candidate artifact and exact map identity:
model/config/vector bytes, prefix IDs, steering layer, exact alpha and log10-T
constants and their interpretation, full vocabulary/state dimensions, evaluator
and helper sources, arithmetic backend, precision, and compatible authenticated
prefix cache. A cache's precision and receiver construction must match the
actual evaluator. A 1024-bit Jacobian over a containing box may serve 2048-bit
nominal centres and later gates, provided those bindings and containment hold.

## Input and output evidence for each shard

Each final-shard plan pins its global phase index, an explicit unique ordered
list of coordinate indices, and exact signed basis columns `e_j` in all 896
coordinates. No random projection, absolute-value tangent, selected vocabulary,
unrecorded direction, or numerical-Jacobian replacement is admitted. Retain the
basis data and its hash, rather than trusting only the index labels.

Construct a containing box with `phase_domain`, then record the **actual live
receiver balls supplied to the evaluator**, their packed mantissas/radii, hash,
and exact coordinate containment witnesses. If a constructor or assignment
widens/rebuilds a ball before evaluation, recompute witnesses and bind that new
actual input. Never evaluate only a rounded centre or shrink the radius.

Crucially, an assembler checks the **original serialized producer input
endpoints** against the exact nominal endpoints with rational arithmetic before
Arb decoding. A wider decoded ball cannot retrospectively repair a producer
input that was too narrow. Subsequent decoding of outputs must also preserve
the original encoded endpoints; extra outward widening is allowed and recorded.

The new final-shard receipt retains the input/domain/map/direction bindings,
source commit, all stages and resource outcomes, all 896 primal balls, and every
`896 x shard_width` signed derivative ball. All scalar divisions, nonlinear
operations and roundoff must satisfy the reviewed evaluator's whole-box
contract. Failure or timeout retains partial artifacts but never counts as a
completed shard. Optional dominant-label witnesses concern only their actual
boxes; they do not replace the orbit proof.

## Assembly and proof boundary

The assembler admits only the new final-shard schema and the exact pinned
candidate-final-box identity. Existing phase-zero prototype pilot receipts are
not substitutable, even if their output dimensions are compatible. It verifies
all receipt inventories, source/map/cache identities, actual input endpoint
containment, finite enclosures and direction data. It must obtain each pair
`(phase i, direction j)` for `i=0..7, j=0..895` exactly once. Missing, duplicate,
conflicting, reordered-without-explicit-mapping, or unsuccessful rows fail.
Different containing receiver boxes are mathematically admissible only when
each individually contains the same assigned nominal phase box and is bound
to its own evaluator output; no larger-domain claim is inferred by decoding.

Bind the distinct hash conventions explicitly from the **same exact nominal
data**: `phase_domain` hashes canonical JSON with period/dimension/centres/radii;
`shooting_gates.domain_sha256` uses labeled streamed exact-ball records. These
strings are not expected to equal. Record both and an audited exact-data
cross-binding, plus each actual-receiver hash. On output, independently decode
and hash the assembled Jacobians under the gate's enclosure convention; never
rename a shard-file hash as a matrix or enclosure hash.

A separate whole-proof manifest binds point-image receipts to **these exact
centres** and the same map. Point images from earlier centres, prefix prototypes,
or float-converted centres do not qualify. The assembled whole-box Jacobians
and point images can then discharge the conditional shooting enclosure
contract, together with its analytic regularity/evaluator review. Root
inclusion/uniqueness, primitive-period separation and dynamical attraction
remain separate gates; successful derivative coverage alone is no certificate.


## Implementation addendum after the measured 64-column pilot

The parent subsequently authorized implementation and synthetic tests, with no
candidate-domain selection or target execution. `derivative_shard.py` now fixes
one global phase and one aligned contiguous positive-basis block of 64 columns,
1024-bit Arb arithmetic, all 896 state coordinates, all 151936 vocabulary rows,
and vocabulary chunks of 1024. Its plan requires CPU1/nice10/18 GiB/3600 seconds;
the separate launcher enforces the memory/time envelope. A complete collection
would contain 14 shards per phase, 112 in total. This note does not authorize
those runs or imply their cost has been approved.

The externally pinned definition uses schema `codex-llm-candidate-final-box-v1`
and role `candidate_final_shooting_boxes`. It contains exactly `schema`, `role`,
`period`, `dimension`, `centers`, `radii`, `map`, `map_sha256`, and nonempty
`selection_provenance`. The original bytes are copied verbatim. Current pinned
assets/prefix IDs and fixed alpha/log10-T constants independently determine the
map descriptor; current source files and arithmetic package metadata determine
the evaluator-bundle/backend anchors. They are compared against plan pins, not
accepted merely because the candidate file supplies strings. The unchanged
prefix loader checks its pinned receipt, historical source, assets, cache and
actual receiver construction. A valid saved prefix may originate in an attempt
whose later, unrelated box stage failed; prefix validity is checked directly.

The first `step_start` callback records the actual live scalar Jet inputs and
basis tangents before decoder layer zero. Exact rational endpoint witnesses
bind these balls to the nominal phase box. The driver checks all 24 ordered
layer starts/ends and both complete ordered vocabulary passes. It retains the
full primal/derivative output even if subsequent diagnostic bookkeeping fails;
only the final successful status and external receipt can admit a shard. The
all-competitor readout diagnostic is preserved separately and bound to the
actual-input/design/map/phase identities. Exact endpoint comparisons check its
margin consistency and strict sign. This readout result is not an orbit gate.

Implementation source SHA-256:
`3fe44a583176e0551bb209093fec08cd059d055fcb1a57702ac8799c7cb7cc50`.
Synthetic test source SHA-256:
`c4fe5c9a64b343e23cddfd03cd3c23582e8be52c3e58062aebc179e5228a637b`.
The tests exercise exact low-precision containment, independent map/backend/
source pins, prototype refusal, original live receiver capture, signed basis
coverage, all-row retention, collector compatibility, head bookkeeping, and
failure status with a synthetic evaluator. No checkpoint or target-model
forward pass was performed during implementation.
