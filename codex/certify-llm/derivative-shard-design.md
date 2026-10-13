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
