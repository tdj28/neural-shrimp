# Independent derivative and prefix-cache review

Review date: 13 October 2026 UTC. Reviewer: Codex inference subagent, independent
of the Jet implementation and signed-pilot driver author. This review used
source inspection and CPU1 synthetic computations only. It did not load a target
checkpoint, evaluate the target model, replay a target artifact, or launch a job.

## Verdict and scope

No blocking defect was found for the proposed bounded, one-phase, eight-column
signed-JVP feasibility pilot after the receiver-construction fix below. This is
approval of the implementation for that exploratory measurement, not an orbit
certificate, a full Jacobian enclosure, or a full-box contraction conclusion.
The prior pilot's actual outcome and receipt were audited by the parent agent;
this review does not independently reproduce that computation.

| Component | Evidence | Scope / unproved obligation |
|---|---|---|
| Jet formulas and inherited map | Source audit plus synthetic cross-checks | Depends on Arb arithmetic and the previously reviewed exact-real map; no formal proof assistant |
| Prefix import | Hash-bound importer with synthetic corruption/refusal controls | Original prefix computation, reviewed historical source, and file-to-array construction must be established separately |
| Eight directional columns | Implemented first-order interval AD | Supplies only the specified directions; no omitted-direction or nonlinear-remainder bound |
| Pilot/launcher | Source and synthetic admission review | No target outcome in this review; resource admission and private push are parent responsibilities |
| Generic proof gates | 42 synthetic tests; independent mathematical review by the other subagent | Returns only `conditional_gates_passed`; upstream enclosure truth cannot be inferred from metadata |

## Signed derivative architecture

`BallQwenJet` inherits `BallQwen.forward`, decoder blocks, attention, rotary
application, steering placement, prefix handling, tied output embeddings, and
full-vocabulary readout. All six arithmetic hooks are replaced consistently:
scalar construction, coercion, normalization, softmax, affine sequences and
finiteness checks. No second transformer architecture was copied into the Jet
class. The inherited forward call returns tuple-valued tangents, which the Jet
wrapper restores as an output-by-direction Arb matrix.

The product, reciprocal, exponential and reciprocal-square-root derivatives
are correct. Softmax computes each signed action as
`p_i * (dz_i - sum_j p_j dz_j)`. Its common midpoint shift is a conditioning
constant; differentiating that choice is unnecessary. RMSNorm differentiates
the actual quadratic variance and includes the same exact epsilon. A linear
sequence packs each token as one primal column followed by all tangent columns;
bias is added only to each primal column. Signed cancellation is retained in
those affine products. Matrix directions have shape `(d,K)`; stored binary
NumPy directions have shape `(K,d)` and are admitted without numerical matrix
products. Interval state/direction inputs enclose the specified actions, subject
to the arithmetic contract.

The fixed-prefix tangent is exactly zero with respect to the one-token state,
even when the prefix primal is an interval enclosure. Steering and temperature
are fixed exact mathematical parameters here; these zeros are not appropriate
for parameter derivatives. This pilot does not request such derivatives.

The source rejects mixed direction counts, nonfinite inputs, zero-containing
divisors and unproved positive RMS denominators. The inherited final finiteness
gate remains active. The signed driver's progress callback correctly treats
`.tangent` as eight Arb scalars rather than using the scalar-Dual callback.

## Prefix-import fix and remaining construction contract

The original importer checked saved metadata against supplied asset hashes but
did not connect the receiving map to the independently checked in-memory
construction. An equal-shaped map with different weights or steering could
therefore have imported the wrong prefix despite matching metadata. At the
parent's request this was fixed before the signed run:

- Four explicit receiver inputs are now mandatory: `expected_config`,
  `expected_weights`, `expected_vector`, `expected_steering_layer`.
- The receiver must have equal config contents, the identical weight dictionary,
  and the identical steering-vector object supplied from the independent loader.
- Both steering-layer values must be exact Python integers equal to 12. This
  importer is deliberately restricted to the reviewed original pilot, whose
  source fixes that default; it is not a generic cross-layer importer.
- The signed driver supplies the config and layer from
  `ManualQwenSoftMap.from_pinned_files`, and the very same NumPy weight/vector
  objects used to construct `BallQwenJet`.

`from_pinned_files` independently checks config, safetensors and vector-file
hashes before converting the stored weights and selecting the steering vector.
This construction code and its pinned assets, not an identity comparison alone,
establish the file-to-array relationship. The loader now states that distinction
in its docstring and return record. Arrays, scalar attributes, config and all
aliases must remain unchanged during cache use. Read-only NumPy views are useful
accident protection, not a defense against mutation through other Torch aliases.
The reviewed driver does not perform such mutations.

The importer retains receipt and artifact size/hash checks, safe contained
artifact paths, original map-source binding, source-commit consistency, prefix,
exact parameter ratios, precision, backend versions and exact-real constant
conventions. It checks every layer/token/head/coordinate shape, decodes complete
midpoint/radius pairs, and installs the cache only after successful completion.
No radius is discarded; every restored state tangent is zero. A failed later
box computation does not invalidate a previously saved, finite prefix by itself.
The original producer's complete source and arithmetic remain part of the
reviewed provenance contract, not something proven by checking a source hash.

## Signed driver and launch review

The plan is restricted to phase zero, coordinate directions 0 through 7,
1024-bit arithmetic and the assigned parameters. It binds an owned previous
attempt, its receipt, reviewed map source and proposal bytes. The proposal must
be finite binary64 with shape `(9,896)`. The driver constructs fixed one-hot
inputs, imports the prefix through the strengthened contract, retains all
151936 logits, saves every primal and signed derivative ball, and compares with
an independent binary64 implementation only as a consistency diagnostic.
It explicitly declines to require binary64 results to fall inside exact-real
balls, since those are different mathematical maps. No floating norm or
comparison is used as a proof gate. Completion and failure records both set
`certificate_passed` to false; that pilot field cannot signal success of a proof.

The launcher requires the relevant source files and selected plan to equal
committed bytes, fetches private `origin/main`, and checks that the source commit
is an ancestor before copying it to an exclusive attempt directory. It executes
the copied driver with isolated Python imports, one numerical thread, no CUDA,
offline model access, 18 GiB address-space limit, niceness 10 and an at-most-one-hour
watchdog. Timeout termination targets its own process group. It records command,
source hashes, process ownership, remote ancestry, exit status, timeout state and
artifact hashes. A resource limit is an operational bound, not a mathematical
enclosure or a promise that the run will finish. The committed plan's stated
one-hour envelope must be the one the parent actually dispatches.

## Synthetic validation retained in this review

The affected tests passed: **78 tests**, comprising cache import, Jet arithmetic
and generic proof gates. The cache additions exercise missing construction inputs,
equal-but-distinct weight dictionaries/vector copies, config and steering
mismatches, and restored eight-component zero tangents with unchanged nonzero
primal radii. Existing Jet tests include affine bias placement, exact signed
cancellation, interval-state/direction interiors, full tiny-map comparison against
each scalar-Dual direction, malformed inputs and inherited cache guards.

An additional CPU1 check compared eight arbitrary signed columns of the complete
synthetic two-layer, four-state transformer against the independently written
Torch map. Maximum primal midpoint discrepancy was `3.469446951953614e-18`;
maximum signed-column midpoint discrepancy was `2.7755575615628914e-17`.
This is a consistency check of a synthetic map, not an exact containment result.
The signed-plan validator admitted one synthetic valid plan and refused ten
changes to scope, precision, parameters, bindings and owned location. Python
source compilation passed for all six files listed below. No additional target
outcomes were inspected to choose any of these tests.

The generic gate scope wording was also corrected after the independent
mathematical reviewer pointed out a phase distinction: a passing root gate
proves one unique `F^p` fixed point in the nominal box, belonging to a primitive
attracting orbit. It does not place all phases of that orbit inside that box.

## Reviewed source hashes

| File in `codex/certify-llm/` | SHA-256 |
|---|---|
| `ball_jet.py` | `46b14d8bd65ed6c80174758daf37d00110cfa908813917bdb46c13749f999719` |
| `ball_qwen.py` | `fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80` |
| `cache_io.py` | `f4e25b2134bef8a71c27d3156baf01be9d25c593d7b780f90a66b9c1141abdef` |
| `signed_pilot.py` | `8f43241dac3fd54586f25b2b94d8a417a59b60369b67c4cd86aba5fa5fba653c` |
| `launch.py` | `808892e424140b13c3353c45a76e69071c836b62410f84c01ab62563cd1a94c5` |
| `proof_gates.py` | `ee6678b7dd5153ef095fd73d5711fcfc605f7a228c92b1335a9541cc3d7b2c6a` |
