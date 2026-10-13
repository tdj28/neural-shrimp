# Authenticated derivative evidence adapter

`derivative_evidence.authenticate_and_collect(definition_path,
definition_sha256, *, expected_map, expected_backend,
expected_source_sha256, expected_prefix, attempts)` accepts explicit external
trust anchors. `attempts` is an iterable of `{attempt, receipt_sha256}` records.
`expected_source_sha256` must pin every entry in
`derivative_shard.SOURCE_PATHS`, not only its entrypoint. The expected prefix
object has `attempt`, `receipt_sha256`, `map_source_sha256`, `cache_sha256`, and
`completion`; the last field explicitly selects
`retained_prefix_before_box_failure` or `completed_prefix_point`.

The adapter authenticates the frozen candidate bytes and independently supplied
map/backend/source-bundle objects. It authenticates the complete original
receipt inventory and captured source manifest for each attempt, all evaluator
dependencies, the captured exact plan, source commit, resource envelope,
producer status, live receiver and directions, helper-domain artifact, prefix
binding, and full ordered decoder and vocabulary passes. It reconstructs saved
live-input containment witnesses with exact rational arithmetic directly from
original serialized endpoints; it never repairs narrowing by decoding first.
Each successful original shard is yielded immediately into the existing pure
coverage collector. Only compact audit records accumulate alongside the native
assembled derivative matrices; 112 complete shard JSON objects do not.

The prefix is a separate evidence class. The 1024-bit `arb-pilot-002` prefix was
saved successfully before its box diagnostic failed. Its full externally
pinned receipt/source/cache inventory, exact map metadata, runtime versions,
all decoder stages and finite original cache balls are checked. This does not
admit failed derivative attempts: every derivative receipt must have exit code
zero, no timeout, and the complete final-shard status. A 2048-bit prefix is
refused for these 1024-bit derivatives.

Captured source bytes are authenticated against the external evaluator bundle and
complete receipt/manifest. The declared source commit has syntactic and cross-file
checks, but this module does not call Git to compare blobs with that revision.
The report explicitly sets `git_commit_blob_verification_performed=false`; a
separate archive audit can establish that provenance.

No pretrained model is instantiated. Checkpoint/config/weights-to-cache
correctness remains a reliance on the independently reviewed, externally pinned
producer and cache loader, whose construction and actual receiver binding are
also recorded. This adapter does not claim to recompute the prefix, reconstruct
checkpoint arrays from their hashes, prove evaluator correctness from metadata,
or authenticate arbitrary centre-selection history. Its inputs and source tree
must remain immutable during the read. The external anchors must come from a
reviewed frozen design, never be selected from the first shard.

The returned matrices have full conditional derivative coverage. The returned
report has `artifact_authentication_completed=true` and
`certificate_passed=false`; no shooting root, primitive-period or attraction
gate is evaluated. Full-vocabulary labels concern the actual receiver boxes;
they are not labels of an established periodic orbit. Original versus assembly
precision and both nominal-domain hash conventions remain explicit.

Synthetic validation: 55 tests passed, using two phases, four state coordinates,
two signed columns per shard and 2,051 vocabulary entries (three chunks in each
head pass). The successful path uses the real producer's exact domain, live
receiver and artifact writers with a tiny exact linear map, complete source
snapshots and complete receipts. Refusals include resealed inconsistent map,
source, backend, original endpoint/witness, precision, failed status, incomplete
head/decoder stages, malformed numerical outputs, prototype roles, missing or
duplicate coverage, and the failed-prefix/successful-derivative distinction.
No target model, shard or candidate was evaluated or admitted.
