# S1R bootstrap saved-outcome audit card, 8 October 2026

This audit asks whether the new bootstrap and saved-lineage comparison is
supported by the retained full states and continuation witnesses. It cannot
recover the historical runs' missing full states, prove uniqueness, or infer a
behavioral benefit from geometry. This is an exploratory, outcome-masked audit
freeze: the collection already exists, but the reviewer has received only its
paths, hashes, source and availability. No profile, collection or producer
receipt numerical outcomes were opened before this source/card commit.

## Frozen inputs and execution

Producer mathematical source and plan basis:
`e0fccd73df025f81130d07dfbe60dfe170ef4840`. Independent producer-source review:
`ff2ec8dc133d4fabf766ec5d776e19af1143f2f7`. Collection execution HEAD:
`586e74927215ff7ab5ebc81fbc29720c9322f291`; all eleven reviewed source/plan files
must be byte-identical to the basis and local audit dependencies. Exact source
inventories are checked against Git, including the distinct execution HEADs.

- Collection: `/data2/neural-shrimp-data/explore/s1r-bootstrap-association/attempt-001-collection`.
  Manifest SHA-256: `391092f55c06b7289b852ca58603dd680f7718fa91f747bfc2d749704d5d5974`.
- Required resource profile: adjacent `attempt-001-profile`.
  Manifest SHA-256: `9a9fe3ff9db30bf7a37cae8848e51c5ff1b7cb49e5761072e622bf6440c88599`.
  Exact execution HEAD, supplied as metadata only:
  `e0fccd73df025f81130d07dfbe60dfe170ef4840`.
- Original saved-lineage input: `/data2/neural-shrimp-data/explore/s1r-branch-association/attempt-001`.
  Manifest SHA-256: `45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b`;
  source `db10bd1e1b25b0e24b1f0f8f97cb84035742c776`, central node0004.
- Exclusive new audit root: `/data2/neural-shrimp-data/audits/s1r-bootstrap-association/attempt-001`.
  External launcher/log prefix:
  `/data2/neural-shrimp-data/logs/s1r-bootstrap-association-audit-attempt-001`.
  Parent owns independent backup of all successful/failed attempts and logs.

Launcher code only was read before the freeze; launcher receipts/log contents
remain unopened. Both companion launcher directories are bound by fixed
metadata-only manifest hashes:

- `/data2/neural-shrimp-data/logs/s1r-bootstrap-association-attempt-001-profile`:
  `c93f1ec8adedcb623ff5ef6d32f288cd6442069c4bb9de3e690712fdbd5600b5`.
- `/data2/neural-shrimp-data/logs/s1r-bootstrap-association-attempt-001-collection`:
  `adca7ab0dcbb858cc89c30bdfea14e3b33713bcfe686ceeaeddc6642a5483d30`.

The parser checks the exact four-file launcher inventory, reviewed launcher
source hash, immutable start fields, source/review/command/out/cwd identities,
CPU-thread settings, mode, recorded resource observations, exit/status
agreement, output hashes/counts/bytes and stdout-result equality. It checks
recorded child elapsed time within the exact timeout plus grace, launcher time
at least as large as child time, and monotonic/wall-clock duration agreement
within5 seconds. Launcher startup resource queries have no separate time cap;
the exact timeout command bounds the child. Recorded owned PIDs must include
the child and timeout process, with no owned PIDs recorded alive at completion.
Historical PID values are not queried now, because PID reuse is possible.
The receipt's authorization flag is checked for consistency, not treated as
a substitute for trusted root launch authorization. Original historical-journal
kinds use a separate explicit allowlist; the bootstrap schema is not loosened.

The audit binds every raw file, exact manifest inventory, journal entry,
array shape/dtype/order/byte hash, producer runtime, review/source identities,
immutable-file status, source files and original input manifest. Deterministic
S1R development seed28002 must reproduce the original model, x/y/theta0 bytes;
central pair bytes must match the saved node. No protected seed, subsetB,
foreign process, GPU, paid service, training or continuation is used.
The supplied profile must have exactly2,000 successful fixed-state map calls
and128 successful fixed-state JVP calls and pass its unchanged timing formula.
The audit does not rerun either profile operation.

## Numerical and control-flow scope

1. Inspect both starts separately and in their fixed order, hidden log rates1.1
   and0.95 at output log rate0.6271428571428572. Check every saved1,000-step
   checkpoint, terminal counter, attempted/accepted distinction, tail index,
   JVP attempted/returned distinction, and final-state binding. A completed
   surviving start has40,000 attempted/accepted updates and1,200 JVP returns.
   Across both starts, the80,000 claim is **recorded counter consistency**, not
   independent replay of all80,000 unsaved updates. Missing/incomplete starts
   remain labeled as such; the audit has no requirement that both succeed.
2. Recompute the smallest recurrence period from every complete saved384-state
   tail using the historical full-state tolerance and repeat count. Verify
   terminal seeds are the separately stored states at updates40,000 and39,999.
   Independently evaluate all adjacent finite saved-tail transitions (up to383
   per start) and the correspondingly shifted pre-update losses. This is a
   fixed evaluation at each saved state, never rolling a new trajectory. The
   first tail loss lacks its preceding saved state and is not independently
   reconstructed. Recheck final loss and tail-mean bookkeeping.
3. Check every completed node's two native corrected phases, primitive
   separation, closure and transition gates. Reconstruct all completed-pair
   monodromies with the reviewed shared model Jacobian, check both full72-mode
   left/right eigensystems and residuals, their multiset match, and each saved
   stability label. Rejected nodes retain their actual failed gate. Validate
   saved correction ownership, consecutive iterations, step arithmetic,
   candidate distance, line-search decrease, final undamped correction gate,
   and durable pre-spectrum pair. No Newton solve/correction is called; the
   stored intermediate residuals/linear systems are not newly recomputed.
4. Independently replay the saved control flow for all attempted direct,
   midpoint, half-endpoint and reverse witnesses. Verify exact controls and
   seeds, complete pairs, both cyclic costs, acceptance/rejection reasons,
   explicit carried phases, hidden-axis secant predictors, every retry/halving,
   minimum-step semantics, the12-macro-link schedule, and global19-link/160-phase
   caps. Rejected attempts cannot become accepted links; continuation cannot
   cross a recorded unresolved gap. A resource stop can leave an unfinished
   node/edge with no association verdict.
5. Compare the endpoint with the1.1 bootstrap only after both native pairs
   qualify and their float64 parameter and rate bytes are identical. Recompute
   same-pair/distinct/unresolved and both-attracting decisions using the exact
   existing tolerances. Keep the0.95 start separate: no lower-start bridge is
   authorized. The common-control comparison says nothing about unknown
   historical full-state ancestry.
6. Recompute gradient and complete72x72 Hessian matrices from the independent
   sample prediction-Jacobian plus residual-curvature formula in the prior
   reviewed `audit_s1r_branch_outcome.py` helper. Selection is fixed by role:
   every qualified bootstrap node, saved central node, and the first, middle
   (floor index n/2), and last qualified direct-witness node, deduplicated, at
   most six pairs. Both phases are checked. This selection is by recorded role
   and order, never angle, stability or favorable association. Independently
   reconstruct monodromies, full eigenvalues and eigenvector residuals. The
   other completed nodes use the shared reviewed model derivative.

Fixed thresholds: producer closure1e-9 +1e-8 relative; primitive separation1e-6;
last relative Newton correction1e-10; candidate distance0.02; link displacement
min(0.01,0.25 separation); phase-cost gap0.5 separation; replay agreement1e-7;
distinct-pair threshold1e-5; eigenpair residual1e-10; phase spectrum match1e-7;
stability margin1e-7. Independent analytic cross-check caps are2e-11 for
monodromy maximum entry error and normalized tail update error,2e-12 for tail
loss relative error,1e-8 for fresh spectrum matching, and unchanged1e-10
left/right residuals. These numerical checks are not interval certificates or
forward eigensystem error bounds. FTLE growth sequences, chaotic/drift labels,
unsaved update intervals and all intermediate Newton residual equations are
outside the independent numerical replay scope.

Historical runs were recorded with a dirty worktree and did not save these
full states. Matching the committed historical procedure does not authenticate
unknown dirty-worktree bytes or prove identity with either historical family.
No local continuity result proves distant branch association or global shrimp
boundary organization.

## Resources, records and stop conditions

One CPU/BLAS/OpenMP thread, nice10, internal45 seconds, external60 seconds plus
5-second TERM grace, raw16MiB. This conservative cap is based on the earlier
21-pair saved-state audit/continuation unit taking approximately2.5 seconds and
its narrower independent audit taking under1 second, not on a new target
profile. The reviewed bootstrap cap allows at most79 completed pair nodes;
only the predeclared representative subset gets independent fresh eigenvalues.
No time-, file-, tolerance-, node- or mode-limit relaxation and no automatic
retry. Recheck CPU load, RAM and GPU occupancy before the separately authorized
launch; GPU observations confer no allocation. Track only owned PIDs and
confirm their termination. Synthetic tests need no target data.

The audit uses an exclusive Store with atomic/fsynced per-stage events,
source/input/runtime manifest, final result and raw manifest. A mismatch or
resource stop is retained as a failed audit; no scientific verdict is emitted
from a missing/invalid manifest. A hash-valid interrupted producer attempt can
pass **incomplete-attempt consistency**, never completed association. An
external SIGKILL may leave partial files: preserve them and report the audit
incomplete. After a normal or caught failure finalization, files become0444 and
the attempt directory0555. Compact source-bound `docs/reviews/s1r-bootstrap-outcome-receipt.json` and
`docs/reviews/s1r-bootstrap-outcome.md` outcome notes will be committed only after root authorizes this one replay. Parent
may bind the receipt into the paper; arithmetic agreement is separate from the
sampled analytic checks and their shared-primitives limitations.

Source/card commit and private push precede any target read; root independently
reviews the source and authorizes the exact command. Preparing this card is not
launch authorization. Execute
from the clean committed audit worktree only after that authorization:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src timeout --signal=TERM --kill-after=5s 60s nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/audit_s1r_bootstrap_outcome.py --collection /data2/neural-shrimp-data/explore/s1r-bootstrap-association/attempt-001-collection --manifest-sha256 391092f55c06b7289b852ca58603dd680f7718fa91f747bfc2d749704d5d5974 --producer-commit 586e74927215ff7ab5ebc81fbc29720c9322f291 --profile /data2/neural-shrimp-data/explore/s1r-bootstrap-association/attempt-001-profile --profile-manifest-sha256 9a9fe3ff9db30bf7a37cae8848e51c5ff1b7cb49e5761072e622bf6440c88599 --profile-commit e0fccd73df025f81130d07dfbe60dfe170ef4840 --profile-launcher /data2/neural-shrimp-data/logs/s1r-bootstrap-association-attempt-001-profile --collection-launcher /data2/neural-shrimp-data/logs/s1r-bootstrap-association-attempt-001-collection --original /data2/neural-shrimp-data/explore/s1r-branch-association/attempt-001 --out /data2/neural-shrimp-data/audits/s1r-bootstrap-association/attempt-001
```
