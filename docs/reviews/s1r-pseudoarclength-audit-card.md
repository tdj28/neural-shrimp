# Prospective saved-only pseudoarclength outcome audit

2026-10-08 UTC. Preparation only; no producer outcome was opened. A separate
root authorization with externally pinned metadata is required for one audit.
Producer source is `482cf9c7720936503658206f98c960f074aed847`, independently
reviewed at `9df5e138fbba1b283c18a8d47f49ebe9a69b15ae`. Exact 17-file producer
inventory, settings and tolerances are in
`s1r-pseudoarclength-producer-source.json`. This auditor imports no producer
numerical routines. It reuses its author's previously reviewed lower-gap
saved-record decoder and scalar reductions, with that entire source bound.

## One fixed execution

Only `/data2/neural-shrimp-data/explore/s1r-pseudoarclength/attempt-001` and its
`attempt-001-launcher` sibling are target inputs. A separately SHA256-pinned
JSON supplies producer execution commit, source-review commit and full raw
and launcher manifest hashes. Raw manifest may be null only if absent; a
present malformed manifest retains its actual hash and inventory-only scope.
The execution and reviewed source must have identical bound bytes. Historical
inputs are exactly the consumed-development records and train arrays listed
in the frozen producer plan; no holdouts or new trajectory seeds are opened.

The exclusive output is
`/data2/neural-shrimp-data/audits/s1r-pseudoarclength-independent/attempt-001`
plus its `attempt-001-launcher` sibling. CPU/BLAS one thread, nice at least10,
no GPU or paid resource. Worker timer30s; external TERM40s plus KILL5s;
launcher45s with owned-group cleanup. Audit artifacts total at most16MiB:
15MiB worker (14MiB before finalization),1MiB launcher,128KiB log. Full input
manifests cap producer raw at127MiB plus1MiB launcher. The auditor performs
only saved-array products, scalar reductions and metadata checks. No model,
derivative, JVP, correction, linear solve, decomposition, eigensolve,
continuation or numerical timing profile is permitted. No retry or setting
change after outcomes. A stop or error preserves all outputs and means no
complete arithmetic qualification.

Invoke the committed launcher once from a clean checkout:

```
nice -n 10 /data2/neural-shrimp/.venv/bin/python \
  tools/reviews/launch_s1r_pseudoarclength_audit.py \
  AUDIT_SOURCE_SHA PINNED_METADATA_PATH METADATA_SHA256
```

## Independent checks

The audit checks all raw and launcher files, source bytes, actual runtime,
thread/nice/library settings, owned PID exit, storage totals and exact input
plan. It independently decodes original full-state phase arrays and their
historical aligned-parent convention: old match.phase names the child's
native carried phase, whereas XOR with old parent_phase is native pair
exchange. New pseudoarclength pairs remain native and require XOR carry.

For every retained operation, it reconstructs actual binary64 block rates,
`F=theta-Dg`, `J=I-DH`, hidden-log parameter term, actual two-step native
`R,M,A,b`, and the raw Euclidean73-coordinate bordered residual/derivative.
Saved solves are checked by true forcing, Frobenius backward error and the
nearzero abstention rule; no solve is rerun. Newton/trial inputs and accepted
state retention, actual-slope Armijo factors, local boxes, predictor correction
radius, parameter/state correction and plane convergence gates are checked.
Common-control closure uses actual saved map2, never a reconstructed
`theta+R` substitute or a mere close-h comparison.

All complete retained SVD triplets are checked for residuals, reconstruction,
orthogonality and sorted nonnegative singular values. Tangent null residual,
rank convention, raw orientation, parameter-inclusive native transport and
full saved eigenpair left/right residuals are recomputed. Spectrum agreement
uses the saved bijection, not a fresh assignment solve. Complete eigensystem
coverage rests on the bound producer's full n-mode eig call; residual tests
alone do not independently prove spectral completeness. Dimensions are at
least2 in these helpers; the target is72 and the retained synthetic is2.

Every accepted edge needs four qualified legs, all passing cyclic links,
both canonical-h state-only witnesses, one fixed parent plane, correct
midpoint/direct orientation references and a durable edge_end. Exactly one
half-step retry follows the first rejected edge; a second failure or two
accepted edges ends work. Qualified but unwitnessed nodes stay separate from
last_witnessed. Attempted/returned, assembly and stage counters distinguish
calls from durable evidence and retain narrow increment-before-emit and
final edge/checkpoint boundaries. Failed numerical gates remain evidence of
rejection; they may only unwind, never qualify a dependent node. Interrupted
extra files are restricted to the next record and force partial scope.

Scalar/array mirrors use relative2e-11 plus absolute5e-14 agreement; these
allow independent reduction order, and do not relax any scientific gate.
Full original state/rate/cache/control identities use bytes where specified.
Shared saved gradients/Hessians are **not independently regenerated**. The
combined chart, predictor and globalization change does not isolate which
change helps. No fold, lower-family connection, ancestry, boundary geometry,
training utility or inference utility claim follows from this audit.

## Synthetic controls and durability

The retained gzip fixture is a synthetic two-dimensional polynomial
period-two trace, not target evidence. Its provenance carries the exact
producer draft-core hash used to generate it. The final fixture was exported
before replay mutation; an earlier contaminated draft fixture was detected
and replaced before the source freeze. Current replay copies input metadata.
The complete765-event trace has two witnessed edges and six qualified but
unwitnessed auxiliary nodes. Tests replay it with NumPy decomposition/solve
APIs forbidden, check the complete JSON/NPZ record join, and mutate native
assemblies, bordered equations, cache bytes, common controls, assignments,
Armijo values and terminal/retry structure. No synthetic timing is treated
as a target profile or proof of feasibility.

The worker writes start, provenance, historical inventory, arithmetic result,
final result and exact manifest; the launcher saves source, command, PIDs,
stdout, exit and its manifest, and always cleans only its created group.
Final artifacts become read-only. Root and an independent agent review the
frozen source/card before the one target audit is authorized.
