# Prospective saved-data audit of the six-state lower-gap diagnostic

8 October 2026 UTC. Source and synthetic controls only. This evaluator is prepared
before opening the new diagnostic's artifacts. It introduces no model evaluation,
gradient, Hessian, JVP, HVP, SVD, eigensolver, corrective solve, trajectory or pulse.
The producer is frozen at `5cecd5f0785f119a2631ae13ac375f9d044b2f26`.
Independent source review and a separate root authorization must precede the
single saved-data execution; source preparation does not authorize that run.

## Inputs and provenance

The fixed new locations are
`/data2/neural-shrimp-data/explore/s1r-lower-gap/attempt-001` and its
`attempt-001-launcher` sibling. Root will provide a metadata-only JSON file with
exactly `execution_commit`, `source_review_commit`, `raw_manifest_sha256` and
`launcher_manifest_sha256`; the command and authorization bind its SHA-256.
Both commit fields are full40-hex existing commits. Manifest fields are64-hex;
only an absent scientific manifest uses null. A present malformed manifest must
retain its actual hash. The root authorization identifies the approved producer
review; arbitrary commit existence alone is not evidence of a review.

The evaluator imports standard-library modules and NumPy only. No producer
numerical routine or codec is imported. Twelve producer import/card files must
match both frozen producer and actual execution commits byte for byte. Its
archived launcher, source/runtime/settings/review mirrors, exact raw inventories,
record order, arrays and array-member identities are independently decoded and
checked. New outputs remain inside their externally pinned inventories; unknown
completed files or unreferenced archive members fail closed.

The original collection is independently bound through the producer's frozen
`docs/plans/s1r-lower-gap-inputs.json`: full collection manifest and inventory,
metadata, selected JSON/NPZ identities, historical execution/reviewed source
bytes, encoded state/control/rate descriptors and saved model/data bindings.
The complete original inventory is hashed; only the declared selected records
are decoded. This does not run the old model or import its analytic helper.
Its unchanged gradient/Hessian helper is covered by the producer source list.
The old analytic values and new returned gradient/Hessian arrays remain shared,
source-bound numerical inputs, not independently regenerated derivatives.

## Fixed independent arithmetic

For each of the six fixed states, check both saved gradient/Hessian returns and
phase records. Reassemble the two map images, Jacobians, hidden-log-rate partials,
R, M, A=M−I and b by explicit NumPy arithmetic from the saved arrays. The rate
partial uses `math.log(10)` on the hidden64 coordinates and zero on the output8.
There is no differentiation of rounded rate construction. Phase input state
bytes, original selected states and the thirteen trial states are exact checks.

Check all24 retained decompositions using saved U, singular values and full Vh.
The augmented right basis must have73 rows and columns. Recompute both singular
vector equations, reconstruction and orthogonality with Euclidean/Frobenius
norms, without a decomposition or solve. Scalar norms/dots use Python
`math.fsum`; matrix products use NumPy. Preserve every producer decision bound:
SVD/null residual1e-10; relative rank windows1e-12 and1e-10; isolated soft ratio.1;
orientation cosine1e-8; closure absolute1e-9/relative1e-8; primitive separation
1e-6; and endpoint/stalled residual agreement2e-11 absolute.

The audit's scalar/array mirror tolerance is fixed prospectively at relative
2e-11 and absolute5e-14. It compares independently accumulated values to saved
values; it does **not** replace any scientific threshold. Classification and
abstention flags must agree under the original thresholds. A boundary mismatch
fails this audit without retuning. Exact dtype/shape/C-order byte relationships,
including signed zero, are used for recorded identities and carried states.

Reconstruct D=max(1,abs(last native phase0)) and the fixed h scale.000125.
Check D^-1 A D and D^-1[A,b]diag(D,h_scale), convert right nulls by the column
scale, and orient them with the inverse-column-scaled carried secant. Derive
native phase exchange as parent_phase XOR parent_match.phase. Both failed states
inherit the failed edge's carried child phase. Keep scaled left coupling in its
own coordinates. Check saved native phase transport using only saved J0 and
partial_h_F. No soft singular value or oriented null is a fold certificate;
failed-state tangents retain only a local residual-level-set interpretation.

At the stalled state, reconstruct Ap+R and Ap+R_saved, true relative forcing,
absolute residual and Frobenius backward error. Preserve nearzero abstention
at1e-12 max(1,||theta||), material forcing1e-6, backward threshold1e-12 and the
relative reconstruction-resolution warning. Independently reduce all13 original
call473/iteration3 trial residuals with the strict unscaled infinity-norm rule
and exact theta+2^-k p states. No trial is reevaluated or accepted as an update.

## Incomplete evidence and operation accounting

The journal must be a prefix of the fixed six-role stage order. Attempted calls
must match12 maps/12 Jacobians/24 SVDs at completion; each corresponding durable
return is counted separately. The source increments its returned counter just
before emitting that record. On a failed emit, at most the one pending operation
may be counted returned without a durable journal record; preserve that difference.
Likewise a durable role-complete record can precede its result-list append by
one role during interrupted finalization. Complete attempts permit no such gap.
Uncommitted record tails or temporary checkpoint files remain explicitly listed,
never silently consumed as completed stages.

A complete attempt must pass all six assemblies, all24 decompositions, native
phase/scale/secant joins and all13 trials. A source-consistent negative result
can receive `saved_outcome_audit_pass`. Missing/malformed scientific manifests
receive `incomplete_inventory_only`, bound by the launcher inventory, with no
arithmetic verdict. For other incomplete attempts the complete durable prefix
is checked. A retained invalid terminal return may be recorded as an unvalidated
partial role; it never earns a complete arithmetic verdict. Contradictory
provenance/order/counters or a failed check on a complete attempt fail closed.
No missing matrices, directions or decisions are invented, and there is no retry.

## Resource and execution card

Use CPU/BLAS1, nice>=10 for worker and launcher, no GPU or paid resources. Exactly
one attempt:15seconds internal timer, GNU timeout20seconds TERM plus5seconds
KILL, supervisor25seconds. Bounded owned-group cleanup and reaping each permit
one additional bookkeeping second. Foreign groups are never signalled. The
launcher records exact command, process/group IDs, observed owned processes,
exit, timestamps and full surviving inventory; remaining live owned processes
prevent success. A hard kill or filesystem failure may prevent final metadata.

The producer's actual wrapper time must be<=26.1seconds and observed completed
child-group RSS<=2GiB. Source-bound cooperative15second checks include a final
check; final-record time over15 is reported separately, since descheduling or
bookkeeping does not itself prove extra scientific work. Sampled RSS cannot
prove absence of transient overshoot; launcher memory is additional.

Total new audit artifacts are capped at8MiB:7MiB evaluator plus1MiB launcher.
Ordinary evaluator writes stop at6MiB, reserving1MiB for terminal result/manifest.
The combined stdout log cap is128KiB. Writes are exclusive and fsynced; final
inventories bind all retained bytes, which become read-only. The launcher is
adapted from reviewed CIFAR auditor source `4de644d379884ba85ce71bf39611c7be6ab90d10`
with new names/location and the8MiB storage allocation; timing and owned-group
cleanup remain unchanged. No scientific producer source is modified.

After review, root must pin the metadata and exact clean audit HEAD before:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
nice -n 10 /data2/neural-shrimp/.venv/bin/python \
  tools/reviews/launch_s1r_lower_gap_audit.py \
  AUDIT_SOURCE_FULL_SHA /ABSOLUTE/METADATA_PINS.json METADATA_SHA256
```

Outputs are `/data2/neural-shrimp-data/audits/s1r-lower-gap-independent/attempt-001`
and its `attempt-001-launcher` sibling; both must be absent. Current source
preparation creates neither directory. The original evidence remains immutable.

## Synthetic controls and limits

Thirty-five synthetic tests pass in0.50seconds. They include a hand-constructed
six-role72D saved-algebra graph and all24 complete closed-form bases, without
calling SVD or any model; corrupt matrices/bases/scales/phase labels, rank
windows and abstentions, thirteen trial conditions, signed-zero identities,
journal/counter/partial-return boundaries, clean-source/nice guards, incomplete
finalization, Python-optimization-safe launcher guards and owned-group cleanup.
The first full-graph fixture used mathematical negation instead of the declared
subtraction and was corrected to preserve signed-zero input bytes; no audit
threshold was changed. There has been no target artifact, profile or outcome read.

This is independent saved arithmetic and provenance, not an interval proof.
The analytic gradient/Hessian primitive, returned arrays and their input source
bindings are reused. It cannot establish derivative correctness from first
principles, branch ancestry, a turning-point theorem, absence of a branch,
training utility or inference utility. Any next target work needs a distinct
reviewed design and authorization; this unit stops after the one audit.


## 2026-10-08 UTC — outcome-blind source review amendment

Independent review of initial audit source c009109741b49e02519b64c249f7b2e9dbaaead4
found two incomplete-attempt defects and one completed-phase validation gap.
The producer remains unlaunched; no target evidence has been inspected.

The auditor now retains an attempted-counter increment only for the one next
expected operation when its start emission failed before journaling. Such an
operation cannot have returned. A returned-counter increment remains limited
to an already durably started, pending operation. Both reported and durable
counts, and any unjournaled start, remain explicit in the audit receipt.
A terminal checkpoint may lag the journal by exactly one event only on an
incomplete attempt with otherwise exact durable counters, reflecting the
source's append-before-checkpoint/latest ordering. Wider or contradictory
lags fail. Completed-role append lag likewise requires that the role-complete
event is the journal's final event.

A failed numeric check can be retained as unvalidated partial evidence only
for the actual terminal unfinished role. A durable complete role, any earlier
role followed by later evidence, or a complete attempt remains strict.
Completed phases require finite numeric, non-boolean losses; paired null
losses cannot masquerade as a valid abstention. Rejected terminal raw returns
remain retained with no arithmetic consistency claim for that role.

The repaired suite includes synthetic reproductions of these boundaries,
impossible attempted/returned deltas, checkpoint lags, completed-role
corruption followed by failure, and null/boolean/nonfinite phase losses.
No producer source, scientific gate, input selection or resource cap changed.

Final repaired focused suite:57 synthetic controls passed in0.66seconds under CPU/BLAS1 and nice10.
