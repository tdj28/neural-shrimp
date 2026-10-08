# Independent saved-dose outcome audit — 8 October 2026 UTC

Status: prospective source/card, awaiting root review and separate authorization.
Actual producer output is unopened. Only root-supplied execution/manifest
metadata below has been seen. The four-gradient producer has run; no numerical
outcome was supplied to this auditor before this design. This is a source-bound
exploratory verification, not a new utility experiment or confirmatory claim.

## Exact inputs and scope

Reviewed producer source:075a76ecddddaeeae92f714d8ebbf170812b0500;
source review:167d55bfe638ff788dae3343b62204ca960a6dbb;
execution:977514e4f53025f485bcd47ce947c9adef044843.
All ten producer-bound source/card bytes must match the reviewed historical
commit and execution commit. Never compare future working-tree bytes as proof
of the old execution.

Producer raw directory:
`/data2/neural-shrimp-data/explore/utility-dose-feasibility/attempt-001`;
manifest d0b6a4fd5a0c96c3a1ac95bbfb33cb3661b82202062e37fac238f00c92bf67d2.
Sibling `attempt-001-launcher` manifest:
b3092fb6fee21f53fde9a09b7fa87665b618543e7225d5bf69969799d79de9e2.
These externally supplied full hashes are mandatory CLI inputs, never derived
from whatever receipt is currently present. Root will authorize this exact pair.
Original four-fork inventory remains manifest
2f962ced4b6f32aca588f1012985591acc670437ff944e8e15375acf4fd2e03f,
with its input review42d901c0957221180aec5966b2c1726ea999b11b.

Verify complete directory inventories, sizes, safe paths/no symlinks, all hashes,
producer/launcher source mirrors, saved launcher bytes, command, execution and
review identities, PID fields, UTC ordering, exit, untruncated stdout and caps.
The launcher inventory must bind the entire raw inventory, result and manifest.
Require all four completed producer gradient returns in exact fixed event order,
complete four-fork input retention and terminal checkpoint. A failed/incomplete
producer is retained and reported as unaudited/incomplete, never interpreted as
successful feasibility or as a utility failure. No repair, retry or new target.

Reconstruct only S2-28011's24-row training inputs from original teacher/train
streams and verify both frozen train hashes. No validation/test streams are
instantiated. Bind the exact13-coordinate native state and both binary64 recovery
rates for each fixed row4100/origin fork. All pending states/rates must match the
bound originals bit-for-bit; all-gradient order and counts must be four.

At each unchanged state, compute one independent analytic gradient by explicit
sample/coordinate loops for tanh2x3x1, half-MSE and ridge0.001 on all13 parameters.
W0 occupies0:6, hidden bias6:9, output weights9:12, output bias12:13. No producer
gradient, map, loss, trajectory, HVP or Hessian routine is called. The same NumPy
binary64 tanh is used; this is independent formula/operation ordering, not an
independent precision backend or an exact-real certificate. At most four audit
gradients; retain each return before checking it or checking the deadline.

Compare every original gradient entry to the independent gradient with
`abs(error)<=1e-12+1e-10*abs(reference entry)`. Independently replay every scaled
gradient, block/full norm, R=.05||P*g||, reduction fraction, treatment-minus-sham
increment, total update increment/length, dose and reason/decision, first from
the saved gradient and then from the independent one. For derived scalar/vector
entries use `abs(error)<=1e-11+1e-9*abs(reference entry)`. Keys, shapes, dtypes,
labels and Boolean decisions must agree exactly; no eligible field is dropped.
Uniform reduction.05, maximum group reduction.25 and the original dose-tolerance
formula remain literal checks. Fractions above.25 stay above.25: comparison
tolerances never relax the scientific feasibility gate. Any decision disagreement
or missing/nonfinite comparison causes audit failure. Full and block norms must
not be substituted for the sham-relative treatment dose; total step lengths
remain unequal generic covariates.

A complete passing audit says only that the fixed dose feasibility decision and
retention were independently reproduced. If infeasible, this rejects the fixed
candidate design, not utility generally. Passing cannot acquire missing controls,
show basin/geometry specificity, authorize a pulse or alter EXP-027 restrictions.

## Bounded execution, only after source review

CPU/BLAS1,nice10; internal cooperative5s, external10s TERM plus5s KILL; no GPU,
paid call, reserved seed or protected outcome. One exclusive attempt, no retry.
Own output directory:
`/data2/neural-shrimp-data/audits/utility-dose-feasibility/attempt-001`;
launcher sibling `attempt-001-launcher`. Total new audit raw1MiB:768KiB Store
(576KiB ordinary/192KiB finalization) plus256KiB launcher, stdout<=128KiB.
Save complete source/input binding, all stage/gradient/comparison outputs,
PID/command/exit, failure, result and manifests. Atomic/fsynced writes and
read-only final artifacts; forced KILL can leave explicitly incomplete evidence.
Audit and source review are separate from permission to launch.

After root approval, replace AUDIT_SOURCE/AUDIT_REVIEW with full committed hashes:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_utility_dose_audit.py --source-commit AUDIT_SOURCE --review-commit AUDIT_REVIEW --execution-commit 977514e4f53025f485bcd47ce947c9adef044843 --raw-manifest d0b6a4fd5a0c96c3a1ac95bbfb33cb3661b82202062e37fac238f00c92bf67d2 --launcher-manifest b3092fb6fee21f53fde9a09b7fa87665b618543e7225d5bf69969799d79de9e2 --out /data2/neural-shrimp-data/audits/utility-dose-feasibility/attempt-001 --launcher /data2/neural-shrimp-data/audits/utility-dose-feasibility/attempt-001-launcher
```

Prospective synthetic controls include independent finite differences on toy
loss, all bias/ridge terms, boundary/next-float/zero/nonfinite dose cases,
every-entry comparison, external-pin and inventory corruption, source/process
contradictions, record/array association, durable interruption and a complete
toy producer-to-auditor replay. They do not read the real target output.
Future compact companions are `docs/reviews/utility-dose-independent-outcome.md`
and `docs/reviews/utility-dose-independent-outcome-receipt.json`; original
producer receipts and raw evidence stay unchanged.
