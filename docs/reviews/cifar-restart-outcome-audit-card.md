# Prospective independent saved-outcome audit: fixed CIFAR restart

8 October 2026 UTC. Prepared before the new restart outcome exists or has been
opened. This is an exploratory evidence audit, not a preregistration or a new
solver experiment. Root authorized source preparation only; source review,
producer launch and this audit's single execution each need their separate
root authorization. No target/profile/gradient/JVP/model/QR/solver operation is
performed by the auditor or its synthetic controls.

## Frozen evidence boundary and externally supplied metadata

Producer source is `8aa146fdbeca2f7d1cd4957995ced9b8bda719ee`; its independent
source review is `1f4f2abe8b4d4d4b8ad6e50cf1c9416d568d65f9`. The twelve source
and plan paths in `SOURCES` must have identical bytes at both that source and
the actual execution commit. Execution HEAD may include subsequent metadata.
The evaluator imports only standard-library modules and NumPy. It does not
import the producer's numerical implementation or SciPy/Torch.

After the producer attempt, root supplies a metadata-only JSON file with
exactly these keys, and pins that file's SHA-256 in the authorization/command:

```json
{
  "execution_commit": "ACTUAL_PRODUCER_FULL_40_HEX_SHA",
  "source_review_commit": "1f4f2abe8b4d4d4b8ad6e50cf1c9416d568d65f9",
  "raw_manifest_sha256": "FULL_64_HEX_SHA_OR_NULL_ONLY_IF_ABSENT",
  "launcher_manifest_sha256": "FULL_64_HEX_SHA"
}
```

A present malformed scientific manifest still receives its actual bytes hash;
null is allowed only when the file is absent. No numerical result is used to
choose or change this audit. Both fixed source locations are under
`/data2/neural-shrimp-data/explore/cifar/restart-sensitivity/`:
`attempt-001` and `attempt-001-launcher`. The complete old reference bundle,
corrector and restart50/four-cycle baseline inventories are independently
pinned in evaluator `PINS`; all original source-file hashes are checked too.
There is no acquisition of new input data or replay of the old computation.

## Checks and fixed interpretations

The auditor checks complete regular-file inventories, exact output-name scope,
all twelve producer source bytes, archived launcher source, review/HEAD/runtime
mirrors, fixed settings, NumPy/SciPy/Torch versions and GMRES function hash.
It verifies launcher command/thread/nice limits, byte caps, owned process
identities and exit, recorded start/finish ordering, actual elapsed times and
observed RSS. All saved arrays require float64 and the declared dimension.
State/residual/trial relationships use exact dtype, shape and C-order bytes,
including signed zero. Model, normalized data, rates, original theta and the
four replayed initial arrays are bound to the old immutable evidence.

The event audit checks attempted versus returned counts and proper nesting:
A-action -> one two-step JVP -> two HVPs; direct gradients are top-level;
parity precedes actions; callback i follows product i; trial evaluations follow
the cycle-end and independent true-residual actions. Completed counts satisfy
A=callbacks+2, J2=A+1, HVP=2*J2+1 and direct gradients=11, within the reviewed
202/203/407/11 caps. Interrupted open operation stacks remain explicit. Both
legitimate early convergence after at least50 callbacks and a full200-step
cycle are supported; fewer than50 callbacks cannot qualify the comparison.
The first50 prefix rule remains absolute1e-12 + relative1e-10 of the saved
baseline, without any tolerance or threshold selected from new results.

Saved gradient/HVP/J2 parity retains the producer's componentwise
1e-11 + 1e-9*abs(reference) scale. Independent norm and dot reductions use
Python `math.fsum`; scalar mirror comparisons use relative1e-12/absolute1e-20.
The evaluator independently reconstructs e=Ap+R, its true forcing ratio q,
R dot Ap, all four absolute merits and Armijo decisions, relative closure,
primitive separation, and the aggregate candidate list. All four trials use
the original theta and fixed alphas1,1/2,1/4,1/8. SciPy info0, independently
requested true tolerance1e-10, loose q<=1/2, strict descent, and Armijo1e-4
remain distinct. Closure<1e-8 and separation>=1e-7 remain unchanged screens.
An inexact-direction candidate is never an accepted orbit or continuation.
The old forcing ratio is independently reduced from its saved arrays.

A complete consistent attempt receives `saved_outcome_audit_pass`, regardless
of whether the solver/trials improve the scientific diagnostic. A consistent
incomplete attempt receives `incomplete_saved_records_consistent` with no
qualified comparison and no invented later fields. Missing or malformed final
scientific manifests can receive only `incomplete_inventory_only`, pinned by
the launcher's full retained-byte inventory; no scientific arithmetic/counter
verdict is asserted in that case. Contradictions fail closed as `audit_failed`.
Invalid, interrupted and mismatching evidence is retained, not retried.

Timing interpretation follows the producer's cooperative45-second checks at
operation boundaries, including its final check. The final elapsed record is
made just afterward: the audit reports any excess over45 separately instead
of interpreting bookkeeping as a new scientific deadline event. It verifies
worker time <= launcher time and launcher <=66.1 seconds (65-second supervisor,
one-second reap allowance, 0.1-second polling allowance). Actual completed
worker and sampled owned-group RSS must be <=2GiB. Incomplete memory-guard
records can report the observed excess; RSS monitoring is not a kernel hard
limit and does not prove absence of a transient unsampled excess.

## Exact single-run card; execution not yet authorized

Run only from this audit branch at its clean, exact committed HEAD after
independent source review and separate root authorization. Both output
directories must be absent. Root resolves and records the three placeholders:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
nice -n 10 /data2/neural-shrimp/.venv/bin/python \
  tools/reviews/launch_cifar_restart_audit.py \
  AUDIT_FULL_SOURCE_SHA /ABSOLUTE/PIN_METADATA.json METADATA_FILE_SHA256
```

Outputs are `/data2/neural-shrimp-data/audits/cifar-restart-independent/attempt-001`
and its `attempt-001-launcher` sibling. CPU/BLAS1, launcher and worker nice>=10;
no GPU, paid compute or protected outcomes. The worker has a15-second timer;
GNU timeout sends TERM at20 seconds and KILL after5. The launcher independently
bounds inherited pipes/live leaders at25 seconds, then uses bounded one-second
reaping/owned-group cleanup bookkeeping. It always signals only its own group
and records any surviving owned PIDs; those prevent successful exit. The
launcher also performs read-only hashing under nice>=10. There is one child
attempt, no retry, no resizing, no new model evaluation and no target QR.

Total raw cap4MiB = 3.5MiB audit + 0.5MiB launcher, including manifests and
archived launcher source. Ordinary audit writes stop at3MiB, reserving0.5MiB
for final result/manifest; combined log cap128KiB. Files are exclusive, fsynced,
hash-inventoried and made read-only at finalization. A killed process may leave
partial files; its launcher retains the complete surviving inventory and
process exit. Preflight/source/pin rejection happens before output creation.
The original evidence is never modified. Final outcome notes must report both
source and execution commits, external pins, actual runtime/exit/bytes and all
failures without changing this frozen evaluator.

## Limits and source-only validation

This checks saved arithmetic and provenance. It cannot independently regenerate
A*p, map images, derivatives, Arnoldi basis/actions or the full solver path;
those are retained source-bound inputs. It proves neither numerical periodic
existence nor attraction, and yields no causal claim about restart or input
span beyond this one matched saved diagnostic. New model or solver work would
need its own reviewed card and authorization.

Fifty synthetic controls passed in0.29 seconds on CPU/BLAS1, nice10. They cover
complete50/200 histories, interrupted nesting, callback/order/counter corruption,
prefix boundaries, signed-zero and array identity errors, separate info/true
residual/forcing/Armijo gates, nonfinite/overflow controls, resource/PID/time
contradictions, complete and malformed inventories, clean/nice/source guards,
durable finalization, an owned synthetic descendant after leader/pipes exit,
and production guards under Python optimization. Independent source review
will add a fabricated complete-file audit fixture before any target reading.
