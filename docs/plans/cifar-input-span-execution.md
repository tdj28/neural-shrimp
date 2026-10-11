# CIFAR input-span execution card — 8 October 2026 UTC

Status: source and synthetic controls only; profile and saved-data projection are
not launched. Root must independently review the exact pushed source before the
one resource profile. A passing profile alone does not authorize the target:
root must review its receipt and authorize the fixed collection. This implements
`docs/plans/cifar-input-span-diagnostic.md` at c530287c, without changing its
scientific question or triage threshold. No paid/GPU use, model evaluation,
trajectory, gradient/JVP/HVP, correction, Krylov solve, spectrum or held-out data.

## Fixed source and input contract

The runner requires a clean committed checkout and records the exact execution
commit, file SHA256 values, review commit, PID, UTC endpoints, CPU/environment,
NumPy/BLAS configuration and thread limits. Its six bound files are the algebra
module, runner, test, current `cifar.py` packing/loss source, original plan and
this card. Later compact receipts and research-log entries may change, but these
six files must remain byte-identical for profile reuse. Historical input source
bytes are checked against the pinned commits, not the future integration HEAD.

Only the two complete manifest inventories named in the original plan are
consumed, at their fixed absolute paths. Every file, including unused historical
reference derivatives, is hashed; exact inventory, safe basename, no symlinks,
finite binary64 shapes, subset-A x/y array identities, full binary64 eta/lambda,
model packing and original state identity must pass. Only x/y/theta are loaded
from `reference.npz`; no derivatives are newly evaluated. The five saved pairs
are original, trial0, trial1, trial2, trial3. Each R must equal saved map2 minus
its theta, and trial states must exactly equal original theta plus saved
alpha*p for alpha 1, .5, .25, .125. Saved p, Ap and e belong only to the original
basepoint; e must equal Ap+R bitwise. No trial derivative is inferred.

The nonlinear synthetic fixture uses the actual small CIFAR MLP/Problem class
and verifies all first-layer C-order entries, both-layer training and ridge
gradient lambda*theta. The target runner imports only NumPy/standard-library
code. The fixed profile seed28240 and synthetic-test seed28241 are unprotected;
no seed29000–29999 or subset-B pixels/LM assessment are opened.

## One resource-only profile, pending independent review

Command (substitute the exact source-review commit after approval):

```
env PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=2s 8s nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/diagnose_cifar_input_span.py profile --out /data2/neural-shrimp-data/explore/cifar/input-span/profile-001 --review-commit REVIEW_SHA
```

Exactly one seeded standard-normal synthetic 3072x512 reduced QR, retaining all
512 columns. Internal timer5s; external TERM8s, KILL2s later, so the outer ceiling
is10s. This stricter TERM point leaves cleanup inside the plan's10s ceiling.
No target arrays are opened by this mode. Retain compact synthetic x/Q/R byte
hashes, shape, all basis checks, measured QR time, total internal time and RSS,
source/environment/review binding, result and manifest. Synthetic Q/R arrays
are not retained. Require finite positive QR time <= actual elapsed <=5s, all
basis checks, and the fixed projection `2*QR_seconds+5 <=30`. Failure stops;
no retry, alternative rank, thread count, seed or favorable-outcome gate.

## One saved-data collection, pending profile receipt review

Command, only after separate root authorization:

```
env PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 45s nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/diagnose_cifar_input_span.py collect --out /data2/neural-shrimp-data/explore/cifar/input-span/attempt-001 --review-commit REVIEW_SHA --profile-manifest-sha256 APPROVED_PROFILE_SHA256
```

Internal30s, external45s plus5s cleanup. Recheck host ownership/load before each
launch, CPU/BLAS1 and nice10; do not disturb foreign jobs. Record wrapper PID,
child PID, exact argv, start/end UTC/monotonic time, exit and log hash in sibling
launch records. No retry or automatic follow-on. An interruption retains its
exclusive attempt and whatever was written; SIGKILL may leave it incomplete.

Revalidate the profile's complete inventory, source bytes, environment, review,
actual metrics and numeric gates (not just its status bit). Compute one target
reduced QR; never materialize3072x3072 P. Retain Q/R, exact-Fraction eta/lambda
identities and q, plus rounded q and q²−1 used in this floating diagnostic.
Retain all512 columns even if rank deficient. Record dimensions98666/16746/81920.
In addition to the original orthogonality/representation gates, require QR
reconstruction relative error <=1e−12 and implicit P(PQ)−PQ relative norm
<=1e−10. These are implementation checks, not a rank test or exact enclosure.

For all13 saved vectors (five states, five residuals, p/Ap/e), retain the full
active and complementary vectors, norms, squared fractions, reconstruction,
implicit projector idempotence, Q^T N, XN and Pythagorean errors. Only the first
98304 entries reshape into3072x32 W1; b1/W2/b2 stay in the active coordinates.
Checks use the original plan's `1e−12+1e−8*||full vector||` scale. The additional
XN check uses the same absolute/relative formula with `||XW||`; Pythagoras uses
`1e−12+1e−8*||full vector||²`. Retain seven predicted complementary vectors:
five `(q²−1)N`, one `(q²−1)p_perp`, one `(q²−1)(p_perp+N_original)`.
Check full, active and complementary e=Ap+R separately at the original basepoint.
These are numeric identities against saved arrays, never a new map/JVP evaluation.

All gates passing and original complementary squared R fraction >=.5 is only
priority triage for a separately reviewed reduced solver. It does not prove
convergence, explain the GMRES stall, qualify an orbit or preserve unseen-input
predictions. Report small share or failed identity without weakening thresholds.

## Retention and source controls

Both modes create an exclusive directory. Every JSON/NPY write is flushed and
fsynced, and the final manifest hashes all retained files, including a partial
array after a caught interruption. Ordinary writes cannot consume the final
1MiB reserve; final writes leave512KiB for external launch metadata/logs. Bound
all newly retained output for the entire unit to64MiB; the launcher must retain
no more than512KiB in combined profile/target sibling logs/records. The profile
uses compact JSON only; target NPY payload is40,727,888 bytes (Q/R plus33 full
vectors), plus35 NPY headers and compact metadata. Post-run receipts must report
actual total bytes including both modes, manifests and launch records. Any
unexpected cap exceedance is a failed unit, never grounds to delete evidence.

Prospective synthetic controls cover exact dimensions/packing; actual nonlinear
ridge and direct-product residual/JVP identities; q²=1 non-elimination; rank-
deficient QR completion; finite/dtype and saved-order checks; complete inventory
corruption/symlinks; rehashed inconsistent profile metrics/source/settings;
reserved-byte finalization and partial-array/interruption retention. None runs
the full-size QR or opens saved target arrays. Independent source review and
later independent saved-output audit remain required before scientific claims.

### Pre-execution review amendment

Root hardened the source after independent review, before any profile or target
run: collection now requires the exact external profile-manifest SHA256 from
the post-profile approval, and all named exact saved-array relationships compare
binary64 bytes, including signed zeros. These tighten provenance and do not
change the numerical identities or scientific thresholds.
