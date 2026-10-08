# Prospective independent saved-CIFAR input-span audit — 2026-10-08 UTC

No profile or collection numerical outcome has been opened while preparing this
source. Parent supplied only the eight execution/review/manifest/wrapper identities
in `cifar-input-span-outcome-audit-pins.json`. The fixed six producer sources are
those reviewed at17453ecc08b1d0b08c159f506b9447b1b6061968; source review is
97d0cad7bae51450dde17cd4f99dd1409c2b7458. Root must separately authorize the audit
with this committed source SHA and the pins-file SHA before the command below.

Run once in the clean owned review checkout:

```
/data2/neural-shrimp/.venv/bin/python tools/reviews/launch_cifar_input_span_audit.py AUDIT_SOURCE_COMMIT docs/reviews/cifar-input-span-outcome-audit-pins.json PINS_FILE_SHA256
```

The launcher fixes exclusive outputs under
`/data2/neural-shrimp-data/audits/cifar-input-span-independent/attempt-001` and
`attempt-001-launcher`, CPU/BLAS1, nice10,15s internal,20s external TERM plus5s
KILL/owned-process-group/inherited-pipe supervision. No retry. Raw output is at
most4MiB including the launcher (3.5MiB audit,512KiB launcher;128KiB log). Ordinary
audit writes reserve512KiB for terminal metadata. Every returned failure is
retained and hashed; external kill can leave a partial bundle, which is a failed
audit. This is saved-data arithmetic, not a fresh model or QR profile.

The evaluator independently hashes complete pinned profile, collection, both
launcher inventories and both original-input inventories; verifies executed
sources, exact commands, review/profile bindings, environment, retained bytes,
PID/exit/log metadata and the unchanged resource/profile gate. It reuses saved
Q/R to check all512 columns, orthogonality, X representation, idempotence and
QR reconstruction without recomputing QR or selecting rank. All13 saved vectors
are independently reprojected with saved Q, with packing, Pythagoras, Q^T N and
XN checks; all five period2 R-perpendicular relations and original p/Ap/e
relations are checked. Full saved residual/trial/e-sum identities are byte exact.
Exact Fraction arithmetic distinguishes q²−1 over dyadic input parameters from
the producer's rounded binary64 factor. The fixed inclusive fraction>=.5 triage
is reconstructed only when all original numerical gates pass.

Independent norms use explicit sum-of-squares instead of the producer's BLAS
norm. Saved scalar agreement uses5e-12 relative/2e-14 absolute tolerance. Only the
cancellation-prone squared-norm error additionally allows256 eps times
max(1,||v||²); reconstructed arrays allow2e-12 times max(1,||a||,||b||). These are
cross-implementation audit tolerances, not changes to scientific gates: the
producer's1e-12+1e-8 scale, basis1e-10/1e-12 and triage.5 gates are independently
recomputed and their Boolean decisions must agree. A completed target with failed
identity checks may receive a consistency pass; that does not approve its
scientific gates. An incomplete producer bundle can only receive a scoped
provenance-only result, with no triage verification.

No training-map, gradient/HVP, linear solve, trajectory, new QR, protected data,
paid compute or GPU call is permitted. The original saved map images and Ap are
bound inputs, not independently regenerated. Rank and solver-stall causality are
not established. Final companions will be
`docs/reviews/cifar-input-span-independent-outcome.md` and
`docs/reviews/cifar-input-span-independent-outcome-receipt.json`.

Synthetic controls precede freezing and cover rank-deficient orthonormal
completion, projection/packing mutations, nonfinite precision/shape failures,
exact rational versus rounded factors, threshold endpoints, inventory/link
corruption, launcher binding corruption, source cleanliness, nice enforcement,
interruption retention and exclusive outputs. They never open fixed data paths.

Preparation validation:34 synthetic controls passed in0.09s at ordinary nice0
(with permitted-run fixtures mocking nice10), then34 passed in0.09s under actual
nice10. The production guard still rejects nice0 before output creation. No
profile or target outcome was opened for these tests.
