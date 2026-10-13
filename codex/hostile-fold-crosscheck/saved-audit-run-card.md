# Task7 independent saved-array audit — prepared 2026-10-13

Scope: audit the one completed, qualified `fold-001` attempt after its owner has verified process closure and its original receipt, source commit and Git blobs. No model construction, model replay, optimizer action, passage trajectory or certificate claim. The producing raw attempt is read-only; write a fresh sibling audit directory. A failed or interrupted audit remains evidence and is not silently retried.

The auditor authenticates every original receipt artifact, the captured eleven-file source bundle, original plan, actual hostile steering/checkpoint/prefix metadata and NPZ descriptors. It explicitly reports that it has **not** verified Git blobs; the separate launch-owner audit supplies that provenance. Source correctness and the trusted numerical backend are assumptions, not consequences of hashes.

It independently recomputes the complete saved monodromy spectrum and left/right eigenpair residuals, critical real +1 isolation and complementary radius, both phase closures and proper divisor, augmented residual, eigenvector normalization/pairing/gauge, nested-AD scalar contractions, gauge-invariant `ab` and passage coefficient, both dense-versus-JVP directions, and each fine finite-difference coefficient check. This is evidence rung 2, with `certificate_passed=false` throughout.

## Fixed resource envelope

One local CPU thread, exact nice 10, at most 4 GiB address space, fixed 180-second deadline. A stricter inherited soft memory limit is preserved; an inherited hard limit below 4 GiB is refused. All three BLAS/OpenMP environment limits must equal 1 before imports. The CLI arms a 180-second timer and retains a failed audit report on exceptions; the caller should also apply its usual bounded process supervision because delivery of Python signals can wait for an in-progress native call to return. No subprocesses are created by the auditor. No GPU, paid resources or protected data.

The parent approved this envelope after one synthetic-only 896-dimensional profile on 2026-10-13. Full left/right eigendecomposition, four full residual products, spectrum assignment and a representative 64,241,454-byte NPZ write/read/hash took 1.412872 s, with peak RSS 205,944 KiB. All synthetic residual checks passed. This is a measured engineering observation, not a runtime/memory upper bound for the real record or a scientific result. Original profile source, output and synthetic arrays are retained at:

- `/tmp/task7-audit-cost.py`, SHA256 `6f40f74c577356d8c4e0d12badb8aa3f66c2d103acebcf9e6f1f47fb71b509da`.
- `/tmp/task7-audit-profile-001/report.json`, SHA256 `d0de78622ce72b7cb1dc42defaa4acec855add99c78b411c60e8b47ede760775`.
- `/tmp/task7-audit-profile-001/synthetic-arrays.npz`, SHA256 `d8c858203a5eb8ddb8dde43f675ab7b4cdd8ada4b1e64ee22a1d2618d7ff7f02`.

## Release and execution boundary

Prepared source SHA256: `63d68c5b4ff2e83e9b666c1e42a8b69847309f9bab67bc864ee8bc404efd5598` (`saved_audit.py`).
Tests SHA256: `73f1765e827da86c3064ba80abc05ecaeecf95c19ba15b1dd820d2202c2593c9` (`tests/test_codex_hostile_saved_audit.py`).
All **32** tiny controls passed in 3.83 s. They include the actual saved-format loader and full synthetic success chain, resealed provenance/scientific corruptions, and resource refusals. No real target record was admitted.

Before the actual audit, the Git owner privately checkpoints this source/card, supplies externally authenticated `--receipt-sha256`, `--source-commit` and `--plan-sha256`, confirms successful original fold closure, and chooses a fresh `--output` directly under the owned Task7 data root. The CLI arguments are `--attempt`, those three external pins, and `--output`. The original fold attempt must not be modified. A numerical audit can pass only if every recomputed numerical gate and every saved-metadata comparison passes.
