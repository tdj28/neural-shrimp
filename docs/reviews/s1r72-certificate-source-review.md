# Independent S1R72 certificate adapter review

8 October 2026 UTC. **Pass for the fixed resource-only profile at producer
`2c00544ca35a5cf4edb28bacc0a80e285e284073`; no profile or target was run by this
reviewer.** Actual certificate collection still needs the measured resource
gate and root authorization. The old source `7b7dea941533ea7c14616235e351eccbed31594b`
had a profile-reuse blocker; it is repaired below.

**TLDR:** The 72-parameter tanh network fits the existing exact-real proof
machinery without a mathematical change. The adapter binds the selected saved
state, data, model and actual rates. This earns permission to measure feasibility,
not a new orbit result. A future successful checker result would prove a small
box contains a unique attracting primitive period-two orbit of the supplied
exact-real map; it would not prove branch identity, geometry or utility.

## Binding and mathematical scope

Read the new adapter, card and tests, all three C1 files, the saved-state decoder,
model/data generator and float analytic map. The three proof files
`rigorous_orbits.py`, `certify_c1.py`, `verify_certificates.py` are byte-identical
to authoritative 960ab535. All 11 adapter-bound files match producer 2c00544c.

A read-only preparation check hashes the complete raw inventory, decodes its
72-sample/8-input data and node0004 phase 0, regenerates S1R-28002 against the
frozen EXP-028 plan, and compares exact rational strings with `Fraction(float)`.
No target map, gradient, correction or proof was evaluated. Exact identities:

- Raw manifest: `45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b`.
- Selected `00082-node.json`: `7b60f3626190490a27324eb6d0f5f834396cdddcc286ee34e7dcc53ccc43a2ba`.
- Prepared input: `b05c7456ef9db17aec5bd1f72e76b0f5b3be3f5d6050e8bd950c1067888ff443`.
- Actual hidden/output rates: `1870594557782953/140737488355328` and
  `2385682499131913/562949953421312`, not rates reconstructed from decimal logs.

There are 64 row-major hidden weights and 8 output weights, no biases or optimizer
state. Arb encloses tanh, the gradient and all Hessian terms for the exact
binary64-derived data/scales/ridge/rates. The sample-Jacobian plus residual-
curvature formula matches the 72D synthetic map and Jacobian in producer tests.

The proof argument has no 13D restriction: for `H=F^2-I`, the exact midpoint
preconditioner R and interval bound on `I-R*DH(X)` establish a contraction
mapping `theta-R*H(theta)` of the box into itself. The norm bound also makes R
nonsingular, giving an actual root and uniqueness in that box. Interval
separation from `F(theta)=theta` excludes the only proper divisor. A rigorous
infinity-norm bound below 1 for a power of the enclosed full monodromy proves
strict attraction. Proposal Newton steps and numerical spectra cannot pass
these proof gates. Assume correct Arb/FLINT enclosures and the usual smooth-map
contraction/local stability theorems; this is not proof-assistant formalization.

## Found blocker and repaired controls

At 7b7dea94, profile reuse checked file hashes but trusted the stored timing
projection and did not require completed runtime-end metadata. Eleven independent
controls reproduced acceptance of rehashed contradictory projection/timing/RSS,
wrong action/count/source/status, absent completion, or false manifest total.
Initial focused outcome: 14 passed, 11 failed in 0.64s. This was a resource/provenance
gap, not a false mathematical certificate.

Producer 2c00544c recomputes `2*14*map_seconds+10`, checks literal cost constants,
finite positive timing/RSS within caps, matching completed profile metadata,
exact source/settings/input identity and actual manifested bytes. Independent
controls now also cover bool/nonfinite timings, wrong PID, over-budget elapsed,
unlisted data and symlinks. All 32 focused tests pass in 0.58s: 11 producer/core
plus 21 independent cases. Independent 72D ridge-only analytic controls pass an
attracting fixed point, reject a false primitive period 2, and reject attraction
of an unstable fixed point. No target data enter those numerical controls.

Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src nice -n 10
/data2/neural-shrimp/.venv/bin/python -m pytest -q -p no:cacheprovider
 tests/test_s1r_period2_certificate.py tests/test_rigorous_orbits.py
 tests/test_s1r72_certificate_independent.py` (join wrapped lines).

## Remaining limitations and launch boundary

The profile is one map/Hessian call, not a proof attempt. Its factor 2/10s allowance
does not guarantee matrix algebra, interval-width behavior or runtime; the hard
collection deadline still matters. Storage is checked after writing, and external
termination may leave an incomplete manifest as disclosed in the card. Preserve
partial artifacts and launcher exits. No extension of precision, box, power,
node or phase is implicit in a failure.

A certificate applies to its returned box after Newton proposals. Quantify
whole-box proximity to the selected binary64 seed before linking it to that
numerical state; even close proximity is not historical branch identity. The
[prospective outcome card](s1r72-certificate-outcome-card.md) fixes the 1e-7 bound
before any outcome. There is no parameter-neighborhood, boundary, circle, chaos,
floating-point-training or utility proof here. **No remaining source blocker
for the stated profile; no certificate outcome or feasibility claim.**
