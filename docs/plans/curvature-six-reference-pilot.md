# Six-reference curvature pilot: implementation and exploratory run card

2026-10-09 UTC. Prepared from root0fd62d2d after the user requested resuming
small exploratory computation. This card implements the reviewed fixed panel,
fit and gates in `curvature-subspace-mechanism.md` and its input inventory;
there is no new point/rank/bulk selection. It follows the adaptive protocol:
committed source and this prospective research-log card before execution;
a separate prospective outcome auditor is not required for this exploration.
Root owns resource checks, source review and any profile/panel launch. Author
preparation opened no empirical input files or arrays and ran only synthetic
controls. This source commit is not itself launch authorization.

## Fixed computation and distinctions

The six slots and their ordering are exactly the inventory's four stable rows
and two crossing-neighborhood references. Load only the previously consumed
training x/y; verify all13 historical input identities, original shape-plus-byte
data hashes, current array descriptors, full model/data/rate/state identities,
and committed transitive runtime sources. Worktree-local metadata references
are relocated by repository-relative path with unchanged hash pins. Do not
regenerate data or protected splits. Historical criticality execution gaps
remain gaps. S2 crossing rate reconstruction uses Python float scalar
`10.0 ** log10_value`, as in its historical producer, not vectorized power.

One analytic tanh adapter computes loss, gradient and Hessian together, including
both biases, ridge, loss normalization and nonunit input/output scales. Hidden
bias is outside input scaling; output bias is outside output scaling. Each
reference needs acquisitions at theta0, theta1, F(theta0) and F(theta1), at most
four calls. S2's single saved phase derives theta1=F(theta0). Exactly repeated
inputs may share a return only under exact state/model/data/rate identities;
near equality never caches. This is at most24 **full fused analytic calls**,
not12 Hessians plus hidden uncounted Hessians. The profile's reused first return
counts against this24. Both actual native monodromies are retained as
DF(F(theta_j)) DF(theta_j), together with physical closure and stored-pair
transition checks. Old S1R comparison matrices are historical stored-phase
products, not assumed exact native matrices at approximate closure.

The existing helper is unchanged. Its one full native0 eigensystem/reference
is the declared target; actual native1 matrix/closure and both formation
agreements are retained without claiming a separate native1 diagonalization.
Analysis costs at most12 symmetric eigensystems,6 full general eigensystems
with left/right vectors and6 full basis SVDs. Decomposition observers persist
inputs and returns before checks. Derivative returns likewise persist before
finite checks, image construction or cooperative deadline checks. Nonfinite
arrays remain in NPZ; nonfinite scalar returns are explicitly tagged in JSON.
No new trajectory, corrector, continuation, scan or adaptive rank expansion.

A slot's scientific failure records its reason and continues to later fixed
slots. Resource failures stop further computation but retain all six slot IDs,
marking incomplete slots explicitly. No replacement or automatic retry occurs.
Aggregate positive interpretation requires every frozen gate: qualified six
slots, correct real/complex classes, pair/angle errors, both25%relative and
.001absolute mean gains over mean(per-slot minimum commuting error), and
positive floating attribution for the **four complex/crossing slots only**.
The two real controls cannot require the helper's complex-only attribution.
`analysis_complete` alone is insufficient. Reports distinguish qualification
or loose-bound abstentions from empirical engineering misses; neither rules
out every curvature mechanism. No training/inference benefit follows.

## Resource-only profile and exact reuse

Explicit `--mode profile` acquires only the first S1R panel phase (stable-real,
row17, phase0), then times fixed synthetic72D factors with a=b=.5,
alpha=beta=1.2, u=e1, v=.6e1+.8e2. The synthetic helper uses2eigh/1eig/1SVD;
these are additional profile-only engineering operations, not extra network
references. Projection is twice the remaining23 fused analytic costs plus
six reference decomposition costs (2 times the slower observed eigh, one eig,
one SVD), then5seconds overhead. Accept timing only if projection<=60seconds.
Retain the first return and timings even when this gate fails. No fit outcome
controls eligibility. The panel requires an externally pinned immutable profile
manifest plus identical runtime source bytes, Python/NumPy/SciPy, CPU host,
platform, model/data/rates/state. Keep source/card files unchanged between the
two runs; append-only execution cards in research-log are excluded from the
runtime source inventory. No target profile was run during preparation.

CPU/BLAS1 and nice>=10; profile internal5s/external10+5s, panel internal60s/
external75+5s. The owned-group supervisor separately bounds inherited pipes at
15/80s, monitors group RSS against2GiB (polling is not a kernel hard limit),
and touches no foreign process. Science Store limit16MiB, with12MiB ordinary
allowance plus4MiB finalization reserve; launcher allowance1MiB/log128KiB.
Output directories are exclusive. Retain source/argv/resources/PIDs/exit,
every available partial and full manifest, verify inventories, then freeze
files0444/directories0555. No paid/GPU/holdout work.

## Commands, outputs and preparation validation

Run from the reviewed, clean execution checkout. Set `CURVATURE_REVIEW_SHA`
to root's actual full source-review commit. `--source-commit` binds the current
execution HEAD separately; cherry-picking unchanged reviewed sources is allowed.

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_curvature_pilot.py --source-commit "$(git rev-parse HEAD)" --review-commit "$CURVATURE_REVIEW_SHA" --mode profile --out /data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-profile --launcher /data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-profile-launcher
```

Only after inspecting the retained profile timing gate, set
`CURVATURE_PROFILE_SHA` to its exact manifest SHA256 and run:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_curvature_pilot.py --source-commit "$(git rev-parse HEAD)" --review-commit "$CURVATURE_REVIEW_SHA" --mode run --profile /data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-profile --profile-sha256 "$CURVATURE_PROFILE_SHA" --out /data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-panel --launcher /data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-panel-launcher
```

Synthetic preparation command: CPU/BLAS1, nice10, timeout30+5s, venv pytest
`-q -s -p no:cacheprovider tests/test_curvature_pilot.py tests/test_curvature_mechanism.py`.
The suite includes nonunit-scale biased analytic/HVP parity, native intermediate
composition, exact/context cache controls, nonfinite/deadline return retention,
six-slot failure continuation, real/complex attribution and both mean-gain
thresholds, profile rejection retention, launcher bindings, and a full storage
fixture. Its three72D plus three13D supplied-array fixture executes the complete
24/12/6/6 operation inventory. Observed final-check ordinary storage10,929,032B; final
10,976,599B/286files, below both limits. These synthetic payloads test storage,
not orbit realism. Final focused test result and exact file hashes are in the
source-only receipt. No empirical conclusion is available yet.
