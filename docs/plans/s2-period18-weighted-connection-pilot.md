# 10 October 2026 UTC — one wider weighted quadratic chart pair

The saved-only diagnostic at b41b465e separates the last quadratic chart's
forcing norm1.5793e-10 from propagated radius2.4629e-8. One exact positive
weight vector reduces its saved contraction from0.413205 to0.107849, with
minimum weight about0.1384 and normalized forcing below0.002964. This supports
a single disclosed exploratory next setting; it does not prove a larger domain.

## Fixed scope and settings

Same original 13D model, consumed data, primitive18 center and actual hidden
rate as previous certificates. Fixed output1848862009810323/2251799813685248,
center1254985102288567/17592186044416. Source reconstructs the complete original
C1 certificate/input inventory and exact native-phase anchor. No log-ray
reinterpretation, new data/trajectory or phase permutation.

Two quadratic-predictor charts, halfwidth2^-6, overall radius2^-24,256 bits,
two proposal Newton steps. Apply the SAME first-chart weight vector to both
charts. The intended connected width, if all gates pass, is1/16. Second chart
runs only after first root/primitive/anchor gates pass, with an exact endpoint
root inside its234-coordinate uniqueness domain. Attraction is not evaluated.

Weights come only from
`/data2/neural-shrimp-data/explore/certificates/s2-period18-quadratic-weights-2026-10-10/attempt-001/run/chart-0.json`,
SHA256d2ea032791154ca47d6c57a58af8c6616ef224a10ca49bf3904adb4d47210e5a;
full worker manifestcbc4c267601135a732268bccf99c6ce91163f7a01171cf0f8c3b5fc1bdd05afa,
executionb41b465e3aa10ebdf2b7c2cdee4de9fbf43a3f00. Require all234 weights to be
finite, strictly positive exact dyadics and<=1. Recheck source/inventory/status.
No new eigenvalue/weight search is performed by this target.

## Mathematical change

Historical quadratic and C1 kernels stay unchanged. New
`rigorous_weighted_connection.py` retains the exact curved-path Taylor
remainder but uses correction radii r_i=r*w_i. It acquires the TRUE new-domain
Jacobian on q(D)+E at the larger parameter interval. All inclusion and anchor
tests use r_i, and contraction requires

`max_i sum_j |(I-C*A)_ij|*w_j/w_i < 1`.

Every division/max is outward-enclosed; numerical weight origins do not decide
any gate. The fixed positive weighted infinity norm is equivalent to an
ordinary norm, so strict inclusion/contraction proves the same uniform root
theorem. Full-phase divisor exclusions and exact containment joins remain
mandatory. No old smaller-domain Jacobian is extrapolated to a larger domain.

The preparation heuristic is forcing×64≈0.190 and contraction×4≈0.431 when
the parameter halfwidth grows fourfold. These estimates merely choose one
trial; no proof gate uses them. Retain actual forcing, propagated radius,
weighted contraction, coordinate inclusions and every failed attempt. If a
bound fails, diagnose it before adapting, not an automatic grid or retry.

## Profile, execution and controls

New runner `tools/certify_s2_period18_weighted_connection.py`. First one synthetic
profile with the same zero-label24-sample ridge1 network, zero fixed point
repeated18 times, and the fixed saved weights. Synthetic rates are1/4 rather
than1/128 so the larger parameter interval remains positive and the profile
covers both weighted-root paths. This is an engineering change only; target
rates/data never change. No target objective is evaluated in the profile.

Complete two-chart counts:144 wrapped C1/288 unit Hessian calls and108 scalar
path jets,300tick cap. Profile20s internal25+5external; target120s internal
135+5external; CPU/BLAS1,nice10,2GiB RSS at checkpoints,128MiB worker with120MiB
ordinary allowance plus1MiB launcher. Preserve all sources, coefficient defects,
matrices, raw/decompressed hashes, operation counters and incomplete outputs.
Profile source/library/thread/settings must match; full paths and both roots
pass. Require measured4*time+2<100s and max(3*profilebytes,94,025,634)<120MiB.
Check launcher exit/PIDs/immutable bits and fresh shared-host memory before
target. Coordinate with the independent reviewer to stagger larger jobs.

Source/tests/card freeze and root review precede the profile. One exclusive
profile and one gated target under
`/data2/neural-shrimp-data/explore/certificates/s2-period18-weighted-connection-2026-10-10/`,
named `attempt-001-profile` and `attempt-001-target`. Use runner `--mode profile
--out <profile>`; after committing measured receipt use `--mode target --out
<target> --profile <profile/run> --profile-manifest <exact SHA>`. Interpreter
`/data2/neural-shrimp/.venv/bin/python`, `PYTHONPATH=src`, CPU/BLAS1,nice10.

Controls include exact weighted root/join, coordinate-specific anchor failure,
nonlinear primitive2 and independently recomputed weighted norm, rejected
nonpositive/nondyadic/malformed weights, and pinned input inventory. Passing
proves a larger primitive branch only, pending independent review; no attraction,
fold/flip/cusp connection, fifth/new orbit or whole shrimp claim.
