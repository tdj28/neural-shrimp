# S1R saved-outcome audit card

8 October 2026 UTC. Outcome-aware review requested after producer completed the
fixed pilot. This is an audit of existing numerical evidence, not confirmation
on fresh data. Producer source db10bd1e1b25b0e24b1f0f8f97cb84035742c776;
source review cdf2ad51034db515fab99846fbd88e61a4528a46; outcome receipt
4e50f004ad6f5db80be0dea01fdf404eec71bd1b. Bind raw manifest SHA256
45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b.

Before numerical replay, commit/push this card and tools/audit_s1r_branch_outcome.py
with its synthetic controls. Audit full raw inventory, original source and
input bindings, auxiliary launcher/checker/preparation-failure hashes; rerun the
unchanged producer saved-state checker. Disable its Newton corrector entry
point: the checker may solve for the remaining undamped correction at an
unchanged state, but it must not apply a correction or simulate a trajectory.
Independently decode retained arrays and reconstruct all hidden/output phase
costs, direct/half/reverse witnesses and inherited/secant predictor seeds.

Use a separately written prediction-Jacobian plus residual-curvature Hessian
for six declared representatives: original anchor node0000 and the five main
nodes0004,0005,0009,0013,0017. Check both full-state phase transitions and
closures with the original1e-9+1e-8 componentwise scale and primitive1e-6.
Reconstruct both72x72 monodromies; matrix agreement tolerance2e-11, complete
spectrum assignment tolerance1e-8, stored eigenpair residual limit1e-10 and
original stability margin1e-7. These are numerical cross-check tolerances,
not outward error bounds or an interval continuation proof. The untouched
producer checker also audits all21 stored pairs with its original gates.

One exclusive audit attempt under CPU1/BLAS1 nice10, internal45s,
external55s SIGTERM+5s kill grace, <=60s total. No new target correction,
trajectory, seed, GPU, held-out data or paid call. Preserve any failure before
amending code/tolerances; no automatic retry. Suggested command:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src timeout --signal=TERM --kill-after=5s 55s nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/audit_s1r_branch_outcome.py --out /data2/neural-shrimp-data/audits/s1r-branch-outcome/attempt-001`.
A pass supports local consistency at the tested nodes and links only; the
historical start families, global branch uniqueness and distant resonance or
window association remain unresolved.
