# Clark finite-window response pilot — 9 October 2026

Exploratory method transfer after directly reading Clark, arXiv:2610.12426v1,
§§2.1–2.3 and Appendix D. This tests the generic finite-dimensional response
construction, not the random-RNN cavity closure or any training utility claim.
No fresh model, derivative, trajectory, proof, GPU, paid, or protected-data work.

## Fixed inputs and question

Use exactly two already-consumed 13-dimensional C1 certificates: p10 row35
(`certificates/c1-row35-receipt.json`, SHA256
`07f51a37e8c2304934f0ef60e6e985117cf0e76bf3cf994457374e87bdf6e1f0`)
and p18 row111 (`certificates/s2-period18-receipt.json`, SHA256
`f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277`).
Require every receipt inventory hash, certificate/input join, dimension, period,
and passing certificate status. Convert the exact saved Jacobian ball midpoints
to binary64; retain the arrays/byte hashes, ball-radius diagnostic and comparison
with the saved monodromy midpoint. These define floating periodic linear
surrogates; the new calculations do not inherit interval certification.

Does the finite-window regularized response recover known spectral counts, at
what cost and with what boundary/regularization failures? Preserve full signed
and complex multipliers, angles, eigen residuals, per-phase smallest singular
values, and repeated-QR references (128 and512 periods). A normalized CDF cannot
replace isolated critical modes, primitive-period tests, normal forms or geometry.
Direct-product floating eigenvalues and finite QR can themselves be inaccurate;
report disagreements and singular/neutral thresholds without integer rounding.

## Method and fixed grid

Time is one optimizer update. Let A_t=exp(-s)M_t and
(Kv)_t=v_(t+1)-A_t v_t. For an impulse I at the central phase-zero equation,
solve [eta I,K;-K^T,eta I][u;v]=[I;0], equivalent to minimizing
||Kv-I||²+eta²||v||². Read trace(v_(k+1))/13, using all13 coordinate impulses
(no stochastic trace error). Report true augmented residual and unrounded count.
The true-relative-residual screen is1e-8; it qualifies the numerical solve only,
not finite-window convergence or the infinite-time identity.

Use periods {4,16,64}, shifts {−.5,−.05,−.005,0}, eta {1e−2,1e−4,1e−6},
and boundaries {free,periodic,causal}:216 solves, plus two matched direct solves
for the matrix-free comparison. Free has T+1 unconstrained state blocks and
T equations; periodic has T states with v_T=v_0; causal fixes v_0=0.
Causal is an intentional wrong-limit control: at fixed finite T and eta→0,
the one-step response tends to I even for unstable maps. Periodic resonances
and free endpoint leakage also create finite-window bias. No finite-window
agreement is identified with the infinite-time/zero-regularizer theorem.
Exact neutral or numerically singular phases fail the identity-assumption screen;
near-neutral finite-window errors remain visible. Threshold screen1e-10 and phase
singular-value ratio1e-12 are numerical diagnostics, not proof tolerances.

One additional callback K/K^T LSMR probe per slot uses16 periods, s=−.05,
eta=.01, a deterministic e0 impulse, maxiter300, atol=btol=1e−10 and conlim1e12.
Record iteration/operator/JVP/VJP counts, true normal residual and readout error
against the matched sparse direct solve. It is not a trace estimate. For future
plain-GD application, M=I−D H while M^T=I−H D; tests enforce this rate ordering.
Callbacks only multiply saved small matrices here. No HVP speedup, preconditioner,
large-model scalability, or random-matrix closure is assumed.

## Controls, cost gate and durability

Independent synthetic dense-SVD ridge oracle; nonnormal oblique stable projector;
noncommuting phases; negative and complex multiplier information; causal wrong
limit; window/eta bias; exact-neutral/singular/empty/nonfinite abstention; callback
transpose and direct-column parity; profile tampering/invalid resource metadata;
output finalization reserve. Source tests precede any target arithmetic.

First one exclusive synthetic profile: deterministic13D/18-phase upper-triangular
maps, largest1152-step window, eta1e−6, s−.05, all3 boundaries, plus the fixed
matrix-free probe. Profile internal15s/external20+5s. Conservative panel projection
is218 times the largest solve time plus2 probe times plus5 seconds. Require
projection<45s, profile elapsed<15s, RSS<400MiB, same exact source files, library
versions, thread variables, and niceness; finite nonnegative metadata throughout.
No empirical inputs opened by the profile. If this fails, preserve it and disclose
any amended next cost unit; do not silently shorten the scientific panel.

Panel internal60s/external75+5s, CPU/BLAS1, nice≥10. Worker hard address-space
cap480MiB plus monitored480MiB group RSS, keeping the small launcher within the
root's512MiB allocation. Address space and monitored RSS are distinct limits.
Worker raw≤16MiB (1MiB finalization reserve), launcher/log≤1MiB (log≤128KiB),
aggregate≤17MiB. No foreign process signals. Supervisor covers inherited pipes
and kills only its new process group; record source/argv/resources/PIDs/exit and
all inventories. Save each attempted row config and returned response block before
the next budget check; preserve scientific failures and continue fixed slots when
resources allow. A killed sparse/iterative library call may leave its start record
without a return; do not claim complete attempted-iterate retention. All finished
raw is made read-only. Code plus this card committed before profile or panel.

Raw roots (exclusive):
`/data2/neural-shrimp-data/explore/clark-response/attempt-001-profile` and
`.../attempt-001-panel`. Source inventory includes this card but excludes append-only
research-log updates, so profile and panel bind equal executable/card bytes.

Command prefix for both stages:
`env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_response_spectrum.py`.
Append `--mode profile --out <profile-root>`; then after root source review and a
passing profile, `--mode panel --out <panel-root> --profile <profile-root>/run
--profile-manifest <exact profile run manifest SHA256>`.

Report continuous count errors by shift/window/eta/boundary, reference disagreements,
solver failures and matrix-free cost independently. The next decision is whether a
bounded callback/HVP cost test is warranted, not whether this establishes a training
or shrimp mechanism. Independent result review remains pending.
