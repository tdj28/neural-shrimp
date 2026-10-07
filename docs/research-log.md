# Research log

Append-only record of runs, decisions and amendments. See [the adaptive research protocol](adaptive-research-protocol.md).

## 2026-10-07 05:54 UTC: Retrospective summary of 6 October 2026 (written before EXP-028 results)

Assessment recommended option C; user chose it. Program C plan, gdmaps engine, EXP-028 design. Adversarial pre-freeze review found 2 blockers (cascade-band gate, ridge-free S1 drift) and 15 further issues; all fixed before freeze (docs/EXP-028-design-audit.md). Artificial-fixture S2 test (seed 99991) found two audit-confirmed embedded period-4 windows. EXP-028 freeze 91db3654 pushed at 5725820c; run launched 18:05 local (attempt-001, 14 workers). EXP-029 autonomous survey frozen at 6a162ce2, waiting for CPU. CUDA float64 gradient-descent backend qualified on fixtures (branch claude/gdmap-cuda-backend); throughput benchmark deferred while another agent uses the GPU. Driven SoA engine and ray continuation added. User pointed to Jones (2012, arXiv:1201.4343): hub shape from unimodal conjugacy, shrimp centers doubly superstable at a second critical point; folded into the A2/A3 design as testable predictions (MSS ordering, symbolic words, doubly superstable centers, critical curvature 1/eta). User directive: adopt an adaptive research protocol in place of mandatory pre-run machine plans.

_commit `7729eb822f6c` (dirty worktree)_

## 2026-10-07 05:54 UTC: Open adaptation candidate: EXP-028 stage-3 timing

Zoom stage is slower than estimated (shared load near 20; zoom cells never diverge early). Projected completion about 01:15-01:30 local versus the 02:05 wall ceiling. If the ceiling stops stage 3, all coarse and zoom maps are already saved; the planned adaptation is a successor that runs certification, ring confirmation and chaos reruns from the saved maps, labeled as such. No outcomes have been inspected.

_commit `7729eb822f6c` (dirty worktree)_

## 2026-10-07 06:07 UTC: Novelty check received (before EXP-028 outcomes)

Verified shrimps with full-state cycles and fold/flip continuation in per-layer learning-rate planes: novel (C1). Do not claim first period-colored training map (Sveleba et al. 2026) or first gradient-descent continuation (Hofmann 2026, 1-D reductions). Fractal-boundary mechanism (C2): novel but needs crisis analysis and a window-masking test. Reproducible islands (C3): partly anticipated by Ly and Gong 2025; needs uncertainty-exponent quantification. Orbit averaging (C4): largely anticipated; dropped as headline. Recorded in docs/novelty-check-2026-10-06.md; A3 design amended.

_commit `6b62f4eafffc` (dirty worktree)_

## 2026-10-07 06:09 UTC: amendment to `docs/EXP-028-protocol.md`

- **Change:** If attempt-001 stops at its 8-hour wall ceiling, run attempt-002 with tools/continue_exp028.py: verify attempt-001's receipt, copy its coarse maps, zoom specifications and hash-verified zoom chunks unchanged, recompute only missing zoom chunks with the frozen chunk worker, then run stage 3 exactly as frozen. Wall ceiling for the continuation: the plan's 8 hours, counted afresh.
- **Why:** Zoom chunks take about 31 minutes each because zoom cells never diverge early; the projected completion (about 02:20 local) exceeds the 02:05 ceiling. Recomputation is deterministic: on a synthetic interruption the tool reproduced dropped chunks byte-for-byte and stage-3 outputs exactly.
- **Outcomes seen:** None. No EXP-028 map, label, certificate or count has been opened; only chunk file counts and timestamps were checked for progress.
- **Claim impact:** None if invoked; EXP-028 remains prospectively frozen. Results will be labeled as attempt-001 maps completed by the attempt-002 continuation.

_commit `74be4db8e85a` (dirty worktree)_

## 2026-10-07 09:06 UTC: EXP-028 attempt-001 stopped at the wall ceiling; continuation invoked

attempt-001 ended at 09:05 UTC with status failed (wall ceiling) during stage 3b ring-confirmation reruns. Stage 1 and stage 2 completed (all 256 zoom chunks and four zoom maps saved at 09:02 UTC) and certificates.json was written. Invoking the pre-logged conditional amendment: attempt-002 via tools/continue_exp028.py. Outcomes seen: none (only file names, counts and modification times).

_commit `86b4014e5a85`_

## 2026-10-07 09:09 UTC: GPU throughput benchmark (GPU free)

CUDA float64 backend at 64af0462, build sha256 062c5084..., artificial fixtures: S1-type 0.36 us/cell-step plain, 0.63 with FTLE tangent (65,536 cells); width-16 3.3 and 5.9 us. One CPU core under shared load: 21 and 38 us (S1-type), 149 and 327 us (width 16). Receipt: neural-shrimp-data/audits/gdmap-cuda-qualification/benchmark-001/benchmark.json. Full 20,000-update qualification (attempt-003) started on fixtures.

_commit `726dc84cd522`_

## 2026-10-07 09:15 UTC: GPU backend full-horizon qualification (artificial fixtures)

attempt-003, 20,000 updates, plan screening settings, build 062c5084: S2-like 576 cells 99.8% status agreement, 170/171 periodic orbits equal status and period (25 at shifted phase), divergent 139/139, chaotic 14/14, unresolved 252/252. S1-like 97.9% agreement: divergent 150/151 (+1 GPU-only), chaotic 208/215, drift 206/210. Width-16 (600 updates) 100%. Repeat, rechunked and host-screen runs bit-identical. Decision: GPU approved for exploratory maps (A2 local maps, high-resolution figures); labels of chaotic or drifting cells can differ in a few percent of cells, so confirmatory statements about chaotic labels use CPU or both. Receipt: neural-shrimp-data/audits/gdmap-cuda-qualification/attempt-003/qualification.json.

_commit `726dc84cd522` (dirty worktree)_

## 2026-10-07 09:23 UTC: EXP-028 frozen analysis run (outcomes now seen)

attempt-002 continuation complete (154 certificates, 2,801 ring reruns, 160 random chaos reruns; all 266 map files reused). Frozen analysis: producer gate positive with 62 qualifying embedded components across S1R-28001/28002, S2-28011/28012, coarse and zoom maps; periods from 4 to 58 including odd periods 5, 11, 13, 17, 29. S1 (no ridge, published setting) has zero periodic cells: divergent, chaotic or drift. The published binary score labels chaotic and period-2 training as convergent. Final gate awaits the independent audit (running). Summary: docs/EXP-028-summary.json; figures: neural-shrimp-data/figures/EXP-028/attempt-002.

_commit `34e6293615cf` (dirty worktree)_

## 2026-10-07 09:24 UTC: EXP-029 launched

Frozen plan e2f8a007 (commit 6a162ce2), run from HEAD fc219a7a after preflight (load 6.5, 49 GiB free). 14 CPU workers; raw ../neural-shrimp-data/raw/EXP-029/attempt-001.

_commit `fc219a7a4619`_

## 2026-10-07 11:18 UTC: run (exploratory)

- **Question:** Do the S2-28011 zoom-window stripes resolve into shrimp-shaped domains with period-adding order at 4x finer resolution?
- **Setup:** GPU backend (build 062c5084), EXP-028 settings, S2 seed 28011, log10 hidden 1.6946-2.0593, log10 output -0.4475 to -0.0828, 1024x1024
- **Output:** `../neural-shrimp-data/explore/s2-28011-zoom-1024`
- **Would surprise us:** Stripes that do not separate into distinct isoperiodic domains, or periods that do not increase systematically across the stripes
- **Command:** `.venv/bin/python tools/explore_gpu_map.py --name s2-28011-zoom-1024 --system S2 --seed 28011 --hidden 1.6945644359381693 2.0592703182911105 --output -0.4474707242107141 -0.08276484185777289 --resolution 1024 --library ../neural-shrimp-data/builds/gdmap-cuda-64af0462/gdmap_cuda.so`

_commit `f2cc6a9ff968` (dirty worktree)_

## 2026-10-07 11:20 UTC: EXP-028 independent audit: gate confirmed

Audit validation-001: receipt (276 files), plan hash, inputs, zoom rule and axes reproduce; all 154 certificates recheck (closure, finite-difference spectral radius within 1e-6, stability, primitivity, rings, seeds, rates). Audit-qualifying 63; producer 62; confirmed by both 61 (one producer-only and two audit-only components sit at the 0.5 ring-fraction threshold). Independent single-cell re-simulation: S1R and S2 regular cells 71/72 agree; S1 (no ridge) fixed-point labels are fragile (8/13 and 12/15). Results: docs/EXP-028-results.md.

_commit `f2cc6a9ff968` (dirty worktree)_

## 2026-10-07 11:21 UTC: run (exploratory)

- **Question:** Are confirmed EXP-028 windows bounded by both fold and flip boundaries, as shrimps are?
- **Setup:** 32 rays per window in log-rate coordinates scaled to 3x component extent; continuation with exact-Jacobian Newton; pilot target S2-28011:189 (period 4)
- **Output:** `../neural-shrimp-data/explore/a2-rays-pilot`
- **Would surprise us:** A window bounded only by flips or only by folds, or rays that never lose stability within 3x the component extent
- **Command:** `.venv/bin/python tools/explore_a2_rays.py --targets S2-28011:189 --workers 8 --out ../neural-shrimp-data/explore/a2-rays-pilot`

_commit `2b62ff902b1c` (dirty worktree)_

- run started 2026-10-07 11:21 UTC finished 11:21 UTC with exit status 0

## 2026-10-07 11:22 UTC: run (exploratory)

- **Question:** Same pilot with bisection tolerance 1e-9: do the unresolved rays resolve as folds or flips?
- **Setup:** S2-28011:189 (period 4), 32 rays, bisection 1e-9
- **Output:** `../neural-shrimp-data/explore/a2-rays-pilot-2`
- **Would surprise us:** Unresolved rays persisting with multipliers well inside the unit circle
- **Command:** `.venv/bin/python tools/explore_a2_rays.py --targets S2-28011:189 --workers 8 --out ../neural-shrimp-data/explore/a2-rays-pilot-2`

_commit `a6c98e4c27c0` (dirty worktree)_

- run started 2026-10-07 11:22 UTC finished 11:22 UTC with exit status 0

## 2026-10-07 11:22 UTC: run (exploratory)

- **Question:** Across six confirmed windows (S2 and S1R, periods 4-18), what bifurcation types bound them?
- **Setup:** 64 rays each, bisection 1e-9; targets S2-28011:189 (p4), S2-28011:168 (p5), S2-28011:132 (p11), S2-28011-zoom:587 (p18), S1R-28002:113 (p12), S1R-28001-zoom:13 (p4)
- **Output:** `../neural-shrimp-data/explore/a2-rays-six`
- **Would surprise us:** Flip boundaries absent in every window
- **Command:** `.venv/bin/python tools/explore_a2_rays.py --rays 64 --workers 12 --targets S2-28011:189 S2-28011:168 S2-28011:132 S2-28011-zoom:587 S1R-28002:113 S1R-28001-zoom:13 --out ../neural-shrimp-data/explore/a2-rays-six`

_commit `a6c98e4c27c0` (dirty worktree)_

- run started 2026-10-07 11:22 UTC finished 11:24 UTC with exit status 0

- run started 2026-10-07 11:18 UTC finished 11:39 UTC with exit status 0

## 2026-10-07 11:52 UTC: EXP-029 complete and audited

757,760 cell runs; global map: periodic 62%, chaotic 38%, no equilibria; spiral organization visible; start agreement 98%. Audit: 92/96 labels, 32/32 periodic labels and periods, transverse multipliers within 1e-4 for 30/32 (two at 1.9e-4 and 3.3e-4). Results: docs/EXP-029-results.md.

_commit `a6c98e4c27c0` (dirty worktree)_

## 2026-10-07 11:53 UTC: run (exploratory)

- **Question:** What does the period-18 window S2-28011-zoom:587 (fold and flip boundaries) look like at 512x512, and do its typed ray boundaries trace its edges?
- **Setup:** GPU, EXP-028 settings, S2 seed 28011, log10 hidden 1.66882-2.03782, log10 output -0.11996 to -0.05130 (center +/- ray scale)
- **Output:** `../neural-shrimp-data/explore/s2-28011-w587-512`
- **Would surprise us:** Typed boundaries that do not coincide with the period-18 domain edge; no period-36 cells beyond flip boundaries
- **Command:** `.venv/bin/python tools/explore_gpu_map.py --name s2-28011-w587-512 --system S2 --seed 28011 --hidden 1.66882 2.03782 --output -0.11996 -0.05130 --resolution 512 --library ../neural-shrimp-data/builds/gdmap-cuda-64af0462/gdmap_cuda.so`

_commit `c148d4ab3514`_

- run started 2026-10-07 11:53 UTC finished 11:58 UTC with exit status 0

## 2026-10-07 11:59 UTC: A2 exploratory: window S2-28011-zoom:587 is shrimp-consistent

64 rays: fold 46, flip 15, Neimark-Sacker 3. 512x512 GPU local map: ray boundaries match the period-18 domain edge within 2 cells on 57/64 rays; the first cell beyond 8/15 flip boundaries has period 36 (none beyond folds, which open onto chaos, unresolved or drift cells). Morphology: fold-bounded body, flip edge into a period-36 region, leg and tail extensions. Passes all draft A2 criteria. Across six windows, boundary types vary: fold-dominated (p4), flip and Neimark-Sacker only (p5), and mixtures; Neimark-Sacker boundaries are common, unlike two-dimensional map shrimps. Figure: neural-shrimp-data/explore/s2-28011-w587-512/overlay.png. Next: systematic A2 survey over all 61 confirmed windows.

_commit `2d1024692cda` (dirty worktree)_

## 2026-10-07 11:59 UTC: A2 systematic survey launched (exploratory)

All 61 audit-confirmed EXP-028 windows (33 S1R, 28 S2): 64 rays each on CPU (12 workers) and 256x256 GPU local maps spanning center +/- 3x component extent; analysis applies the draft A2 criteria scaled to 64 rays. Would surprise us: shrimp-consistent organization in none or nearly all windows; flip boundaries absent in most windows.

_commit `92485ec0f840`_

## 2026-10-07 12:02 UTC: A3 exploratory: naive critical-curvature symbols fail

Words for 148 certified periodic orbits (EXP-028): at every orbit point J = I - P H has 2 to 19 negative eigenvalues and the smallest lies near -1 or below (minimum |lambda_min| 0.88 to 2.0), never near the critical value 0. Labeling R by lambda_min < 0 therefore gives all-R words (only period-2 words are trivially admissible). Interpretation: at these rates training reverses orientation along many directions, so the single-stiffest-direction unimodal reduction is not valid as defined. The mechanism hypothesis is not falsified; it needs an empirical reduction (principal-axis return maps of nearby chaotic trajectories, likely of the two-step map F^2 given the mostly even periods). Output: neural-shrimp-data/explore/a3-words/words.json.

_commit `d164bd02672f` (dirty worktree)_

## 2026-10-07 12:02 UTC: run (exploratory)

- **Question:** Is chaotic training just outside the period-18 window's fold effectively one-dimensional with a unimodal return map?
- **Setup:** S2-28011 at log10 rates (1.83010, -0.08563), 15% beyond the lower fold along the downward ray; 40,000 updates, last 20,000 projected on the first principal axis; return maps at lags 1, 2, 4
- **Output:** `../neural-shrimp-data/explore/a3-return-maps/w587-below-fold`
- **Would surprise us:** A thin single-humped curve (strong unimodal reduction) or a structureless cloud (no low-dimensional reduction)
- **Command:** `.venv/bin/python tools/explore_a3_return_map.py --system S2 --seed 28011 --log-hidden 1.83009726253427 --log-output -0.08562528015073712 --out ../neural-shrimp-data/explore/a3-return-maps/w587-below-fold`

_commit `4fbd4d3ebdf9`_

- run started 2026-10-07 12:02 UTC finished 12:02 UTC with exit status 0

## 2026-10-07 12:04 UTC: A3 exploratory: phase locking on invariant circles of the two-step map

Just outside the period-18 window S2-28011-zoom:587 (15% beyond its lower fold), training lies on thin closed curves in principal-axis return maps (first two axes 98% of variance), with the two-step map rotating at 0.21936 turns per step (FTLE 0.0058, weakly chaotic). The certified period-18 orbit makes exactly 2 net turns over its 9 two-step points: rotation number 2/9 = 0.2222. Across 115 certified orbits with at least 5 points (two-step points for even periods), 65 are well-ordered (each step jumps exactly m positions in angular order), versus 0.7% for random angles; 44 of 80 embedded windows. Working hypothesis: edge-of-stability period-2 orbit -> Neimark-Sacker -> invariant circle -> Arnold tongues (fold-bounded periodic windows of period 2q, rotation m/q) -> torus breakdown with period doubling inside tongues. Odd-period S2 windows are mostly not well-ordered and may follow the fold-flip (Gallas/Jones) route. Caveats: two-dimensional projection; integer winding is automatic and was not counted as evidence. Next: rotation-number staircase along slices (devil's staircase, Farey order) and novelty check for quasi-periodic gradient-descent dynamics.

_commit `e24027696cd0` (dirty worktree)_

## 2026-10-07 12:05 UTC: run (exploratory)

- **Question:** Does the rotation number of the two-step map form a devil's staircase with Farey-ordered plateaus along a slice through the 2/9 window?
- **Setup:** S2-28011, output rate fixed at log10 -0.0856 (window 587 center column), hidden rate log10 1.70-2.05, 2,000 cells, 40,000 updates, last 4,000 states
- **Output:** `../neural-shrimp-data/explore/a3-staircase/s2-28011-out-0.0856`
- **Would surprise us:** No plateaus at rationals; plateau at 2/9 not between 1/5 and 1/4; rotation numbers non-monotone across periodic windows
- **Command:** `.venv/bin/python tools/explore_a3_staircase.py --system S2 --seed 28011 --vary hidden --fixed -0.08562528015073712 --range 1.70 2.05 --out ../neural-shrimp-data/explore/a3-staircase/s2-28011-out-0.0856`

_commit `dee72e05fa1f`_

- run started 2026-10-07 12:05 UTC finished 12:12 UTC with exit status 0

## 2026-10-07 12:13 UTC: A3 exploratory: devil's staircase through the 2/9 window

S2-28011, output rate fixed at the window column, 2,000 hidden-rate cells: period-2 region (rotation 0) up to log10 1.737; quasi-periodic cells (EXP-028 label unresolved, near-zero exponent) with rotation rising from 0.217; a wide plateau at exactly 2/9 (period 18, log10 1.833-1.879); weakly chaotic cells with rotation rising 0.225-0.24; a plateau at 1/4 (period 8, log10 1.909-1.918); then chaos. Monotone staircase with 2/9 between 1/5 and 1/4 (Farey mediant). Interpretation: quasi-periodic route to chaos in large-rate gradient descent (period-2 -> Neimark-Sacker -> invariant circle -> Arnold tongues -> torus breakdown). Next: other slices, S1R, other seeds.

_commit `dee72e05fa1f` (dirty worktree)_

## 2026-10-07 12:13 UTC: run (exploratory)

- **Question:** Does the 72-parameter S1R network show the same rotation-number staircase through its largest locked window (1/8, period 16)?
- **Setup:** S1R-28002, output rate fixed at log10 0.93852, hidden rate log10 1.25-1.45, 2,000 cells, 40,000 updates
- **Output:** `../neural-shrimp-data/explore/a3-staircase/s1r-28002-out-0.9385`
- **Would surprise us:** No staircase; rotation numbers not monotone; no plateau at 1/8
- **Command:** `.venv/bin/python tools/explore_a3_staircase.py --system S1R --seed 28002 --vary hidden --fixed 0.93852 --range 1.25 1.45 --out ../neural-shrimp-data/explore/a3-staircase/s1r-28002-out-0.9385`

_commit `7057b490a771`_

## 2026-10-07 12:29 UTC: Novelty check 2: quasi-periodic training and tongues are novel for optimizers

Quasi-periodic training with verified rotation numbers, Arnold tongues and devil's staircases in gradient-descent hyperparameter planes, and torus-breakdown routes are not reported for optimizer-on-loss maps; tongues and Neimark-Sacker are documented in learning-in-games maps. Hofmann (2026) partly anticipates Neimark-Sacker of the edge-of-stability 2-cycle (complex multipliers, no rotation numbers). Recorded in docs/novelty-check-2026-10-06.md.

_commit `7057b490a771` (dirty worktree)_

- run started 2026-10-07 12:13 UTC finished 12:56 UTC with exit status 0

## 2026-10-07 12:57 UTC: A3 exploratory: S1R staircase is chaos-dominated with locked plateaus and coexisting branches

S1R-28002, output fixed at log10 0.9385, hidden 1.25-1.45: quasi-periodic branches with rotation rising 0.11-0.14, locked plateaus at 1/8 and near 2/15, many chaotic cells with ill-defined rotation, and two interleaved branches at the same rates (likely coexisting attractors from the same initialization). Same ingredients as S2, less clean.

_commit `22dab26f7bd9` (dirty worktree)_
