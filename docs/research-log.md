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

## 2026-10-07 12:57 UTC: A3 exploratory: S1R locked periods equal twice the rotation denominator; Farey mediants

S1R-28002 slice: periodic cells on rotation plateaus have periods 2q (or doubled): 1/9 -> 18 (36), 2/17 -> 34, 1/8 -> 16 (64), 3/23 -> 46, 2/15 -> 30 (60), 2/19 -> 38, 1/10 -> 40. Plateaus follow Farey mediants: 2/17 = mediant(1/9, 1/8), 3/23 = mediant(1/8, 2/15), 2/15 = mediant(1/8, 1/7). The phase-locking mechanism (Arnold tongues of the two-step map's invariant circle, with period doubling inside tongues) holds in both S2 and S1R.

_commit `ee873a1305fd`_

## 2026-10-07 12:57 UTC: A2 exploratory: boundary types across all 61 confirmed windows

3,904 rays (64 per window): fold 1,316, Neimark-Sacker 1,395, flip 1,012, still attracting at range edge 168, unresolved 13. 15/61 windows have at least 8 fold and 8 flip rays (shrimp-type counts); 41/61 have at least 8 Neimark-Sacker rays; 9 are fold-dominated; some are bounded entirely by Neimark-Sacker (S2-28012:46, 64/64). Consistent with tongue organization: fold edges, with flip and secondary Neimark-Sacker inside tongues. Draft A2 criteria await the GPU local maps (6/61 done).

_commit `e26c6f4f4108`_

## 2026-10-07 12:59 UTC: A3 exploratory: Neimark-Sacker of the edge-of-stability 2-cycle starts quasi-periodic training in both networks

Continuing the period-2 orbit along hidden-rate slices: S2-28011 (output log10 -0.0856) loses stability by Neimark-Sacker at log10 hidden 1.738439 with two-step multiplier 0.1987+0.9801i, predicting rotation 0.21816; the staircase's quasi-periodic band starts at 1.737 with rotation about 0.2175. S1R-28002 (output log10 0.9413) loses stability by Neimark-Sacker at 1.121671 with multiplier 0.6745+0.7383i, predicting rotation 0.1322, matching the 0.12-0.135 staircase branches. Bisection to 1e-10; |mu| = 1 - 1e-7.

_commit `23c41e3942da`_

## 2026-10-07 12:59 UTC: run (exploratory)

- **Question:** Where along the Neimark-Sacker curve of the period-2 orbit does the birth rotation number pass through 2/9, 1/4 and other rationals, and do confirmed windows emanate from those points?
- **Setup:** S2-28011, start log10 hidden 1.65 inside period 2, 40 output rates log10 -0.45 to -0.08, continuation to stability loss, bisection 1e-10
- **Output:** `../neural-shrimp-data/explore/a3-ns-curve/s2-28011`
- **Would surprise us:** Stability lost by flip or fold instead of Neimark-Sacker; birth rotation numbers not spanning the window rotation numbers
- **Command:** `.venv/bin/python tools/explore_a3_ns_curve.py --system S2 --seed 28011 --start 1.65 --output -0.45 -0.08 --out ../neural-shrimp-data/explore/a3-ns-curve/s2-28011`

_commit `0e3e07e1723b`_

- run started 2026-10-07 12:59 UTC finished 13:00 UTC with exit status 0

## 2026-10-07 13:00 UTC: A3 exploratory: Neimark-Sacker curve of the period-2 orbit traced (S2-28011)

Across 35 output-rate slices (log10 -0.45 to -0.08) where the start was period 2, the period-2 orbit always loses stability by Neimark-Sacker; the curve is U-shaped with minimum near log10 hidden 1.650, and the birth rotation number theta/2pi runs from 1/6 at the bottom to 0.240 (left branch) and 0.221 (right branch), passing 2/11, 1/5, 2/9 and 3/13. The curve bounds the period-2 region in the 1024x1024 map, with quasi-periodic cells and locked stripes beyond it. Tying individual stripes to predicted rational roots needs per-stripe rotation numbers (next). Figure: neural-shrimp-data/explore/a3-ns-curve/s2-28011/tongues.png.

_commit `0e3e07e1723b` (dirty worktree)_

## 2026-10-07 20:45 UTC: run (exploratory)

- **Question:** Which confirmed EXP-028 windows meet the draft A2 shrimp-consistency criteria (fold and flip each on at least 8 of 64 rays, at least half of flip rays opening onto period 2p, ray/map edge agreement on at least 75% of compared rays)?
- **Setup:** rays.json (64 rays per window) plus 61 local 256x256 maps from the CUDA float64 backend; analysis code tools/explore_a2_survey.py analyze
- **Output:** `../neural-shrimp-data/explore/a2-survey/survey.json`
- **Would surprise us:** Edge agreement below 75% for most windows (would mean ray continuation and maps disagree, a method problem)
- **Command:** `.venv/bin/python tools/explore_a2_survey.py analyze`

_commit `5416c8d4f191`_

- run started 2026-10-07 20:45 UTC finished 20:45 UTC with exit status 0

## 2026-10-07 20:48 UTC: run (exploratory)

- **Question:** Inside each confirmed window's ray-continued stability region, are map cells without the window's period long transients or coexisting attractors?
- **Setup:** All 61 windows; up to 120 sampled non-period-p cells inside 0.9x the interpolated ray boundary; rerun 80,000 updates from the shared initial parameters, and 20,000 from the certified orbit point; EXP-028 screening options
- **Output:** `../neural-shrimp-data/explore/a2-survey/speckle.json`
- **Would surprise us:** Orbit-start runs failing to return period p inside the ray-continued region (would contradict the ray continuation)
- **Command:** `.venv/bin/python tools/explore_a2_speckle.py --workers 10`

_commit `2508f1913658` (dirty worktree)_

## 2026-10-07 20:49 UTC: Reserved for confirmatory tests: seeds 29000-29999 in every system

To keep the eventual confirmatory test on fresh data, exploratory work from now on must not run any network, dataset or initialization drawn from seeds 29000-29999 (any system, including new real-data networks). Exploration uses the EXP-028/029 seeds (28001, 28002, 28011, 28012 and earlier) or seeds below 29000. Any accidental use is to be disclosed here and the affected seeds removed from the reserved pool.

_commit `2508f1913658` (dirty worktree)_

## 2026-10-07 20:49 UTC: A2 survey: draft shrimp criteria pass 2/61; map edges unreliable where windows coexist with other attractors

Draft criteria (fold and flip each on at least 8 of 64 rays, at least half of flip rays opening onto period 2p in the local map, ray/map edge agreement at least 75%) pass S2-28011-zoom:587 (period 18) and S2-28011-zoom:985 (period 16). Edge agreement fails for most S1R windows because their local maps (from the shared initial parameters) are speckled inside the ray-continued stability region. First diagnostic (S1R-28001-zoom:931, period 8): of 20 speckled cells, 18 return period 8 when started from the certified orbit point but only 1 after 4x longer runs from the shared start: coexisting attractors, not long transients. Fixed-start maps therefore show basin selection layered on the window's existence region, and map-based edge criteria need orbit-start maps. The flip-to-2p criterion fails often even in clean S2 windows (subcritical flips or 2p bands thinner than a map cell are both possible); not yet diagnosed. Boundary census over all windows: S2 windows are mostly Neimark-Sacker and flip bounded (tongue-type); S1R windows mostly fold bounded. Full speckle survey running.

_commit `2508f1913658` (dirty worktree)_

## 2026-10-07 21:04 UTC: run (exploratory)

- **Question:** A3b consequence pilot, S2-28011 slice: do reproducibility, loss and stopping-time sensitivity differ between ordinary period-2, quasi-periodic, locked and chaotic regimes, and does rotation/locking information predict them beyond FTLE, sharpness, rate and oscillation amplitude?
- **Setup:** S2-28011, log10 output -0.08562528015073712, 750 hidden-rate cells log10 1.55-2.05; base run plus 30 copies perturbed at initialization (1e-12..1e-4, 6 directions, seeds 28700-28705); horizons 2k/10k/40k; step-decay anneal x0.1 3k + x0.01 1k; tail 8,192 steps with two tangents (seeds 28710-1), on-attractor 1e-10 copies (28720-1), sharpness at 16 states; anneal from 64 consecutive tail states. Regime rules fixed in neural_shrimp.regimes.classify before analysis. 6 workers, nice 10.
- **Output:** `/data2/neural-shrimp-data/explore/consequence/s2-28011-out-0.0856`
- **Would surprise us:** Locked or quasi-periodic cells showing chaotic-level reproducibility loss at the attractor (not just from transients); rotation features adding substantial out-of-sample R2 beyond FTLE, sharpness, rate and amplitude
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_consequence.py simulate --name s2-28011-out-0.0856 --system S2 --seed 28011 --output -0.08562528015073712 --hidden 1.55 2.05 --cells 750 --chunks 15 --workers 6`

_commit `1e705775726e`_

- run started 2026-10-07 21:04 UTC finished 21:32 UTC with exit status 0

## 2026-10-07 21:32 UTC: run (exploratory)

- **Question:** A3b consequence pilot, S1R-28002 slice: same regime and incremental-value question as the S2 slice, in the 72-parameter network with coexisting attractors
- **Setup:** S1R-28002, log10 output 0.9413, 160 hidden-rate cells log10 1.00-1.45 (Neimark-Sacker of the period-2 orbit at 1.121671); reduced profile: copies at 1e-12/1e-8/1e-4 x 4 directions (seeds 28700-28703), horizons 2k/10k/40k, anneal x0.1 2k + x0.01 500, tail 4,096 with two tangents and on-attractor copies, sharpness at 8 states, anneal from 32 consecutive tail states; probe inputs N(0,1) seed 28900 (random labels, so no held-out loss). 5 workers, nice 10.
- **Output:** `/data2/neural-shrimp-data/explore/consequence/s1r-28002-out-0.9413`
- **Would surprise us:** Same as S2; also locked cells with O(1) disagreement from on-attractor perturbations
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_consequence.py simulate --name s1r-28002-out-0.9413 --system S1R --seed 28002 --output 0.9413 --hidden 1.00 1.45 --cells 160 --chunks 16 --workers 5 --profile reduced`

_commit `b0416454a7fb` (dirty worktree)_

## 2026-10-07 21:32 UTC: run (exploratory)

- **Question:** Noisy precursor: with small additive noise on every update, does the even-step training-loss spectrum peak at arg(mu)/2pi of the period-2 orbit's two-step multiplier before the Neimark-Sacker point, growing as |mu| approaches 1, and at the rotation number beyond it?
- **Setup:** S2-28011, log10 output -0.08562528015073712, 40 hidden-rate cells log10 1.70-1.80 (Neimark-Sacker at 1.738439); settle 40,000 noiseless steps; two-step multipliers from dense Jacobians for period-2 cells; record 16,384 steps with Gaussian update noise sigma 1e-7 and 1e-5 (seeds 28730-28731). 1 process, nice 10.
- **Output:** `/data2/neural-shrimp-data/explore/consequence/precursor-s2-28011`
- **Would surprise us:** Peak frequency away from arg(mu)/2pi in period-2 cells with |mu| > 0.95, or no peak growth toward the bifurcation
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_consequence_precursor.py --name precursor-s2-28011 --system S2 --seed 28011 --output -0.08562528015073712 --hidden 1.70 1.80 --cells 40 --noise 1e-7 1e-5`

_commit `b0416454a7fb` (dirty worktree)_

- run started 2026-10-07 21:32 UTC finished 21:32 UTC with exit status 0

## 2026-10-07 21:33 UTC: run (exploratory)

- **Question:** Analysis of the S2-28011 consequence slice: per-regime outcomes against period-2 and chaotic controls; incremental value of rotation and locking features beyond FTLE classes, sharpness, rate and amplitude
- **Setup:** tools/analyze_consequence.py with the pre-specified regime rules and models; 300 block-bootstrap draws; Neimark-Sacker point 1.738439 for the beyond-NS subset
- **Output:** `/data2/neural-shrimp-data/explore/consequence/analysis.json`
- **Would surprise us:** Rotation features with delta R2 intervals clearly above zero for magnitude outcomes
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/analyze_consequence.py s2-28011-out-0.0856 --ns 1.738439 --draws 300`

_commit `b0416454a7fb` (dirty worktree)_

- run started 2026-10-07 21:33 UTC finished 21:34 UTC with exit status 0

## 2026-10-07 21:35 UTC: run (exploratory)

- **Question:** Noisy precursor, second pass (adapted after the first run showed peak picking is noisy far from the bifurcation): does a least-squares AR(6) fit of the noisy even-step loss recover the period-2 orbit's two-step multiplier (modulus and angle) from the loss alone, across the period-2 region up to the Neimark-Sacker point?
- **Setup:** S2-28011, log10 output -0.08562528015073712, 60 hidden-rate cells log10 1.60-1.80; settle 40,000; record 16,384 with noise sigma 1e-7 and 1e-5 (seeds 28730-28731); AR(6) dominant complex root vs Jacobian multipliers. 1 process, nice 10.
- **Output:** `/data2/neural-shrimp-data/explore/consequence/precursor-s2-28011-ar`
- **Would surprise us:** AR modulus or angle far from the Jacobian values for period-2 cells with |mu| > 0.9
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_consequence_precursor.py --name precursor-s2-28011-ar --system S2 --seed 28011 --output -0.08562528015073712 --hidden 1.60 1.80 --cells 60 --noise 1e-7 1e-5`

_commit `0c73e956063a` (dirty worktree)_

- run started 2026-10-07 21:35 UTC finished 21:36 UTC with exit status 0

## 2026-10-07 21:36 UTC: A3b exploratory: S2-28011 consequence slice — regimes differ sharply, but rotation/locking adds no robust prediction beyond FTLE classes, sharpness, rate and amplitude

750 cells, log10 hidden 1.55-2.05 at output -0.0856: period-2 282 (to 1.7376), quasi-periodic 125 (1.738-1.826 with interleaved small windows), locked 97 (rotations 1/5, 7/32, 5/23, 2/9 on 1.834-1.879, 1/4 on 1.911-1.919; 96/97 have period 2q x 2^k), chaotic 223, irregular 2, divergent 21. Medians (log10): function disagreement at 40k steps from 1e-8 initial perturbations: period-2 -15.9, quasi-periodic -6.1 (linear response: slope 1.00 vs perturbation size), locked -14.7 but bimodal (IQR -15.2 to -0.75: the upper half of the 2/9 window shows O(1) phase slips inherited from chaotic transients), chaotic -0.63 (slope 0). After the step-decay anneal: -15.8 / -8.7 / -9.4 (bimodal) / -2.3. Stopping-time spread of annealed models: -4.98 / -3.73 / -3.24 / -2.36, with a 10x dip inside the 1/4 window. Training and test losses rise smoothly with the rate; annealed test MSE is about 15% lower after chaotic training (-3.05 vs -2.98). Loss spectra: line fraction 1.0 / 0.99 / 0.96 / 0.10; the strongest peak of the even-step loss equals the rotation number within 1e-3 in 125/125 quasi-periodic and 97/97 locked cells. Nested blocked-CV models (n=729): rotation/locking features add delta R2 ~0 (intervals spanning 0) for every loss and reproducibility outcome; reproducibility is explained by FTLE classes plus a transient-amplification covariate (R2 0.92); the only interval above zero is stopping-time spread (delta R2 +0.045 [+0.001, +0.107]), sign-unstable in the beyond-NS subset and confounded with rate nonlinearity on a single slice. 8 quasi-periodic labels have clearly negative FTLE (possible slow locked orbits); excluding them changes nothing. Verdict for this slice: the consequences are regime-dependent but generic (Lyapunov sign, transient chaos, attractor size).

_commit `0c73e956063a` (dirty worktree)_

## 2026-10-07 21:36 UTC: A3b exploratory: noisy precursor — the training loss alone measures the edge-of-stability 2-cycle's Neimark-Sacker multiplier

S2-28011, output -0.0856, 60 cells log10 hidden 1.60-1.80, Gaussian update noise 1e-7 or 1e-5. For the 41 period-2 cells (|mu| 0.954-0.9985), a least-squares AR(6) fit of the noisy even-step training loss recovers the two-step multiplier from Jacobians: angle/2pi within median 1.8e-4 (max 9.4e-4), modulus within median 0.0014 (max 0.0045); spectral peak height tracks -log(1-|mu|) (r = 0.94). Beyond the Neimark-Sacker point the fit gives modulus ~1 and frequency equal to the rotation number (0.2158-0.2182). So the distance to the quasi-periodic transition and the rotation number of the training that follows are readable from the loss curve before the transition; FTLE (ln|mu|/2) carries only the modulus. First pass used raw peak picking (median gap 3e-4, max 2.5e-3 far from the bifurcation); the AR estimator was added after seeing that (adapted). Caveat: isotropic additive noise stands in for minibatch noise.

_commit `0c73e956063a` (dirty worktree)_

## 2026-10-07 21:37 UTC: run (exploratory)

- **Question:** Noisy precursor in the 72-parameter network: does the AR(6) loss-only estimate recover the period-2 orbit's two-step multiplier in S1R-28002 as in S2?
- **Setup:** S1R-28002, log10 output 0.9413, 40 hidden-rate cells log10 0.95-1.15 (Neimark-Sacker at 1.121671, birth angle 0.1322); settle 40,000; record 16,384 with update noise 1e-7 and 1e-5 (seeds 28730-28731). 1 process, nice 10.
- **Output:** `/data2/neural-shrimp-data/explore/consequence/precursor-s1r-28002-ar`
- **Would surprise us:** AR estimates far from Jacobian multipliers in S1R (would limit the diagnostic to the 13-parameter network)
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_consequence_precursor.py --name precursor-s1r-28002-ar --system S1R --seed 28002 --output 0.9413 --hidden 0.95 1.15 --cells 40 --noise 1e-7 1e-5`

_commit `23d96b67b686`_

- run started 2026-10-07 21:37 UTC finished 21:38 UTC with exit status 0

## 2026-10-07 21:38 UTC: A3b exploratory: the loss-only multiplier readout also works in the 72-parameter network

S1R-28002, output 0.9413, 40 cells log10 hidden 0.95-1.15, update noise 1e-7 or 1e-5. For 34 period-2 cells (|mu| 0.888-0.998), the AR(6) fit of the noisy even-step loss recovers the two-step multiplier angle within median 3.6e-4 (max 2.2e-3) and modulus within median 0.0023 (max 0.018, largest far from the bifurcation); peak height tracks -log(1-|mu|) (r = 0.93-0.96). Beyond the Neimark-Sacker point (1.1217) the fit gives modulus 1.0 and frequencies 0.11-0.14, the S1R rotation numbers.

_commit `23d96b67b686` (dirty worktree)_

- run started 2026-10-07 21:32 UTC finished 21:43 UTC with exit status 0

## 2026-10-07 21:44 UTC: run (exploratory)

- **Question:** A3b consequence pilot, second S2-28011 slice at a different output rate, to decouple rotation number from hidden rate in the nested models
- **Setup:** S2-28011, log10 output -0.30 (Neimark-Sacker near log10 hidden 1.686 with birth angle about 0.170 from the traced curve), 600 hidden-rate cells log10 1.58-2.05; full profile as the first S2 slice; 5 workers, nice 10.
- **Output:** `/data2/neural-shrimp-data/explore/consequence/s2-28011-out-0.30`
- **Would surprise us:** Rotation features adding robust predictive value once two slices with different rotation-rate relations are pooled
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_consequence.py simulate --name s2-28011-out-0.30 --system S2 --seed 28011 --output -0.30 --hidden 1.58 2.05 --cells 600 --chunks 15 --workers 5`

_commit `814224e3eed8` (dirty worktree)_

## 2026-10-07 21:47 UTC: run (exploratory)

- **Question:** Analysis of both consequence slices (S2-28011 at -0.0856 and S1R-28002 at 0.9413) with the same pre-specified rules and models
- **Setup:** tools/analyze_consequence.py, 300 block-bootstrap draws, Neimark-Sacker points 1.738439 and 1.121671
- **Output:** `/data2/neural-shrimp-data/explore/consequence/analysis.json`
- **Would surprise us:** Rotation features with delta R2 intervals clearly above zero in S1R
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/analyze_consequence.py s2-28011-out-0.0856 s1r-28002-out-0.9413 --ns 1.738439 1.121671 --draws 300`

_commit `814224e3eed8` (dirty worktree)_

- run started 2026-10-07 21:47 UTC finished 21:48 UTC with exit status 0

## 2026-10-07 21:50 UTC: A3b exploratory: S1R-28002 consequence slice repeats the S2 pattern; nested comparisons underpowered

160 cells, log10 hidden 1.00-1.45 at output 0.9413 (reduced profile): period-2 49 (to 1.119, plus a coexisting period-2 branch at 1.187-1.200), quasi-periodic 31, locked 19 (1/9, 2/17, 1/8, 3/23, 2/15; 19/19 with period 2q x 2^k), chaotic 59 (from 1.255), irregular 2. Medians (log10) of function disagreement at 40k steps from 1e-8 initial perturbations: period-2 -15.4, quasi-periodic -6.25 (slope 0.98), locked -14.0 bimodal (9/19 cells with phase slips; their transient amplification is 10^11.9 in 2k steps vs 10^1.0 without slips), chaotic -0.45. Stopping-time spread -5.04 / -3.17 / -3.09 / -2.36; loss line fraction 1.0 / 0.98 / 0.96 / 0.39; loss peak equals rotation number in 31/31 quasi-periodic and 19/19 locked cells. Rule flaw found: the quasi-periodic rule bounds FTLE only from above, so 12 cells with clearly negative FTLE (down to -0.009; rotation near 0.115, likely locked orbits with period above the 64-step screen) carry the quasi-periodic label; the pre-specified exclusion check was run. Nested models: n = 160 (6 CV blocks) gives degenerate bootstrap intervals; point estimates of delta R2 range -0.08 to +0.07 for reproducibility, stopping-time and loss outcomes, i.e. no evidence that rotation/locking adds value beyond FTLE classes, sharpness, rate and amplitude.

_commit `b3fc42abda51` (dirty worktree)_
