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

## 2026-10-07 20:56 UTC: run (exploratory)

- **Question:** Real-data calibration: sharpness at initialization and at an L-BFGS minimum (flip threshold 2/lambda_max) for a 64-8-1 tanh MLP on 1,024 UCI digits (halves task, no weight decay), and the cost of one batched step
- **Setup:** Seed 28101; standard parameterization, PyTorch-default init; MSE; float64
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves/calibrate.json`
- **Would surprise us:** lambda_max at the minimum far from lambda_max at init (would make a fixed-init sweep hard to anchor)
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py calibrate --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves`

_commit `6228425e9cd9`_

- run started 2026-10-07 20:56 UTC finished 20:56 UTC with exit status 0

## 2026-10-07 20:57 UTC: run (exploratory)

- **Question:** Real-data coarse scan, no weight decay: with one global learning rate from a fixed initialization, which attractors (fixed, period 2, higher periods, quasi-periodic, chaotic) occur for the 64-8-1 digits MLP?
- **Setup:** Seed 28101, n=1024 halves task, ridge 0; 160 log10 eta in [-0.9, 0.1]; 20,000 updates with EXP-028 screening, then 4,096 recorded updates (loss, 32-d random projection); rotation numbers and loss-spectrum peaks; 4 workers
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves/scan-coarse`
- **Would surprise us:** Only fixed points and divergence (no oscillatory regime at all), or a direct jump from period 2 to divergence
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py scan --range -0.9 0.1 --cells 160 --chunk 20 --workers 4 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves/scan-coarse`

_commit `6228425e9cd9` (dirty worktree)_

## 2026-10-07 20:57 UTC: run (exploratory)

- **Question:** Real-data coarse scan, weight decay 1e-3: same question as the no-decay scan, with minima made isolated and slow directions damped
- **Setup:** Seed 28101, n=1024 halves task, ridge 1e-3; 160 log10 eta in [-0.9, 0.1]; 20,000 updates with EXP-028 screening, then 4,096 recorded updates; 4 workers
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-coarse`
- **Would surprise us:** Weight decay removing the oscillatory regime entirely
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py scan --ridge 1e-3 --range -0.9 0.1 --cells 160 --chunk 20 --workers 4 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-coarse`

_commit `6228425e9cd9` (dirty worktree)_

## 2026-10-07 21:00 UTC: run (exploratory)

- **Question:** Real-data branch, weight decay 1e-3: where does the polished minimum flip, at 2/lambda_max, and how does the period-2 orbit born there lose stability as one global learning rate grows: fold, flip or Neimark-Sacker, with what multiplier angle?
- **Setup:** Seed 28101, 64-8-1 tanh MLP, n=1024 halves task, ridge 1e-3; L-BFGS then Newton polish; period-2 seeds at eta/eta_flip in 1.002, 1.01, 1.03, 1.1 with a 1e-3 kick along the top Hessian eigenvector, 20,000 settling updates; continuation in log10 eta up to +0.3 with bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/branch`
- **Would surprise us:** Subcritical flip with no attracting period-2 orbit near the flip, or immediate loss of the orbit
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py branch --ridge 1e-3 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/branch`

_commit `767251c96bcd` (dirty worktree)_

- run started 2026-10-07 21:00 UTC finished 21:01 UTC with exit status 0

## 2026-10-07 21:02 UTC: run (exploratory)

- **Question:** Real-data branch rerun, weight decay 1e-3: with the period-2 seed gate relaxed so Newton certifies, how does the period-2 orbit born at the flip lose stability as the global learning rate grows?
- **Setup:** As the first branch run but seed gate two-step gap below 1e-5, factors 1.01, 1.03, 1.1; minimum polished again from the same initialization; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/branch-2`
- **Would surprise us:** Newton failing to certify a period-2 orbit at gaps near 1e-7, which would mean the slow drift is not convergence to an orbit
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py branch --ridge 1e-3 --factors 1.01 1.03 1.1 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/branch-2`

_commit `8f46ca07a35d` (dirty worktree)_

- run started 2026-10-07 20:57 UTC finished 21:21 UTC with exit status 0

- run started 2026-10-07 20:57 UTC finished 21:22 UTC with exit status 0

## 2026-10-07 21:25 UTC: run (exploratory)

- **Question:** Real-data fine scan of the transition, weight decay 1e-3: between the edge-of-stability period-2 oscillation near log10 eta -0.40 and broadband dynamics near -0.25, is there a slow invariant-circle regime with a well-defined rotation number that grows with eta, and locked windows?
- **Setup:** Seed 28101, 64-8-1, n=1024 halves, ridge 1e-3, fixed initialization; 120 log10 eta in -0.44 to -0.24; 60,000 updates with EXP-028 screening, then 16,384 recorded updates for rotation numbers and loss spectra; 6 workers, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-fine`
- **Would surprise us:** Slow loss modulations without a monotone two-step rotation, or rotation numbers that jump irregularly with eta
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py scan --ridge 1e-3 --range -0.44 -0.24 --cells 120 --chunk 20 --steps 60000 --keep 16384 --workers 6 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-fine`

_commit `f0e059e5a220` (dirty worktree)_

## 2026-10-07 21:25 UTC: Real-data coarse scans: no exactly periodic cells; edge-of-stability period-2 oscillation with slow drift, then slow loss modulations, broadband dynamics, divergence

64-8-1 digits MLP, one global learning rate, fixed initialization, 160 log10 eta in -0.9..0.1, 20,000 updates. Without weight decay: 132 drift, 4 chaotic, 24 divergent cells; with weight decay 1e-3: 92 unresolved, 22 drift, 20 chaotic, 25 divergent, and no fixed or periodic labels in either. Loss spectra with weight decay: for log10 eta -0.9..-0.40 a line at 1/2 carries up to 80% of the power on top of slow convergence (an edge-of-stability period-2 oscillation not resolved by the screen because slow weight-decay directions, curvature near 4e-4, are still relaxing); from about -0.37 slow modulations appear with harmonic peaks, e.g. 0.0029, 0.0059, 0.0088 cycles per update at -0.347 with FTLE -0.00025; from about -0.27 several incommensurate low-frequency peaks and broadband power; positive FTLE from -0.20; divergence from -0.05. The plot's first quasi-periodic labels came from sub-turn angle drifts and were withdrawn (criterion now needs three monotone turns). Interpretation pending a finer, longer scan of -0.44..-0.24 and continuation of the period-2 orbit.

_commit `f0e059e5a220` (dirty worktree)_

- run started 2026-10-07 21:02 UTC finished 21:32 UTC with exit status 0

## 2026-10-07 21:33 UTC: run (exploratory)

- **Question:** Real-data continuation of the period-2 orbit that gradient descent reaches from its initialization, weight decay 1e-3: starting at log10 eta -0.42, how does it lose stability as eta grows, and does the boundary match the onset of slow loss modulations near -0.37?
- **Setup:** Seed 28101, 64-8-1, n=1024 halves, ridge 1e-3; 60,000 settling updates from the initialization at log10 eta -0.42; period 2 certified by Newton even if the screen misses it; continuation up to +0.2 in log10 eta, h0 0.002, bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/continue-0.42`
- **Would surprise us:** A fold or flip boundary instead of Neimark-Sacker, or a boundary far from -0.37
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py continue --ridge 1e-3 --start -0.42 --period 2 --steps 60000 --s-max 0.2 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/continue-0.42`

_commit `fc4a3584d247` (dirty worktree)_

## 2026-10-07 21:33 UTC: Real-data branch, weight decay 1e-3: the sharp L-BFGS minimum's period-2 orbit ends in a fold, not Neimark-Sacker

Polished minimum: gradient norm 8e-16, loss 0.0412, lambda_max 23.64, lambda_min 3.9e-4, so the flip sits at eta 0.0846 (log10 -1.0727). The period-2 orbit seeded at 1.01 x eta_flip (Newton-certified despite a two-step gap of 6e-8 after 20,000 updates) stays attracting along 30 continuation points and loses stability at log10 eta -0.8952 with dominant multiplier +0.99999990 (real): a fold of the period-2 orbit. This minimum is much sharper than where gradient descent from the initialization settles at these rates, so the result speaks to this branch only.

_commit `fc4a3584d247` (dirty worktree)_

- run started 2026-10-07 21:33 UTC finished 21:40 UTC with exit status 0

## 2026-10-07 21:40 UTC: Real-data: the edge-of-stability period-2 orbit reached from the initialization loses stability by Neimark-Sacker with one global learning rate

64-8-1 tanh MLP on 1,024 UCI digits, standard parameterization, weight decay 1e-3, seed 28101. After 60,000 updates from the initialization at log10 eta -0.42 the screen reports unresolved because of slow drift, but Newton certifies an attracting period-2 orbit; continued in log10 eta it stays attracting until -0.3885464, where the dominant two-step multiplier is 0.998814 + 0.048693i with modulus 1 - 1.0e-7 (bracket 9e-10): a Neimark-Sacker bifurcation. Predicted rotation of the newborn circle arg(mu)/2pi = 0.00775 turns per two-step iteration, i.e. a loss modulation near 0.0039 cycles per update, about 258 updates per cycle. The coarse scan shows slow harmonic loss modulations from about -0.37 (0.0027-0.0029 cycles per update at -0.372 and -0.347, from the initialization, possibly a different branch). The finer, longer scan now running will test the predicted rotation just beyond the bifurcation. Exploratory.

_commit `3c993e36d58b` (dirty worktree)_

- run started 2026-10-07 21:25 UTC finished 21:44 UTC with exit status 0

## 2026-10-07 21:46 UTC: run (exploratory)

- **Question:** Real-data rotation numbers just beyond the Neimark-Sacker point: started from the certified period-2 state at the bifurcation, do cells settle on an invariant circle whose two-step rotation number tends to the predicted 0.00775 as eta approaches the bifurcation from above?
- **Setup:** Seed 28101, 64-8-1, n=1024 halves, ridge 1e-3; every cell starts from the last attracting period-2 state of the continuation, at log10 eta -0.388546; 90 log10 eta in -0.3885 to -0.37; 40,000 updates with EXP-028 screening, then 32,768 recorded; 5 workers, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-ns`
- **Would surprise us:** Cells close to the bifurcation leaving for a distant attractor, which would suggest a subcritical bifurcation, or rotation numbers far from 0.00775 near onset
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py scan --ridge 1e-3 --init /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/continue-0.42/continue--0.42000-state.npy --range -0.3885 -0.37 --cells 90 --chunk 18 --steps 40000 --keep 32768 --workers 5 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/scan-ns`

_commit `00c9ca177b55` (dirty worktree)_

## 2026-10-07 21:46 UTC: run (exploratory)

- **Question:** Real-data adiabatic sweep along the Neimark-Sacker branch: following one attractor from the bifurcation upward in eta, how does the rotation number evolve, are there plateaus at rationals with exactly periodic orbits of period 2q, and where does the circle break down?
- **Setup:** Seed 28101, 64-8-1, n=1024 halves, ridge 1e-3; start from the certified period-2 state at log10 eta -0.388546; 60 log10 eta in -0.3885 to -0.33, each starting from the previous final state; 20,000 settling updates, 16,384 recorded, FTLE over the last 4,000; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/sweep-ns`
- **Would surprise us:** Rotation number decreasing with eta, or chaos immediately after the bifurcation
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py sweep --ridge 1e-3 --init /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/continue-0.42/continue--0.42000-state.npy --range -0.3885 -0.33 --cells 60 --steps 20000 --keep 16384 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/sweep-ns`

_commit `00c9ca177b55` (dirty worktree)_

## 2026-10-07 21:46 UTC: Real-data fine scan from the initialization: slow invariant-circle candidates in patches, rotation 0.004-0.012

120 log10 eta in -0.44..-0.24, 60,000 updates plus 16,384 recorded, weight decay 1e-3. Below the Neimark-Sacker point (-0.3885) cells relax onto the period-2 oscillation (one screened period-2 cell at -0.398, the rest still drifting slowly). Above it, 18 cells meet the quasi-periodic candidate criterion (unresolved, |FTLE| below 0.005, forward two-step angle steps on at least 99% of iterations, at least three turns): rotation numbers 0.0039-0.0124 turns per two-step iteration, in runs of adjacent cells with nearly equal values (-0.3963..-0.3929: 0.0039-0.0041; -0.3492..-0.3459: 0.0060; -0.2971..-0.2938: 0.0103-0.0121), rank correlation with eta +0.63. The patches suggest that gradient descent from the initialization lands on different period-2 branches at different rates (loss levels jump between patches), each with its own circle. The three cells at -0.396..-0.393 lie below the continued branch's bifurcation, so they belong to another branch or a coexisting circle. FTLE estimates near -3e-4 cannot separate quasi-periodic orbits from weakly attracting locked orbits with periods above 64; the branch-following sweep records recurrence up to lag 2,048.

_commit `00c9ca177b55` (dirty worktree)_

- run started 2026-10-07 21:46 UTC finished 21:56 UTC with exit status 0

## 2026-10-07 21:59 UTC: run (exploratory)

- **Question:** Real-data phase portraits: is the attractor after the change near log10 eta -0.380 a deformed or period-doubled invariant circle, or something else?
- **Setup:** Seed 28101, ridge 1e-3; from the period-2 state at the Neimark-Sacker point; log10 eta -0.3880, -0.3820, -0.3790, -0.3700, -0.3450; 40,000 settling updates, 8,192 recorded; two-step points on their principal plane colored by two-step parity; rotation numbers of the two- and four-step maps; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/portraits.png`
- **Would surprise us:** A single smooth loop at -0.379 with a non-monotone angle, which would mean the rotation estimate failed rather than the dynamics changing
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/plot_realdata_portrait.py --init /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/continue-0.42/continue--0.42000-state.npy --rates -0.3880 -0.3820 -0.3790 -0.3700 -0.3450 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-halves-wd1e-3/portraits.png`

_commit `330a449c8716` (dirty worktree)_

- run started 2026-10-07 21:46 UTC finished 22:00 UTC with exit status 0

- run started 2026-10-07 21:59 UTC finished 22:00 UTC with exit status 0

## 2026-10-07 22:01 UTC: run (exploratory)

- **Question:** Real-data robustness, second data subset and initialization: does the edge-of-stability period-2 orbit reached from the initialization again lose stability by Neimark-Sacker as one global learning rate grows?
- **Setup:** Seed 28102, 64-8-1 tanh MLP, n=1024 halves task, ridge 1e-3; 60,000 updates from the initialization at log10 eta -0.45, period 2 certified by Newton; continuation up to +0.3 in log10 eta, bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/continue-0.45`
- **Would surprise us:** A fold or flip ending instead of Neimark-Sacker
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py continue --seed 28102 --ridge 1e-3 --start -0.45 --period 2 --steps 60000 --s-max 0.3 --out /data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/continue-0.45`

_commit `330a449c8716` (dirty worktree)_

## 2026-10-07 22:01 UTC: run (exploratory)

- **Question:** Real-data robustness, second task: for digit parity instead of halves, does the edge-of-stability period-2 orbit reached from the initialization lose stability by Neimark-Sacker?
- **Setup:** Seed 28101, 64-8-1 tanh MLP, n=1024 parity task, ridge 1e-3; 60,000 updates from the initialization at log10 eta -0.45, period 2 certified by Newton; continuation up to +0.3 in log10 eta, bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/continue-0.45`
- **Would surprise us:** A fold or flip ending instead of Neimark-Sacker
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py continue --task parity --ridge 1e-3 --start -0.45 --period 2 --steps 60000 --s-max 0.3 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/continue-0.45`

_commit `330a449c8716` (dirty worktree)_

## 2026-10-07 22:01 UTC: Real-data: rotation number of the newborn circle matches the Neimark-Sacker multiplier angle; circle grows, wrinkles, and a second smooth circle segment follows

From the certified period-2 state at the bifurcation (log10 eta -0.388546), independent cells (40,000 updates, 32,768 recorded) and an adiabatic sweep (each rate from the previous final state) agree. At log10 eta -0.3885 the attractor is a smooth invariant circle of the two-step map: rotation 0.007751 (scan 127 turns, sweep 63 turns), forward angle steps 100%, planar share 0.993-1.000, no recurrence up to lag 2,048, FTLE -0.0003 to -0.0006 (zero within the finite-time error), versus the predicted arg(mu)/2pi = 0.0077528. The rotation number then decreases smoothly (0.006886 at -0.3816 in the scan, 0.006705 at -0.3806 in the sweep) while the loss modulation grows from 1e-4 to 3e-3. Near -0.380 the loop becomes wrinkled and non-star-shaped (phase portraits at -0.379 and -0.370), so the centroid-angle rotation estimate fails there (forward 0.45-0.9); FTLE stays within +-0.0015. From -0.347 to -0.330 the sweep again follows a smooth monotone circle with rotation rising 0.00584 to 0.00753 and a regular slow loss oscillation with two spikes per cycle (about 267 updates per cycle at -0.330). No exactly periodic orbits up to lag 2,048, so no locked windows were resolved: at these small rotation numbers the relevant tongues have q above 100 and would be very thin. Figures: sweep-ns/sweep.png, portraits.png under explore/realdata/w8-n1024-halves-wd1e-3.

_commit `330a449c8716` (dirty worktree)_

- run started 2026-10-07 22:01 UTC finished 22:02 UTC with exit status 0

- run started 2026-10-07 22:01 UTC finished 22:02 UTC with exit status 0

## 2026-10-07 22:10 UTC: Real-data robustness: continuation could not start at log10 eta -0.45 for seed 28102 halves or seed 28101 parity

Both runs ended start_not_attracting: Newton did not certify an attracting period-2 orbit from the state after 60,000 updates at -0.45. The edge-of-stability range was located only for seed 28101 halves, so -0.45 may sit outside the period-2 regime for these networks. Coarse scans of log10 eta -0.8..-0.2 follow to place the regime before continuing; the continuation now records the start certificate's reason.

_commit `6c41af8bef16` (dirty worktree)_

## 2026-10-07 22:10 UTC: run (exploratory)

- **Question:** Real-data robustness scan, seed 28102 halves: where are the edge-of-stability period-2 regime and the onset of slow loss modulations?
- **Setup:** Seed 28102, 64-8-1, n=1024 halves, ridge 1e-3; 120 log10 eta in -0.8 to -0.2 from the initialization; 20,000 updates with EXP-028 screening, then 4,096 recorded; 3 workers, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/scan-coarse`
- **Would surprise us:** No line at 1/2 anywhere in the range
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py scan --seed 28102 --ridge 1e-3 --range -0.8 -0.2 --cells 120 --chunk 20 --workers 3 --out /data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/scan-coarse`

_commit `aab87e578c3f` (dirty worktree)_

## 2026-10-07 22:10 UTC: run (exploratory)

- **Question:** Real-data robustness scan, seed 28101 parity: where are the edge-of-stability period-2 regime and the onset of slow loss modulations?
- **Setup:** Seed 28101, 64-8-1, n=1024 parity, ridge 1e-3; 120 log10 eta in -0.8 to -0.2 from the initialization; 20,000 updates with EXP-028 screening, then 4,096 recorded; 3 workers, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/scan-coarse`
- **Would surprise us:** No line at 1/2 anywhere in the range
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py scan --task parity --ridge 1e-3 --range -0.8 -0.2 --cells 120 --chunk 20 --workers 3 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/scan-coarse`

_commit `aab87e578c3f` (dirty worktree)_

- run started 2026-10-07 22:10 UTC finished 22:16 UTC with exit status 0

- run started 2026-10-07 22:10 UTC finished 22:16 UTC with exit status 0

## 2026-10-07 22:17 UTC: run (exploratory)

- **Question:** Real-data robustness, seed 28102 halves: starting inside the period-2 regime located by the coarse scan, how does the period-2 orbit lose stability as eta grows?
- **Setup:** Seed 28102, 64-8-1, n=1024 halves, ridge 1e-3; 60,000 updates from the initialization at log10 eta -0.7092, where the loss line at 1/2 carries 72% of the power; period 2 certified by Newton; continuation up to +0.3, bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/continue-0.7092`
- **Would surprise us:** Fold or flip instead of Neimark-Sacker; boundary far from the first slow modulations near -0.634
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py continue --seed 28102 --ridge 1e-3 --start -0.7092 --period 2 --steps 60000 --s-max 0.3 --out /data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/continue-0.7092`

_commit `41be47d81b68` (dirty worktree)_

## 2026-10-07 22:17 UTC: run (exploratory)

- **Question:** Real-data robustness, parity task: starting inside the period-2 regime, how does the period-2 orbit lose stability as eta grows?
- **Setup:** Seed 28101, 64-8-1, n=1024 parity, ridge 1e-3; 60,000 updates from the initialization at log10 eta -0.4067, where the loss line at 1/2 carries 72% of the power; period 2 certified by Newton; continuation up to +0.3, bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/continue-0.4067`
- **Would surprise us:** Fold or flip instead of Neimark-Sacker
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py continue --task parity --ridge 1e-3 --start -0.4067 --period 2 --steps 60000 --s-max 0.3 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/continue-0.4067`

_commit `41be47d81b68` (dirty worktree)_

## 2026-10-07 22:17 UTC: run (exploratory)

- **Question:** Real-data robustness, parity task, second start: from log10 eta -0.4521, does the period-2 orbit reached there lose stability by Neimark-Sacker?
- **Setup:** Seed 28101, 64-8-1, n=1024 parity, ridge 1e-3; 60,000 updates from the initialization at log10 eta -0.4521, line at 1/2 with 79% of the power; period 2 certified by Newton; continuation up to +0.3, bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/continue-0.4521`
- **Would surprise us:** Fold or flip instead of Neimark-Sacker
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py continue --task parity --ridge 1e-3 --start -0.4521 --period 2 --steps 60000 --s-max 0.3 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/continue-0.4521`

_commit `41be47d81b68` (dirty worktree)_

## 2026-10-07 22:17 UTC: Real-data robustness scans: period-2 regime and slow loss modulations recur for seed 28102 halves and for the parity task

Coarse scans (120 log10 eta in -0.8..-0.2, 20,000 updates, weight decay 1e-3) of the loss spectrum. Seed 28102 halves: the line at 1/2 carries 44-73% of the power from -0.78 to -0.68; slow harmonic modulations first at -0.634 (0.0037, 0.0044 cycles per update), then in most cells with harmonic sets such as 0.0024/0.0049/0.0073 interleaved with period-2 cells (-0.543, -0.422, -0.392, -0.377); positive FTLE up to 0.0037 near -0.21. Parity: period-2 cells scattered from -0.74 to -0.41 (line at 1/2 up to 79%), slow harmonic modulations from -0.42 (0.0044/0.0088/0.0176), positive FTLE (up to 0.0087, one chaotic cell) from -0.24. Continuations from -0.7092 (seed 28102) and from -0.4067 and -0.4521 (parity) follow.

_commit `41be47d81b68` (dirty worktree)_

- run started 2026-10-07 22:17 UTC finished 22:18 UTC with exit status 0

## 2026-10-07 22:18 UTC: run (exploratory)

- **Question:** Real-data robustness, seed 28102 halves, second start: Newton did not converge at -0.7092 (last spectral radius 1.0005); from -0.7546, deeper in the period-2 regime, how does the orbit lose stability?
- **Setup:** Seed 28102, 64-8-1, n=1024 halves, ridge 1e-3; 60,000 updates from the initialization at log10 eta -0.7546, line at 1/2 with 67% of the power; continuation up to +0.3, bisection 1e-9; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/continue-0.7546`
- **Would surprise us:** Fold or flip instead of Neimark-Sacker
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py continue --seed 28102 --ridge 1e-3 --start -0.7546 --period 2 --steps 60000 --s-max 0.3 --out /data2/neural-shrimp-data/explore/realdata/s28102-w8-n1024-halves-wd1e-3/continue-0.7546`

_commit `200ab0fadd8f` (dirty worktree)_

- run started 2026-10-07 22:18 UTC finished 22:19 UTC with exit status 0

- run started 2026-10-07 22:17 UTC finished 22:23 UTC with exit status 0

- run started 2026-10-07 22:17 UTC finished 22:23 UTC with exit status 0

## 2026-10-07 22:24 UTC: run (exploratory)

- **Question:** Real-data parity: the period-2 orbit ends in a fold at log10 eta -0.39831. Is the fold a saddle-node on an invariant circle, so that just beyond it training rotates slowly with rotation number growing from zero like the square root of the distance, or does training jump to a distant attractor?
- **Setup:** Seed 28101, 64-8-1, n=1024 parity, ridge 1e-3; adiabatic sweep from the last attracting period-2 state of the continuation from -0.4067; 20 log10 eta in -0.39830 to -0.39400; 20,000 settling updates, 16,384 recorded; single process, nice 10
- **Output:** `/data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/sweep-fold`
- **Would surprise us:** Small rotation numbers that do not vanish toward the fold, or immediate jumps to large-amplitude dynamics
- **Command:** `nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explore_realdata.py sweep --task parity --ridge 1e-3 --init /data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/continue-0.4067/continue--0.40670-state.npy --range -0.39830 -0.39400 --cells 20 --steps 20000 --keep 16384 --out /data2/neural-shrimp-data/explore/realdata/w8-n1024-parity-wd1e-3/sweep-fold`

_commit `200ab0fadd8f` (dirty worktree)_

## 2026-10-07 22:24 UTC: Real-data robustness continuations: parity branches end in folds; seed 28102 starts not certified

Parity task (seed 28101): period-2 orbits certified after 60,000 updates from the initialization at log10 eta -0.4067 and -0.4521 lie on different branches and both end in folds: at -0.398313 (multiplier +0.99999985) and at -0.412628 (+0.99999987), not Neimark-Sacker. Seed 28102 halves: Newton did not certify a period-2 orbit at -0.7092 (newton_not_converged, last spectral radius 1.0005) or -0.7546 (line_search_stalled), so that subset's branch structure is unresolved. Slow harmonic loss modulations exist in all three configurations' scans, so the Neimark-Sacker branch of seed 28101 halves is one route into them. A fold of the period-2 orbit lying on an invariant circle of the two-step map (the edge of a rotation-0 locking region, a saddle-node on a circle) is a candidate second route; an adiabatic sweep just beyond the parity fold tests it.

_commit `200ab0fadd8f` (dirty worktree)_

- run started 2026-10-07 22:24 UTC finished 22:28 UTC with exit status 0

## 2026-10-07 22:29 UTC: Real-data parity fold sweep: no slow rotation beyond the fold; gradient descent stays on a period-2 orbit

Adiabatic sweep from the last attracting state at the parity fold (log10 eta -0.398313) over 20 rates in -0.39830..-0.39400: every rate is exactly period-2 by the recurrence check (projected states repeat at lag 2), rotation 6e-5 with half a turn in the record (no rotation), FTLE -0.00064, and loss values varying smoothly from 0.05861 to 0.06000. So the fold is not a saddle-node on an invariant circle here: past it, gradient descent settles on a nearby period-2 orbit, another branch or an imperfect-pitchfork partner, and the period-2 regime is a set of branches joined by folds. The parity scan's slow modulations from -0.42 onward must come from other branches; that task's Neimark-Sacker hypothesis is untested.

_commit `f671ef42137f` (dirty worktree)_
