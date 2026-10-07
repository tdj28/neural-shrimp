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

## 2026-10-07 20:52 UTC: run (exploratory)

- **Question:** Do locked windows (period 2q, rotation m/q) emanate from the points of the Neimark-Sacker curve of the period-2 orbit where the birth rotation arg(mu)/2pi equals m/q (Arnold tongue roots), and is the rotation number continuous across the quasi-periodic band?
- **Setup:** S2-28011, log10 hidden 1.60-2.06 x log10 output -0.46 to -0.07, 256x256 cells, EXP-028 screening (20,000 updates) then 256 two-step samples for the plane and 6,000 for the rotation number; compare with the traced Neimark-Sacker curve (a3-ns-curve/s2-28011)
- **Output:** `../neural-shrimp-data/explore/a3-rotation-map/s2-28011`
- **Would surprise us:** Rotation just beyond the Neimark-Sacker curve differing from arg(mu)/2pi; locked regions with rotation m/q not approaching the predicted root; period not 2q on rational plateaus
- **Command:** `.venv/bin/python tools/explore_a3_rotation_map.py --system S2 --seed 28011 --hidden 1.60 2.06 --output -0.46 -0.07 --resolution 256 256 --workers 6 --out a3-rotation-map/s2-28011`

_commit `b80365216008`_

## 2026-10-07 20:53 UTC: run (exploratory)

- **Question:** Does the minimal two-parameter factorization model with weight decay and separate rates show a Neimark-Sacker bifurcation of the edge-of-stability 2-cycle (invariant circles drawable in the (u, v) plane)?
- **Setup:** L = (uv-1)^2 + 0.05(u^2+v^2); log10 rates -0.6..0.2 for both, 400x400 map with EXP-028-style screening; period-2 continuation along the u-rate axis at 24 values of the v rate, bisection 1e-11
- **Output:** `../neural-shrimp-data/explore/a3-uv-model`
- **Would surprise us:** Not expected either way: a 2-D gradient map can lose the 2-cycle by flip, fold or Neimark-Sacker
- **Command:** `.venv/bin/python tools/explore_a3_uv_model.py`

_commit `64cc016cf75c` (dirty worktree)_

## 2026-10-07 20:57 UTC: run (exploratory)

- **Question:** Reproduce the well-ordering result with committed code, and classify the 61 confirmed windows: which are Birkhoff circle orbits with rotation m/q?
- **Setup:** All verified EXP-028 certificates with period >= 3; two-step points for even periods; principal-plane angles; rigid-shift test with gcd(m, q) = 1; null from 10,000 uniform random angle sets per point count (seed 20261007)
- **Output:** `../neural-shrimp-data/explore/a3-well-ordering/well-ordering.json`
- **Would surprise us:** Committed code not reproducing 65 of 115; confirmed windows not more often well-ordered than other certified orbits
- **Command:** `.venv/bin/python tools/explore_a3_well_ordering.py`

_commit `7c7dabcbeb29` (dirty worktree)_

- run started 2026-10-07 20:57 UTC finished 20:57 UTC with exit status 0

## 2026-10-07 20:58 UTC: run (exploratory)

- **Question:** With curve (tour) ordering and period-doubled levels, how many confirmed windows are Birkhoff orbits of an m/q tongue, against a multiplicity-corrected null?
- **Setup:** Same certificates; tests on the plane and tour orders at levels k = 0..3 (points of F^(2^(k+1))); only tests on at least 5 points count; null = sum of per-test random-angle pass rates, capped at 1 per orbit
- **Output:** `../neural-shrimp-data/explore/a3-well-ordering/well-ordering.json`
- **Would surprise us:** Refinements raising the pass count no more than the null expectation rises
- **Command:** `.venv/bin/python tools/explore_a3_well_ordering.py --refined`

_commit `9a48cc25e8ee` (dirty worktree)_

- run started 2026-10-07 20:58 UTC finished 20:58 UTC with exit status 0

## 2026-10-07 20:59 UTC: A3 exploratory: 35 of 56 testable confirmed windows are Birkhoff orbits of m/q tongues (null 3.6)

Committed code (tools/explore_a3_well_ordering.py) reproduces the earlier inline result: 65/115 certified orbits with at least 5 points well-ordered in the principal plane (random-angle null expectation 1.56); confirmed windows 31/61 (null 4.92; orbits of 2 or 3 points pass trivially and are not counted as evidence). With curve (tour) ordering and period-doubled levels (points of F^(2^(k+1))), 35 of the 56 confirmed windows having a test on at least 5 points pass, against a multiplicity-corrected null expectation of 3.58 (all certified: 78/115 vs 7.43). Period doubling inside tongues is directly visible: six period-20 S1R-28001-zoom windows are well-ordered at the four-step level with rotation 2/5 (= 2 x 1/5), and a period-44 window gives 4/11 (= 2 x 2/11). Unexplained: all 5 odd-period windows (S2: periods 5, 11, 13, 17, 29) and about 16 even windows (S1R periods 34-58, S2 periods 16-50). Caveat: the base rotation of a doubled orbit should also be checked against an adjacent locked window of base period; principal-plane and tour orders are both projections of a curve in 13 or 72 dimensions.

_commit `9a48cc25e8ee` (dirty worktree)_

- run started 2026-10-07 20:48 UTC finished 21:17 UTC with exit status 0

## 2026-10-07 21:18 UTC: A2 speckle survey: S1R windows coexist with other attractors; S2 windows have long chaotic transients (orbit-start test confounded)

Inside 0.9x the ray-continued stability region of each confirmed window: S1R (33 windows) shows the window period in 26.5% of map cells from the shared start; of 3,864 sampled other cells, 4x longer runs from the shared start recover period p in 10.3% but runs from the certified orbit point recover 86.5%: coexisting attractors, so fixed-start maps hide most of each S1R window. S2 (28 windows) shows the window period in 55.6% of cells; of 2,224 sampled other cells, 4x longer runs recover 54.6% (long chaotic transients) and orbit-point starts only 44.4%, failing mostly as chaotic labels (e.g. zoom:230 long 83/120 vs orbit 1/120). The pre-stated surprise (orbit starts failing inside the ray-continued region) therefore occurred for S2, but the comparison is confounded: orbit-start runs had 20,000 updates against 80,000 for the long runs, and the window-center orbit point may lie outside the basin of the continued orbit at distant cells; the angular interpolation of ray boundaries can also include cells outside thin curved regions. Follow-up queued: orbit-point starts with 80,000 updates for S2, and starts from the continued orbit along each ray. Implication either way: map-based shrimp criteria must use orbit-following (continuation) maps, and paper figures from fixed starts must say they show basin selection.

_commit `f0ae8272b59d` (dirty worktree)_

## 2026-10-07 21:18 UTC: run (exploratory)

- **Question:** Is the S2 orbit-start failure an artifact of the shorter horizon? Rerun the same sampled cells from the certified orbit point with 80,000 updates (equal to the long runs).
- **Setup:** S2 windows only (28), same sampled cells (seeds tied to rays.json order), orbit-point starts with 4 map horizons (80,000 updates), EXP-028 screening
- **Output:** `../neural-shrimp-data/explore/a2-survey/speckle-s2-orbit80k.json`
- **Would surprise us:** Orbit-point starts still recovering period p in under half the sampled cells (would point to basin limits of the window-center orbit point or to region-mask errors)
- **Command:** `.venv/bin/python tools/explore_a2_speckle.py --systems S2 --orbit-factor 4 --skip-long --workers 4 --name speckle-s2-orbit80k.json`

_commit `fdb2aea8c409`_

## 2026-10-07 21:22 UTC: run (exploratory)

- **Question:** High-resolution rotation-number map of the S2-28011 plane: is the rotation number continuous across the quasi-periodic band, equal to arg(mu)/2pi just beyond the Neimark-Sacker curve, and do m/q plateaus (period 2q) emanate from the predicted roots?
- **Setup:** GPU (RTX 2000 Ada, qualified float64 backend), S2-28011, log10 hidden 1.60-2.06 x log10 output -0.46 to -0.07, 1024x1024, EXP-028 screening (20,000 updates) then 256 two-step samples (plane) and 6,000 (rotation)
- **Output:** `../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-1024`
- **Would surprise us:** Same as the 256x256 CPU run: rotation beyond the curve differing from arg(mu)/2pi; m/q regions not approaching their roots; periods not 2q on plateaus
- **Command:** `.venv/bin/python tools/explore_gpu_rotation_map.py --name a3-gpu-rotation/s2-28011-1024 --system S2 --seed 28011 --hidden 1.60 2.06 --output -0.46 -0.07 --resolution 1024 1024 --library ../neural-shrimp-data/builds/gdmap-cuda-64af0462/gdmap_cuda.so`

_commit `92ae8586eebd` (dirty worktree)_

- run started 2026-10-07 21:18 UTC finished 21:23 UTC with exit status 0

## 2026-10-07 21:27 UTC: Informative failure: two-parameter model scan killed by its 30-minute timeout

The 400x400 scan of the factorization model (run card above, commit before 92ae8586) was killed by timeout 1800 (exit 143) while the CPU was oversubscribed (load above 30, processes at nice 10); no output was written and research_log could not append its finish line. Rerunning at 160x160 with the same settings.

_commit `92ae8586eebd` (dirty worktree)_

## 2026-10-07 21:27 UTC: run (exploratory)

- **Question:** Does the minimal two-parameter factorization model show a Neimark-Sacker bifurcation of the edge-of-stability 2-cycle? (rerun of the timed-out scan at lower resolution)
- **Setup:** L = (uv-1)^2 + 0.05(u^2+v^2); log10 rates -0.6..0.2 for both, 160x160 map, 20,000 updates, EXP-028-style screening; period-2 continuation along the u-rate axis at 24 v-rate values, bisection 1e-11
- **Output:** `../neural-shrimp-data/explore/a3-uv-model`
- **Would surprise us:** Not expected either way: a 2-D gradient map can lose the 2-cycle by flip, fold or Neimark-Sacker
- **Command:** `.venv/bin/python tools/explore_a3_uv_model.py --resolution 160`

_commit `0d8f714583ba` (dirty worktree)_

- run started 2026-10-07 21:27 UTC finished 21:30 UTC with exit status 0

## 2026-10-07 21:30 UTC: A3 exploratory: the two-parameter factorization model follows the period-doubling route, not the quasi-periodic one

L = (uv-1)^2 + 0.05(u^2+v^2) with separate rates, 160x160 log-rate map (20,000 updates): fixed 6,836 cells, period 2 6,875, period 4 1,190, period 8 195, period 16 86, odd windows (3: 159, 5: 5, 7: 8) and chaos 5,983; no unresolved or drift cells, i.e. no quasi-periodic training anywhere in the plane. Continuing the period-2 orbit along the u-rate axis at 24 v-rate values, it loses stability by flip every time (multiplier -1 to 1e-7). Interpretation (exploratory): with two parameters the dynamics near the balanced minimum reduce to an effectively one-dimensional cubic map (Feigenbaum route, odd windows, as in one-dimensional reductions of edge-of-stability dynamics), whereas the 13- and 72-parameter networks show the Neimark-Sacker route. Next candidate minimal model for the quasi-periodic route: a two-mode depth-2 linear network (the setting where Hofmann 2026 reports complex multipliers), to test for invariant circles and tongues.

_commit `63cfde7454b8` (dirty worktree)_

## 2026-10-07 21:31 UTC: A2 speckle follow-up: with equal horizons, S2 orbit-point starts recover 58% (shared start 55%)

Same 2,224 sampled S2 cells, orbit-point starts with 80,000 updates: period p recovered in 1,289 (58.0%), against 987 (44.4%) with 20,000 updates and 1,214 (54.6%) for 80,000 updates from the shared start. So the earlier S2 orbit-start deficit was mostly the shorter horizon: S2 speckle is dominated by long chaotic transients (about 42% of sampled cells are still not period p after 80,000 updates from either start). Window-specific exceptions remain where the window-center orbit point rarely returns period p but the shared start does (zoom:230 1/120 vs 83/120; zoom:363 14 vs 88; zoom:455 18 vs 98; zoom:587 5/32; zoom:411 4/95), consistent with the center orbit point lying outside the basin of the continued orbit at distant cells. The pre-stated surprise (orbit starts under half) did not occur overall but did for these windows. Conclusion for figures and criteria: fixed-start 20,000-update maps understate S2 windows (transients) and S1R windows (coexistence); orbit-following maps from continued orbits are needed for boundary figures.

_commit `797b62a677ec`_

- run started 2026-10-07 20:52 UTC finished 21:43 UTC with exit status 0

## 2026-10-07 21:43 UTC: run (exploratory)

- **Question:** Analyze the 256x256 S2-28011 rotation map: continuity with the Neimark-Sacker birth angle and m/q locking of even-period cells
- **Setup:** tools/analyze_a3_rotation_map.py on a3-rotation-map/s2-28011 with the traced curve a3-ns-curve/s2-28011; offsets 0.004, 0.008, 0.016 in log10 hidden beyond the curve; FTLE threshold 5e-3
- **Output:** `../neural-shrimp-data/explore/a3-rotation-map/s2-28011/analysis.json`
- **Would surprise us:** Median |rho - birth angle| above 0.01 at the smallest offset; fewer than 80% of even-period cells locked at m/q with q = period/2
- **Command:** `.venv/bin/python tools/analyze_a3_rotation_map.py --map ../neural-shrimp-data/explore/a3-rotation-map/s2-28011/rotation.npz --ns ../neural-shrimp-data/explore/a3-ns-curve/s2-28011/ns-curve.json --out ../neural-shrimp-data/explore/a3-rotation-map/s2-28011`

_commit `753a7e6ce578` (dirty worktree)_

- run started 2026-10-07 21:43 UTC finished 21:43 UTC with exit status 0

## 2026-10-07 21:44 UTC: run (exploratory)

- **Question:** Rerun the locking test with the corrected tolerance and known-period denominators
- **Setup:** Same 256x256 map and curve; tolerance 1/samples; q = p/2 or p/2^(k+1)
- **Output:** `../neural-shrimp-data/explore/a3-rotation-map/s2-28011/analysis.json`
- **Would surprise us:** Fewer than 80% of even-period cells locked at m/q (directly or period-doubled)
- **Command:** `.venv/bin/python tools/analyze_a3_rotation_map.py --map ../neural-shrimp-data/explore/a3-rotation-map/s2-28011/rotation.npz --ns ../neural-shrimp-data/explore/a3-ns-curve/s2-28011/ns-curve.json --out ../neural-shrimp-data/explore/a3-rotation-map/s2-28011`

_commit `208f8bd1829e` (dirty worktree)_

- run started 2026-10-07 21:44 UTC finished 21:44 UTC with exit status 0
## 2026-10-07 21:47 UTC: Manuscript draft paper/gd-route first compiles (exploratory and frozen results labeled)

New working draft (branch claude/paper-gd-route): 14 pages, six figures generated from saved evidence with per-figure receipts and byte-identical --verify, 80 generated number macros with sources and hashes in paper/gd-route/data/numbers.json, and scripts/verify.py passing. Section 4 carries the prospectively frozen, audited map study; Section 5 the exploratory route, with placeholders for the rotation-number map, real-data network, minimal two-parameter model and consequence streams. Building the numbers surfaced a discrepancy: the raw random-chaos diagnostics and docs/EXP-028-summary.json record 157 of 160 reruns confirmed, while docs/EXP-028-results.md says 155 of 160 (the draft quotes 157 from the receipt). The Farey-neighbour pattern of staircase plateaus is weak evidence in S2 (3 of 3 pairs against 49% per pair for random fractions) and uninformative in S1R (null 100%, every small-denominator fraction in range appears). All citations remain unverified at primary sources.

_commit `57264965e3b5`_

## 2026-10-07 22:01 UTC: run (exploratory)

- **Question:** Do locked m/q regions approach their predicted roots on the Neimark-Sacker curve (256x256 map)?
- **Setup:** tools/analyze_a3_tongue_roots.py on a3-rotation-map/s2-28011 with a3-ns-curve/s2-28011; q <= 13; locked = period 2q and |rho - m/q| <= 1/samples; null = 2,000 random points along the traced curve
- **Output:** `../neural-shrimp-data/explore/a3-rotation-map/s2-28011/tongue-roots.json`
- **Would surprise us:** No rational with locked cells near its root
- **Command:** `.venv/bin/python tools/analyze_a3_tongue_roots.py --map ../neural-shrimp-data/explore/a3-rotation-map/s2-28011/rotation.npz --ns ../neural-shrimp-data/explore/a3-ns-curve/s2-28011/ns-curve.json --out ../neural-shrimp-data/explore/a3-rotation-map/s2-28011`

_commit `012bc873cc8e`_

- run started 2026-10-07 22:01 UTC finished 22:01 UTC with exit status 0

## 2026-10-07 22:01 UTC: A3 exploratory: tongue-root test at 256x256 is mixed; tips unresolved and the curve trace is incomplete

1/5 (period 10): locked cells 3.2 map cells from both predicted roots (random curve points: median 31-33 cells, one-sided p about 0.045 each); 2/11 (period 22): 2.1 cells from the right-branch root (p = 0.007) but 131.5 cells from the left-branch root (p = 0.64); 2/9 (period 18; only the left-branch root lies in the traced range) 217 cells (p = 0.90); 3/13 (2 locked cells) 224 cells (p = 0.94). Two limitations explain the misses without testing the hypothesis: tongues narrow like distance^((q-2)/2) toward their roots, so 256x256 cannot resolve tips of q >= 9 tongues; and the traced curve ends at log10 output -0.08, before the right-branch 2/9 root (birth rotation 0.2211 at the trace end), where the known 2/9 window's tongue should start. Next: extend the curve trace and map high-resolution zooms around each predicted root on the GPU.

_commit `012bc873cc8e` (dirty worktree)_

## 2026-10-07 22:01 UTC: run (exploratory)

- **Question:** Extend the Neimark-Sacker curve of the period-2 orbit along the right branch (S2-28011) to locate the 2/9 and 1/4 roots
- **Setup:** start log10 hidden 1.65, 30 output rates log10 -0.08..0.25, continuation to stability loss, bisection 1e-10, 6 workers
- **Output:** `../neural-shrimp-data/explore/a3-ns-curve/s2-28011-right`
- **Would surprise us:** The right branch turning back or ending in a flip/fold before the birth rotation reaches 2/9
- **Command:** `.venv/bin/python tools/explore_a3_ns_curve.py --system S2 --seed 28011 --start 1.65 --output -0.08 0.25 --count 30 --workers 6 --out ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-right`

_commit `b7b4a941f91c`_

- run started 2026-10-07 22:01 UTC finished 22:01 UTC with exit status 0

## 2026-10-07 22:01 UTC: run (exploratory)

- **Question:** Continue the right branch of the Neimark-Sacker curve beyond log10 output -0.035 from a lower start (hidden 1.58)
- **Setup:** S2-28011, start log10 hidden 1.58, 20 output rates log10 -0.05..0.25, bisection 1e-10, 6 workers
- **Output:** `../neural-shrimp-data/explore/a3-ns-curve/s2-28011-right-1.58`
- **Would surprise us:** No period-2 start found at this hidden rate
- **Command:** `.venv/bin/python tools/explore_a3_ns_curve.py --system S2 --seed 28011 --start 1.58 --output -0.05 0.25 --count 20 --workers 6 --out ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-right-1.58`

_commit `b7b4a941f91c` (dirty worktree)_

- run started 2026-10-07 22:01 UTC finished 22:01 UTC with exit status 0

## 2026-10-07 22:01 UTC: run (exploratory)

- **Question:** Continue the right branch of the Neimark-Sacker curve beyond log10 output -0.035 from a lower start (hidden 1.50)
- **Setup:** S2-28011, start log10 hidden 1.50, 20 output rates log10 -0.05..0.25, bisection 1e-10, 6 workers
- **Output:** `../neural-shrimp-data/explore/a3-ns-curve/s2-28011-right-1.50`
- **Would surprise us:** No period-2 start found at this hidden rate
- **Command:** `.venv/bin/python tools/explore_a3_ns_curve.py --system S2 --seed 28011 --start 1.50 --output -0.05 0.25 --count 20 --workers 6 --out ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-right-1.50`

_commit `b7b4a941f91c` (dirty worktree)_

- run started 2026-10-07 22:01 UTC finished 22:01 UTC with exit status 0

## 2026-10-07 22:02 UTC: run (exploratory)

- **Question:** Densify the Neimark-Sacker curve around its local maximum of birth rotation (log10 output about -0.07) to locate the 2/9 roots relative to the known 2/9 window
- **Setup:** S2-28011, start hidden 1.65, 24 output rates log10 -0.095..-0.050, bisection 1e-10
- **Output:** `../neural-shrimp-data/explore/a3-ns-curve/s2-28011-max`
- **Would surprise us:** Birth rotation never reaching 2/9 near the maximum
- **Command:** `.venv/bin/python tools/explore_a3_ns_curve.py --system S2 --seed 28011 --start 1.65 --output -0.095 -0.050 --count 24 --workers 6 --out ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-max`

_commit `b7b4a941f91c` (dirty worktree)_

- run started 2026-10-07 22:02 UTC finished 22:02 UTC with exit status 0

## 2026-10-07 22:02 UTC: A3 exploratory: the Neimark-Sacker curve is W-shaped; birth-rotation extrema sit just beyond 2/9 and 1/6, next to the largest windows

Extending the S2-28011 trace (start hidden 1.65, 1.58 and 1.50): after the left U (minimum hidden 1.650 near log10 output -0.18; birth rotation minimum 0.16699 near -0.25), the curve rises to a local maximum of birth rotation 0.22281 near output -0.0726 (hidden 1.744), dips to hidden 1.652 and birth 0.2104 near -0.003, then climbs steeply (output 0.06: hidden 1.881, birth 0.2987); for output above 0.076 the start diverges. All boundaries are Neimark-Sacker. Birth rotation exceeds 2/9 only for output -0.0772 to -0.0679, giving two 2/9 roots 0.009 apart just right of the wide 2/9 window (center column -0.0856, hidden 1.833-1.879); its minimum on the left U (0.16699) lies just above 1/6, near which period-12 (1/6) cells are abundant (1,151 in the 256x256 map) and window S2-28011:209 is 1/6. Speculative mechanism for the prominence of these windows: a near-tangency of the birth rotation with a low-order rational (pairs of nearby roots, possibly merging tongues). Test planned: GPU rotation zoom from the 2/9 roots up to the window.

_commit `b7b4a941f91c` (dirty worktree)_
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

- run started 2026-10-07 21:44 UTC finished 21:52 UTC with exit status 0

## 2026-10-07 21:53 UTC: run (exploratory)

- **Question:** Pooled S2 analysis: with two slices whose rotation-rate relations differ (output -0.0856 and -0.30), do rotation and locking features add predictive value beyond FTLE classes, sharpness, rate, amplitude and slice intercepts?
- **Setup:** tools/analyze_consequence.py --pool on s2-28011-out-0.0856 and s2-28011-out-0.30 (Neimark-Sacker points 1.738439 and 1.6854 interpolated from the traced curve); 300 block-bootstrap draws; CV blocks within slices
- **Output:** `/data2/neural-shrimp-data/explore/consequence/analysis-s2-pooled.json`
- **Would surprise us:** Pooled delta R2 intervals clearly above zero for any magnitude outcome
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/analyze_consequence.py s2-28011-out-0.0856 s2-28011-out-0.30 --ns 1.738439 1.6854 --pool --report analysis-s2-pooled.json --draws 300`

_commit `e9cb2cffe905` (dirty worktree)_

- run started 2026-10-07 21:53 UTC finished 21:55 UTC with exit status 0

## 2026-10-07 21:56 UTC: run (exploratory)

- **Question:** Pooled S2 robustness (adapted after seeing the first pooled fit, where rotation features gained delta R2 0.03-0.15 with intervals above zero): do the gains survive slice-specific slopes for rate, amplitude and sharpness, and a transient-amplification covariate for reproducibility?
- **Setup:** tools/analyze_consequence.py --pool on the two S2 slices, variants beyond_eos_slice_slopes and beyond_eos_slice_slopes_transient; 300 draws
- **Output:** `/data2/neural-shrimp-data/explore/consequence/analysis-s2-pooled-robust.json`
- **Would surprise us:** Gains surviving both checks with intervals above zero
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/analyze_consequence.py s2-28011-out-0.0856 s2-28011-out-0.30 --ns 1.738439 1.6854 --pool --report analysis-s2-pooled-robust.json --draws 300`

_commit `36814781fc5a` (dirty worktree)_

- run started 2026-10-07 21:56 UTC finished 21:58 UTC with exit status 0

## 2026-10-07 21:59 UTC: A3b exploratory: second S2 slice and pooled models — rotation/locking adds little beyond generic features; two locking effects survive robustness checks

S2-28011 at output -0.30 (600 cells, 1.58-2.05): period-2 132 (to 1.6828; traced Neimark-Sacker 1.6854), quasi-periodic 33 (narrow band from 1.684, rotation from about 0.170), locked 75 (1/5, 1/6, 2/13, 1/7, 3/20, 1/10, 1/11, 1/12, 1/18, 1/20; 75/75 with period 2q x 2^k; loss peak equals rotation in 69/75 locked and 33/33 quasi-periodic cells), other periodic 19, chaotic 338. Here most locked windows sit inside chaos and slip phases (median disagreement -1.0). Pooling both S2 slices (n = 1,329) with slice intercepts gave delta R2 +0.03 to +0.15 with intervals above zero for several outcomes; after adding slice-specific slopes for rate, amplitude and sharpness (robustness check added after seeing that fit) and transient amplification for reproducibility, the gains vanish or become negligible for disagreement at 40k (+0.008 [-0.002, +0.024]), disagreement after anneal (+0.002 [-0.004, +0.018]), training and held-out losses (<= +0.008, R2 already 0.97-0.996) and annealed losses (+0.012 to +0.034, intervals spanning zero). Two gains survive: the shape of the reproducibility dose-response (slope of final disagreement vs perturbation size, +0.18 [+0.07, +0.36]; locked partial r +0.44), reflecting perturbation-size thresholds for phase slips in windows embedded in chaos, and stopping-time spread (+0.047 [+0.011, +0.099]; locked partial r +0.48), reflecting finite phase sets of locked attractors. Both are locking/discreteness effects, partly confounded with distance to chaos, not rotation-number effects; exploratory.

_commit `36814781fc5a` (dirty worktree)_

## 2026-10-07 22:01 UTC: A3b exploratory: first quasi-periodic rotation number equals the Neimark-Sacker birth angle in all three slices

Rotation number of the first quasi-periodic cell vs the period-2 orbit's two-step multiplier angle at the Neimark-Sacker point: S2 output -0.0856: 0.21807 vs 0.21816; S2 output -0.30: 0.16976 vs 0.17009 (angle interpolated from the traced curve); S1R output 0.9413: 0.13217 vs 0.13218. Stream summary written to docs/explore-consequence-2026-10-07.md.

_commit `73e55461d873` (dirty worktree)_

## 2026-10-07 22:28 UTC: Correction: the 91% m/q-locking statistic of the rotation map is largely definitional

The 256x256 rotation-map analysis reported that 91% of even-period periodic cells are m/q-locked (74% with q = period/2, 17% period-doubled inside a tongue). As the references consult (and the roadmap consult's anticipated referee attack) points out, this is largely true by construction: any period-2q orbit sampled by the two-step map visits q points, so its accumulated principal-plane angle over full cycles is an integer number m of turns and the estimate equals m/q within 1/S automatically. The only non-trivial content is that m is non-zero and coprime to q (the orbit winds primitively about its centroid), and for the doubled class that the base level does. The statistic is withdrawn as evidence of locking. Meaningful locking evidence remains: the Birkhoff well-ordering test against its random-angle null (35 of 56 testable confirmed windows vs 3.58), continuity of rotation between quasi-periodic cells and plateaus along slices, and the resonance-ancestry test now running. The value was quoted to the human on 7 October as evidence; this note corrects it.

_commit `547f6ab16875`_

- run started 2026-10-07 21:22 UTC finished 22:37 UTC with exit status 0

## 2026-10-07 22:37 UTC: run (exploratory)

- **Question:** 1024x1024 GPU rotation map: continuity with the birth angle along the merged Neimark-Sacker curve, and tongue roots at higher resolution
- **Setup:** analyze_a3_rotation_map.py and analyze_a3_tongue_roots.py on a3-gpu-rotation/s2-28011-1024 with the merged curve (69 points, log10 output -0.45 to 0.06); the locking count is definitional except for primitive winding and is not used as evidence
- **Output:** `../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-1024`
- **Would surprise us:** Median |rho - birth angle| above 0.005 at the smallest offset; no rational with locked cells within a few cells of its root
- **Command:** `bash -c .venv/bin/python tools/analyze_a3_rotation_map.py --map ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-1024/rotation.npz --ns ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-merged/ns-curve.json --offsets 0.002 0.004 0.008 0.016 --out ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-1024 && .venv/bin/python tools/analyze_a3_tongue_roots.py --map ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-1024/rotation.npz --ns ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-merged/ns-curve.json --out ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-1024`

_commit `2e15406746b9` (dirty worktree)_

- run started 2026-10-07 22:37 UTC finished 22:37 UTC with exit status 0

## 2026-10-07 22:38 UTC: A3 exploratory: 1024x1024 GPU rotation map; continuity to 6e-4 and roots near the birth-rotation maximum supported

S2-28011, log10 hidden 1.60-2.06 x output -0.46 to -0.07 (4,503 s on the RTX 2000 Ada; the rotation stage was host-bound). Continuity with the merged Neimark-Sacker curve: at 0.002 decades beyond it, median |rho - arg(mu)/2pi| = 0.00058 over 39 curve points (0.0008 at 0.004, 0.0018 at 0.008, 0.0035 at 0.016: smooth finite-detuning drift). Tongue-root test (locked = period 2q with rho within 1/S of m/q; the count itself is definitional except primitivity and is not used as evidence): both 1/5 roots have locked cells within 0.4 and 3.0 cells (one-sided null p 0.001 and 0.004 against random curve points); the two 2/9 roots at the birth-rotation maximum 0.7 and 24.5 cells (p 0.001, 0.029); one 3/13 root 0.7 cells (p 0.001); one 2/11 root 8.1 cells (p 0.007). Left-branch roots of 2/9, 2/11 and 3/13 have no locked cells nearby (p 0.63-0.95); two roots (1/4 and the far-right 2/9) lie outside the mapped output range, a bounds check the tool lacks. The map shows a smooth rotation field rising from 0.16 to 0.22 above the right half of the U, a thin band on the left branch, and long low-rotation stripes (about 0.15-0.17, colormap clipped below 0.15) running into the chaotic region. Figure: a3-gpu-rotation/s2-28011-1024/rotation-map.png. Map-based evidence remains secondary to the continuation-based ancestry test.

_commit `2e15406746b9` (dirty worktree)_

## 2026-10-07 22:39 UTC: run (exploratory)

- **Question:** Complete the W: rotation-number map over the right half of the Neimark-Sacker curve (2/9 roots at the birth-rotation maximum, the second dip, and the steep right branch with 1/4, 3/11, 2/7 roots)
- **Setup:** GPU, S2-28011, log10 hidden 1.60-2.06 x output -0.10..0.08, 1024 x 400, EXP-028 screening then 256 + 6,000 two-step samples (torch bookkeeping)
- **Output:** `../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-right`
- **Would surprise us:** No locked regions touching the curve at the right-branch roots; rotation just beyond the curve not continuous with arg(mu)/2pi
- **Command:** `.venv/bin/python tools/explore_gpu_rotation_map.py --name a3-gpu-rotation/s2-28011-right --system S2 --seed 28011 --hidden 1.60 2.06 --output -0.10 0.08 --resolution 1024 400 --library ../neural-shrimp-data/builds/gdmap-cuda-64af0462/gdmap_cuda.so`

_commit `b30a116919c5`_

- run started 2026-10-07 22:39 UTC finished 22:45 UTC with exit status 1

## 2026-10-07 22:45 UTC: Informative failure: right-W rotation map died in torch batched eigh

The 1024 x 400 GPU rotation map (run card above) finished screening (300 s) and then failed: torch.linalg.eigh on 409,600 13x13 covariance blocks requested a 232 GB cuSOLVER workspace. The decomposition now runs on the host (numpy), with planes returned to the GPU; the 40-cell test slice still reproduces the CPU rotation numbers. Rerunning.

_commit `b3845d3b98dd` (dirty worktree)_

## 2026-10-07 22:45 UTC: run (exploratory)

- **Question:** Complete the W (rerun after the eigh fix): rotation-number map over the right half of the Neimark-Sacker curve
- **Setup:** GPU, S2-28011, log10 hidden 1.60-2.06 x output -0.10..0.08, 1024 x 400, EXP-028 screening then 256 + 6,000 two-step samples (torch bookkeeping, host eigh)
- **Output:** `../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-right`
- **Would surprise us:** No locked regions touching the curve at the right-branch roots; rotation just beyond the curve not continuous with arg(mu)/2pi
- **Command:** `.venv/bin/python tools/explore_gpu_rotation_map.py --name a3-gpu-rotation/s2-28011-right --system S2 --seed 28011 --hidden 1.60 2.06 --output -0.10 0.08 --resolution 1024 400 --library ../neural-shrimp-data/builds/gdmap-cuda-64af0462/gdmap_cuda.so`

_commit `9bfb18c079ec`_
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
## 2026-10-07 22:26 UTC: run (exploratory)

- **Question:** Controls for the minimal-model scan: does the pipeline type an engineered gradient-map Neimark-Sacker correctly and find a supercritical setting; does the two-parameter factorization model's 2-cycle ever have a complex dominant pair (D < 0)?
- **Setup:** Engineered planar gradient map with prescribed 2-cycle and two-step derivative r R_phi (phi/2pi = 0.13), r from 0.95, criticality probes at r = 1.002 and 1.008 for kappa in {-1, -0.2, 0, 0.2, 1}; factorization model (uv, ridge 0.1) 2-cycle continued along the u-rate axis at 8 v-rates, D = (tr M)^2 - 4 det M along each branch
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/controls.json`
- **Would surprise us:** Engineered crossing not typed Neimark-Sacker at r = 1 with theta/2pi = 0.13; any D < 0 on the factorization branches
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_model.py controls`

_commit `93bb78c7a8e4`_

- run started 2026-10-07 22:26 UTC finished 22:27 UTC with exit status 0

## 2026-10-07 22:27 UTC: run (exploratory)

- **Question:** Rerun the minimal-model controls with the largest complex-multiplier modulus recorded on the factorization branches
- **Setup:** Same controls as the previous card, plus max |mu| while D < 0 on each factorization branch
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/controls.json`
- **Would surprise us:** A factorization branch whose complex pair reaches the unit circle (would be a Neimark-Sacker missed by ray typing)
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_model.py controls`

_commit `0630f7d06fcf` (dirty worktree)_

- run started 2026-10-07 22:27 UTC finished 22:29 UTC with exit status 0

## 2026-10-07 22:29 UTC: run (exploratory)

- **Question:** In the coupled four-weight model, how does the edge-of-stability 2-cycle lose stability across mode coupling c, ridge and rate ratio; does any case reach Neimark-Sacker?
- **Setup:** L = 1/2 z^T C z - d^T z + lam/2 |theta|^2, z = (a1 b1, a2 b2), d = C z* + lam (z* = (1, 0.5)); c in {0, 0.3, 0.6, 0.9}, lam in {0.05, 0.2}, eta_b/eta_a = 2^(k/2), k = -3..4; start 0.2-10% past the fixed-point flip, follow the 2-cycle up to 10^1.5 x the start rate (h0 0.002 decades, bisection 1e-10)
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/grid.json`
- **Would surprise us:** Any Neimark-Sacker at c = 0 (decoupled modes should reproduce the factorization model's flips)
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_model.py grid --workers 4`

_commit `0630f7d06fcf` (dirty worktree)_

- run started 2026-10-07 22:29 UTC finished 22:29 UTC with exit status 0

## 2026-10-07 22:32 UTC: Minimal model (exploratory): EOS-connected Neimark-Sacker already in the scalar factorization with stronger weight decay and unequal layer rates

Coupled four-weight model (d = C z* + lam, z* = (1, 0.5)), 64 cases from the minimum through the fixed-point flip and along the 2-cycle: 58 end in flip; 6 end in Neimark-Sacker, all at lam = 0.2 with rate ratios 0.354, 2.828 or 4: (c, ratio, theta/2pi) = (0, 0.354, 0.4288), (0, 2.828, 0.4288; the a-b mirror case, identical), (0, 4, 0.3401; 0.007 from the strong 1/3 resonance), (0.3, 4, 0.3834), (0.6, 4, 0.3759), (0.9, 4, 0.4075); crossings transversal (slope of log|mu| per decade of rate 4.3-7.2). Criticality probes 1e-3 and 4e-3 decades past the crossing: bounded attractors near the old orbit with amplitude ratio 2.08 (c = 0) and 2.18 (c = 0.3), close to the sqrt(delta) value 2, rotation 0.425 and 0.381 near the birth angles (supercritical-like); c = 0.6 and 0.9 leave the neighbourhood at the larger offset. The pre-stated surprise occurred: Neimark-Sacker at c = 0, where the modes decouple into two independent scalar factorizations, so the two-weight model 0.5 (uv - d)^2 + 0.5 lam (u^2 + v^2) with lam = 0.2 and a rate ratio about 2.8 already has an EOS-born 2-cycle with a supercritical Neimark-Sacker; at lam = 0.05 (the earlier uv control's effective ridge) all cases flip. Controls: engineered planar gradient 2-cycle typed Neimark-Sacker at r = 1 with theta/2pi = 0.13 and supercritical for all kappa tested (amplitude ratio 1.6-2.0); the earlier factorization model (ridge 0.1) has complex 2-cycle multipliers on 5 of 8 branches but with modulus at most 0.65, ending in flips. Coupling c is not needed; weight decay strength and rate asymmetry are.

_commit `0a46892663b4`_

## 2026-10-07 22:34 UTC: run (exploratory)

- **Question:** Hofmann rotational edge, task 1: does the aligned 2-cycle of the 2x2 depth-two linear network lose stability at a(s1+s2) = 2 through a transversal complex-pair crossing, and what are the other multipliers?
- **Setup:** l = 0.5 ||W2 W1 - diag(1, s2)||^2, no weight decay, single rate; exact two-step derivative at the closed-form aligned cycle; bisection on the complex pair modulus for s2/s1 in 0.7, 0.8, 0.9
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/hofmann/edge.json`
- **Would surprise us:** Crossing away from a(s1+s2) = 2, a real crossing, or another multiplier above 1 before the edge
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_hofmann.py edge`

_commit `4958e6b9b3ed`_

- run started 2026-10-07 22:34 UTC finished 22:34 UTC with exit status 0

## 2026-10-07 22:34 UTC: run (exploratory)

- **Question:** Hofmann rotational edge, task 2: is the post-edge misaligned oscillatory state an attracting invariant circle of the two-step map with rotation continuous from the birth angle, and amplitude growing like sqrt(delta)?
- **Setup:** No weight decay; a = edge (1 + delta), delta in 1e-3..1.6e-2; start at the aligned cycle plus 1e-3 noise; 200,000-update transient, 4,000 two-step samples of rotation-invariant observables (W2 W1, W1^T W1, W2 W2^T); top 3 Lyapunov exponents (one is the exact zero of the hidden-rotation symmetry)
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/hofmann/circle-lam0.0.json`
- **Would surprise us:** Chaotic or divergent state just past the edge; rotation not approaching the birth angle; amplitude not scaling like sqrt(delta)
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_hofmann.py circle`

_commit `4958e6b9b3ed` (dirty worktree)_

- run started 2026-10-07 22:34 UTC finished 22:34 UTC with exit status 0

## 2026-10-07 22:36 UTC: run (exploratory)

- **Question:** Hofmann edge: is the onset hard (bistable, subcritical) or soft? Follow the misaligned state from just past the edge to just before it, and measure small-delta amplitude scaling
- **Setup:** No weight decay; sequential attractor following through delta = +2e-3, +1e-3, +3e-4, -3e-4, -1e-3, -2e-3, -4e-3 (relative to the edge rate), 400,000-update transient per rate, 4,000 two-step samples
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/hofmann/circle-lam0.0-hysteresis.json`
- **Would surprise us:** None either way; a misaligned state persisting before the edge means a hard onset
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_hofmann.py circle --sequential --deltas 2e-3 1e-3 3e-4 -3e-4 -1e-3 -2e-3 -4e-3 --transient 400000 --tag hysteresis`

_commit `8e9054adc89c` (dirty worktree)_

- run started 2026-10-07 22:36 UTC finished 22:36 UTC with exit status 2

## 2026-10-07 22:36 UTC: run (exploratory)

- **Question:** Hofmann edge: is the onset hard (bistable, subcritical) or soft? Follow the misaligned state from just past the edge to just before it (rerun with decimal deltas after an argument-parsing failure)
- **Setup:** No weight decay; sequential attractor following through delta = +0.002, +0.001, +0.0003, -0.0003, -0.001, -0.002, -0.004 (relative to the edge rate), 400,000-update transient per rate, 4,000 two-step samples
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/hofmann/circle-lam0.0-hysteresis.json`
- **Would surprise us:** None either way; a misaligned state persisting before the edge means a hard onset
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_hofmann.py circle --sequential --deltas 0.002 0.001 0.0003 -0.0003 -0.001 -0.002 -0.004 --transient 400000 --tag hysteresis`

_commit `8e9054adc89c` (dirty worktree)_

- run started 2026-10-07 22:36 UTC finished 22:38 UTC with exit status 0

## 2026-10-07 22:38 UTC: run (exploratory)

- **Question:** With weight decay (lam = 0.01, removing the neutral mode-2 continuum), does the post-edge circle amplitude scale like sqrt(delta) (generic Neimark-Sacker) instead of the delta^(1/4) seen at lam = 0, and is the onset still soft?
- **Setup:** lam = 0.01, edge located by bisection on the exact aligned-cycle multipliers; sequential following through delta = +0.003, +0.001, +0.0003, +0.0001, -0.0003, -0.001; 400,000-update transient, 4,000 two-step samples
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/hofmann/circle-lam0.01-hysteresis.json`
- **Would surprise us:** Amplitude still scaling like delta^(1/4), or a hard onset
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_hofmann.py circle --lam 0.01 --sequential --deltas 0.003 0.001 0.0003 0.0001 -0.0003 -0.001 --transient 400000 --tag hysteresis`

_commit `8e9054adc89c` (dirty worktree)_

- run started 2026-10-07 22:38 UTC finished 22:40 UTC with exit status 0

## 2026-10-07 22:40 UTC: run (exploratory)

- **Question:** Hofmann model, task 3: beyond the rotational edge, does the rotation number form a staircase with locked windows of period 2q at rotation m/q (lam = 0.01)?
- **Setup:** lam = 0.01; 400 rates from 1.0005 x edge to the end of the 2-cycle window, each from the aligned cycle plus 1e-3 noise; 200,000-update transient, 4,000 two-step samples of rotation-invariant observables; period from two-step recurrence (tolerance 1e-7); top 3 Lyapunov exponents; s2/s1 in 0.7, 0.8, 0.9
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/hofmann/staircase-lam0.01.json`
- **Would surprise us:** No periodic windows at all, or periodic windows whose rotation is not m/q with q = period/2
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_hofmann.py staircase --lam 0.01`

_commit `8e9054adc89c` (dirty worktree)_

- run started 2026-10-07 22:40 UTC finished 22:42 UTC with exit status 0

## 2026-10-07 22:43 UTC: Hofmann rotational edge (exploratory): a supercritical Neimark-Sacker of the edge-of-stability 2-cycle once weight decay removes a neutral continuum

2x2 depth-two linear network, l = 0.5 ||W2 W1 - diag(1, s2)||^2, single rate a. (1) Exact multipliers of the closed-form aligned 2-cycle: for s2/s1 = 0.7, 0.8, 0.9 the misalignment complex pair crosses the unit circle at a(s1 + s2) = 2 to machine precision, transversally (d|mu|/da = 1.85, 1.90, 1.95, empirically (3 + s2/s1)/2), with birth angles theta/2pi = 0.1380, 0.1082, 0.0737 (far from 1/4, 1/3, 1/2); the other multipliers are two exact 1s (hidden-rotation symmetry; neutral continuum of second-mode minima uv = s2 without weight decay) and the rest below 0.9. (2) Past the edge (no weight decay) training settles on a bounded, non-periodic state with nonzero loss and two near-zero Lyapunov exponents (symmetry plus circle) and a negative third: an attracting invariant circle of the two-step map in rotation-invariant coordinates; its rotation tends to the birth angle (0.13744, 0.10753, 0.07287 at delta = 3e-4 vs 0.13800, 0.10817, 0.07368); the onset is soft (following the state back below the edge returns the aligned 2-cycle, no bistability) but the amplitude scales like delta^0.24, a degenerate (quartic-amplitude) onset. With weight decay lam = 0.01 (edge shifts to a = 1.1908, 1.1239, 1.0641; only the symmetry multiplier stays at 1) the amplitude scales like delta^0.49 (ratios 1.71-1.78 per tripling, sqrt(3) = 1.73), the rotation at delta = 1e-4 is within 3e-5 of the birth angle, and the onset is soft: a generic supercritical Neimark-Sacker modulo the rotation symmetry. At lam = 0, s2/s1 = 0.7 and delta = 1.6e-2 the state already has a positive exponent (0.017).

_commit `8e9054adc89c` (dirty worktree)_

## 2026-10-07 22:44 UTC: run (exploratory)

- **Question:** Hofmann model: is there a resolvable 1/10 tongue (period 20) at s2/s1 = 0.8, where the coarse scan passed rotation 0.1 without a visible plateau?
- **Setup:** lam = 0.01, s2/s1 = 0.8, 400 rates in a = 1.136..1.148 (spacing 3e-5), same settings as the coarse staircase
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/hofmann/staircase-lam0.01-fine08.json`
- **Would surprise us:** Rotation crossing 0.1 with no locked cells even at this resolution
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_hofmann.py staircase --lam 0.01 --ratios 0.8 --a-range 1.136 1.148 --tag fine08`

_commit `02ddf7ffc57a`_

- run started 2026-10-07 22:44 UTC finished 22:45 UTC with exit status 0

## 2026-10-07 22:45 UTC: Hofmann rotational edge (exploratory), task 3: phase locking beyond the edge

lam = 0.01, 400 rates from 1.0005 x edge to the end of the 2-cycle window per spectral ratio, each from the aligned cycle plus 1e-3 noise. s2/s1 = 0.7: rotation falls monotonically from the birth angle 0.1388 to 0.1154 with a plateau at exactly 1/8 (27 cells of period 16, a = 1.2146-1.2183) and isolated locked cells at 2/17 (period 34, 2 cells) and 5/42 (period 84, 1 cell); all 30 periodic cells have rotation m/q with q = period/2, and 5/42 is the Farey mediant of 2/17 and 3/25, between the Farey neighbours 2/17 and 1/8. No chaotic cells. s2/s1 = 0.8 and 0.9: rotation falls smoothly from the birth angles (0.1086 to 0.079; 0.0737 to 0.044) with no plateau resolved at 400 cells; a fine scan at 0.8 (spacing 3e-5) resolves a 1/10 tongue (6 cells of period 20, a = 1.14093-1.14108, about 25 times narrower than the 1/8 tongue at 0.7). From a = 1.197 (0.8) and 1.108 (0.9) blocks of starts diverge, interleaved with bounded circle states (coexistence with escape; not analysed further). Interpretation: Hofmann's rotational edge is the Neimark-Sacker onset of the same quasi-periodic route seen in the trained networks, with Arnold tongues beyond it, in an eight-weight linear model.

_commit `02ddf7ffc57a` (dirty worktree)_

## 2026-10-07 22:45 UTC: run (exploratory)

- **Question:** Do the frozen predictions (30 flip, 2 Neimark-Sacker) hold for the held-out couplings c = 0.45 and 0.75?
- **Setup:** Same pipeline as the grid; 32 held-out cases; predictions committed before this run (docs/explore-minimal-model-heldout-predictions.json)
- **Output:** `/data2/neural-shrimp-data/explore/minimal-model/heldout.json`
- **Would surprise us:** Fewer than 28 of 32 types predicted correctly
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_minimal_model.py heldout --workers 4`

_commit `44bc54b0d5d3`_

- run started 2026-10-07 22:45 UTC finished 22:45 UTC with exit status 0

## 2026-10-07 22:56 UTC: Synthesis: the Neimark-Sacker onset of quasi-periodic training is not specific to the two toy networks

Three independent exploratory streams (merged 7 October): (1) real data, single global learning rate (claude/explore-realdata): 64-8-1 tanh MLP, standard parameterization, 1,024 UCI digit images, weight decay 1e-3, float64 full-batch GD; the Newton-certified edge-of-stability 2-cycle loses stability at log10 eta = -0.3885464 with multiplier 0.998814 + 0.048693i; predicted rotation 0.0077528 turns per two updates, measured 0.007751 on a smooth invariant circle (FTLE about 0); one of five period-2 branches examined (others end in folds); no locking resolved (q > 100 at this rotation). (2) Hofmann's two-mode depth-two linear network (claude/explore-minimal-model): the rotational edge a(s1 + s2) = 2 is a transversal complex-pair crossing; beyond it an attracting invariant circle with rotation tending to the birth angle; with weight decay 0.01 the amplitude grows like delta^0.49 (supercritical, soft onset); locked windows 1/8 (period 16), 2/17 (34) and 5/42 (84, the Farey mediant of 2/17 and 3/25) at s2/s1 = 0.7 and a 1/10 tongue at 0.8. (3) Coupled four-weight model: 6 of 64 cases end in Neimark-Sacker (ridge 0.2), including c = 0, i.e. the plain two-weight factorization 1/2 (uv - d)^2 + 0.1 (u^2 + v^2) with rate ratio about 2.8; held-out couplings 32/32 types predicted. Together with S2 and S1R this places the route in nonlinear and linear networks, per-layer and single learning rates, synthetic and real data. All exploratory; ancestry test still running.

_commit `77299eb1e52f` (dirty worktree)_

- run started 2026-10-07 22:45 UTC finished 22:58 UTC with exit status 0

## 2026-10-07 22:58 UTC: run (exploratory)

- **Question:** Right half of the W: continuity with the birth angle and tongue roots (2/9 pair at the maximum, 3/13, 1/4, 3/11, 2/7 on the steep branch)
- **Setup:** analyze_a3_rotation_map.py and analyze_a3_tongue_roots.py (with bounds check) on a3-gpu-rotation/s2-28011-right and the merged curve
- **Output:** `../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-right`
- **Would surprise us:** Median |rho - birth| above 0.005 at the smallest offset; no locked cells near in-range right-branch roots
- **Command:** `bash -c .venv/bin/python tools/analyze_a3_rotation_map.py --map ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-right/rotation.npz --ns ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-merged/ns-curve.json --offsets 0.002 0.004 0.008 0.016 --out ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-right && .venv/bin/python tools/analyze_a3_tongue_roots.py --map ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-right/rotation.npz --ns ../neural-shrimp-data/explore/a3-ns-curve/s2-28011-merged/ns-curve.json --out ../neural-shrimp-data/explore/a3-gpu-rotation/s2-28011-right`

_commit `a694db980dae` (dirty worktree)_

- run started 2026-10-07 22:58 UTC finished 22:58 UTC with exit status 0

## 2026-10-07 22:59 UTC: A3 exploratory: soft and hard edges - along the Neimark-Sacker curve, crossing gives quasi-periodic training in the middle and divergence on the right

Right-W GPU map (1024 x 400, 731 s; continuity: median |rho - arg(mu)/2pi| 0.00036 at 0.002 decades over 31 curve points; tongue roots: both 2/9 roots at the birth-rotation maximum 0.4 and 0.6 cells, p 0.001-0.002; steep-branch 3/13 3.0 cells, p 0.002; 1/4 20.6 cells, p 0.04). Classifying cells at 0.002-0.016 decades beyond each curve point (both maps): output -0.45 to -0.36 mostly chaotic at once (vanishing quasi-periodic band); -0.35 to -0.046 quasi-periodic or slow-drift at every offset (soft onset); -0.034 to +0.06 divergent at every offset (hard loss of stability straight to divergence). Hypothesis: the period-2 Neimark-Sacker is supercritical (safe boundary) on the middle arc and subcritical or immediately escaping (dangerous boundary) on the right arc, with a Chenciner point (first Lyapunov coefficient zero) near output -0.04; on the right arc the trainability (divergence) boundary would coincide with the Neimark-Sacker curve. Caveats: fixed-start labels; divergence could also reflect basin escape from the shared start; criticality not yet computed. Test planned: first Lyapunov coefficient along the curve, bistability and amplitude scaling on both arcs.

_commit `a694db980dae` (dirty worktree)_

## 2026-10-07 23:05 UTC: run (exploratory)

- **Question:** Positive control for the criticality computation. Does the sign of d at r equal 1 for the engineered planar gradient map 2-cycle predict a soft or hard loss of stability just beyond r equal 1, with kappa tuning the cubic terms
- **Setup:** EngineeredTwoCycle with phi 2 pi times 0.13 and kappa in -4 -2 -1 0 1 2 4 8. d from the Kuznetsov formula with finite difference forms of the exact two step JVP. Dynamics at r 1.001 and 1.004 from p0 plus 1e-3, 200000 two step iterations
- **Output:** `../neural-shrimp-data/explore/criticality/control/control.json`
- **Would surprise us:** Sign of d disagreeing with the dynamical outcome for any kappa
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_criticality.py control`

_commit `97d6ba4fea7a`_

## 2026-10-07 23:08 UTC: run (exploratory)

- **Question:** First Lyapunov coefficient d of the period-2 Neimark-Sacker bifurcation along the merged S2-28011 curve. Where is it negative and soft, where positive and hard, and does the sign change near log10 output -0.04 where fixed-start maps switch from quasi-periodic to divergent
- **Setup:** S2-28011, all 69 Neimark-Sacker points of the merged trace. Period-2 orbit re-found 0.02 to 0.2 below the traced boundary, continued to the boundary with bisection 1e-10, d evaluated at the last attracting point from the exact dense two step Jacobian and Richardson finite difference forms, second step size as a check, 8 workers at nice 10
- **Output:** `../neural-shrimp-data/explore/criticality/s2-28011-curve/curve.json`
- **Would surprise us:** d negative on the hard arc or positive on the soft arc, or no sign change between them
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_criticality.py curve --workers 8`

_commit `63eddebb6acb` (dirty worktree)_

- run started 2026-10-07 23:08 UTC finished 23:08 UTC with exit status 1

## 2026-10-07 23:10 UTC: run (exploratory)

- **Question:** First Lyapunov coefficient d of the period-2 Neimark-Sacker bifurcation along the merged S2-28011 curve, rerun after fixing the data path. Where is it negative and soft, where positive and hard, and does the sign change near log10 output -0.04
- **Setup:** S2-28011, all 69 Neimark-Sacker points of the merged trace. Period-2 orbit re-found 0.02 to 0.2 below the traced boundary, continued to the boundary with bisection 1e-10, d at the last attracting point from the exact dense two step Jacobian and Richardson finite difference forms, second step size as a check, 8 workers at nice 10
- **Output:** `/data2/neural-shrimp-data/explore/criticality/s2-28011-curve/curve.json`
- **Would surprise us:** d negative on the hard arc or positive on the soft arc, or no sign change between them
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_criticality.py curve --workers 8`

_commit `a7e3d90de700` (dirty worktree)_

- run started 2026-10-07 23:10 UTC finished 23:11 UTC with exit status 0

## 2026-10-07 23:12 UTC: Criticality exploratory: first Lyapunov coefficient along the S2-28011 Neimark-Sacker curve

d computed at 66 of 69 merged curve points with Kuznetsov's formula (exact two step Jacobian, Richardson finite difference forms; a second step size agrees to all printed digits; eigen-gaps 0.88 to 1.01; no strong resonance). d is negative, supercritical, from log10 output -0.4121 to -0.0026, most negative near -0.062 at -5.08, and positive, subcritical, at -0.45 to -0.4215 and at +0.0132. Two sign changes, generalized Neimark-Sacker or Chenciner points: between -0.4215 and -0.4121 and between -0.0026 and +0.0132, near the bottom of the second dip. The three steep right-branch points found no period-2 start within 0.2 below the curve. Pre-stated surprise partly occurred: on -0.034 to -0.003, where fixed-start maps diverge just beyond the curve, d is clearly negative, -1.81, -1.33, -0.11, so subcriticality does not explain that part of the hard edge. Dynamical checks follow.

_commit `f74ed0e39489` (dirty worktree)_

## 2026-10-07 23:12 UTC: run (exploratory)

- **Question:** Dynamical checks of criticality at eight points of the S2-28011 Neimark-Sacker curve. Beyond the curve, a small circle with amplitude growing like the square root of the offset, or escape. Before the curve, an escape threshold that shrinks toward the curve if subcritical. And an up and down sweep for hysteresis
- **Setup:** Outputs -0.44, -0.30, -0.20, -0.08, -0.034, -0.018, -0.0026, +0.013 from the criticality curve run. Offsets 1e-4, 4e-4, 1.6e-3 in log10 hidden rate. Beyond, 40000 two step iterations from the boundary orbit point plus 1e-6 noise. Before, bisection on the perturbation amplitude along the critical eigenvector with 10000 two step iterations per trial. Sweep over 18 rates carrying the state
- **Output:** `/data2/neural-shrimp-data/explore/criticality/s2-28011-dynamics/dynamics.json`
- **Would surprise us:** Supercritical points escaping beyond the curve while starting from the 2-cycle, or subcritical points showing a small circle beyond
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_criticality.py dynamics --outputs -0.44 -0.30 -0.20 -0.08 -0.034 -0.018 -0.0026 0.013 --workers 8`

_commit `52a2c4448e35`_

## 2026-10-07 23:12 UTC: run (exploratory)

- **Question:** Quantitative positive control. On the engineered gradient map 2-cycle, does the normal form amplitude of the circle at small offsets approach the square root of delta over minus d, for several kappa
- **Setup:** EngineeredTwoCycle with phi 2 pi times 0.13, kappa -2, 0, 2. d at r 1. Dynamics at r 1.001 and 1.004 from p0 plus 1e-3, and normal form amplitude at r 1 plus 1e-5 and 1e-6 over 400000 two step iterations started on the predicted circle
- **Output:** `/data2/neural-shrimp-data/explore/criticality/control/control.json`
- **Would surprise us:** Measured normal form amplitude differing from the prediction by more than 10 percent at the smallest offset
- **Command:** `/data2/neural-shrimp/.venv/bin/python tools/explore_criticality.py control --kappas -2 0 2`

_commit `52a2c4448e35` (dirty worktree)_

- run started 2026-10-07 23:12 UTC finished 23:12 UTC with exit status 0

## 2026-10-07 23:14 UTC: Informative failure: first criticality dynamics run measured distances from the wrong reference point

The first dynamics run at eight curve points measured amplitudes and escape thresholds relative to the boundary orbit point, but the 2-cycle shifts with the rate, so all before-curve thresholds came out below 1e-6 and the smallest beyond-curve offset was unconverged. Still informative: between offsets 4e-4 and 1.6e-3 beyond the curve the asymptotic amplitude doubles at every supercritical point (ratios 1.99 to 2.19, square-root scaling), including -0.034 and -0.018 where fixed-start maps diverge, while at the subcritical points -0.44 and +0.013 trajectories jump to distant attractors (distance about 0.64 and 0.20, nearly offset-independent) without diverging. Rerunning with the 2-cycle re-found by Newton at each offset and normal-form amplitudes compared with sqrt(log modulus over -d).

_commit `1bd9aeb6ce66` (dirty worktree)_
