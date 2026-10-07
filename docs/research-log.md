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
