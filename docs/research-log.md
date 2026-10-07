# Research log

Append-only record of runs, decisions and amendments. See [the adaptive research protocol](adaptive-research-protocol.md).

## 2026-10-07 05:54 UTC: Retrospective summary of 6 October 2026 (written before EXP-028 results)

Assessment recommended option C; user chose it. Program C plan, gdmaps engine, EXP-028 design. Adversarial pre-freeze review found 2 blockers (cascade-band gate, ridge-free S1 drift) and 15 further issues; all fixed before freeze (docs/EXP-028-design-audit.md). Artificial-fixture S2 test (seed 99991) found two audit-confirmed embedded period-4 windows. EXP-028 freeze 91db3654 pushed at 5725820c; run launched 18:05 local (attempt-001, 14 workers). EXP-029 autonomous survey frozen at 6a162ce2, waiting for CPU. CUDA float64 gradient-descent backend qualified on fixtures (branch claude/gdmap-cuda-backend); throughput benchmark deferred while another agent uses the GPU. Driven SoA engine and ray continuation added. User pointed to Jones (2012, arXiv:1201.4343): hub shape from unimodal conjugacy, shrimp centers doubly superstable at a second critical point; folded into the A2/A3 design as testable predictions (MSS ordering, symbolic words, doubly superstable centers, critical curvature 1/eta). User directive: adopt an adaptive research protocol in place of mandatory pre-run machine plans.

_commit `7729eb822f6c` (dirty worktree)_

## 2026-10-07 05:54 UTC: Open adaptation candidate: EXP-028 stage-3 timing

Zoom stage is slower than estimated (shared load near 20; zoom cells never diverge early). Projected completion about 01:15-01:30 local versus the 02:05 wall ceiling. If the ceiling stops stage 3, all coarse and zoom maps are already saved; the planned adaptation is a successor that runs certification, ring confirmation and chaos reruns from the saved maps, labeled as such. No outcomes have been inspected.

_commit `7729eb822f6c` (dirty worktree)_
