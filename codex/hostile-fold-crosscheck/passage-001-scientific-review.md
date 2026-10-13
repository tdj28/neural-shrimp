# Task7 controlled hostile-fold passages — completed scientific review

The independently reproduced bottleneck gives clear full-channel and midpoint
transits. Their durations are close to the frozen local predictions, but the
primary all-six accuracy claim **fails**: the smallest-offset midpoint exits two
F² iterates earlier than the iterated quadratic predicts. Five of six rows meet
the one-iterate criterion. We retain that failure without tuning or substitution.

All six trajectories completed. The independent saved-evidence audit passes:
654 original artifacts and 10 source Git blobs were authenticated by the lifecycle
owner; all 896-coordinate step inputs/outputs, state chains, fixed parameters,
mode projections, first strict exits, event clocks and forecasts were checked.
There are 310 G=F² steps, hence 620 F evaluations, with no censored rows. The
projection recomputation discrepancy is zero. Prior-to-exit states are at least
6.80×10⁻⁶ below U, and crossing states at least 9.41×10⁻⁶ above U. This is a
completed scientific mismatch, not a timeout, missing row or ambiguous exit.

## Frozen comparison, every row retained

At alpha=0.083, p_c=log10T=−0.04624653567444522, a=−1.1683279225482623,
b=44.52540973728358, and U=0.1/b=0.0022459085854579907. Starts are x_c−U r
or x_c, with u=l·(x−x_c), ||r||₂=1 and l·r=1. Count the first strict u>+U
after a whole G step, with no burn-in. One G count is two F evaluations.

| Physical log10T offset | Entrance | Measured G | Quadratic G | Error | Primary | Finite arctan G | Arctan error | Asymptotic G |
|---:|:---|---:|---:|---:|:---|---:|---:|---:|
| −0.00001 | full |119|118|+1|pass|118.077459|+0.922541|137.741095|
| −0.00001 | midpoint |59|61|−2|**fail**|59.038730|−0.038730|68.870548|
| −0.00003 | full |61|61|0|pass|60.477534|+0.522466|79.524858|
| −0.00003 | midpoint |31|32|−1|pass|30.238767|+0.761233|39.762429|
| −0.0001 | full |27|27|0|pass|26.230817|+0.769183|43.557559|
| −0.0001 | midpoint |13|14|−1|pass|13.115409|−0.115409|21.778779|

The finite-endpoint arctan predictions were also frozen beforehand. Their errors
are all below one G iterate (maximum 0.922541), a **secondary descriptive** result;
the plan did not give this comparison a replacement all-six success gate. It
cannot rescue the failed quadratic primary. This test does not isolate the cause
of the midpoint discrepancy or show that the continuous approximation wins in
other settings.

The finite-endpoint correction is substantial: the arctan prediction is 14.28%,
23.95% and39.78% below the infinite-channel asymptotic prediction at the three
offsets. Therefore a direct comparison of the observed periodic-window slope
2.235 with the asymptotic coefficient sqrt(|ab|)/pi=2.2958127729 does not identify
natural full-channel reinjection. These six trajectories were deliberately
placed at specified entrances; none measures where the dynamics reinjects in
the P16–20 windows. No natural-reinjection or interval-proof claim follows.

## Exact evidence and allowed manuscript statement

- Producing commit: `b6c352e370331666f5f3ec400c37cde949130264`.
- Prospective plan SHA256: `db0028cc4983b4bbe5ec5cbbd52f67adb7fff77b6dcab9221f6be3611782bcc7`.
- Original receipt: `/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck/passage-001/receipt.json`, SHA256 `2ee6cda10c7216a63155c1a0cd1b5aa8657311130ca80aaafaae82a390722160`.
- Saved audit: `passage-001-saved-audit-001/audit.json` under the same data root, SHA256 `6cf869fe0bdd1866307c8c005729b1fcc74d07988468f29455e002d92174298c`; auditor source `fab3e516c51543c3de7e7cfa2e73456a314aa43af5b3c88db4a6af2a6fcde9b4`.
- Authority archive manifest: `audits-passage-001-authority-001/manifest.json`, SHA256 `7055896d2e594ca8d985e03c73fb133343682c7465918c2dc12ae2852be40690`.
- The accompanying JSON retains every unrounded row, full scientific source hashes, method, resource data and limits.

A precise manuscript statement is: “In an independent float64 implementation,
controlled full/midpoint passages at log10T offsets −10⁻⁵, −3×10⁻⁵ and −10⁻⁴ took
119/59, 61/31 and 27/13 iterates of F². The prospectively frozen iterated quadratic
forecasts were 118/61, 61/32 and 27/14: five of six were within one iterate, while the
smallest-offset midpoint missed by two, failing the aggregate criterion.
Separately frozen finite-endpoint arctan predictions differed by at most 0.923
iterates. These tests measure controlled local transit times, not natural
reinjection in the periodic windows.”

The worker took 161.678s with 5,385,204KiB peak RSS; the owning launcher completed
and reaped in 165.018s. The saved audit took 0.168s/37,524KiB. Original and audit
process groups are closed. All certificate flags remain false. Task7 is complete:
no further test, model replay, retry or post-outcome tuning is requested.
