# E2/E3 status (program PI, local GPU)

Complete (17 Oct 2026, 05:20 UTC). Results: docs/research-log.md, "2026-10-17 05:20 UTC: E2 and E3 results".
Receipts: docs/pi/e2-summary.json, docs/pi/e3-summary.json. Raw: /data2/neural-shrimp-data/explore/pi/e2e3-dev, e2, e3.

- [x] dev (s = 1e-3; E3 at 512 problems)
- [x] E2: 4 arms + T 0.5 control (tools/pi_e2ctl_run.sh, added after the outcome)
- [x] E3: soft0, 8 perturbed, greedy, 8 sampled
- [x] receipts and results entry
Re-analysis: tools/pi_e2e3_analyze.py e2|e3 (commands in the receipts' run lists).
