# E2/E3 status (program PI, local GPU)

Run card: docs/research-log.md, "2026-10-17 00:45 UTC: Run cards E2 and E3".
Runner: tools/pi_e2e3_run.sh dev|e2|e3 [PROBLEMS_JSON]; reruns resume from records.jsonl.
Raw: /data2/neural-shrimp-data/explore/pi/e2e3-dev, e2, e3.

- [ ] dev (timing; cap and perturbation checks)
- [ ] E2 (4 arms x 512 problems x layouts 8, 16)
- [ ] E3 (soft0, 8 perturbed, greedy, 8 sampled; layout 16)
- [ ] analysis receipts docs/pi/e2-summary.json, docs/pi/e3-summary.json; results entry in the log
