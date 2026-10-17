# C12-4c status (confirmatory rerun of C12-4)

Design: docs/pi-c12-4-confirmatory.md. Launcher: tools/c12_4c_run.sh. Raw outputs:
/data2/neural-shrimp-data/explore/pi/c12-4c/ (run-a, run-b, confirm.json, check.json, logs/).

## State

- 2026-10-17 00:55 UTC: design, code, tests (33 pass) and instance list ready.
- 2026-10-17 01:00 UTC: freeze pushed at 76ae09fe (verified on origin/main); run card at d1745366.
- 2026-10-17 01:06 UTC: tools/c12_4c_run.sh launched (log: /data2/neural-shrimp-data/explore/pi/c12-4c/logs/run.log);
  machine load about 25 on 32 logical CPUs at launch. Run A (24 batches) first, then run B (12 batches).

## Resume

- Check whether the job is still running: `pgrep -af c12_4c_run` (this task's own job only; never touch other
  processes).
- Otherwise resume in place: `RESUME=1 nohup tools/c12_4c_run.sh >> /data2/neural-shrimp-data/explore/pi/c12-4c/logs/run.log 2>&1 &`.
  Completed batches are kept (done markers in records.jsonl). A batch that was interrupted mid-way has no done
  marker and is recomputed in full; its partial rows stay in the file, and `confirm` and the checker keep only
  the completed pass of each batch (complete_rows; the discarded count is a gate field).
- After the run: write docs/pi/c12-4c-summary.json (receipt spec with instance_set "conf"), append results to
  docs/research-log.md, and update this file.
- 2026-10-17 01:18 UTC: run A batch 0 took 683 s (load 36 from other agents' jobs; dry-run estimate 430-500 s).
  Projection: about 4.6 h for A and 2.3 h for B, near the 7 h estimate, finishing about 08:00 UTC. The design's
  reduction rule applies only before data, so the run continues unchanged.
