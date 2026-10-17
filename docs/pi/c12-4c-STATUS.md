# C12-4c status (confirmatory rerun of C12-4)

Design: docs/pi-c12-4-confirmatory.md. Launcher: tools/c12_4c_run.sh. Raw outputs:
/data2/neural-shrimp-data/explore/pi/c12-4c/ (run-a, run-b, confirm.json, check.json, logs/).

## State

- 2026-10-17 00:55 UTC: design, code, tests (32 pass) and instance list ready; freeze commit pending push. No
  confirmatory data exists.

## Resume

- Check whether the job is still running: `pgrep -af c12_4c_run` (this task's own job only; never touch other
  processes).
- Otherwise resume in place: `RESUME=1 nohup tools/c12_4c_run.sh >> /data2/neural-shrimp-data/explore/pi/c12-4c/logs/run.log 2>&1 &`.
  Completed batches are kept (done markers in records.jsonl). A batch that was interrupted mid-way has no done
  marker and is recomputed in full; its partial rows stay in the file, and `confirm` and the checker keep only
  the completed pass of each batch (complete_rows; the discarded count is a gate field).
- After the run: write docs/pi/c12-4c-summary.json (receipt spec with instance_set "conf"), append results to
  docs/research-log.md, and update this file.
