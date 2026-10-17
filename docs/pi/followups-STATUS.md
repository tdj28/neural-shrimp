# PM2 follow-ups (items 3, 4, 6; human 17 Oct 2026): status and resume notes

Run card: docs/research-log.md, "Run card PM2X". Item 4 design: docs/pi/pm2cs-design.md. Costs: private ledger only.

## Pods (job sets in tools/runpod_soft_runner.sh)

Commit ae1712a0; all six pods created 00:46-00:48 UTC 17 Oct; every pod ends itself at its run limit.

| pod | item | RunPod id, GPU | job set and args | run limit | output (under /data2/neural-shrimp-data/explore/llm-soft-loop/pod64/pi/) | state |
|---|---|---|---|---|---|---|
| X3 | 3 diagnosis | 17z174u3bkbj0a, H100 NVL | pm2diag `llama-diag 95` | 110 min | llama-diag/<condition>/ | done, terminated; log entry and docs/pi/pm2x-llama-diag.json |
| X3b | 3b corrected R1-Llama GSM8K | x54879k3ktq53j, H100 NVL | preason `pm2fix r1llama8b 65 1536 0 gsm8k:256` | 80 min | pm2fix/r1llama8b-gsm8k-{soft,plain}/ | done, terminated; docs/pi/pm2fix-summary.json |
| X4a | 4 R1-Qwen-7B GSM8K | gackfd5xhkxdv7, H100 PCIe | preason `pm2cs-q7g r1qwen7b 65 4096 0.01 gsm8k:128` | 80 min | pm2cs-q7g/ | done, terminated (ran on an H100 PCIe); docs/pi/pm2cs-summary.json |
| X4b | 4 R1-Qwen-7B MATH | bkchb2wixdhtvz, H100 NVL | preason `pm2cs-q7m r1qwen7b 255 4096 0.01 math500:64` | 270 min | pm2cs-q7m/ | done, terminated; docs/pi/pm2cs-summary.json |
| X4c | 4 QwQ-32B GSM8K | rq4ysxnp4h78mc, H200 | preason `pm2cs-qwq qwq32b 105 4096 0.01 gsm8k:64` | 120 min | pm2cs-qwq/ | done, terminated; docs/pi/pm2cs-summary.json |
| X6 | 6 QwQ float64 twins | daykr51idbd04k, H200 | pm2f64x `pm2f64b qwq32b 160 96 4 8` | 175 min | pm2f64b/qwq32b/ | done, terminated; docs/pi/pm2f64b-summary.json |
| X3c | 3c corrected R1-Llama MATH-500 (amendment 1) | ytpiizjb3nyy06, H100 NVL | preason `pm2fix-math r1llama8b 155 1536 0 math500:64` | 170 min | pm2fix-math/r1llama8b-math500-{soft,plain}/ | done, terminated; docs/pi/pm2fix-math-summary.json |

Launcher: tools/runpod_soft_float64.py with `--name neural-shrimp-pmx-<pod> --commit ae1712a0 --jobset <set> --jobset-args "<args>" --run-minutes <limit> --ready-minutes 15`, retried on no capacity.

## Resume

- Orchestrator logs are ../neural-shrimp-data/runpod/PMX-<pod>-orchestrator.log. The last JSON line gives the pod id, out dir, `hash_verified` and the runner rc.
- If a results.tgz was not extracted (extraction clash), extract it by hand into a fresh directory.
- Check pods with `uv run --no-sync python tools/runpod_pod.py inventory`. Terminate only pods listed in the ledger (`runpod_pod.py terminate --pod ID`).
- Every tool resumes from its records.jsonl. A rerun with the same --out skips finished batches.

## Analysis commands

- Item 3: `tools/pi_llama_diag_analyze.py --diag <pi>/llama-diag --f64 <pi>/pm2f64/r1llama8b`.
- Item 3b/3c: `tools/pi_matrix2_analyze.py` with the corrected cells in place of PM2's B cells.
- Item 4: `tools/pi_pm2cs_analyze.py --cell NAME:SOFT:PLAIN:PM2_SOFT:PM2_PLAIN`. PM2 pod A's dirs are under explore/pi/pm2-A/explore/llm-soft-loop/pod64/pi/pm2/.
- Item 6: `tools/pi_f64_reason_summary.py --set new:<pi>/pm2f64b/qwq32b --set all:<pi>/pm2f64/qwq32b,<pi>/pm2f64b/qwq32b`.

## State at 03:42 UTC 17 Oct

All seven PM2X pods are done, their results hash-verified and extracted, and every pod terminated (inventory: none running). Items 3, 3b, 3c, 4 and 6 have log entries and receipts. Nothing remains to resume.
