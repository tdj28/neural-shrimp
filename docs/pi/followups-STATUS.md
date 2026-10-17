# PM2 follow-ups (items 3, 4, 6; human 17 Oct 2026): status and resume notes

Run card: docs/research-log.md, "Run card PM2X". Item 4 design: docs/pi/pm2cs-design.md. Costs: private ledger only.

## Pods (job sets in tools/runpod_soft_runner.sh; launcher loop: scratchpad pmx_launch.sh, copied below)

| pod | item | job set and args | output (under /data2/neural-shrimp-data/explore/llm-soft-loop/pod64/pi/) | state |
|---|---|---|---|---|
| X3 | 3 diagnosis | pm2diag `llama-diag 115` | llama-diag/<condition>/ | planned |
| X3b | 3b corrected R1-Llama GSM8K | preason `pm2fix r1llama8b 75 1536 0 gsm8k:256` | pm2fix/r1llama8b-gsm8k-{soft,plain}/ | planned |
| X4a | 4 R1-Qwen-7B GSM8K | preason `pm2cs-q7g r1qwen7b 75 4096 0.01 gsm8k:128` | pm2cs-q7g/... | planned |
| X4b | 4 R1-Qwen-7B MATH | preason `pm2cs-q7m r1qwen7b 195 4096 0.01 math500:64` | pm2cs-q7m/... | planned |
| X4c | 4 QwQ-32B GSM8K | preason `pm2cs-qwq qwq32b 115 4096 0.01 gsm8k:64` | pm2cs-qwq/... | planned |
| X6 | 6 QwQ float64 twins | pm2f64x `pm2f64b qwq32b 170 96 4 8` | pm2f64b/qwq32b/ | planned |

## Resume

- Orchestrator logs are ../neural-shrimp-data/runpod/PMX-<pod>-orchestrator.log. The last JSON line gives the pod id, out dir, `hash_verified` and the runner rc.
- If a results.tgz was not extracted (extraction clash), extract it by hand into a fresh directory.
- Check pods with `uv run --no-sync python tools/runpod_pod.py inventory`. Terminate only pods listed in the ledger (`runpod_pod.py terminate --pod ID`).
- Every tool resumes from its records.jsonl. A rerun with the same --out skips finished batches.
