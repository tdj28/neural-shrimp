# PM2 follow-ups (items 3, 4, 6; human 17 Oct 2026): status and resume notes

Run card: docs/research-log.md, "Run card PM2X". Item 4 design: docs/pi/pm2cs-design.md. Costs: private ledger only.

## Pods (job sets in tools/runpod_soft_runner.sh)

Commit ae1712a0; all six pods created 00:46-00:48 UTC 17 Oct; every pod ends itself at its run limit.

| pod | item | RunPod id, GPU | job set and args | run limit | output (under /data2/neural-shrimp-data/explore/llm-soft-loop/pod64/pi/) | state |
|---|---|---|---|---|---|---|
| X3 | 3 diagnosis | 17z174u3bkbj0a, H100 NVL | pm2diag `llama-diag 95` | 110 min | llama-diag/<condition>/ | running |
| X3b | 3b corrected R1-Llama GSM8K | x54879k3ktq53j, H100 NVL | preason `pm2fix r1llama8b 65 1536 0 gsm8k:256` | 80 min | pm2fix/r1llama8b-gsm8k-{soft,plain}/ | running |
| X4a | 4 R1-Qwen-7B GSM8K | gackfd5xhkxdv7, H100 PCIe | preason `pm2cs-q7g r1qwen7b 65 4096 0.01 gsm8k:128` | 80 min | pm2cs-q7g/ | running |
| X4b | 4 R1-Qwen-7B MATH | bkchb2wixdhtvz, H100 NVL | preason `pm2cs-q7m r1qwen7b 255 4096 0.01 math500:64` | 270 min | pm2cs-q7m/ | running |
| X4c | 4 QwQ-32B GSM8K | rq4ysxnp4h78mc, H200 | preason `pm2cs-qwq qwq32b 105 4096 0.01 gsm8k:64` | 120 min | pm2cs-qwq/ | running |
| X6 | 6 QwQ float64 twins | daykr51idbd04k, H200 | pm2f64x `pm2f64b qwq32b 160 96 4 8` | 175 min | pm2f64b/qwq32b/ | running |

Launcher: tools/runpod_soft_float64.py with `--name neural-shrimp-pmx-<pod> --commit ae1712a0 --jobset <set> --jobset-args "<args>" --run-minutes <limit> --ready-minutes 15`, retried on no capacity.

## Resume

- Orchestrator logs are ../neural-shrimp-data/runpod/PMX-<pod>-orchestrator.log. The last JSON line gives the pod id, out dir, `hash_verified` and the runner rc.
- If a results.tgz was not extracted (extraction clash), extract it by hand into a fresh directory.
- Check pods with `uv run --no-sync python tools/runpod_pod.py inventory`. Terminate only pods listed in the ledger (`runpod_pod.py terminate --pod ID`).
- Every tool resumes from its records.jsonl. A rerun with the same --out skips finished batches.
