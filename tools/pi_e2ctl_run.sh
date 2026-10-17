#!/bin/bash
# E2 control arm (amendment 2026-10-17 02:15 UTC, added after E2's outcome): soft top-15 at a uniform T = 0.5,
# uncapped, layouts 8 and 16, on the E2 problems. Kept apart from tools/pi_e2e3_run.sh, which was running.
set -o pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/data2/neural-shrimp-data/explore/pi/e2/t0.5
t0=$(date +%s)
uv run --no-sync python tools/pi_harness.py run --split train --indices docs/pi/e2e3-problems.json --mode soft \
  --top-n 15 --temperature 0.5 --soft-steps 64 --layouts 8 16 --out $O > $O.log 2>&1
echo "t0.5 exit $? seconds $(( $(date +%s) - t0 )) $(date -u +%H:%M:%S)"
