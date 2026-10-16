#!/bin/bash
# Run PM (docs/pi-matrix-confirmatory.md, frozen before any data): the confirmatory soft-thinking matrix,
# Qwen2.5-7B-Instruct bfloat16 on the local GPU, the first 256 GSM8K train problems, layouts 8 and 16.
# The arms run in the frozen order. A rerun resumes each arm from its records.
# Usage: tools/pi_matrix_run.sh [OUTDIR]   (default ../neural-shrimp-data/explore/pi/pm)
set -o pipefail
cd "$(dirname "$0")/.."
D=${1:-../neural-shrimp-data/explore/pi/pm}
mkdir -p "$D"
COMMON="--split train --limit 256 --layouts 8 16"
run() {
  name=$1; shift
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run --no-sync python tools/pi_harness.py run "$@" $COMMON \
    --out "$D/$name" > "$D/$name.log" 2>&1
  echo "$name exit $? $(date -u +%H:%M:%S)"
}
run top15-s64 --mode soft --top-n 15 --temperature 0.6 --soft-steps 64 --save-embeddings
run replay --mode soft --top-n 15 --temperature 0.6 --soft-steps 64 --replay-from "$D/top15-s64"
run greedy --mode hard --temperature 0 --max-new 320
run top1-s64 --mode soft --top-n 1 --temperature 0.6 --soft-steps 64
run full-s64 --mode soft --top-n 0 --temperature 0.6 --soft-steps 64
run top15-s16 --mode soft --top-n 15 --temperature 0.6 --soft-steps 16
run full-s16 --mode soft --top-n 0 --temperature 0.6 --soft-steps 16
echo done
