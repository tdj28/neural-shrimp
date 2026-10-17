#!/bin/bash
# Program PI runs E2 (gain-capped soft thinking) and E3 (vote over perturbed soft runs, against sampled
# self-consistency) on the local GPU: Qwen2.5-7B-Instruct bf16, run card in docs/research-log.md (2026-10-17).
# Usage: tools/pi_e2e3_run.sh dev|e2|e2ctl|e3 [PROBLEMS_JSON]   (a rerun resumes each run from its records)
set -o pipefail
cd "$(dirname "$0")/.."
STAGE=$1
D=/data2/neural-shrimp-data/explore/pi
PY="uv run --no-sync python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SOFT="--mode soft --top-n 15 --temperature 0.6 --soft-steps 64"
run() {
  out=$1; shift
  mkdir -p "$(dirname "$out")"
  local t0=$(date +%s)
  $PY tools/pi_harness.py run --split "$SPLIT" --indices "$IDX" "$@" --out "$out" > "$out.log" 2>&1
  echo "$(basename "$out") exit $? seconds $(( $(date +%s) - t0 )) $(date -u +%H:%M:%S)"
}
case $STAGE in
  dev)
    SPLIT=test; IDX=docs/pi/e2e3-dev-problems.json; O=$D/e2e3-dev
    for cap in 0 0.37 0.10 0.015; do run $O/e2-cap$cap $SOFT --layouts 8 16 --gain-cap $cap; done
    for s in 0 1e-4 1e-3 1e-2; do run $O/e3-p$s-k1 $SOFT --layouts 16 --perturb $s --perturb-seed 1001000; done
    run $O/e3-p1e-3-k2 $SOFT --layouts 16 --perturb 1e-3 --perturb-seed 2001000
    run $O/e3-greedy --mode hard --temperature 0 --max-new 320 --stop-eos --layouts 16
    run $O/e3-sample-k1 --mode hard --temperature 0.7 --max-new 320 --stop-eos --layouts 16 --seed 21261016
    ;;
  e2)
    SPLIT=train; IDX=${2:-docs/pi/e2e3-problems.json}; O=$D/e2
    for cap in 0 0.37 0.10 0.015; do run $O/cap$cap $SOFT --layouts 8 16 --gain-cap $cap; done
    ;;
  e2ctl)                                   # added after E2's outcome: a uniform lower temperature, no cap
    SPLIT=train; IDX=${2:-docs/pi/e2e3-problems.json}; O=$D/e2
    run $O/t0.5 --mode soft --top-n 15 --temperature 0.5 --soft-steps 64 --layouts 8 16
    ;;
  e3)
    SPLIT=train; IDX=${2:-docs/pi/e2e3-problems.json}; O=$D/e3; S=${E3_PERTURB:-1e-3}
    run $O/soft0 $SOFT --layouts 16
    for k in 1 2 3 4 5 6 7 8; do run $O/soft-k$k $SOFT --layouts 16 --perturb $S --perturb-seed $((1000 + k * 1000000)); done
    run $O/greedy --mode hard --temperature 0 --max-new 320 --stop-eos --layouts 16
    for k in 1 2 3 4 5 6 7 8; do run $O/sample-k$k --mode hard --temperature 0.7 --max-new 320 --stop-eos --layouts 16 \
      --seed $((20261016 + k * 1000000)); done
    ;;
  *) echo "usage: $0 dev|e2|e2ctl|e3 [PROBLEMS_JSON]"; exit 2 ;;
esac
echo "stage $STAGE done $(date -u +%H:%M:%S)"
