#!/usr/bin/env bash
# C12-4c confirmatory runs (docs/pi-c12-4-confirmatory.md): run A (direction v), run B (direction v2), then the
# frozen analysis and the independent check. CPU only, float32, 6 threads, nice 10. Resumable: rerun with RESUME=1.
set -euo pipefail
cd /data2/neural-shrimp
D=/data2/neural-shrimp-data/explore/pi/c12-4c
V1=/data2/neural-shrimp-data/explore/pi/c12-4/vectors-qwen3b
V2=$D/vectors-v2-qwen3b
CH=docs/pi/c12-4-choices.json
V1SHA=1aae51535d2485bb0c010ed9801a0374bb88179e01f735cfca1ff8bf56148475
V2SHA=4554a8d1d816826baa2ed942df0eef8d0c550d89261087a6312d41b9adb161d5
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
R=${RESUME:+--resume}
echo "$V1SHA  $V1/vectors.npz" | sha256sum -c -
echo "$V2SHA  $V2/vectors.npz" | sha256sum -c -
PY="nice -n 10 uv run --no-sync python tools/c12_steering_history.py"
$PY histories --split conf --instances docs/pi/c12-4c-instances.json --choices $CH --vectors $V1 --starts 8 \
    --dwell 64 --histories direct up down zero random unsteered --instances-per-batch 2 --threads 6 \
    --out $D/run-a $R
$PY histories --split conf --instances docs/pi/c12-4c-instances.json --choices $CH --vectors $V2 --starts 8 \
    --dwell 64 --histories direct up down --instances-per-batch 4 --threads 6 --out $D/run-b $R
$PY confirm --run-a $D/run-a --run-b $D/run-b --choices $CH --vectors-a-sha256 $V1SHA --json $D/confirm.json
uv run --no-sync python tools/c12_4c_check.py --run-a $D/run-a --run-b $D/run-b --confirm $D/confirm.json \
    --json $D/check.json
echo "C12-4c done"
