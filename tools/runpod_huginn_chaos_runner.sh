#!/bin/bash
# Dedicated pinned Huginn runtime; invoked only for its allowlisted toolseq job.
set -euo pipefail
name=$1
minutes=$2
shift 3
cd /root/neural-shrimp
export NEURAL_SHRIMP_DATA=/root/neural-shrimp-data
export PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
OUT=$NEURAL_SHRIMP_DATA/explore/llm-soft-loop/pod64
mkdir -p "$OUT/logs"
pack() { tar czf /root/results.tgz -C "$NEURAL_SHRIMP_DATA" explore/llm-soft-loop/pod64; sha256sum /root/results.tgz; }
trap pack EXIT
curl -LsSf https://astral.sh/uv/install.sh | sh > "$OUT/logs/setup-uv.log" 2>&1
uv venv -q -p 3.12 /root/venv-huginn
uv pip install -q -p /root/venv-huginn/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 > "$OUT/logs/setup-torch.log" 2>&1
uv pip install -q -p /root/venv-huginn/bin/python transformers==4.55.0 numpy==2.5.4 scipy==1.18.1 > "$OUT/logs/setup-packages.log" 2>&1
PY=/root/venv-huginn/bin/python
HF_HUB_OFFLINE=0 $PY -c "from huggingface_hub import snapshot_download; snapshot_download('tomg-group-umd/huginn-0125',revision='bb6621b65e90b6a4b9b29ef88dc83866d450470c',cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" > "$OUT/logs/setup-model.log" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/logs/hardware.txt"
uv pip freeze -p "$PY" > "$OUT/logs/libraries.txt"
timeout --signal=TERM --kill-after=10s "${minutes}m" "$PY" -u tools/huginn_chaos_map.py "$@" 2>&1 | tee "$OUT/logs/$name.log"
