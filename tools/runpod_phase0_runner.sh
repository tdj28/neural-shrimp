#!/usr/bin/env bash
# RunPod Phase 0 runner (executed ON the pod, from an extracted git archive).
#
# Ports the gdmap CUDA engine to the pod GPU, compares every output array bit
# for bit with the RTX 2000 Ada reference, runs the CUDA backend tests, and
# times the float64 torch workload of the bounded real-data pilot. No
# credentials are needed or used. Everything is written under $OUT and packed
# into $OUT.tgz with its SHA-256 printed at the end.
#
# Environment (defaults suit an H100 pod):
#   ROOT  extracted repository archive        (/root/neural-shrimp)
#   REF   sm_89 reference from crossgpu_parity (/root/reference-sm89)
#   OUT   results directory                    (/root/phase0-results)
#   ARCH  CUDA architecture for the build      (sm_90)
#   CAP   compute capability being qualified   (9.0; empty for a qualified device)
#   SKIP_SETUP=1  reuse an existing environment (local rehearsal)
set -euo pipefail
ROOT=${ROOT:-/root/neural-shrimp}
REF=${REF:-/root/reference-sm89}
OUT=${OUT:-/root/phase0-results}
ARCH=${ARCH:-sm_90}
CAP=${CAP-9.0}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/runner.log") 2>&1
echo "phase0 runner start $(date -u +%FT%TZ) arch=$ARCH cap=$CAP"
nvidia-smi
nvidia-smi -q > "$OUT/nvidia-smi-q.txt"
lscpu > "$OUT/lscpu.txt" || true
nvcc --version | tee "$OUT/nvcc.txt"

cd "$ROOT"
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  if ! command -v g++-12 > /dev/null; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++-12 > /dev/null
  fi
  if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv venv -q --python 3.12 .venv-pod
  uv pip install -q --python .venv-pod/bin/python numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.1 pytest
fi
PY=${PY:-$ROOT/.venv-pod/bin/python}
"$PY" -c "import sys, numpy, scipy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)" | tee "$OUT/versions.txt"
g++-12 --version | head -1 | tee -a "$OUT/versions.txt"

BUILD=${BUILD:-/root/build-$ARCH}
"$PY" tools/build_gdmap_cuda.py "$BUILD" --arch "$ARCH" | tee "$OUT/build.txt"
cp "$BUILD/gdmap_cuda-build.json" "$OUT/"

echo "parity compare $(date -u +%FT%TZ)"
NEURAL_SHRIMP_CUDA_QUALIFYING=$CAP "$PY" -u tools/crossgpu_parity.py compare \
  --library "$BUILD/gdmap_cuda.so" --reference "$REF" --out "$OUT/parity"

echo "cuda backend tests $(date -u +%FT%TZ)"
NEURAL_SHRIMP_CUDA_QUALIFYING=$CAP NEURAL_SHRIMP_GDMAP_CUDA_LIBRARY="$BUILD/gdmap_cuda.so" \
  "$PY" -m pytest -q -p no:cacheprovider tests/test_gdmaps_cuda.py > "$OUT/pytest-cuda.txt" 2>&1 || true
tail -3 "$OUT/pytest-cuda.txt"

echo "torch float64 benchmark $(date -u +%FT%TZ)"
TORCH_PY=${TORCH_PY:-python3}
"$TORCH_PY" -u tools/bench_torch_float64.py --cells 1 32 128 256 --out "$OUT/torch-bench.json" || true

echo "phase0 runner end $(date -u +%FT%TZ)"
tar czf "$OUT.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
sha256sum "$OUT.tgz"
