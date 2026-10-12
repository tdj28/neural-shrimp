#!/bin/bash
# Runner for float64 soft-token continuation on a RunPod pod. tools/runpod_soft_float64.py uploads the code and
# the start data, calls this script over SSH, retrieves /root/results.tgz and terminates the pod.
# Usage: bash tools/runpod_soft_runner.sh JOBSET [ARGS...]   (job sets listed in the case below)
# Everything runs in float64 on the GPU (norms and rotary tables patched by keep_float64_norms): dense
# monodromies, JVPs, residuals and multipliers, so no mixed-precision correction is involved.
set -uo pipefail
JOBSET=${1:-skeleton}
shift || true
JOBARGS=("$@")              # scanband: OUT-NAME MINUTES then scan arguments
cd /root/neural-shrimp
export NEURAL_SHRIMP_DATA=/root/neural-shrimp-data
export PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
OUT=$NEURAL_SHRIMP_DATA/explore
LOGS=$OUT/llm-soft-loop/pod64/logs
mkdir -p "$LOGS"
log() { echo "$(date -u +%FT%TZ) $*"; }

log "setup start"
curl -LsSf https://astral.sh/uv/install.sh | sh > "$LOGS/setup-uv.log" 2>&1
uv venv -q -p 3.12 /root/venv
uv pip install -q -p /root/venv/bin/python "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126 \
  > "$LOGS/setup-torch.log" 2>&1
uv pip install -q -p /root/venv/bin/python "transformers==5.19.0" "numpy==2.5.3" "scipy==1.18.1" matplotlib \
  > "$LOGS/setup-packages.log" 2>&1
PY=/root/venv/bin/python
REV=7ae557604adf67be50417f59c2c2f167def9a775
HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='$REV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
  > "$LOGS/setup-model.log" 2>&1
# offline loading resolves 'main' through refs/main, which a download by commit hash does not write
REFS=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/refs
mkdir -p "$REFS" && printf %s "$REV" > "$REFS/main"
$PY -c "import torch, transformers, numpy, scipy; print('versions', torch.__version__, torch.version.cuda,
transformers.__version__, numpy.__version__, scipy.__version__, torch.cuda.get_device_name(0))" > "$LOGS/versions.txt" \
  || { log "setup failed"; tail -5 "$LOGS"/setup-*.log; exit 3; }
cat "$LOGS/versions.txt"
HF_HUB_OFFLINE=1 $PY -c "
import sys; sys.path.insert(0, 'tools')
import explore_llm_attractors as ela
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(ela.MODELS['qwen0.5b'], cache_dir=ela.CACHE); print('model cache ok')" \
  || { log "model cache not loadable offline"; tail -5 "$LOGS/setup-model.log"; exit 4; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$LOGS/hardware.txt"
log "setup done"

run() {   # run NAME MINUTES ARGS...: one explore_llm_soft_loop call under a wall-time limit, echoing its
          # progress file once a minute so that the streamed log shows the time per point
  local name=$1 minutes=$2
  shift 2
  local outdir
  outdir=$(echo "$@" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
  log "start $name"
  timeout "${minutes}m" $PY tools/explore_llm_soft_loop.py "$@" > "$LOGS/$name.log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if [ -f "$OUT/$outdir/progress.jsonl" ]; then
      echo "$(date -u +%T) progress $name $(wc -l < "$OUT/$outdir/progress.jsonl") $(tail -1 "$OUT/$outdir/progress.jsonl" | cut -c1-220)"
    fi
  done
  wait $pid
  local status=$?
  log "end $name status $status"
  grep -v -i -e warn -e Loading -e crop -e jit "$LOGS/$name.log" | tail -1 | cut -c1-700
}

CURVE="curve --device cuda --dtype float64 --reference-dtype none --no-exact-jvp --gmres-tol 1e-10 --tol 1e-10
       --gtol 1e-9 --max-newton 10 --ds-max 0.05 --chunk 896"
BRANCH="branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp --free alpha --log-temperature
        -0.115 --direction 1 --bounds 0.05 0.09 --chunk 896"
P8UP=llm-soft-loop/branch-p8-a0.0717-up2
P8DOWN=llm-soft-loop/branch-p8-a0.0717-down
P2=llm-soft-loop/branch-p2-t-0.115-alpha-up
D=llm-soft-loop/pod64

case "$JOBSET" in
  smoke)       # timed units: 16x16 scans in float32 and float64, the P8 flip-curve start and two points
    run smoke-scan32 15 scan --device cuda --dtype float32 --chunk 256 --alpha 0.060 0.082 --log-temperature \
      -0.18 -0.05 --resolution 16 16 --tol 1e-4 --eps 1e-4 --out $D/smoke-scan32
    run smoke-scan64 20 scan --device cuda --dtype float64 --chunk 256 --alpha 0.060 0.082 --log-temperature \
      -0.18 -0.05 --resolution 16 16 --tol 1e-8 --eps 1e-6 --out $D/smoke-scan64
    run smoke-p8-down 20 $CURVE --start $P8DOWN --direction 1 --steps 3 --log-temperature-bounds -0.4 0.3 \
      --alpha-bounds 0 0.12 --out $D/smoke-curve-p8-down
    ;;
  skeleton)    # groups given after the job set name (default all): p8down = P8 flip curve both ways,
               # p8upminus = P8 fold curve toward the stripe's lower end, p2fold = the 2-cycle's fold curve both
               # ways, ladder = fold edges of the stripes P6..P20 and the 2-cycle branch at log10 T -0.115
    GROUPS=${JOBARGS[*]:-p8down p8upminus p2fold ladder}
    for g in $GROUPS; do
      case $g in
        p8down) for dir in 1 -1; do
            name=$([ $dir = 1 ] && echo plus || echo minus)
            run curve-p8-down-$name 150 $CURVE --start $P8DOWN --direction $dir --steps 120 \
              --log-temperature-bounds -0.4 0.3 --alpha-bounds 0 0.12 --out $D/curve-p8-down-$name
          done ;;
        p8downplus|p8downminus)
          dir=$([ $g = p8downplus ] && echo 1 || echo -1); name=${g#p8down}
          run curve-p8-down-$name 150 $CURVE --start $P8DOWN --direction $dir --steps 120 \
            --log-temperature-bounds -0.4 0.3 --alpha-bounds 0 0.12 --out $D/curve-p8-down-$name ;;
        p8upminus) run curve-p8-up-minus 150 $CURVE --start $P8UP --direction -1 --steps 120 \
              --log-temperature-bounds -0.4 0.3 --alpha-bounds 0 0.12 --out $D/curve-p8-up-minus ;;
        p2fold) for dir in 1 -1; do
            name=$([ $dir = 1 ] && echo plus || echo minus)
            run curve-p2-fold-$name 90 $CURVE --start $P2 --kind fold --direction $dir --steps 150 \
              --log-temperature-bounds -0.6 0.3 --alpha-bounds 0 0.12 --out $D/curve-p2-fold-$name
          done ;;
        ladder)
          run ladder-p2-up 30 $BRANCH --alpha 0.0620 --ds 0.0005 --ds-max 0.002 --beyond 0.001 \
            --out $D/ladder-t-0.115/p2-up
          for spec in "20 0.063335 1.5e-5 4e-5 3e-6" "18 0.063534 1.5e-5 4e-5 3e-6" "16 0.063857 5e-5 1.3e-4 1e-5" \
                      "14 0.064443 1.5e-4 4e-4 3e-5" "12 0.065550 4e-4 1.2e-3 8e-5" "10 0.067766 8e-4 2.4e-3 1.7e-4" \
                      "8 0.071674 1.2e-3 3.6e-3 2.5e-4" "6 0.077872 1.4e-3 4.2e-3 3e-4"; do
            set -- $spec
            run ladder-p$1-up 40 $BRANCH --alpha $2 --ds $3 --ds-max $4 --beyond $5 --steps 80 \
              --out $D/ladder-t-0.115/p$1-up
          done ;;
      esac
    done
    ;;
  maps)        # 256x256 float64 maps: the period-adding stripes, and the high-entropy region with
               # two-dimensional 4-cycles (second multipliers 0.06-0.09) where shrimps could live
    SCAN="scan --device cuda --dtype float64 --chunk 2048 --resolution 256 256 --tol 1e-8 --eps 1e-6"
    run map-stripes 150 $SCAN --alpha 0.060 0.082 --log-temperature -0.17 -0.04 --out $D/map-stripes-256
    run map-highT 150 $SCAN --alpha 0.045 0.085 --log-temperature -0.02 0.14 --out $D/map-highT-256
    ;;
  p4curves)    # fold and flip curves of the period-4 body orbit from the p4a branches (uploaded with
               # --extra-data): SET A = both folds of the S-shaped temperature branch (cusp candidate), C = the
               # alpha fold and the lower temperature fold, E = the first flip of the alpha branch toward the legs
    P=llm-soft-loop/pod64/p4
    C4="$CURVE --steps 70"
    for set_ in ${JOBARGS[0]}; do
      case $set_ in
        A) for dir in 1 -1; do n=$([ $dir = 1 ] && echo plus || echo minus)
             run p4-foldA-$n 45 $C4 --start $P/branch-T-up --kind fold --direction $dir \
               --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-foldA-$n
             run p4-foldB-$n 45 $C4 --start $P/branch-T-up --kind fold --start-index 53 --direction $dir \
               --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-foldB-$n
           done ;;
        C) for dir in 1 -1; do n=$([ $dir = 1 ] && echo plus || echo minus)
             run p4-foldC-$n 45 $C4 --start $P/branch-a-up --kind fold --direction $dir \
               --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-foldC-$n
             run p4-foldD-$n 45 $C4 --start $P/branch-T-down --kind fold --direction $dir \
               --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-foldD-$n
           done ;;
        E) for dir in 1 -1; do n=$([ $dir = 1 ] && echo plus || echo minus)
             run p4-flipE-$n 45 $C4 --start $P/branch-a-down --kind flip --direction $dir \
               --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-flipE-$n
           done ;;
      esac
    done
    ;;
  scanband)    # one band of a large float64 map split across pods: scanband OUT MINUTES SCAN-ARGS...
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    run scanband-$name "$minutes" scan --device cuda --dtype float64 --chunk 2048 --tol 1e-8 --eps 1e-6 \
      "${JOBARGS[@]:2}" --out $D/bands/$name
    ;;
  highT)       # 256x256 float64 map of the high-entropy region with attractor identities (token loops), so that
               # different period-4 orbits are told apart; then the same window zoomed on the period-4 body
    SCAN="scan --device cuda --dtype float64 --chunk 2048 --tol 1e-8 --eps 1e-6"
    run map-highT 150 $SCAN --resolution 256 256 --alpha 0.045 0.085 --log-temperature -0.02 0.14 \
      --out $D/map-highT-256-families
    run map-p4body 100 $SCAN --resolution 192 192 --alpha 0.055 0.080 --log-temperature 0.0 0.07 \
      --out $D/map-p4body-192-families
    ;;
  p4a)         # shrimp candidate (period-4 domain at high temperature, zoom-p4): float64 orbits at the body,
               # legs and the period-8 island, then one-parameter branches out of the body in T and in alpha
    ORB="orbit --device cuda --dtype float64 --tol 1e-12 --max-newton 12"
    for spec in "0.0453 0.06484 body" "0.0505 0.06800 body2" "0.0779 0.07684 leg-ur" "0.0579 0.06421 leg-ul" \
                "0.0558 0.05632 leg-l" "0.0053 0.07811 band-low" "0.0295 0.07053 island-p8"; do
      set -- $spec
      run orbit-p4-$3 15 $ORB --log-temperature $1 --alpha $2 --out $D/p4/orbit-$3
    done
    BR="branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp --steps 150 --chunk 896"
    for dir in 1 -1; do
      name=$([ $dir = 1 ] && echo up || echo down)
      run p4-branch-T-$name 30 $BR --free temperature --log-temperature 0.0453 --alpha 0.06484 --direction $dir \
        --ds 0.002 --ds-max 0.01 --beyond 0.003 --bounds -0.1 0.2 --out $D/p4/branch-T-$name
      run p4-branch-a-$name 30 $BR --free alpha --log-temperature 0.0453 --alpha 0.06484 --direction $dir \
        --ds 0.002 --ds-max 0.01 --beyond 0.0005 --bounds 0.04 0.09 --out $D/p4/branch-a-$name
    done
    ;;
  *) echo "unknown job set $JOBSET"; exit 2 ;;
esac

tar czf /root/results.tgz -C "$NEURAL_SHRIMP_DATA" explore/llm-soft-loop/pod64
sha256sum /root/results.tgz
log "runner done"
