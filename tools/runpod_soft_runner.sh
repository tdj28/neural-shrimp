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
  "accelerate==1.15.0" \
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
REV15=989aa7980e4cf806f80c7fef2b1adb7bc71aa306                 # Qwen2.5-1.5B-Instruct, the local snapshot
HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='$REV15', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
  >> "$LOGS/setup-model.log" 2>&1
REFS15=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs
mkdir -p "$REFS15" && printf %s "$REV15" > "$REFS15/main"
if [ "$JOBSET" = p1control ] || [ "$JOBSET" = p1detect ] || [ "$JOBSET" = p1commit ] || [ "$JOBSET" = p1x ]; then   # the pinned toxicity classifier
  HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('unitary/toxic-bert', revision='4d6c22e74ba2fdd26bc4f7238f50766b045a0d94',
                  cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" >> "$LOGS/setup-model.log" 2>&1
fi
GPUCHK="torch.cuda.get_device_name(0)"
[ "$JOBSET" = certcal ] && GPUCHK="'cpu-only job set'"     # certificate shards use no GPU; a faulty host GPU must not stop them
$PY -c "import torch, transformers, numpy, scipy; print('versions', torch.__version__, torch.version.cuda,
transformers.__version__, numpy.__version__, scipy.__version__, $GPUCHK)" > "$LOGS/versions.txt" \
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

get7b() {  # pinned Qwen2.5-7B-Instruct into the pod's HF cache (as in job set bp)
  local REV7=a09a35458c702b33eeacc393d103063234e8bc28
  log "download Qwen2.5-7B-Instruct $REV7"
  HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='$REV7', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
    > "$LOGS/setup-model7.log" 2>&1
  local REFS7=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/refs
  mkdir -p "$REFS7" && printf %s "$REV7" > "$REFS7/main"
}

getreasoner() {  # getreasoner MODEL: the pinned reasoning model, GSM8K and MATH-500 into the pod's HF cache (as pm2)
  local model=$1 REPO MREV
  case $model in
    r1qwen7b)  REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; MREV=916b56a44061fd5cd7d6a8fb632557ed4f724f60 ;;
    r1llama8b) REPO=deepseek-ai/DeepSeek-R1-Distill-Llama-8B; MREV=6a6f4aa4197940add57724a7707d069478df56b1 ;;
    qwq32b)    REPO=Qwen/QwQ-32B; MREV=976055f8c83f394f35dbd3ab09a285a984907bd0 ;;
    *) log "unknown model $model"; exit 5 ;;
  esac
  for try in 1 2 3; do
    log "download $REPO $MREV (try $try)"
    HF_HUB_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=120 $PY -c "
from huggingface_hub import snapshot_download
c = '$NEURAL_SHRIMP_DATA/models/hf/hub'
snapshot_download('$REPO', revision='$MREV', cache_dir=c)
snapshot_download('openai/gsm8k', repo_type='dataset', revision='740312add88f781978c0658806c59bc2815b9866', cache_dir=c)
snapshot_download('HuggingFaceH4/MATH-500', repo_type='dataset', revision='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
                  cache_dir=c)" >> "$LOGS/setup-reasoner.log" 2>&1 && break
  done
  local MDIR=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO//\//--}
  mkdir -p "$MDIR/refs" && printf %s "$MREV" > "$MDIR/refs/main"
  uv pip install -q -p $PY "pandas==3.0.6" "pyarrow==26.0.0" > "$LOGS/setup-pandas.log" 2>&1
  mkdir -p /data2 && ln -sfn "$NEURAL_SHRIMP_DATA" /data2/neural-shrimp-data   # pi_harness's fixed cache path
}

run() {   # run NAME MINUTES ARGS...: one explore_llm_soft_loop call (or tools/$TOOL) under a wall-time limit,
          # echoing its progress file once a minute so that the streamed log shows the time per point
  local name=$1 minutes=$2
  shift 2
  local outdir
  outdir=$(echo "$@" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
  log "start $name"
  timeout "${minutes}m" $PY tools/${TOOL:-explore_llm_soft_loop.py} "$@" > "$LOGS/$name.log" 2>&1 &
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
    JOBGROUPS=${JOBARGS[*]:-p8down p8upminus p2fold ladder}   # (not GROUPS: a bash builtin array)
    for g in $JOBGROUPS; do
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
  superstable) # superstable parameters of a sequence of bands with tools/llm_feigenbaum.py on the GPU (float64):
               # JOBARGS = NAME MINUTES then superstable arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=llm_feigenbaum.py run superstable-$name "$minutes" superstable --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/superstable/$name
    ;;
  edges)       # fold and flip edges of periodic windows with tools/llm_feigenbaum.py on the GPU (float64):
               # JOBARGS = NAME MINUTES then edges arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=llm_feigenbaum.py run edges-$name "$minutes" edges --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/edges/$name
    ;;
  retmap)      # return maps of chaotic attractors (tools/return_map.py, float64 on the GPU): JOBARGS = NAME MINUTES then
               # tool arguments (vectors uploaded with --extra-data)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=return_map.py run retmap-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/retmap/$name
    ;;
  xo)          # orbit-preserving readout/network crossover (run XO; tools/crossover.py, float64 on the GPU): JOBARGS =
               # NAME MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    case " ${JOBARGS[*]} " in *" qwen7b "*) get7b ;; esac
    TOOL=crossover.py run xo-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/xo/$name
    ;;
  reprotask)   # answers of continuous-feedback inference under batch-layout changes (run REPRO-T; tools/repro_task.py;
               # frozen design docs/repro-task-test.md): JOBARGS = NAME MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    case " ${JOBARGS[*]} " in *" qwen7b "*) get7b ;; esac
    TOOL=repro_task.py run reprotask-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/repro/$name
    ;;
  repro70)     # run REPRO on Llama-3.3-70B-Instruct (gated; float64 with bf16 storage over all GPUs): JOBARGS = NAME
               # MINUTES then tool arguments (use --model llama3.3-70b --device auto --weights bf16; vectors uploaded)
    REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)" > "$LOGS/setup-big.log" 2>&1
    unset HF_TOKEN
    mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
    mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
    log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1), free $(df -h --output=avail / | tail -1)"
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=repro_noise.py run repro-$name "$minutes" "${JOBARGS[@]:2}" --out $D/repro/$name
    ;;
  reprollama)  # run REPRO on Llama-3.1-8B-Instruct (gated: token on stdin, unset after the download; revision as llama8b):
               # JOBARGS = NAME MINUTES then tool arguments (use --model llama3.1-8b --vectors none)
    LREV=0e9e39f249a16976918f6564b8830bc894c89659
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', revision='$LREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'])" > "$LOGS/setup-llama.log" 2>&1
    unset HF_TOKEN
    lrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/refs
    mkdir -p "$lrefs" && printf %s "$LREV" > "$lrefs/main"
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=repro_noise.py run repro-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/repro/$name
    ;;
  repro)       # incidental numerical differences against the dynamical regime (run REPRO; tools/repro_noise.py, on the
               # GPU): JOBARGS = NAME MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    case " ${JOBARGS[*]} " in *" qwen7b "*) get7b ;; esac
    TOOL=repro_noise.py run repro-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/repro/$name
    ;;
  tp)          # turning points of return maps over chaotic cells (run TP; tools/turning_scan.py, float64 on the GPU):
               # JOBARGS = NAME MINUTES ARGS [:: NAME MINUTES ARGS ...] (vectors uploaded with --extra-data)
    i=0; n=${#JOBARGS[@]}
    while [ $i -lt $n ]; do
      name=${JOBARGS[$i]}; minutes=${JOBARGS[$((i+1))]}; i=$((i+2)); margs=()
      while [ $i -lt $n ] && [ "${JOBARGS[$i]}" != "::" ]; do margs+=("${JOBARGS[$i]}"); i=$((i+1)); done
      i=$((i+1))
      TOOL=turning_scan.py run tp-$name "$minutes" --device cuda "${margs[@]}" --out $D/tp/$name
    done
    ;;
  mech)        # gain split along steering, steering block and a logit bias (run MECH; tools/steer_gain.py, float64 on
               # the GPU): JOBARGS = NAME MINUTES ARGS [:: NAME MINUTES ARGS ...], one tool call per group, in order
               # (vectors uploaded with --extra-data)
    i=0; n=${#JOBARGS[@]}
    while [ $i -lt $n ]; do
      name=${JOBARGS[$i]}; minutes=${JOBARGS[$((i+1))]}; i=$((i+2)); margs=()
      while [ $i -lt $n ] && [ "${JOBARGS[$i]}" != "::" ]; do margs+=("${JOBARGS[$i]}"); i=$((i+1)); done
      i=$((i+1))
      TOOL=steer_gain.py run mech-$name "$minutes" --device cuda "${margs[@]}" --out $D/mech/$name
    done
    ;;
  rc)          # memory capacity of the soft-token loop as a reservoir (run RC; tools/reservoir_capacity.py, float64 on
               # the GPU): JOBARGS = NAME MINUTES ARGS [:: NAME MINUTES ARGS ...], one tool call per group, in order
    i=0; n=${#JOBARGS[@]}
    while [ $i -lt $n ]; do
      name=${JOBARGS[$i]}; minutes=${JOBARGS[$((i+1))]}; i=$((i+2)); margs=()
      while [ $i -lt $n ] && [ "${JOBARGS[$i]}" != "::" ]; do margs+=("${JOBARGS[$i]}"); i=$((i+1)); done
      i=$((i+1))
      TOOL=reservoir_capacity.py run rc-$name "$minutes" "${margs[@]}" --device cuda --out $D/rc/$name
    done
    ;;
  dialogue)    # two agents in one soft-token conversation (run DLG; tools/soft_dialogue.py, float64 on the GPU): JOBARGS =
               # NAME MINUTES MODE ARGS [:: NAME MINUTES MODE ARGS ...], one tool call per group, in order (vectors
               # uploaded with --extra-data)
    i=0; n=${#JOBARGS[@]}
    while [ $i -lt $n ]; do
      name=${JOBARGS[$i]}; minutes=${JOBARGS[$((i+1))]}; i=$((i+2)); margs=()
      while [ $i -lt $n ] && [ "${JOBARGS[$i]}" != "::" ]; do margs+=("${JOBARGS[$i]}"); i=$((i+1)); done
      i=$((i+1))
      TOOL=soft_dialogue.py run dlg-$name "$minutes" "${margs[@]}" --device cuda --out $D/dialogue/$name
    done
    ;;
  tbmulti)     # several two_block_map.py calls in order (run TBC): JOBARGS = NAME MINUTES ARGS [:: NAME MINUTES ARGS ...]
    i=0; n=${#JOBARGS[@]}
    while [ $i -lt $n ]; do
      name=${JOBARGS[$i]}; minutes=${JOBARGS[$((i+1))]}; i=$((i+2)); margs=()
      while [ $i -lt $n ] && [ "${JOBARGS[$i]}" != "::" ]; do margs+=("${JOBARGS[$i]}"); i=$((i+1)); done
      i=$((i+1))
      TOOL=two_block_map.py run tb-$name "$minutes" --device cuda "${margs[@]}" --out $D/tb/$name
    done
    ;;
  tb)          # two steering directions at two blocks (run TB; tools/two_block_map.py, float64 on the GPU): JOBARGS = NAME
               # MINUTES then tool arguments (vectors uploaded with --extra-data)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=two_block_map.py run tb-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/tb/$name
    ;;
  bp)          # per-prompt regimes and ordinary answers (run BP; tools/behavior_prompts.py): JOBARGS = NAME MINUTES MODEL
               # PART then part arguments (PART label or generate); for qwen7b the pinned checkpoint is downloaded first
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    bpmodel=${JOBARGS[2]}
    if [ "$bpmodel" = qwen7b ]; then
      REV7=a09a35458c702b33eeacc393d103063234e8bc28
      HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='$REV7', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
        > "$LOGS/setup-model7.log" 2>&1
      REFS7=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/refs
      mkdir -p "$REFS7" && printf %s "$REV7" > "$REFS7/main"
      bplayer=14
    else
      bplayer=12
    fi
    TOOL=behavior_prompts.py run bp-$name "$minutes" ${JOBARGS[3]} --model "$bpmodel" --layer $bplayer --device cuda \
      "${JOBARGS[@]:4}" --out $D/bp/$name
    ;;
  bz)          # ordinary steered inference in chaotic and periodic zones (run BZ; tools/behavior_zones.py run, float32 on
               # the GPU; model-written code is executed here, inside the pod): JOBARGS = NAME MINUTES then run arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=behavior_zones.py run bz-$name "$minutes" run --device cuda "${JOBARGS[@]:2}" --out $D/bz/$name
    ;;
  flipnf)      # flip normal form and its doubled-orbit tests (tools/flip_normal_form.py, float64 on the GPU): JOBARGS = NAME
               # MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=flip_normal_form.py run flipnf-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/flipnf/$name
    ;;
  spectrum)    # Lyapunov spectra by QR iteration (tools/lyapunov_spectrum.py, float64 on the GPU): JOBARGS = NAME MINUTES
               # then spectrum arguments (cells file in docs/, vectors uploaded with --extra-data if not the default)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=lyapunov_spectrum.py run spectrum-$name "$minutes" --device cuda --dtype float64 "${JOBARGS[@]:2}" \
      --out $D/hidden/spectrum-$name
    ;;
  pif64)       # program PI, float64: soft-thinking twins and float64 answers on GSM8K (tools/pi_f64_twins.py; model
               # downloaded by name): JOBARGS = NAME MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    get7b                         # the pinned Qwen2.5-7B-Instruct (the pod runs offline otherwise)
    TOOL=pi_f64_twins.py run pif64-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/pi/$name
    ;;
  pm)          # program PI, run PM (docs/pi-matrix-confirmatory.md, amendment 1): the confirmatory soft-thinking matrix,
               # bfloat16 on the GPU, via tools/pi_matrix_run.sh; JOBARGS = MINUTES
    minutes=${JOBARGS[0]}
    get7b
    log "download GSM8K 740312ad"
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/gsm8k', repo_type='dataset', revision='740312add88f781978c0658806c59bc2815b9866',
                  cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" > "$LOGS/setup-gsm8k.log" 2>&1
    uv pip install -q -p $PY "pandas==3.0.6" "pyarrow==26.0.0" > "$LOGS/setup-pandas.log" 2>&1
    mkdir -p /data2 && ln -sfn "$NEURAL_SHRIMP_DATA" /data2/neural-shrimp-data   # pi_harness's fixed cache path
    log "start pm"
    PY=$PY timeout "${minutes}m" bash tools/pi_matrix_run.sh "$OUT/$D/pi/pm" > "$LOGS/pm.log" 2>&1
    log "end pm status $?"
    cat "$LOGS/pm.log"
    ;;
  pm2f64)      # run PM2F (exploratory): float64 twins of reasoning-model soft thinking (tools/pi_reason_f64.py);
               # JOBARGS = MODEL MINUTES LIMIT GROUP
    model=${JOBARGS[0]}; minutes=${JOBARGS[1]}; lim=${JOBARGS[2]}; grp=${JOBARGS[3]}
    case $model in
      r1qwen7b)  REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; MREV=916b56a44061fd5cd7d6a8fb632557ed4f724f60 ;;
      r1llama8b) REPO=deepseek-ai/DeepSeek-R1-Distill-Llama-8B; MREV=6a6f4aa4197940add57724a7707d069478df56b1 ;;
      qwq32b)    REPO=Qwen/QwQ-32B; MREV=976055f8c83f394f35dbd3ab09a285a984907bd0 ;;
      *) log "unknown model $model"; exit 5 ;;
    esac
    for try in 1 2 3; do
      log "download $REPO $MREV (try $try)"
      HF_HUB_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=120 $PY -c "
from huggingface_hub import snapshot_download
c = '$NEURAL_SHRIMP_DATA/models/hf/hub'
snapshot_download('$REPO', revision='$MREV', cache_dir=c)
snapshot_download('openai/gsm8k', repo_type='dataset', revision='740312add88f781978c0658806c59bc2815b9866', cache_dir=c)" \
        >> "$LOGS/setup-pm2f64.log" 2>&1 && break
    done
    MDIR=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO//\//--}
    mkdir -p "$MDIR/refs" && printf %s "$MREV" > "$MDIR/refs/main"
    uv pip install -q -p $PY "pandas==3.0.6" "pyarrow==26.0.0" > "$LOGS/setup-pandas.log" 2>&1
    mkdir -p /data2 && ln -sfn "$NEURAL_SHRIMP_DATA" /data2/neural-shrimp-data
    log "start pm2f64-$model"
    timeout "${minutes}m" $PY tools/pi_reason_f64.py --model $model --limit $lim --group $grp --device cuda \
      --out "$OUT/$D/pi/pm2f64/$model" > "$LOGS/pm2f64-$model.log" 2>&1
    log "end pm2f64-$model status $?"
    ;;
  pm2diag)     # item 3 (17 Oct): why R1-Distill-Llama-8B answers poorly in PM2 bf16; 64 PM2F problems
               # (docs/pi/llama-diag-indices.json); JOBARGS = OUTNAME MINUTES
    name=${JOBARGS[0]}; minutes=${JOBARGS[1]}
    getreasoner r1llama8b
    end=$(( $(date +%s) + minutes * 60 ))
    D8="--model r1llama8b --task gsm8k --layouts 8 --limit 64"
    R="$OUT/$D/pi/$name"
    while read -r tag tool targs; do
      left=$(( (end - $(date +%s)) / 60 ))
      if [ $left -le 1 ]; then log "time budget spent: $tag skipped"; continue; fi
      log "start $tag"
      timeout "${left}m" $PY tools/$tool $targs --out "$R/$tag" > "$LOGS/$name-$tag.log" 2>&1 < /dev/null
      log "end $tag status $?"
    done <<JOBS
reason-plain-legacy pi_reason.py --model r1llama8b --task gsm8k --arm plain --layouts 1 --limit 64 --indices docs/pi/llama-diag-indices.json --tokenizer legacy
reason-plain pi_reason.py $D8 --arm plain
gen-bf16 pi_generate_diag.py --model r1llama8b --dtype bfloat16 --limit 64 --group 8
gen-fp32 pi_generate_diag.py --model r1llama8b --dtype float32 --limit 64 --group 8
reason-soft pi_reason.py $D8 --arm soft
reason-plain-ans1024 pi_reason.py $D8 --arm plain --answer-budget 1024
reason-plain-eager pi_reason.py $D8 --arm plain --attn eager
gen-bf16-4096 pi_generate_diag.py --model r1llama8b --dtype bfloat16 --limit 64 --group 8 --max-new-tokens 4096
JOBS
    ;;
  preason)     # pi_reason.py soft and plain arms under a new output name (items 3b and 4, 17 Oct);
               # JOBARGS = OUTNAME MODEL MINUTES THINK_BUDGET COLD_STOP_ENTROPY TASK:LIMIT...
               # (gsm8k layouts 8 16, math500 layouts 1 8; Cold Stop length 256 when the entropy is above 0)
    name=${JOBARGS[0]}; model=${JOBARGS[1]}; minutes=${JOBARGS[2]}; tb=${JOBARGS[3]}; cse=${JOBARGS[4]}
    getreasoner $model
    end=$(( $(date +%s) + minutes * 60 ))
    for arm in soft plain; do
      for tl in "${JOBARGS[@]:5}"; do
        task=${tl%%:*}; lim=${tl##*:}
        if [ "$task" = gsm8k ]; then lay="8 16"; else lay="1 8"; fi
        left=$(( (end - $(date +%s)) / 60 ))
        if [ $left -le 1 ]; then log "time budget spent: $model-$task-$arm skipped"; continue; fi
        log "start $model-$task-$arm"
        timeout "${left}m" $PY tools/pi_reason.py --model $model --task $task --arm $arm --layouts $lay \
          --limit $lim --think-budget $tb --cold-stop-entropy $cse --cold-stop-length 256 \
          --out "$OUT/$D/pi/$name/$model-$task-$arm" > "$LOGS/$name-$model-$task-$arm.log" 2>&1
        log "end $model-$task-$arm status $?"
      done
    done
    ;;
  pm2f64x)     # float64 twins extending a PM2F run without overlap (item 6, 17 Oct);
               # JOBARGS = OUTNAME MODEL MINUTES LIMIT GROUP SKIP_GROUPS
    name=${JOBARGS[0]}; model=${JOBARGS[1]}; minutes=${JOBARGS[2]}; lim=${JOBARGS[3]}; grp=${JOBARGS[4]}
    skip=${JOBARGS[5]}
    getreasoner $model
    log "start $name-$model"
    timeout "${minutes}m" $PY tools/pi_reason_f64.py --model $model --limit $lim --group $grp --skip-groups $skip \
      --device cuda --out "$OUT/$D/pi/$name/$model" > "$LOGS/$name-$model.log" 2>&1
    log "end $name-$model status $?"
    ;;
  pm2)         # run PM2 (docs/pi-matrix2-confirmatory.md): Soft Thinking with a reasoning model under two batch
               # layouts, bfloat16 (tools/pi_reason.py); JOBARGS = MODEL MINUTES TASK... (gsm8k, math500)
    model=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    case $model in
      r1qwen7b)  REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; MREV=916b56a44061fd5cd7d6a8fb632557ed4f724f60 ;;
      r1llama8b) REPO=deepseek-ai/DeepSeek-R1-Distill-Llama-8B; MREV=6a6f4aa4197940add57724a7707d069478df56b1 ;;
      qwq32b)    REPO=Qwen/QwQ-32B; MREV=976055f8c83f394f35dbd3ab09a285a984907bd0 ;;
      *) log "unknown model $model"; exit 5 ;;
    esac
    end=$(( $(date +%s) + minutes * 60 ))
    for try in 1 2 3; do
      log "download $REPO $MREV (try $try)"
      HF_HUB_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=120 $PY -c "
from huggingface_hub import snapshot_download
c = '$NEURAL_SHRIMP_DATA/models/hf/hub'
snapshot_download('$REPO', revision='$MREV', cache_dir=c)
snapshot_download('openai/gsm8k', repo_type='dataset', revision='740312add88f781978c0658806c59bc2815b9866', cache_dir=c)
snapshot_download('HuggingFaceH4/MATH-500', repo_type='dataset', revision='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
                  cache_dir=c)" >> "$LOGS/setup-pm2.log" 2>&1 && break
    done
    MDIR=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO//\//--}
    mkdir -p "$MDIR/refs" && printf %s "$MREV" > "$MDIR/refs/main"
    uv pip install -q -p $PY "pandas==3.0.6" "pyarrow==26.0.0" > "$LOGS/setup-pandas.log" 2>&1
    mkdir -p /data2 && ln -sfn "$NEURAL_SHRIMP_DATA" /data2/neural-shrimp-data   # pi_harness's fixed cache path
    for arm in soft plain; do                   # every soft arm first: the primary endpoint needs only those
      for task in "${JOBARGS[@]:2}"; do
        if [ "$task" = gsm8k ]; then lay="8 16"; lim=256; else lay="1 8"; lim=64; fi
        left=$(( (end - $(date +%s)) / 60 ))
        if [ $left -le 1 ]; then log "time budget spent: $model-$task-$arm skipped"; continue; fi
        log "start $model-$task-$arm"
        timeout "${left}m" $PY tools/pi_reason.py --model $model --task $task --arm $arm --layouts $lay \
          --limit $lim --out "$OUT/$D/pi/pm2/$model-$task-$arm" > "$LOGS/pm2-$model-$task-$arm.log" 2>&1
        log "end $model-$task-$arm status $?"
      done
    done
    ;;
  dss)         # superstable curves and doubly superstable points of one orbit (tools/doubly_superstable.py grid, float64
               # on the GPU): JOBARGS = NAME MINUTES then grid arguments (vectors uploaded with --extra-data)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=doubly_superstable.py run dss-$name "$minutes" grid --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/dss/$name
    ;;
  cascade)     # period-doubling flips with tools/llm_feigenbaum.py on the GPU (float64):
               # JOBARGS = NAME MINUTES then cascade arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=llm_feigenbaum.py run cascade-$name "$minutes" cascade --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/cascade/$name
    ;;
  p1ladder)    # misalignment proxy P1, phase 2 (docs/misalignment-dynamics-plan.md): sampled self-dialogue arms on
               # the GPU, 128 replicas x 40 turns at T 0.7 (vectors uploaded with --extra-data); each JOBARGS entry
               # is NAME:CONCEPT:ALPHA:SEED
    for arm in "${JOBARGS[@]}"; do
      IFS=: read -r name concept alpha seed sched <<< "$arm"        # optional 5th field: schedule with '/' separators
      extra=""
      [ -n "$sched" ] && extra="--schedule ${sched//\//:}"
      TOOL=explore_llm_msm.py run p1-$name 30 sample --device cuda --threads 8 --replicas 128 --turns 40 \
        --temperature 0.7 --seed $seed --vectors llm-phase-map/vectors-l12-p1b --concepts $concept --alphas $alpha \
        $extra --out $D/msm/p1-$name
    done
    ;;
  p1control)   # closed-loop exit controller under persistent hostile pressure (docs/misalignment-dynamics-plan.md):
               # each JOBARGS entry is NAME:ALPHA:SEED:MODE:CONCEPT:CALPHA:CTURNS, MODE none|constant|triggered,
               # counter-steer CALPHA along CONCEPT (triggered: for CTURNS turns after a flagged reply)
    for arm in "${JOBARGS[@]}"; do
      IFS=: read -r name alpha seed mode cconcept calpha cturns <<< "$arm"
      TOOL=explore_llm_msm.py run p1-$name 40 sample --device cuda --threads 8 --replicas 128 --turns 40 \
        --temperature 0.7 --seed $seed --vectors llm-phase-map/vectors-l12-p1b --concepts hostility-vs-kindness \
        --alphas $alpha --controller $mode --controller-concept $cconcept --controller-alpha $calpha \
        --controller-turns $cturns --out $D/msm/p1-$name
    done
    ;;
  p1detect)    # detection study: in-context states (layers 12, 18, 23) and look-ahead probes (K replies of the next
               # turn at the dose and at dose + 0.02, 40 tokens, scored by the pinned classifier); each JOBARGS entry
               # is NAME:ALPHA:SEED:REPLICAS:K
    for arm in "${JOBARGS[@]}"; do
      IFS=: read -r name alpha seed replicas k <<< "$arm"
      TOOL=explore_llm_msm.py run p1-$name 120 sample --device cuda --threads 8 --replicas $replicas --turns 40 \
        --temperature 0.7 --seed $seed --vectors llm-phase-map/vectors-l12-p1b --concepts hostility-vs-kindness \
        --alphas $alpha --record-layers 12,18,23 --probe-k $k --probe-delta 0.02 --probe-tokens 40 --probe-chunk 512 \
        --out $D/msm/p1-$name
    done
    ;;
  p1commit)    # detectability ceiling (tools/p1_committor.py rollout): K continuations of H turns from the selected
               # dialogue states (states.json uploaded with --extra-data); JOBARGS: STATES MINUTES then rollout arguments
    TOOL=p1_committor.py run p1-committor ${JOBARGS[1]} rollout --states ${JOBARGS[0]} "${JOBARGS[@]:2}" \
      --out $D/msm/p1-committor
    ;;
  huginn|huginnA|huginnS|hedge)  # Huginn-0125 in its own venv with transformers 4.55, as in the local llm env. huginn: JOBARGS NAME
               # MINUTES then huginn_relax_eval arguments. huginnA (Part A of the frozen test): JOBARGS START COUNT
               # SPEC_MINUTES READ_MINUTES; spectra of the gold rows (fixed-spectrum --arc-ids batch) are computed and
               # saved before the read-out run on the same questions. Exports uploaded with --extra-data.
    HREV=bb6621b65e90b6a4b9b29ef88dc83866d450470c
    if [ ! -x /root/venv-huginn/bin/python ]; then
      uv venv -q -p 3.12 /root/venv-huginn
      uv pip install -q -p /root/venv-huginn/bin/python "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126 \
        > "$LOGS/setup-huginn-torch.log" 2>&1
      uv pip install -q -p /root/venv-huginn/bin/python "transformers==4.55.0" "numpy==2.5.3" "scipy==1.18.1" \
        > "$LOGS/setup-huginn.log" 2>&1
      HF_HUB_OFFLINE=0 /root/venv-huginn/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('tomg-group-umd/huginn-0125', revision='$HREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
        >> "$LOGS/setup-huginn.log" 2>&1
      HREFS=$NEURAL_SHRIMP_DATA/models/hf/hub/models--tomg-group-umd--huginn-0125/refs
      mkdir -p "$HREFS" && printf %s "$HREV" > "$HREFS/main"
      /root/venv-huginn/bin/python -c "import torch, transformers; print('huginn venv', torch.__version__, transformers.__version__)" \
        | tee -a "$LOGS/versions.txt"
    fi
    if [ "$JOBSET" = huginnS ]; then           # spectra only: JOBARGS OUT MINUTES then question indices (Part A)
      PY=/root/venv-huginn/bin/python TOOL=explore_latent_loop.py run spectra-${JOBARGS[0]} "${JOBARGS[1]}" \
        fixed-spectrum --model huginn --dtype float32 --device cuda --arc-split test --arc-indices "${JOBARGS[@]:2}" \
        --arc-ids batch --lam 1.0 --out $D/huginn/${JOBARGS[0]}
    elif [ "$JOBSET" = hedge ]; then           # per-problem stability edges (tools/huginn_edge.py): JOBARGS NAME MINUTES ARGS
      PY=/root/venv-huginn/bin/python TOOL=huginn_edge.py run hedge-${JOBARGS[0]} "${JOBARGS[1]}" "${JOBARGS[@]:2}" \
        --device cuda --out $D/huginn/edge-${JOBARGS[0]}
    elif [ "$JOBSET" = huginn ]; then
      name=${JOBARGS[0]}
      minutes=${JOBARGS[1]}
      PY=/root/venv-huginn/bin/python TOOL=huginn_relax_eval.py run huginn-$name "$minutes" "${JOBARGS[@]:2}" \
        --out $D/huginn/$name
    else
      start=${JOBARGS[0]}; count=${JOBARGS[1]}
      idx=$(seq -s ' ' "$start" $((start + count - 1)))
      PY=/root/venv-huginn/bin/python TOOL=explore_latent_loop.py run partA-spectra "${JOBARGS[2]}" fixed-spectrum \
        --model huginn --dtype float32 --device cuda --arc-split test --arc-indices $idx --arc-ids batch --lam 1.0 \
        --out $D/huginn/partA-spectra
      PY=/root/venv-huginn/bin/python TOOL=huginn_relax_eval.py run partA-readout "${JOBARGS[3]}" arc --split test \
        --start "$start" --count "$count" --steps 24 25 26 27 28 29 30 31 32 --lams 1.0 1.05 1.08 1.10 --seed 0 \
        --dtype bfloat16 --device cuda --out $D/huginn/partA-readout
    fi
    ;;
  hverify)     # float64 verification of candidate windows of the hostility-steered map (vectors-l12-p1b,
               # hostility-vs-kindness): JOBARGS entries orbit:NAME:LOGT:ALPHA (Newton orbit at a cell) or
               # branch:NAME:LOGT:ALPHA:FREE:DIR:LO:HI (continuation in alpha or temperature from that cell)
    HV="--vectors llm-phase-map/vectors-l12-p1b --concept hostility-vs-kindness"
    for spec in "${JOBARGS[@]}"; do
      IFS=: read -r kind name lt al free dir lo hi <<< "$spec"
      case $kind in
        orbit)
          run horbit-$name 15 orbit --device cuda --dtype float64 --tol 1e-12 --max-newton 12 $HV \
            --log-temperature $lt --alpha $al --out $D/hverify/orbit-$name ;;
        branch)
          run hbranch-$name 40 branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp --steps 150 \
            --chunk 896 $HV --free $free --log-temperature $lt --alpha $al --direction $dir --ds 0.002 --ds-max 0.01 \
            --beyond 0.003 --bounds $lo $hi --out $D/hverify/branch-$name ;;
        *) echo "unknown hverify entry $spec" ;;
      esac
    done
    ;;
  hnormal)     # fold normal form of the hostile '!Yes' 2-cycle (accumulation point of the hostile cascade): float64
               # branch in temperature down to its fold from (log10 T -0.0445, alpha 0.083), then the normal form at the
               # branch point nearest the fold, eps in log10 T
    HV="--vectors llm-phase-map/vectors-l12-p1b --concept hostility-vs-kindness"
    run hnormal-branch 60 branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp --steps 200 \
      --chunk 896 $HV --period 2 --free temperature --log-temperature -0.0445 --alpha 0.083 --direction -1 \
      --ds 0.0002 --ds-max 0.001 --beyond 0.0003 --bounds -0.06 -0.04 --out $D/hnormal/branch-p2-fold
    run hnormal-nf 30 normal-form --device cuda --run $D/hnormal/branch-p2-fold --nearest-fold --param temperature \
      --out $D/hnormal/normal-form-p2
    ;;
  hnormal2)    # the exact fold of the hostile '!Yes' 2-cycle from the hnormal branch (curve start: fold located by
               # its test function), then the normal form there, eps in log10 T (run V5 refused a branch point at
               # multiplier 1.012); the uploaded branch is passed with --extra-data
    run hnormal2-curve 40 curve --device cuda --dtype float64 --reference-dtype none --no-exact-jvp --gmres-tol 1e-10 \
      --tol 1e-10 --gtol 1e-9 --max-newton 10 --ds-max 0.05 --chunk 896 --kind fold --start $D/hnormal/branch-p2-fold \
      --direction 1 --steps 3 --log-temperature-bounds -0.1 0.0 --alpha-bounds 0.06 0.10 --out $D/hnormal/curve-p2-fold
    run hnormal2-nf 30 normal-form --device cuda --run $D/hnormal/curve-p2-fold --index 0 --param temperature \
      --out $D/hnormal/normal-form-p2
    ;;
  scalex)      # another model (family): JOBARGS KEY REPO REV LAYER PHIL_MINUTES HOST_MINUTES. Downloads REPO@REV, builds
               # concept vectors (with the hostility:kindness contrast and 3 random directions) at LAYER on the GPU, a
               # timed 8x8 float64 scan, then coarse 64x64 float64 maps along philosophy and hostility-vs-kindness over
               # the same ranges as for Qwen 7B
    key=${JOBARGS[0]}; repo=${JOBARGS[1]}; rev=${JOBARGS[2]}; layer=${JOBARGS[3]}
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', revision='$rev', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" > "$LOGS/setup-$key.log" 2>&1
    refs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${repo//\//--}/refs
    mkdir -p "$refs" && printf %s "$rev" > "$refs/main"
    vec=llm-phase-map/vectors-$key-l$layer
    TOOL=explore_llm_phase_map.py run scalex-vectors 25 vectors --model $key --layer $layer --device cuda \
      --dtype float64 --contrasts hostility:kindness --random 3 --out $vec
    mkdir -p "$OUT/$D/scalex-$key" && cp -r "$OUT/$vec" "$OUT/$D/scalex-$key/"
    SX="scan --device cuda --dtype float64 --chunk 128 --tol 1e-8 --eps 1e-6 --model $key --layer $layer --vectors $vec"
    run scalex-timing 20 $SX --resolution 8 8 --alpha -0.12 0.12 --log-temperature -1.0 0.3 --out $D/scalex-$key/timing-8
    run scalex-philosophy "${JOBARGS[4]}" $SX --resolution 64 64 --alpha -0.12 0.12 --log-temperature -1.0 0.3 \
      --out $D/scalex-$key/coarse-64-philosophy
    run scalex-hostility "${JOBARGS[5]}" $SX --concept hostility-vs-kindness --resolution 64 64 --alpha 0.0 0.12 \
      --log-temperature -0.3 0.3 --out $D/scalex-$key/coarse-64-hostility
    ;;
  scale7)      # scale-up rung 2: Qwen2.5-7B-Instruct (pinned revision), layer 14 of 28: concept vectors (with the
               # hostility:kindness contrast and three random directions) built on the GPU, a timed 8x8 float64 scan,
               # then coarse 64x64 float64 maps along philosophy (the 0.5B and 1.5B range) and hostility-vs-kindness;
               # JOBARGS: MINUTES for the philosophy map, MINUTES for the hostility map (0 skips it)
    REV7=a09a35458c702b33eeacc393d103063234e8bc28
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='$REV7', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
      > "$LOGS/setup-model7.log" 2>&1
    REFS7=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/refs
    mkdir -p "$REFS7" && printf %s "$REV7" > "$REFS7/main"
    minutes=${JOBARGS[0]:-90}
    TOOL=explore_llm_phase_map.py run scale7-vectors 20 vectors --model qwen7b --layer 14 --device cuda \
      --dtype float64 --contrasts hostility:kindness --random 3 --out llm-phase-map/vectors-7b-l14
    mkdir -p "$OUT/$D/scale7" && cp -r "$OUT/llm-phase-map/vectors-7b-l14" "$OUT/$D/scale7/"
    S7="scan --device cuda --dtype float64 --chunk 128 --tol 1e-8 --eps 1e-6 --model qwen7b --layer 14
        --vectors llm-phase-map/vectors-7b-l14"
    run scale7-timing 20 $S7 --resolution 8 8 --alpha -0.12 0.12 --log-temperature -1.0 0.3 --out $D/scale7/timing-8
    run scale7-philosophy "$minutes" $S7 --resolution 64 64 --alpha -0.12 0.12 --log-temperature -1.0 0.3 \
      --out $D/scale7/coarse-64-philosophy
    hminutes=${JOBARGS[1]:-0}
    if [ "$hminutes" -gt 0 ]; then
      run scale7-hostility "$hminutes" $S7 --concept hostility-vs-kindness --resolution 64 64 --alpha 0.0 0.12 \
        --log-temperature -0.3 0.3 --out $D/scale7/coarse-64-hostility
    fi
    ;;
  p1x)         # P1 replication in another model (Task 12): JOBARGS MODEL LAYER LADDER_DOSES(comma) CTRL_SEED TAG; builds the
               # hostility:kindness vectors on the GPU, runs a ladder (128 x 40, seed 1) with a random-direction control
               # at the top dose, scores it on the pod, picks the dose whose hostile share in turns 5-39 is closest to
               # 0.5 (tools/p1_select_dose.py, rule fixed in the run card), then runs none, constant -0.02,
               # regenerate -0.04 (no hold) and post-reply trigger -0.06 x 3 at that dose
    model=${JOBARGS[0]}; layer=${JOBARGS[1]}; doses=${JOBARGS[2]}; cseed=${JOBARGS[3]:-2}; tag=${JOBARGS[4]:-run}
    vec=llm-phase-map/vectors-$model-l$layer-p1
    pre=p1x-$model-$tag                                   # every output folder carries the run tag (no clashes)
    TOOL=explore_llm_phase_map.py run p1x-vectors 20 vectors --model $model --layer $layer --device cuda \
      --dtype float64 --contrasts hostility:kindness --random 3 --out $vec
    mkdir -p "$OUT/$D/msm" && cp -r "$OUT/$vec" "$OUT/$D/msm/$pre-vectors"
    MSM="sample --model $model --layer $layer --device cuda --threads 8 --replicas 128 --turns 40 --temperature 0.7"
    ladder=()
    top=0
    for a in ${doses//,/ }; do
      dtag=$($PY -c "print(f'{round($a * 1000):03d}')")
      TOOL=explore_llm_msm.py run $pre-h$dtag 45 $MSM --seed 1 --vectors $vec --concepts hostility-vs-kindness \
        --alphas $a --out $D/msm/$pre-h$dtag
      $PY tools/hostility_score.py score --sample $D/msm/$pre-h$dtag --threads 8 --device cuda > /dev/null 2>&1
      ladder+=("$D/msm/$pre-h$dtag:$a")
      top=$a
    done
    TOOL=explore_llm_msm.py run $pre-r1 45 $MSM --seed 1 --vectors $vec --concepts random1 --alphas $top \
      --out $D/msm/$pre-r1
    dose=$($PY tools/p1_select_dose.py "${ladder[@]}" --target 0.5 2> "$OUT/$D/msm/$pre-dose.json")
    log "chosen dose $dose"
    cat "$OUT/$D/msm/$pre-dose.json"
    for arm in "none:0:3" "constant:-0.02:3" "regenerate:-0.04:0" "triggered:-0.06:3"; do
      IFS=: read -r mode calpha cturns <<< "$arm"
      TOOL=explore_llm_msm.py run $pre-ctl-$mode 45 $MSM --seed $cseed --vectors $vec \
        --concepts hostility-vs-kindness --alphas $dose --controller $mode --controller-concept hostility-vs-kindness \
        --controller-alpha $calpha --controller-turns $cturns --out $D/msm/$pre-ctl-$mode
    done
    ;;
  hverify7)    # float64 verification at Qwen2.5-7B (revision as in scale7, layer 14, vectors-7b-l14 uploaded with
               # --extra-data): entries as hverify (orbit:NAME:LOGT:ALPHA or branch:NAME:LOGT:ALPHA:FREE:DIR:LO:HI)
    REV7=a09a35458c702b33eeacc393d103063234e8bc28
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='$REV7', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
      > "$LOGS/setup-model7.log" 2>&1
    REFS7=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/refs
    mkdir -p "$REFS7" && printf %s "$REV7" > "$REFS7/main"
    HV7="--model qwen7b --layer 14 --vectors llm-phase-map/vectors-7b-l14 --concept hostility-vs-kindness"
    for spec in "${JOBARGS[@]}"; do
      IFS=: read -r kind name lt al free dir lo hi minutes dsmax <<< "$spec"   # optional MINUTES (50), DSMAX (0.01)
      case $kind in
        orbit)
          run h7orbit-$name 30 orbit --device cuda --dtype float64 --tol 1e-12 --max-newton 12 --chunk 128 $HV7 \
            --log-temperature $lt --alpha $al --out $D/hverify7/orbit-$name ;;
        branch)
          run h7branch-$name "${minutes:-50}" branch --device cuda --dtype float64 --reference-dtype none \
            --no-exact-jvp --steps 120 --chunk 128 $HV7 --free $free --log-temperature $lt --alpha $al \
            --direction $dir --ds 0.002 --ds-max "${dsmax:-0.01}" --beyond 0.003 --bounds $lo $hi \
            --out $D/hverify7/branch-$name ;;
        *) echo "unknown hverify7 entry $spec" ;;
      esac
    done
    ;;
  cusps)       # fold normal form (tools/cusp_check.py, float64 on the GPU) at points of uploaded fold-curve runs; each
               # JOBARGS entry is CURVE_RUN:INDICES (comma-separated), output next to the curve run
    for spec in "${JOBARGS[@]}"; do
      crun=${spec%%:*}; idx=${spec#*:}
      log "start cusp-check $crun $idx"
      timeout 90m $PY tools/cusp_check.py "$crun" "$idx" cuda 2> "$LOGS/cusp-$(basename $crun).log" \
        | grep --line-buffered '^{' > "$OUT/$crun/cusp-check.jsonl"
      log "end cusp-check $crun status $?"
      cut -c1-300 "$OUT/$crun/cusp-check.jsonl"
    done
    ;;
  codim2)      # codimension-two points on uploaded fold and flip curves (run CL, tools/cusp_locate.py, float64 on the GPU):
               # each JOBARGS entry is TEST:CURVE_RUN:LEFT (TEST cusp, crossing or reversal; LEFT the curve index before
               # the sign change), output llm-soft-loop/pod64/codim2/<curve name>-<test>-<left>
    for spec in "${JOBARGS[@]}"; do
      ctest=${spec%%:*}; rest=${spec#*:}; crun=${rest%:*}; left=${rest##*:}
      name=$(basename "$crun")-$ctest-$left
      TOOL=cusp_locate.py run cl-$name 120 "$crun" "$left" --test "$ctest" --device cuda --out $D/codim2/$name
    done
    ;;
  llamactrl)   # steering controls at the Goodfire SAE's layer (run C1): JOBARGS NAME MINUTES SCAN-ARGS...; official gated
               # weights (token on stdin, unset after the download) and the Goodfire layer-19 SAE (revisions as llama8b);
               # concept vectors at layer 19 of 32 built exactly as llama8b's layer-16 ones, then the frozen SAE feature
               # directions (docs/steering-controls-selection.json) and norm-matched Gaussian directions
               # (tools/steering_controls.py build), then one float64 scan into controls/NAME
    LREV=0e9e39f249a16976918f6564b8830bc894c89659
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', revision='$LREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'])" > "$LOGS/setup-llama.log" 2>&1
    unset HF_TOKEN
    lrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/refs
    mkdir -p "$lrefs" && printf %s "$LREV" > "$lrefs/main"
    SAED=$NEURAL_SHRIMP_DATA/models/sae
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Goodfire/Llama-3.1-8B-Instruct-SAE-l19', 'Llama-3.1-8B-Instruct-SAE-l19.pth',
                revision='f6775a221e47b44233af4bac2c7b65189265519a', local_dir='$SAED/goodfire-8b-l19')" \
      > "$LOGS/setup-sae.log" 2>&1
    (cd $NEURAL_SHRIMP_DATA/models && sha256sum sae/goodfire-8b-l19/*.pth \
      hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/$LREV/model-*.safetensors) | tee "$LOGS/llama-sae-sha256.txt"
    cvec=llm-phase-map/vectors-llama3.1-8b-l19
    TOOL=explore_llm_phase_map.py run ctrl-vectors 30 vectors --model llama3.1-8b --layer 19 --device cuda \
      --dtype float64 --contrasts hostility:kindness --random 3 --out $cvec
    TOOL=steering_controls.py run ctrl-build 15 build --concept-vectors $cvec --sae-kind goodfire \
      --sae-path $SAED/goodfire-8b-l19/Llama-3.1-8B-Instruct-SAE-l19.pth --layer 19 \
      --selection docs/steering-controls-selection.json --out llm-soft-loop/controls/vectors-llama8b-l19
    mkdir -p "$OUT/$D/controls/${JOBARGS[0]}"
    cp -r "$OUT/$cvec" "$OUT/$D/controls/${JOBARGS[0]}/concept-vectors"
    cp -r "$OUT/llm-soft-loop/controls/vectors-llama8b-l19" "$OUT/$D/controls/${JOBARGS[0]}/vectors"
    run llamactrl-${JOBARGS[0]} "${JOBARGS[1]}" scan --device cuda --dtype float64 --chunk 128 --tol 1e-8 --eps 1e-6 \
      --model llama3.1-8b --layer 19 --vectors llm-soft-loop/controls/vectors-llama8b-l19 "${JOBARGS[@]:2}" \
      --out $D/controls/${JOBARGS[0]}/scan
    ;;
  selfref)     # self-referential dynamics study (docs/selfref-dynamics-design.md): JOBARGS PART ...; official gated
               # Llama-3.1-8B-Instruct and the Goodfire layer-19 SAE (revisions as llama8b). PART p: per-text maximum
               # SAE activations on the uploaded designed corpus (llm_sae_readout.py profile-csv) for the frozen
               # feature rule (tools/selfref_features.py, run locally afterwards)
    LREV=0e9e39f249a16976918f6564b8830bc894c89659
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', revision='$LREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'])" > "$LOGS/setup-llama.log" 2>&1
    unset HF_TOKEN
    lrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/refs
    mkdir -p "$lrefs" && printf %s "$LREV" > "$lrefs/main"
    SAED=$NEURAL_SHRIMP_DATA/models/sae
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Goodfire/Llama-3.1-8B-Instruct-SAE-l19', 'Llama-3.1-8B-Instruct-SAE-l19.pth',
                revision='f6775a221e47b44233af4bac2c7b65189265519a', local_dir='$SAED/goodfire-8b-l19')" \
      > "$LOGS/setup-sae.log" 2>&1
    GF="goodfire:$SAED/goodfire-8b-l19/Llama-3.1-8B-Instruct-SAE-l19.pth:19"
    if [ "${JOBARGS[0]}" = p ]; then
      TOOL=llm_sae_readout.py run selfref-profile 60 profile-csv --model llama3.1-8b --device cuda --sae $GF \
        --corpus $NEURAL_SHRIMP_DATA/explore/selfref/corpus/public_sae_feature_mapping_corpus.csv \
        --out $D/selfref/profile
    fi
    ;;
  qwen35moe)   # Qwen3.5-35B-A3B chaos check (run M1): JOBARGS NAME MINUTES SCAN-ARGS...; downloads the pinned revision,
               # builds concept vectors at layer 20 of 40 in float64 (tools/qwen35_float64.py: float32 paths patched,
               # float32-stored tensors exact, bfloat16 experts used exactly), then one float64 scan into controls/NAME
    QREV=59d61f3ce65a6d9863b86d2e96597125219dc754
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-35B-A3B', revision='$QREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
      > "$LOGS/setup-qwen35.log" 2>&1
    qrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen3.5-35B-A3B/refs
    mkdir -p "$qrefs" && printf %s "$QREV" > "$qrefs/main"
    (cd $NEURAL_SHRIMP_DATA/models && sha256sum hf/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/$QREV/*.safetensors) \
      > "$LOGS/qwen35-sha256.txt"
    qvec=llm-phase-map/vectors-qwen3.5-35b-a3b-l20
    TOOL=explore_llm_phase_map.py run q35-vectors 40 vectors --model qwen3.5-35b-a3b --layer 20 --device cuda \
      --dtype float64 --contrasts hostility:kindness --random 3 --out $qvec
    mkdir -p "$OUT/$D/controls/${JOBARGS[0]}" && cp -r "$OUT/$qvec" "$OUT/$D/controls/${JOBARGS[0]}/vectors"
    run qwen35-${JOBARGS[0]} "${JOBARGS[1]}" scan --device cuda --dtype float64 --chunk 64 --tol 1e-8 --eps 1e-6 \
      --model qwen3.5-35b-a3b --layer 20 --vectors $qvec "${JOBARGS[@]:2}" --out $D/controls/${JOBARGS[0]}/scan
    ;;
  anymap)      # one float64 scan of an ungated model (run C1): JOBARGS NAME REPO REV MINUTES SCAN-ARGS...; downloads
               # REPO@REV (pinned; REPO '-': already cached), then the scan with free arguments into controls/NAME
    name=${JOBARGS[0]}; repo=${JOBARGS[1]}; rev=${JOBARGS[2]}; minutes=${JOBARGS[3]}
    if [ "$repo" != "-" ]; then
      HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', revision='$rev', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" > "$LOGS/setup-$name.log" 2>&1
      refs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${repo//\//--}/refs
      mkdir -p "$refs" && printf %s "$rev" > "$refs/main"
    fi
    run anymap-$name "$minutes" scan --device cuda --dtype float64 --tol 1e-8 --eps 1e-6 "${JOBARGS[@]:4}" \
      --out $D/controls/$name
    ;;
  llamamap)    # Llama-3.1-8B-Instruct high-resolution map tile (run L14): JOBARGS NAME MINUTES SCAN-ARGS...; official
               # gated weights (token on stdin, unset after the download), vectors uploaded from run L13
               # (llm-soft-loop/pod64/llama8b-b/vectors); one float64 scan tile into llama8b-hires/NAME
    LREV=0e9e39f249a16976918f6564b8830bc894c89659
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', revision='$LREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'])" > "$LOGS/setup-llama.log" 2>&1
    unset HF_TOKEN
    lrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/refs
    mkdir -p "$lrefs" && printf %s "$LREV" > "$lrefs/main"
    run llamamap-${JOBARGS[0]} "${JOBARGS[1]}" scan --device cuda --dtype float64 --chunk 128 --tol 1e-8 --eps 1e-6 \
      --model llama3.1-8b --layer 16 --vectors llm-soft-loop/pod64/llama8b-b/vectors "${JOBARGS[@]:2}" \
      --out $D/llama8b-hires/${JOBARGS[0]}
    ;;
  llama8b)     # Llama-3.1-8B-Instruct with SAEs (run L13): JOBARGS PART MAP_MINUTES. Downloads the official gated model
               # (meta-llama, revision pinned; the human's token arrives on stdin and is unset after the download), the
               # Goodfire layer-19 SAE and two seonglae layer-24 SAEs (pinned revisions); checks the SAEs' hook points
               # (llm_sae_readout validate); builds concept vectors at layer 16 of 32 in float64 exactly as scalex; then
               # PART a: a 64x64 float64 map along gratitude, PART b: hostility-vs-kindness and philosophy (ranges as for
               # Qwen 7B and OLMo-2); then the SAE readout at each map's largest families and most chaotic cells
    part=${JOBARGS[0]}; mapmin=${JOBARGS[1]}
    LREV=0e9e39f249a16976918f6564b8830bc894c89659
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', revision='$LREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'])" > "$LOGS/setup-llama.log" 2>&1
    unset HF_TOKEN
    lrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/refs
    mkdir -p "$lrefs" && printf %s "$LREV" > "$lrefs/main"
    SAED=$NEURAL_SHRIMP_DATA/models/sae
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Goodfire/Llama-3.1-8B-Instruct-SAE-l19', 'Llama-3.1-8B-Instruct-SAE-l19.pth',
                revision='f6775a221e47b44233af4bac2c7b65189265519a', local_dir='$SAED/goodfire-8b-l19')
for sid in ('Llama-3.1-8B_blocks.24.hook_resid_pre_16384_topk_80_6e-05_42_faithful-llama3.1-8b_512',
            'Llama-3.1-8B_blocks.24.hook_resid_pre_16384_topk_80_6e-05_42_fineweb_512'):
    for f in ('cfg.json', 'sae_weights.safetensors'):
        hf_hub_download('seonglae/Llama-3.1-8B-sae', sid + '/' + f, revision='f4a4201e452eb40ba5d10b4177606567eba543db',
                        local_dir='$SAED/seonglae')" > "$LOGS/setup-sae.log" 2>&1
    (cd $NEURAL_SHRIMP_DATA/models && sha256sum sae/goodfire-8b-l19/*.pth sae/seonglae/*/sae_weights.safetensors \
      hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/$LREV/model-*.safetensors) | tee "$LOGS/llama-sae-sha256.txt"
    LDIR=$D/llama8b-$part
    GF="goodfire:$SAED/goodfire-8b-l19/Llama-3.1-8B-Instruct-SAE-l19.pth:19"
    SF="saelens:$SAED/seonglae/Llama-3.1-8B_blocks.24.hook_resid_pre_16384_topk_80_6e-05_42_faithful-llama3.1-8b_512"
    SW="saelens:$SAED/seonglae/Llama-3.1-8B_blocks.24.hook_resid_pre_16384_topk_80_6e-05_42_fineweb_512"
    TOOL=llm_sae_readout.py run llama-sae-validate 20 validate --model llama3.1-8b --device cuda --sae $GF \
      --sae $SF --sae $SW --out $LDIR/sae-validate
    if [ "$part" = p ]; then                      # PART p: feature profiles on held-out category texts
      TOOL=llm_sae_readout.py run llama-sae-profile 20 profile --model llama3.1-8b --device cuda --sae $GF \
        --sae $SF --sae $SW --out $LDIR/sae-profile
    fi
    if [ "$part" = a ] || [ "$part" = b ]; then   # PART v / p: no maps
      lvec=llm-phase-map/vectors-llama3.1-8b-l16
      TOOL=explore_llm_phase_map.py run llama-vectors 30 vectors --model llama3.1-8b --layer 16 --device cuda \
        --dtype float64 --contrasts hostility:kindness --random 3 --out $lvec
      mkdir -p "$OUT/$LDIR" && cp -r "$OUT/$lvec" "$OUT/$LDIR/vectors"
      LSX="scan --device cuda --dtype float64 --chunk 128 --tol 1e-8 --eps 1e-6 --model llama3.1-8b --layer 16"
      LSX="$LSX --vectors $lvec --resolution 64 64"
      if [ "$part" = a ]; then
        maps="gratitude:-0.12:0.12:-1.0:0.3"
      else
        maps="hostility-vs-kindness:0.0:0.12:-0.3:0.3 philosophy:-0.12:0.12:-1.0:0.3"
      fi
      for m in $maps; do
        IFS=: read -r concept alo ahi tlo thi <<< "$m"
        run llama-map-$concept "$mapmin" $LSX --concept $concept --alpha $alo $ahi --log-temperature $tlo $thi \
          --out $LDIR/coarse-64-$concept
        TOOL=llm_sae_readout.py run llama-read-$concept 30 read --model llama3.1-8b --device cuda --layer 16 \
          --vectors $lvec --concept $concept --cells-from-scan $LDIR/coarse-64-$concept --sae $GF --sae $SF --sae $SW \
          --out $LDIR/readout-$concept
      done
    fi
    ;;
  pairmap)     # same architecture, other weights (run S10): JOBARGS KEY REPO REV LAYER MAP_MINUTES. Downloads REPO@REV
               # (REPO '-': already cached), builds concept vectors at LAYER on the GPU in float64 exactly as scalex,
               # two 128x128 float64 maps along gratitude and philosophy (alpha -0.12..0.12, log10 T -1.0..0.3, the
               # philosophy range of the other models), then sampled self-dialogue along gratitude at alpha 0, 0.01,
               # 0.02 and 0.04 (128 replicas x 40 turns, T 0.7, seed 1), identical for every model
    key=${JOBARGS[0]}; repo=${JOBARGS[1]}; rev=${JOBARGS[2]}; layer=${JOBARGS[3]}; mapmin=${JOBARGS[4]}
    if [ "$repo" != "-" ]; then
      HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', revision='$rev', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" > "$LOGS/setup-$key.log" 2>&1
      refs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${repo//\//--}/refs
      mkdir -p "$refs" && printf %s "$rev" > "$refs/main"
    fi
    vec=llm-phase-map/vectors-$key-l$layer-pair
    TOOL=explore_llm_phase_map.py run pair-vectors 25 vectors --model $key --layer $layer --device cuda \
      --dtype float64 --contrasts hostility:kindness --random 3 --out $vec
    mkdir -p "$OUT/$D/pairmap-$key" && cp -r "$OUT/$vec" "$OUT/$D/pairmap-$key/vectors"
    SX="scan --device cuda --dtype float64 --chunk 2048 --tol 1e-8 --eps 1e-6 --transient 500 --samples 300 --record 80"
    SX="$SX --max-period 48 --model $key --layer $layer --vectors $vec --resolution 128 128 --alpha -0.12 0.12"
    run pair-gratitude "$mapmin" $SX --concept gratitude --log-temperature -1.0 0.3 \
      --out $D/pairmap-$key/map128-gratitude
    run pair-philosophy "$mapmin" $SX --concept philosophy --log-temperature -1.0 0.3 \
      --out $D/pairmap-$key/map128-philosophy
    PMSM="sample --model $key --layer $layer --device cuda --threads 8 --replicas 128 --turns 40 --temperature 0.7"
    for a in 0 0.01 0.02 0.04; do
      gtag=$($PY -c "print(f'{round($a * 1000):03d}')")
      TOOL=explore_llm_msm.py run pair-grat$gtag 30 $PMSM --seed 1 --vectors $vec --concepts gratitude \
        --alphas $a --out $D/msm/pair-$key-grat$gtag
    done
    ;;
  pilot2)      # cusp-unfolding pilot stage 2: JOBARGS KEY REPO REV BRANCH_RUN OUT_PREFIX STEPS MINUTES TLO THI ALO AHI:
               # fold curve of the stage-1 branch both ways, cusp checks with c and the unfolding test at every
               # tangent reversal (tools/pilot_stage2.py)
    key=${JOBARGS[0]}; repo=${JOBARGS[1]}; rev=${JOBARGS[2]}
    if [ "$repo" != "-" ]; then
      HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', revision='$rev', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" > "$LOGS/setup-$key.log" 2>&1
      prefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${repo//\//--}/refs
      mkdir -p "$prefs" && printf %s "$rev" > "$prefs/main"
    fi
    log "start pilot-stage2"
    timeout 230m $PY tools/pilot_stage2.py "${JOBARGS[@]:3}" 2> "$LOGS/pilot-stage2.err" | tee "$LOGS/pilot-stage2.out"
    log "end pilot-stage2 status $?"
    ;;
  pilot1)      # cusp-unfolding pilot stage 1 (docs/qwen2-cusp-unfolding-pilot.md): JOBARGS KEY REPO REV LAYER VECTORS
               # CONCEPT LOGT ALPHA. Downloads REPO@REV (REPO '-': cached), then a float64 Newton orbit at the frozen
               # candidate's interior cell and its branch continued in temperature (+-0.05) and in alpha (+-0.02),
               # both ways, with every crossing classified (curves follow in a later stage, from a fold found here)
    key=${JOBARGS[0]}; repo=${JOBARGS[1]}; rev=${JOBARGS[2]}; layer=${JOBARGS[3]}; pvec=${JOBARGS[4]}
    pcon=${JOBARGS[5]}; plt=${JOBARGS[6]}; pal=${JOBARGS[7]}
    if [ "$repo" != "-" ]; then
      HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', revision='$rev', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" > "$LOGS/setup-$key.log" 2>&1
      prefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${repo//\//--}/refs
      mkdir -p "$prefs" && printf %s "$rev" > "$prefs/main"
    fi
    PV="--model $key --layer $layer --vectors $pvec --concept $pcon"
    PD=$D/pilot-$key
    run pilot-orbit 20 orbit --device cuda --dtype float64 --tol 1e-12 --max-newton 12 $PV --log-temperature $plt \
      --alpha $pal --out $PD/orbit
    tlo=$($PY -c "print($plt - 0.05)"); thi=$($PY -c "print($plt + 0.05)")
    alo=$($PY -c "print($pal - 0.02)"); ahi=$($PY -c "print($pal + 0.02)")
    for spec in "temperature:1:Tup:$tlo:$thi" "temperature:-1:Tdown:$tlo:$thi" "alpha:1:aup:$alo:$ahi" \
                "alpha:-1:adown:$alo:$ahi"; do
      IFS=: read -r free dir tag lo hi <<< "$spec"
      run pilot-branch-$tag 35 branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp \
        --steps 150 --chunk 896 $PV --free $free --log-temperature $plt --alpha $pal --direction $dir --ds 0.002 \
        --ds-max 0.01 --beyond 0.003 --bounds $lo $hi --out $PD/branch-$tag
    done
    ;;
  unfold)      # cusp unfolding pilot (run U1; consult 6): tools/cusp_unfolding.py at curve points; JOBARGS entries
               # CURVE_RUN:INDEX; output unfolding-INDEX.json next to the curve run (predictions written before solving)
    for spec in "${JOBARGS[@]}"; do
      crun=${spec%%:*}; idx=${spec#*:}
      log "start unfolding $crun $idx"
      timeout 90m $PY tools/cusp_unfolding.py "$crun" "$idx" cuda 2> "$LOGS/unfold-$(basename $crun)-$idx.log" \
        | tee "$LOGS/unfold-$(basename $crun)-$idx.out"
      log "end unfolding $crun status $?"
    done
    ;;
  cubic)       # cusp nondegeneracy (run V11): fold normal form with the cubic coefficient c (tools/cusp_check.py ...
               # cubic) at points of uploaded fold-curve runs; JOBARGS entries CURVE_RUN:INDICES; output
               # cusp-cubic.jsonl next to the curve run (never over an earlier cusp-check.jsonl)
    for spec in "${JOBARGS[@]}"; do
      crun=${spec%%:*}; idx=${spec#*:}
      log "start cusp-cubic $crun $idx"
      timeout 60m $PY tools/cusp_check.py "$crun" "$idx" cuda cubic 2> "$LOGS/cubic-$(basename $crun).log" \
        | grep --line-buffered '^{' > "$OUT/$crun/cusp-cubic.jsonl"
      log "end cusp-cubic $crun status $?"
      cut -c1-300 "$OUT/$crun/cusp-cubic.jsonl"
    done
    ;;
  hcascade)    # the hostile hub's period-7 interior (run S9): float64 bifurcation lines with attractor projections
               # (a temperature line at alpha 0.0655 and an alpha line at log10 T 0.1830), Newton orbits at period-14,
               # -28 and -21 cells of the M3b map, and the period-14 branch in temperature both ways (needs --extra-data
               # explore/llm-phase-map/vectors-l12-p1b)
    HV="--vectors llm-phase-map/vectors-l12-p1b --concept hostility-vs-kindness"
    SC="scan --device cuda --dtype float64 --chunk 700 --tol 1e-8 --eps 1e-6 --max-period 48 --projections $HV"
    run hcascade-tline 40 $SC --log-temperature 0.1760 0.1935 --alpha 0.0655 0.0655 --resolution 1400 1 \
      --transient 1000 --samples 400 --record 160 --out $D/hcascade/tline-a0.0655
    run hcascade-aline 40 $SC --log-temperature 0.1830 0.1830 --alpha 0.0635 0.0680 --resolution 1 1400 \
      --transient 1000 --samples 400 --record 160 --out $D/hcascade/aline-t0.1830
    for spec in p14:0.18250:0.065518 p28:0.18470:0.065644 p21:0.17980:0.065267; do
      IFS=: read -r name lt al <<< "$spec"
      run hcascade-orbit-$name 20 orbit --device cuda --dtype float64 --tol 1e-12 --max-newton 12 $HV \
        --log-temperature $lt --alpha $al --out $D/hcascade/orbit-$name
    done
    for dir in 1 -1; do
      run hcascade-p14-T$dir 40 branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp \
        --steps 150 --chunk 896 $HV --free temperature --log-temperature 0.18250 --alpha 0.065518 --direction $dir \
        --ds 0.002 --ds-max 0.01 --beyond 0.003 --bounds 0.1790 0.1880 --out $D/hcascade/branch-p14-T$dir
    done
    ;;
  hcurves)     # float64 fold/flip curves of the hostility-steered map from uploaded branch runs (--extra-data); each
               # JOBARGS entry is NAME:START:KIND:DIR:STEPS:MINUTES:DS:DSMAX:TLO:THI:ALO:AHI
    for spec in "${JOBARGS[@]}"; do
      IFS=: read -r name start kind dir steps minutes ds dsmax tlo thi alo ahi <<< "$spec"
      run hcurve-$name "$minutes" $CURVE --steps "$steps" --start "$start" --kind "$kind" --direction "$dir" \
        --ds "$ds" --ds-max "$dsmax" --log-temperature-bounds "$tlo" "$thi" --alpha-bounds "$alo" "$ahi" \
        --out $D/hcurves/$name
    done
    ;;
  scan64)      # one float64 scan with free arguments: scan64 OUT-NAME MINUTES SCAN-ARGS... (zooms to be read without
               # the float32 precision caveat; on an H100 float64 costs about the same as float32 for this model)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    run scan64-$name "$minutes" scan --device cuda --dtype float64 --chunk 2048 --tol 1e-8 --eps 1e-6 \
      "${JOBARGS[@]:2}" --out $D/scan64/$name
    ;;
  sheets)      # attractor sheets over the parameter plane (run SH; tools/sheets_map.py, float64 on the GPU):
               # JOBARGS = NAME MINUTES MODE ARGS...; seeds uploaded with --extra-data explore/sheets/seeds
    TOOL=sheets_map.py run sheets-${JOBARGS[0]} "${JOBARGS[1]}" "${JOBARGS[2]}" --device cuda --threads 8 \
      "${JOBARGS[@]:3}" --out $D/sheets/${JOBARGS[0]}
    ;;
  transects)   # run SH local analyses on the GPU, in order: JOBARGS = NAME MINUTES KIND ARGS [:: NAME MINUTES KIND ARGS ...]
               # KIND transect (tools/sheets_transect.py) or edge (tools/sheets_edge.py); seeds via --extra-data explore/sheets/seeds
    i=0; n=${#JOBARGS[@]}
    while [ $i -lt $n ]; do
      name=${JOBARGS[$i]}; minutes=${JOBARGS[$((i+1))]}; kind=${JOBARGS[$((i+2))]}; i=$((i+3)); margs=()
      while [ $i -lt $n ] && [ "${JOBARGS[$i]}" != "::" ]; do margs+=("${JOBARGS[$i]}"); i=$((i+1)); done
      i=$((i+1))
      TOOL=sheets_$kind.py run $kind-$name "$minutes" --device cuda --threads 8 "${margs[@]}" --out $D/sheets/$kind-$name
    done
    ;;
  p5temp)      # period-5 shrimp pilot: the window's head followed to lower temperature (tools/p5_temperature.py, float64
               # on the GPU): JOBARGS = NAME MINUTES ARGS...; vectors-l4 and vectors-l16 uploaded with --extra-data
    TOOL=p5_temperature.py run p5temp-${JOBARGS[0]} "${JOBARGS[1]}" --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/sheets/p5temp-${JOBARGS[0]}
    ;;
  p5dss)       # period-5 shrimp step 2: window search and superstability grid at one temperature (tools/p5_dss_temperature.py,
               # float64 on the GPU): JOBARGS = NAME MINUTES ARGS...; vectors-l4/l16 uploaded with --extra-data
    TOOL=p5_dss_temperature.py run p5dss-${JOBARGS[0]} "${JOBARGS[1]}" --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/sheets/p5dss-${JOBARGS[0]}
    ;;
  ltrans)      # run LT: long transients behind chaos labels (tools/long_transients.py, float64 on the GPU):
               # JOBARGS = NAME MINUTES ARGS... (cells file docs/lt-cells.json uploaded with the code)
    TOOL=long_transients.py run lt-${JOBARGS[0]} "${JOBARGS[1]}" --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/lt/${JOBARGS[0]}
    ;;
  certcal)     # Task 34: one Codex certificate shard packet (tools/cert_remote.py). JOBARGS = NAME PACKET COMMIT PACKET_SHA256,
               # PACKET relative to the checkout. Uploaded data: codex/remote/NAME/{cpython-3.12.14.tar,src.tar} (the exact
               # interpreter build and a sparse partial clone at COMMIT), the packet's upstream directories and vectors-l12; the
               # Qwen2.5-0.5B snapshot is the pinned revision downloaded in the setup above. Every step refuses, never adapts.
    name=${JOBARGS[0]}; packet=${JOBARGS[1]}; commit=${JOBARGS[2]}; packet_sha=${JOBARGS[3]}
    CO=$OUT/llm-soft-loop/pod64/certcal/$name
    mkdir -p "$CO" /data2/neural-shrimp-runtime/python /data2/neural-shrimp-data
    RD=$NEURAL_SHRIMP_DATA/codex/remote/$name
    tar xf "$RD/cpython-3.12.14.tar" -C /data2/neural-shrimp-runtime/python
    CPY=/data2/neural-shrimp-runtime/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
    uv venv -q -p $CPY /root/certvenv > "$CO/setup-venv.log" 2>&1
    uv pip install -q -p /root/certvenv/bin/python "numpy==2.5.3" "python-flint==0.9.0" "safetensors==0.8.0" \
      "threadpoolctl==3.7.0" "tokenizers==0.23.3" \
      > "$CO/setup-pip.log" 2>&1
    uv pip install -q -p /root/certvenv/bin/python "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130 \
      >> "$CO/setup-pip.log" 2>&1
    CV=/root/certvenv/bin/python
    tar --no-same-owner -xf "$RD/src.tar" -C /root            # owned by the pod's user, so git accepts the repository
    chown -R "$(id -u):$(id -g)" /root/cert-src
    git -C /root/cert-src remote set-url origin /root/cert-src
    $CV -I tools/cert_remote.py prepare --packet /root/cert-src/$packet --src-data $NEURAL_SHRIMP_DATA \
      --record "$CO/prepare.json" > "$CO/prepare.log" 2>&1
    log "prepare status $?"
    CMD=$($CV -I -c "import json,sys; c=json.load(open(sys.argv[1]))['command']; assert c.startswith('python '); print(c)" \
          /root/cert-src/$packet)
    CMD="$CV ${CMD#python }"
    if $CV -I tools/cert_remote.py preflight --packet /root/cert-src/$packet --repo /root/cert-src --commit $commit \
         --packet-sha256 $packet_sha --record "$CO/preflight.json" > "$CO/preflight.log" 2>&1; then
      log "preflight passed: launching $CMD"
      $CV -I tools/cert_remote.py run --repo /root/cert-src --cmd "$CMD" --deadline-seconds 3900 --rss-limit-gib 18 \
        --stdout "$CO/launch.log" --record "$CO/run.json" > "$CO/run.log" 2>&1
      log "run status $?"
    else
      log "preflight refused: no launch"; cat "$CO/preflight.log"
    fi
    $CV -I tools/cert_remote.py pack --packet /root/cert-src/$packet --out "$CO" > "$CO/pack.log" 2>&1
    $CV -I -c "import sys,numpy,torch,flint; print(sys.version); print(numpy.__version__, torch.__version__, flint.__version__, flint.__FLINT_VERSION__)" > "$CO/backend.txt" 2>&1
    uv pip freeze -p /root/certvenv/bin/python > "$CO/pip-freeze.txt" 2>&1
    free -g > "$CO/free.txt"; df -h /data2 > "$CO/df.txt"; nproc > "$CO/nproc.txt"; lscpu > "$CO/lscpu.txt"
    cat "$CO/pack.log"
    ;;
  shrimpc)     # run P5C: doubly superstable point of the period-5 window's reduction (tools/shrimp_centre.py, float64 on the
               # GPU): JOBARGS = NAME MINUTES ARGS...; vectors-l4/l16 uploaded with --extra-data
    TOOL=shrimp_centre.py run shrimpc-${JOBARGS[0]} "${JOBARGS[1]}" --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/shrimp/${JOBARGS[0]}
    ;;
  hidden)      # hidden chaos along a line (tools/hidden_chaos.py): JOBARGS NAME MINUTES ARGS...; float64 on the GPU,
               # Qwen2.5-0.5B (cached), twins at two offsets, leading-token statistics, into hidden/NAME
    TOOL=hidden_chaos.py run hidden-${JOBARGS[0]} "${JOBARGS[1]}" line --device cuda --dtype float64 \
      "${JOBARGS[@]:2}" --out $D/hidden/${JOBARGS[0]}
    ;;
  foldlaw7)    # fold law at Qwen2.5-7B (run FL7, philosophy at layer 14, vectors uploaded): JOBARGS PART ALPHA. Part
               # branch: the period-2 branch in temperature from log10 T -0.25 up to its fold, then the fold's normal
               # form; part line: tools/hidden_chaos.py on docs/fl7-line-cells.json, one twin offset (window positions)
    REV7=a09a35458c702b33eeacc393d103063234e8bc28
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='$REV7', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub')" \
      > "$LOGS/setup-model7.log" 2>&1
    REFS7=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/refs
    mkdir -p "$REFS7" && printf %s "$REV7" > "$REFS7/main"
    M7="--model qwen7b --layer 14 --vectors llm-phase-map/vectors-7b-l14 --concept philosophy"
    case ${JOBARGS[0]} in
      branch) run fl7-branch 100 branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp $M7 \
                --alpha "${JOBARGS[1]}" --log-temperature -0.25 --period 0 --free temperature --direction 1 \
                --steps 60 --ds 0.002 --ds-max 0.006 --beyond 0.004 --bounds -0.32 -0.15 --chunk 512 \
                --out $D/fl7/branch-p2
              run fl7-normal-form 40 normal-form --device cuda --run $D/fl7/branch-p2 --nearest-fold \
                --param temperature --chunk 512 --out $D/fl7/normal-form-p2 ;;
      nf) run fl7-foldpoint 60 curve --device cuda --dtype float64 --reference-dtype none --no-exact-jvp \
            --start $D/fl7/branch-p2 --kind fold --steps 2 --ds 1e-4 --ds-max 1e-4 --tol 1e-10 --gtol 1e-9 \
            --max-newton 10 --chunk 512 --log-temperature-bounds -0.4 0.0 --alpha-bounds 0.0 0.1 \
            --out $D/fl7/curve-fold-p2
          run fl7-normal-form-b 60 normal-form --device cuda --run $D/fl7/curve-fold-p2 --index 0 \
            --param temperature --chunk 512 --out $D/fl7/normal-form-p2 ;;
      line) TOOL=hidden_chaos.py run hidden-fl7-line 140 cells --device cuda --dtype float64 \
              --cells-file docs/fl7-line-cells.json $M7 --transient 2000 --record 400 --eps 1e-6 --tol 1e-8 \
              --chunk 250 --out $D/fl7/line ;;
    esac
    ;;
  foldlaw15)   # fold law at Qwen2.5-1.5B (run FL15, philosophy at layer 14, vectors uploaded): JOBARGS PART ALPHA. Part
               # branch: the period-4 branch in temperature from log10 T -0.105 up to its fold, then the fold's normal
               # form; part line: tools/hidden_chaos.py on docs/fl15-line-cells.json (the window fan above the fold)
    M15="--model qwen1.5b --layer 14 --vectors llm-phase-map/vectors-1.5b-l14 --concept philosophy"
    case ${JOBARGS[0]} in
      branch) run fl15-branch 60 branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp $M15 \
                --alpha "${JOBARGS[1]}" --log-temperature -0.105 --period 0 --free temperature --direction 1 \
                --steps 60 --ds 0.001 --ds-max 0.003 --beyond 0.003 --bounds -0.12 -0.07 --chunk 1536 \
                --out $D/fl15/branch-p4
              run fl15-normal-form 40 normal-form --device cuda --run $D/fl15/branch-p4 --nearest-fold \
                --param temperature --chunk 1536 --out $D/fl15/normal-form-p4 ;;
      nf) run fl15-foldpoint 40 curve --device cuda --dtype float64 --reference-dtype none --no-exact-jvp \
            --start $D/fl15/branch-p4 --kind fold --steps 2 --ds 1e-4 --ds-max 1e-4 --tol 1e-10 --gtol 1e-9 \
            --max-newton 10 --chunk 1536 --log-temperature-bounds -0.2 0.0 --alpha-bounds 0.0 0.1 \
            --out $D/fl15/curve-fold-p4
          run fl15-normal-form-b 40 normal-form --device cuda --run $D/fl15/curve-fold-p4 --index 0 \
            --param temperature --chunk 1536 --out $D/fl15/normal-form-p4 ;;
      line) TOOL=hidden_chaos.py run hidden-fl15-line 110 cells --device cuda --dtype float64 \
              --cells-file docs/fl15-line-cells.json $M15 --transient 3000 --record 600 --eps 1e-6 1e-9 --tol 1e-8 \
              --chunk 850 --out $D/fl15/line ;;
    esac
    ;;
  hiddencells) # tools/hidden_chaos.py on an uploaded cells file (float64 on the GPU): JOBARGS NAME MINUTES CELLS_FILE ARGS...,
               # output hidden/NAME
    TOOL=hidden_chaos.py run hidden-${JOBARGS[0]} "${JOBARGS[1]}" cells --device cuda --dtype float64 \
      --cells-file "${JOBARGS[2]}" "${JOBARGS[@]:3}" --out $D/hidden/${JOBARGS[0]}
    ;;
  horizon)     # horizon check H2 (tools/hidden_chaos.py, float64, twins at 1e-6 and 1e-9, 1000 transient + 2000 steps):
               # JOBARGS LINE_MINUTES CELLS_MINUTES. (1) the exact grid of the hostile cascade line of Figure hubcascade
               # (hcascade/tline-a0.0655: alpha 0.0655, log10 T 0.176..0.1935, 1400 cells); (2) the stratified random
               # cells of two philosophy maps listed in docs/h2-horizon-cells.json
    TOOL=hidden_chaos.py run h2-line "${JOBARGS[0]}" line --device cuda --dtype float64 \
      --vectors llm-phase-map/vectors-l12-p1b --concept hostility-vs-kindness --alpha 0.0655 \
      --log-temperature 0.176 0.1935 --n 1400 --transient 1000 --record 2000 --eps 1e-6 1e-9 --chunk 700 \
      --out $D/hidden/h2-line-exact
    TOOL=hidden_chaos.py run h2-cells "${JOBARGS[1]}" cells --device cuda --dtype float64 \
      --vectors llm-phase-map/vectors-l12 --concept philosophy --cells-file docs/h2-horizon-cells.json \
      --transient 1000 --record 2000 --eps 1e-6 1e-9 --chunk 448 --out $D/hidden/h2-cells
    ;;
  tangentcheck) # tangent check TC (review response): JOBARGS CELLS_MINUTES SPECTRUM_MINUTES. (1) the 448 H2 cells of
               # docs/h2-horizon-cells.json exactly as in H2 (float64, 1000 transient + 2000 steps, twins 1e-6 and 1e-9),
               # plus the exact tangent exponent and the running exponents after 250, 500, 1000 and 2000 steps; (2) the
               # four leading exponents by QR iteration (tools/lyapunov_spectrum.py) at the 8 cells listed in
               # docs/tc-spectrum-cells.json, same horizon
    TOOL=hidden_chaos.py run tc-cells "${JOBARGS[0]}" cells --device cuda --dtype float64 \
      --vectors llm-phase-map/vectors-l12 --concept philosophy --cells-file docs/h2-horizon-cells.json \
      --transient 1000 --record 2000 --eps 1e-6 1e-9 --chunk 448 --tangent --checkpoints 250 500 1000 2000 \
      --out $D/hidden/tc-h2-cells-tangent
    TOOL=lyapunov_spectrum.py run tc-spectrum "${JOBARGS[1]}" --device cuda --dtype float64 \
      --vectors llm-phase-map/vectors-l12 --concept philosophy --cells-file docs/h2-horizon-cells.json \
      --cells-index 10 47 56 143 240 293 242 406 --k 4 --transient 1000 --record 2000 \
      --checkpoints 250 500 1000 2000 --out $D/hidden/tc-spectrum
    ;;
  bigsweep)    # large dense models (runs B1, B2, B4): JOBARGS MODEL MINUTES [name=SUFFIX] [SCAN-ARGS...] (output
               # big/MODEL-alpha0[-SUFFIX]; later scan arguments override the defaults). Downloads the pinned checkpoint
               # (gated Llama: the human's token arrives on stdin and is unset after the download), then an unsteered
               # float64 temperature sweep with the decoder's linear weights kept in their stored bfloat16 and used in
               # float64 (exact; tests/test_bf16_storage.py) and the layers spread over all GPUs (device auto)
    model=${JOBARGS[0]}
    case $model in
      qwen72b)      REPO=Qwen/Qwen2.5-72B-Instruct;         MREV=495f39366efef23836d0cfae4fbe635880d2be31 ;;
      llama3.3-70b) REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
      llama3.1-70b) REPO=meta-llama/Llama-3.1-70B-Instruct; MREV=1605565b47bb9346c5515c34102e054115b4f98b ;;
      *) log "unknown big model $model"; exit 2 ;;
    esac
    log "download $REPO"
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)" > "$LOGS/setup-big.log" 2>&1
    unset HF_TOKEN
    log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1)"
    mrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}/refs
    mkdir -p "$mrefs" && printf %s "$MREV" > "$mrefs/main"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader > "$LOGS/gpus.txt" 2>&1
    suffix=""; extra=("${JOBARGS[@]:2}")
    if [[ "${extra[0]:-}" == name=* ]]; then suffix="-${extra[0]#name=}"; extra=("${extra[@]:1}"); fi
    run bigsweep-$model "${JOBARGS[1]}" scan --device auto --weights bf16 --dtype float64 --model $model --vectors "" \
      --alpha 0 0 --log-temperature -1 0.3 --resolution 128 1 --transient 1000 --samples 2000 --max-period 64 \
      --tol 1e-8 --eps 1e-6 --chunk 128 "${extra[@]}" --out $D/big/$model-alpha0$suffix
    ;;
  bigctrl)     # Llama-3.3-70B-Instruct with the Goodfire layer-50 SAE (runs S70P, S70M; the 8B protocol of run C1 at 70B):
               # JOBARGS PART [MINUTES KEY SCAN-ARGS...]. Gated checkpoint and SAE, pinned; the token arrives on stdin and
               # is unset after the downloads. Float64 with bfloat16 weight storage over all GPUs (exact).
               # PART prof: SAE hook-point check and held-out category profiles (llm_sae_readout validate, profile),
               #   for the frozen selection rule (tools/steering_controls.py select, run locally afterwards).
               # PART map: concept vectors at layer 50 of 80 (hostility:kindness, 3 random), the SAE directions of the
               #   uploaded frozen selection (docs/steering-controls-selection-70b.json) and norm-matched Gaussians
               #   (steering_controls build), then one float64 scan of direction KEY into controls70/KEY
    part=${JOBARGS[0]}
    REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b
    SAED=$NEURAL_SHRIMP_DATA/models/sae
    log "download $REPO and the SAE"
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)
hf_hub_download('Goodfire/Llama-3.3-70B-Instruct-SAE-l50', 'Llama-3.3-70B-Instruct-SAE-l50.pt',
                revision='128ee921ecd1b8b3a87d776cbcc357c0855da134', local_dir='$SAED/goodfire-70b-l50')" \
      > "$LOGS/setup-big.log" 2>&1
    unset HF_TOKEN
    mrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--meta-llama--Llama-3.3-70B-Instruct/refs
    mkdir -p "$mrefs" && printf %s "$MREV" > "$mrefs/main"
    SAEP=$SAED/goodfire-70b-l50/Llama-3.3-70B-Instruct-SAE-l50.pt
    sha256sum $SAEP | tee "$LOGS/sae70-sha256.txt"
    BIG="--model llama3.3-70b --device auto --dtype float64 --weights bf16"
    if [ "$part" = prof ]; then
      TOOL=llm_sae_readout.py run s70-validate 40 validate $BIG --sae goodfire:$SAEP:50 --out $D/controls70/validate
      TOOL=llm_sae_readout.py run s70-profile 60 profile $BIG --sae goodfire:$SAEP:50 --out $D/controls70/sae-profile
    else
      cvec=llm-phase-map/vectors-llama3.3-70b-l50
      TOOL=explore_llm_phase_map.py run s70-vectors 40 vectors $BIG --layer 50 --contrasts hostility:kindness \
        --random 3 --out $cvec
      TOOL=steering_controls.py run s70-build 15 build --concept-vectors $cvec --sae-kind goodfire --sae-path $SAEP \
        --layer 50 --selection docs/steering-controls-selection-70b.json --out llm-soft-loop/controls/vectors-llama70b-l50
      key=${JOBARGS[2]}
      sargs=("${JOBARGS[@]:3}"); kdir=$key        # optional name=SUFFIX (zoom bands): output controls70/KEY-SUFFIX
      if [[ "${sargs[0]:-}" == name=* ]]; then kdir="$key-${sargs[0]#name=}"; sargs=("${sargs[@]:1}"); fi
      mkdir -p "$OUT/$D/controls70/$kdir"
      cp -r "$OUT/$cvec" "$OUT/$D/controls70/$kdir/concept-vectors"
      cp -r "$OUT/llm-soft-loop/controls/vectors-llama70b-l50" "$OUT/$D/controls70/$kdir/vectors"
      if [ "$part" = cells ]; then   # plateau test (run PT): JOBARGS cells MINUTES KEY name=SUFFIX CELLS-FILE [ARGS...]
        TOOL=hidden_chaos.py run s70-$kdir "${JOBARGS[1]}" cells $BIG --layer 50 --chunk 32 --tol 1e-8 \
          --vectors llm-soft-loop/controls/vectors-llama70b-l50 --concept $key --cells-file "${sargs[@]}" \
          --out $D/controls70/$kdir/cells
      else
        run s70-$kdir "${JOBARGS[1]}" scan $BIG --layer 50 --chunk 256 --tol 1e-8 --eps 1e-6 \
          --vectors llm-soft-loop/controls/vectors-llama70b-l50 --concept $key "${sargs[@]}" \
          --out $D/controls70/$kdir/scan
      fi
    fi
    ;;
  bigmap)      # large dense models, one steered float64 map (run B3): JOBARGS MODEL LAYER MINUTES CONCEPT [SCAN-ARGS...].
               # Pinned checkpoint as in bigsweep; concept vectors at LAYER built on the pod (hostility:kindness and three
               # norm-matched random directions, exactly as for every model), then one scan of CONCEPT into
               # big/MODEL-lLAYER-CONCEPT; float64 with bfloat16 weight storage over all GPUs (exact)
    model=${JOBARGS[0]}; layer=${JOBARGS[1]}; concept=${JOBARGS[3]}
    case $model in
      qwen72b)      REPO=Qwen/Qwen2.5-72B-Instruct;         MREV=495f39366efef23836d0cfae4fbe635880d2be31 ;;
      llama3.3-70b) REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
      *) log "unknown big model $model"; exit 2 ;;
    esac
    log "download $REPO"
    HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)" > "$LOGS/setup-big.log" 2>&1
    unset HF_TOKEN
    mrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}/refs
    mkdir -p "$mrefs" && printf %s "$MREV" > "$mrefs/main"
    BIG="--model $model --device auto --dtype float64 --weights bf16"
    cvec=llm-phase-map/vectors-$model-l$layer
    TOOL=explore_llm_phase_map.py run big-vectors 40 vectors $BIG --layer $layer --contrasts hostility:kindness \
      --random 3 --out $cvec
    bdir=$D/big/$model-l$layer-$concept
    mkdir -p "$OUT/$bdir" && cp -r "$OUT/$cvec" "$OUT/$bdir/vectors"
    run bigmap-$model "${JOBARGS[2]}" scan $BIG --layer $layer --chunk 128 --tol 1e-8 --eps 1e-6 --vectors $cvec \
      --concept $concept "${JOBARGS[@]:4}" --out $bdir/scan
    ;;
  gainsplit)   # where the loop's gain comes from (runs G1, G2; tools/gain_split.py): JOBARGS MODE SPEC [SPEC...], MODE small
               # (one GPU, float64) or big (float64 with bf16 storage over all GPUs), SPEC = MODEL:LOG10T_LO:LOG10T_HI;
               # unsteered, 24 temperatures, 500 transient + 1000 recorded steps, twins at relative offset 1e-6. Each
               # pinned checkpoint is downloaded (gated Llama: token on stdin, unset after the downloads), measured and
               # then deleted from the pod's disk
    mode=${JOBARGS[0]}
    if [ "$mode" = big ]; then GDEV="--device auto --weights bf16"; else GDEV="--device cuda --weights native"; fi
    for spec in "${JOBARGS[@]:1}"; do
      IFS=: read -r model lo hi <<< "$spec"
      case $model in
        llama3.1-8b)  REPO=meta-llama/Llama-3.1-8B-Instruct;  MREV=0e9e39f249a16976918f6564b8830bc894c89659 ;;
        qwen7b)       REPO=Qwen/Qwen2.5-7B-Instruct;          MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
        qwen72b)      REPO=Qwen/Qwen2.5-72B-Instruct;         MREV=495f39366efef23836d0cfae4fbe635880d2be31 ;;
        llama3.3-70b) REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
        llama3.1-70b) REPO=meta-llama/Llama-3.1-70B-Instruct; MREV=1605565b47bb9346c5515c34102e054115b4f98b ;;
        *) log "unknown model $model"; continue ;;
      esac
      log "download $REPO"
      # plain downloads (no xet chunk cache, which keeps a second copy and filled the disk in G2)
      HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
      log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1), free $(df -h --output=avail / | tail -1)"
      mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
      mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
      TOOL=gain_split.py run gain-$model 60 --model $model $GDEV --log-temperature $lo $hi --n 24 \
        --transient 500 --record 1000 --eps 1e-6 --out $D/gain/$model
      case "$mdir" in "$NEURAL_SHRIMP_DATA"/models/hf/hub/models--?*) rm -rf -- "$mdir" ;; esac
      rm -rf -- "$HOME/.cache/huggingface/xet"
    done
    unset HF_TOKEN
    ;;
  bighidden)   # large dense models, robust long-horizon line test (runs B6..): JOBARGS MODEL NAME MINUTES ARGS... for
               # tools/hidden_chaos.py line (twins at two offsets, leading-token statistics), unsteered (--vectors-none),
               # float64 with bf16 storage over all GPUs; pinned checkpoint as in bigsweep; into big/MODEL-NAME
    model=${JOBARGS[0]}
    case $model in
      qwen72b)      REPO=Qwen/Qwen2.5-72B-Instruct;         MREV=495f39366efef23836d0cfae4fbe635880d2be31 ;;
      llama3.3-70b) REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
      llama3.1-70b) REPO=meta-llama/Llama-3.1-70B-Instruct; MREV=1605565b47bb9346c5515c34102e054115b4f98b ;;
      *) log "unknown big model $model"; exit 2 ;;
    esac
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)" > "$LOGS/setup-big.log" 2>&1
    unset HF_TOKEN
    log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1)"
    mrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}/refs
    mkdir -p "$mrefs" && printf %s "$MREV" > "$mrefs/main"
    mode=line; hargs=("${JOBARGS[@]:3}")      # optional first argument cells=FILE: cells mode (plateau test, run PT)
    if [[ "${hargs[0]:-}" == cells=* ]]; then mode=cells; hargs=(--cells-file "${hargs[0]#cells=}" "${hargs[@]:1}"); fi
    TOOL=hidden_chaos.py run bighidden-$model "${JOBARGS[2]}" $mode --model $model --device auto --weights bf16 \
      --dtype float64 --vectors-none --alpha 0 "${hargs[@]}" --out $D/big/$model-${JOBARGS[1]}
    ;;
  q72fix)      # consult 8 follow-up after the float64 hook fix (runs QFa, QFb): Qwen2.5-72B-Instruct pinned as in
               # bighidden, float64 with bf16 storage over all GPUs, tools/hidden_chaos.py with exact tangent exponents
               # (--tangent) and the one-step precision probe (--fd-probe). JOBARGS PART MINUTES [ARGS...]:
               # PART cells: cells mode on the uploaded B3 concept vectors (llm-phase-map/vectors-qwen72b-l40-b3,
               #   layer 40, hostility-vs-kindness) with ARGS (cells file, offsets, steps), into big/qwen72b-fix-cells
               # PART lines: unsteered lines of 12 temperatures, ARGS = IDX:LO:HI ... (prefix index of
               #   docs/prefix-scale-prefixes.json and log10 T range), into big/qwen72b-fix-pIDX
    part=${JOBARGS[0]}; minutes=${JOBARGS[1]}
    REPO=Qwen/Qwen2.5-72B-Instruct; MREV=495f39366efef23836d0cfae4fbe635880d2be31
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)" > "$LOGS/setup-big.log" 2>&1
    unset HF_TOKEN
    log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1)"
    mrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/refs
    mkdir -p "$mrefs" && printf %s "$MREV" > "$mrefs/main"
    QF="--model qwen72b --device auto --weights bf16 --dtype float64 --tangent --fd-probe"
    if [ "$part" = clamp ]; then   # run QC: JOBARGS clamp MINUTES LO HI N IDX [IDX...]: entropy-clamped lines
      lo=${JOBARGS[2]}; hi=${JOBARGS[3]}; nt=${JOBARGS[4]}
      for idx in "${JOBARGS[@]:5}"; do
        TOOL=hidden_chaos.py run qc-p$idx "$minutes" line --model qwen72b --device auto --weights bf16 --dtype float64 \
          --fd-probe --vectors-none --alpha 0 --prefix-file docs/prefix-scale-prefixes.json --prefix-index $idx \
          --clamp-entropy $lo $hi --n $nt --eps 1e-5 1e-6 1e-7 --transient 500 --record 1000 --chunk 32 \
          --out $D/big/qwen72b-clamp-p$idx
      done
    elif [ "$part" = cells ]; then
      TOOL=hidden_chaos.py run qf-cells "$minutes" cells $QF --layer 40 --vectors llm-phase-map/vectors-qwen72b-l40-b3 \
        --concept hostility-vs-kindness "${JOBARGS[@]:2}" --out $D/big/qwen72b-fix-cells
    else
      for spec in "${JOBARGS[@]:2}"; do
        IFS=: read -r idx lo hi <<< "$spec"
        TOOL=hidden_chaos.py run qf-p$idx "$minutes" line $QF --vectors-none --alpha 0 \
          --prefix-file docs/prefix-scale-prefixes.json --prefix-index $idx --log-temperature $lo $hi --n 12 \
          --eps 1e-5 1e-6 1e-7 --transient 400 --record 800 --chunk 16 --out $D/big/qwen72b-fix-p$idx
      done
    fi
    ;;
  scaleset)    # scaling series (run SC): JOBARGS MODEL [MODEL...]. For each model, sequentially: pinned checkpoint (plain
               # download; gated ones with the token on stdin), an unsteered float64 sweep (bf16 storage, layers over all
               # GPUs; 160 temperatures over log10 T -1..0.6, 1000 + 2000 steps, twin 1e-6, max period 64) into
               # big/MODEL-alpha0-wide, then the gain split (tools/gain_split.py, 24 temperatures over log10 T -0.3..0.9,
               # 500 + 1000 steps) into gain/MODEL-wide; the checkpoint and the download cache are then deleted
    for model in "${JOBARGS[@]}"; do
      case $model in
        qwen0.5b)    REPO=Qwen/Qwen2.5-0.5B-Instruct;        MREV=7ae557604adf67be50417f59c2c2f167def9a775 ;;
        qwen1.5b)    REPO=Qwen/Qwen2.5-1.5B-Instruct;        MREV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306 ;;
        qwen7b)      REPO=Qwen/Qwen2.5-7B-Instruct;          MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
        llama3.1-8b) REPO=meta-llama/Llama-3.1-8B-Instruct;  MREV=0e9e39f249a16976918f6564b8830bc894c89659 ;;
        olmo2-7b)    REPO=allenai/OLMo-2-1124-7B-Instruct;   MREV=470b1fba1ae01581f270116362ee4aa1b97f4c84 ;;
        qwen3b)      REPO=Qwen/Qwen2.5-3B-Instruct;          MREV=aa8e72537993ba99e69dfaafa59ed015b17504d1 ;;
        qwen14b)     REPO=Qwen/Qwen2.5-14B-Instruct;         MREV=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ;;
        qwen32b)     REPO=Qwen/Qwen2.5-32B-Instruct;         MREV=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd ;;
        olmo2-1b)    REPO=allenai/OLMo-2-0425-1B-Instruct;   MREV=48d788eca847d4d7548f375ad03d3c9312f6139e ;;
        olmo2-13b)   REPO=allenai/OLMo-2-1124-13B-Instruct;  MREV=3a5c85baefbb1896a54d56fe2e76c0395627ddf4 ;;
        olmo2-32b)   REPO=allenai/OLMo-2-0325-32B-Instruct;  MREV=b96024342a77a69aa0dda815c3454a671f477463 ;;
        *) log "unknown model $model"; continue ;;
      esac
      [ "$model" = qwen0.5b ] && true                    # already cached by the setup step
      log "download $REPO"
      HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
      log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1), free $(df -h --output=avail / | tail -1)"
      mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
      mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
      run sweep-$model 150 scan --device auto --weights bf16 --dtype float64 --model $model --vectors "" \
        --alpha 0 0 --log-temperature -1 0.6 --resolution 160 1 --transient 1000 --samples 2000 --max-period 64 \
        --tol 1e-8 --eps 1e-6 --chunk 160 --out $D/big/$model-alpha0-wide
      TOOL=gain_split.py run gain-$model 60 --model $model --device auto --weights bf16 --log-temperature -0.3 0.9 \
        --n 24 --transient 500 --record 1000 --eps 1e-6 --out $D/gain/$model-wide
      if [ "$model" != qwen0.5b ]; then
        case "$mdir" in "$NEURAL_SHRIMP_DATA"/models/hf/hub/models--?*) rm -rf -- "$mdir" ;; esac
      fi
      rm -rf -- "$HOME/.cache/huggingface/xet"
    done
    unset HF_TOKEN
    ;;
  softthink)   # growing-context soft thinking twins (runs ST): JOBARGS MODEL [MODEL...]. Per model (pinned, plain download):
               # tools/soft_thinking_twins.py, unsteered, paper's prompt, 24 temperatures over log10 T -0.3..0.6, 400 steps,
               # 3 twins at eps0 1e-10, float64 with bf16 storage on one GPU, into big/MODEL-softthink
    for model in "${JOBARGS[@]}"; do
      case $model in
        qwen0.5b)    REPO=Qwen/Qwen2.5-0.5B-Instruct;        MREV=7ae557604adf67be50417f59c2c2f167def9a775 ;;
        qwen7b)      REPO=Qwen/Qwen2.5-7B-Instruct;          MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
        *) log "unknown model $model"; continue ;;
      esac
      log "download $REPO"
      HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
      mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
      mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
      TOOL=soft_thinking_twins.py run st-$model 60 --model $model --device auto --weights bf16 \
        --log-temperature -0.3 0.6 --n 24 --steps 400 --twins 3 --eps0 1e-10 --chunk 24 --out $D/big/$model-softthink
    done
    ;;
  softthink72) # growing-context soft thinking twins at 72B (run ST72): JOBARGS [PART...] (default p0 steer p3; p3fine and
               # p3coarse run cells 24-35 and 0-23 of the prefix-3 file, into big/qwen72b-softthink-p3fine/-p3coarse).
               # Qwen2.5-72B-Instruct pinned as in q72fix,
               # float64 with bf16 storage over all GPUs (hook fix), tools/soft_thinking_twins.py, 400 steps, 3 twins at
               # eps0 1e-10, 24 temperatures per batch: (1) paper's prompt, unsteered, 24 temperatures over log10 T
               # -0.3..0.6 (the ST1 grid); (2) paper's prompt, QFa's 8 steered candidates and 3 matched alpha-0 cells
               # (cells 0-10 of docs/q72fix-cells.json; uploaded B3 vectors, layer 40, hostility-vs-kindness); (3) prefix 3,
               # the 36 cells of docs/st72-cells-p3.json (the ST1 grid and the QFb line through its chaotic band); into
               # big/qwen72b-softthink-{p0,steer,p3}
    REPO=Qwen/Qwen2.5-72B-Instruct; MREV=495f39366efef23836d0cfae4fbe635880d2be31
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', 'model-*.safetensors'], max_workers=16)" > "$LOGS/setup-big.log" 2>&1
    unset HF_TOKEN
    log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1)"
    mrefs=$NEURAL_SHRIMP_DATA/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/refs
    mkdir -p "$mrefs" && printf %s "$MREV" > "$mrefs/main"
    ST="--model qwen72b --device auto --weights bf16 --steps 400 --twins 3 --eps0 1e-10 --chunk 24"
    P3="--cells-file docs/st72-cells-p3.json --prefix-file docs/prefix-scale-prefixes.json --prefix-index 3"
    parts=("${JOBARGS[@]}"); [ ${#parts[@]} -eq 0 ] && parts=(p0 steer p3)
    for part in "${parts[@]}"; do
      case $part in
        p0) TOOL=soft_thinking_twins.py run st72-p0 15 $ST --log-temperature -0.3 0.6 --n 24 \
              --out $D/big/qwen72b-softthink-p0 ;;
        steer) TOOL=soft_thinking_twins.py run st72-steer 12 $ST --cells-file docs/q72fix-cells.json \
                 --cells-index 0 1 2 3 4 5 6 7 8 9 10 --layer 40 --vectors llm-phase-map/vectors-qwen72b-l40-b3 \
                 --concept hostility-vs-kindness --out $D/big/qwen72b-softthink-steer ;;
        p3) TOOL=soft_thinking_twins.py run st72-p3 20 $ST $P3 --out $D/big/qwen72b-softthink-p3 ;;
        p3fine) TOOL=soft_thinking_twins.py run st72-p3fine 12 $ST $P3 --cells-index $(seq -s ' ' 24 35) \
                  --out $D/big/qwen72b-softthink-p3fine ;;
        p3coarse) TOOL=soft_thinking_twins.py run st72-p3coarse 15 $ST $P3 --cells-index $(seq -s ' ' 0 23) \
                    --out $D/big/qwen72b-softthink-p3coarse ;;
        *) log "unknown part $part" ;;
      esac
    done
    ;;
  window)      # chaos with a longer fed-back memory (run WIN): JOBARGS MODEL [MODEL...]. Per model (pinned, plain
               # download) and K in 1 2 4 8 16 soft tokens fed back: tools/hidden_chaos.py line, unsteered, paper's prompt,
               # 24 temperatures over log10 T -0.3..0.6, twins 1e-5 and 1e-7, 500 + 1000 steps, float64 with bf16 storage
               # on one GPU, into big/MODEL-window-K
    for model in "${JOBARGS[@]}"; do
      case $model in
        qwen0.5b)    REPO=Qwen/Qwen2.5-0.5B-Instruct;        MREV=7ae557604adf67be50417f59c2c2f167def9a775 ;;
        qwen1.5b)    REPO=Qwen/Qwen2.5-1.5B-Instruct;        MREV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306 ;;
        qwen7b)      REPO=Qwen/Qwen2.5-7B-Instruct;          MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
        *) log "unknown model $model"; continue ;;
      esac
      log "download $REPO"
      HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
      mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
      mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
      for K in 1 2 4 8 16; do
        TOOL=hidden_chaos.py run win-$model-$K 60 line --model $model --device auto --weights bf16 --dtype float64 \
          --vectors-none --alpha 0 --window $K --log-temperature -0.3 0.6 --n 24 --eps 1e-5 1e-7 \
          --transient 500 --record 1000 --chunk 24 --out $D/big/$model-window-$K
      done
    done
    ;;
  damping)     # where the loop damps a change of its fed-back token (run D; tools/damping_locate.py): JOBARGS MODE MODEL
               # [MODEL...], MODE small (one GPU) or big (layers over all GPUs); float64 arithmetic with the decoder
               # linears in their stored bfloat16 (exact). Per pinned checkpoint (gated Llama: token on stdin, unset after
               # the downloads): unsteered, paper's prompt, 24 temperatures over log10 T -0.3..0.9, 500 transient + 500
               # recorded steps, twin 1e-7, knockouts every 25 recorded steps at five temperatures, into
               # damping/d1/MODEL; the checkpoint and the download cache are then deleted
    mode=${JOBARGS[0]}
    if [ "$mode" = big ]; then GDEV="--device auto --weights bf16"; else GDEV="--device cuda --weights bf16"; fi
    for model in "${JOBARGS[@]:1}"; do
      case $model in
        qwen7b)       REPO=Qwen/Qwen2.5-7B-Instruct;          MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
        qwen14b)      REPO=Qwen/Qwen2.5-14B-Instruct;         MREV=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ;;
        qwen32b)      REPO=Qwen/Qwen2.5-32B-Instruct;         MREV=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd ;;
        qwen72b)      REPO=Qwen/Qwen2.5-72B-Instruct;         MREV=495f39366efef23836d0cfae4fbe635880d2be31 ;;
        llama3.1-8b)  REPO=meta-llama/Llama-3.1-8B-Instruct;  MREV=0e9e39f249a16976918f6564b8830bc894c89659 ;;
        llama3.1-70b) REPO=meta-llama/Llama-3.1-70B-Instruct; MREV=1605565b47bb9346c5515c34102e054115b4f98b ;;
        llama3.3-70b) REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
        olmo2-1b)     REPO=allenai/OLMo-2-0425-1B-Instruct;   MREV=48d788eca847d4d7548f375ad03d3c9312f6139e ;;
        olmo2-7b)     REPO=allenai/OLMo-2-1124-7B-Instruct;   MREV=470b1fba1ae01581f270116362ee4aa1b97f4c84 ;;
        olmo2-13b)    REPO=allenai/OLMo-2-1124-13B-Instruct;  MREV=3a5c85baefbb1896a54d56fe2e76c0395627ddf4 ;;
        olmo2-32b)    REPO=allenai/OLMo-2-0325-32B-Instruct;  MREV=b96024342a77a69aa0dda815c3454a671f477463 ;;
        *) log "unknown model $model"; continue ;;
      esac
      log "download $REPO"
      HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
      log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1), free $(df -h --output=avail / | tail -1)"
      mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
      mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
      TOOL=damping_locate.py run damping-$model 120 --model $model $GDEV --log-temperature -0.3 0.9 --n 24 \
        --transient 500 --record 500 --eps 1e-7 --knockout-every 25 --knockout-cells 2 7 12 17 22 --threads 16 \
        --out $D/damping/d1/$model
      case "$mdir" in "$NEURAL_SHRIMP_DATA"/models/hf/hub/models--?*) rm -rf -- "$mdir" ;; esac
      rm -rf -- "$HOME/.cache/huggingface/xet"
    done
    unset HF_TOKEN
    ;;
  moe)         # mixture-of-experts study (run MOE; float64, one GPU): JOBARGS PART [MODEL...]. PART moe: OLMoE-1B-7B
               # (pinned; unsteered sweep of 130 temperatures over log10 T -1..1.2 and run D) and Qwen3.5-35B-A3B
               # (pinned; run D), both with the routing analysis (tools/damping_locate.py, tools/moe_routing.py). PART dense
               # MODEL...: dense Qwen3.5 controls (pinned), each with concept vectors at layer 16 of 32, an unsteered
               # sweep (128 temperatures over log10 T -1..0.3, 1000 + 2000 steps), philosophy and hostility-vs-kindness
               # maps at 32x32 with the MoE maps' ranges and steps, and run D; into moe/MODEL-* and damping/d1/MODEL
    part=${JOBARGS[0]}
    getq35() {   # $1 key, $2 repo, $3 revision: the whole pinned repository (resumed up to 5 times; every shard of the
                 # index checked before use: a run on a partial checkpoint would fail later or, worse, not at all)
      local ok=0
      for attempt in 1 2 3 4 5; do
        log "download $2 $3 (attempt $attempt)"
        HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=120 $PY -c "
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
p = Path(snapshot_download('$2', revision='$3', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub', max_workers=8))
idx = p / 'model.safetensors.index.json'
shards = sorted(set(json.loads(idx.read_text())['weight_map'].values())) if idx.exists() else ['model.safetensors']
missing = [s for s in shards if not (p / s).exists() or (p / s).stat().st_size == 0]
print('shards', len(shards), 'missing', missing)
sys.exit(1 if missing else 0)" >> "$LOGS/setup-$1.log" 2>&1 && { ok=1; break; }
        sleep 20
      done
      log "download $([ $ok = 1 ] && echo done || echo FAILED) $(du -shL $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1), free $(df -h --output=avail / | tail -1)"
      local refs="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${2/\//--}/refs"
      mkdir -p "$refs" && printf %s "$3" > "$refs/main"
      [ $ok = 1 ]
    }
    damp() {     # $1 model key
      TOOL=damping_locate.py run damping-$1 150 --model $1 --device cuda --weights native --log-temperature -0.3 0.9 \
        --n 24 --transient 500 --record 500 --eps 1e-7 --knockout-every 25 --knockout-cells 2 7 12 17 22 --threads 16 \
        --out $D/damping/d1/$1
    }
    if [ "$part" = moe ]; then
      getq35 olmoe-1b-7b allenai/OLMoE-1B-7B-0125-Instruct b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e && \
      run sweep-olmoe 120 scan --device cuda --dtype float64 --chunk 130 --tol 1e-8 --eps 1e-6 --model olmoe-1b-7b \
        --vectors "" --alpha 0 0 --log-temperature -1 1.2 --resolution 130 1 --transient 1000 --samples 2000 \
        --max-period 64 --out $D/moe/olmoe-1b-7b-alpha0-wide
      damp olmoe-1b-7b
      getq35 qwen3.5-35b-a3b Qwen/Qwen3.5-35B-A3B 59d61f3ce65a6d9863b86d2e96597125219dc754
      damp qwen3.5-35b-a3b
    else
      for model in "${JOBARGS[@]:1}"; do
        case $model in
          qwen3.5-4b) REPO=Qwen/Qwen3.5-4B; MREV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a ;;
          qwen3.5-9b) REPO=Qwen/Qwen3.5-9B; MREV=c202236235762e1c871ad0ccb60c8ee5ba337b9a ;;
          *) log "unknown model $model"; continue ;;
        esac
        getq35 $model $REPO $MREV || { log "skip $model: incomplete download"; continue; }
        vec=llm-phase-map/vectors-$model-l16
        TOOL=explore_llm_phase_map.py run vectors-$model 40 vectors --model $model --layer 16 --device cuda \
          --dtype float64 --contrasts hostility:kindness --random 3 --out $vec
        run sweep-$model 60 scan --device cuda --dtype float64 --chunk 128 --tol 1e-8 --eps 1e-6 --model $model \
          --layer 16 --vectors $vec --concept hostility-vs-kindness --alpha 0 0 --log-temperature -1 0.3 \
          --resolution 128 1 --transient 1000 --samples 2000 --max-period 64 --out $D/moe/$model-alpha0
        run phil-$model 90 scan --device cuda --dtype float64 --chunk 64 --tol 1e-8 --eps 1e-6 --model $model \
          --layer 16 --vectors $vec --concept philosophy --alpha -0.12 0.12 --log-temperature -1 0.3 \
          --resolution 32 32 --transient 300 --samples 300 --max-period 32 --out $D/moe/$model-phil
        run host-$model 90 scan --device cuda --dtype float64 --chunk 64 --tol 1e-8 --eps 1e-6 --model $model \
          --layer 16 --vectors $vec --concept hostility-vs-kindness --alpha 0 0.12 --log-temperature -0.3 0.3 \
          --resolution 32 32 --transient 300 --samples 300 --max-period 32 --out $D/moe/$model-host
        damp $model
      done
    fi
    ;;
  upperset)    # upper-temperature extension of the scaling sweeps (exploratory; run SCU): JOBARGS MODEL [MODEL...]. For
               # each model: pinned checkpoint, an unsteered float64 sweep (bf16 storage, layers over all GPUs) of 64
               # temperatures over log10 T 0.61..1.2 (the wide sweeps end at 0.6), settings as in scaleset, into
               # big/MODEL-alpha0-upper; no gain split; the checkpoint and the download cache are then deleted
    for model in "${JOBARGS[@]}"; do
      case $model in
        qwen1.5b)    REPO=Qwen/Qwen2.5-1.5B-Instruct;        MREV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306 ;;
        qwen3b)      REPO=Qwen/Qwen2.5-3B-Instruct;          MREV=aa8e72537993ba99e69dfaafa59ed015b17504d1 ;;
        qwen7b)      REPO=Qwen/Qwen2.5-7B-Instruct;          MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
        qwen14b)     REPO=Qwen/Qwen2.5-14B-Instruct;         MREV=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ;;
        qwen32b)     REPO=Qwen/Qwen2.5-32B-Instruct;         MREV=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd ;;
        olmo2-1b)    REPO=allenai/OLMo-2-0425-1B-Instruct;   MREV=48d788eca847d4d7548f375ad03d3c9312f6139e ;;
        olmo2-7b)    REPO=allenai/OLMo-2-1124-7B-Instruct;   MREV=470b1fba1ae01581f270116362ee4aa1b97f4c84 ;;
        olmo2-13b)   REPO=allenai/OLMo-2-1124-13B-Instruct;  MREV=3a5c85baefbb1896a54d56fe2e76c0395627ddf4 ;;
        olmo2-32b)   REPO=allenai/OLMo-2-0325-32B-Instruct;  MREV=b96024342a77a69aa0dda815c3454a671f477463 ;;
        llama3.1-8b) REPO=meta-llama/Llama-3.1-8B-Instruct;  MREV=0e9e39f249a16976918f6564b8830bc894c89659 ;;
        *) log "unknown model $model"; continue ;;
      esac
      log "download $REPO"
      HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
      log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1), free $(df -h --output=avail / | tail -1)"
      mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
      mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
      run upper-$model 90 scan --device auto --weights bf16 --dtype float64 --model $model --vectors "" \
        --alpha 0 0 --log-temperature 0.61 1.2 --resolution 64 1 --transient 1000 --samples 2000 --max-period 64 \
        --tol 1e-8 --eps 1e-6 --chunk 64 --out $D/big/$model-alpha0-upper
      case "$mdir" in "$NEURAL_SHRIMP_DATA"/models/hf/hub/models--?*) rm -rf -- "$mdir" ;; esac
      rm -rf -- "$HOME/.cache/huggingface/xet"
    done
    unset HF_TOKEN
    ;;
  prefixcoarse) # prompt-blocked stage 1 on one or two GPUs (run PSO; docs/prefix-scale-olmo.md): JOBARGS MODEL IDX [IDX...].
               # Pinned checkpoint; per prefix index of docs/prefix-scale-prefixes.json one unsteered line of 48 temperatures
               # over log10 T -0.4..0.9 (tools/hidden_chaos.py, twins 1e-5 and 1e-7, 1000 + 2000 steps), layers over all GPUs
    model=${JOBARGS[0]}
    case $model in
      olmo2-7b)  REPO=allenai/OLMo-2-1124-7B-Instruct;  MREV=470b1fba1ae01581f270116362ee4aa1b97f4c84 ;;
      olmo2-32b) REPO=allenai/OLMo-2-0325-32B-Instruct; MREV=b96024342a77a69aa0dda815c3454a671f477463 ;;
      *) log "unknown model $model"; exit 2 ;;
    esac
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
    mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
    mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
    for idx in "${JOBARGS[@]:1}"; do
      TOOL=hidden_chaos.py run psc-$model-$idx 60 line --model $model --device auto --weights bf16 --dtype float64 \
        --vectors-none --alpha 0 --transient 1000 --record 2000 --eps 1e-5 1e-7 \
        --prefix-file docs/prefix-scale-prefixes.json --prefix-index $idx --log-temperature -0.4 0.9 --n 48 --chunk 48 \
        --out $D/prefix/$model-p$idx-coarse
    done
    ;;
  prefix72base) # base against Instruct at 72B (run PB72; docs/prefix-scale-72b-base.md): JOBARGS IDX [IDX...]. Qwen2.5-72B
               # (pretrained base) at the pinned revision, stage 1 as PS72 (48 temperatures over log10 T -0.4..0.9, twins at
               # 1e-5 and 1e-6) with 500 + 1000 steps, one line per prefix index, layers over all GPUs (float64 hook fix)
    REPO=Qwen/Qwen2.5-72B; MREV=efba10c8e54e91e0d9570ab5f7b51a958474d4cb
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=16)" > "$LOGS/setup-qwen72b-base.log" 2>&1
    log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1)"
    mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
    mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
    for idx in "${JOBARGS[@]}"; do
      TOOL=hidden_chaos.py run pb72-$idx 45 line --model qwen72b-base --device auto --weights bf16 --dtype float64 \
        --vectors-none --alpha 0 --transient 500 --record 1000 --eps 1e-5 1e-6 \
        --prefix-file docs/prefix-scale-prefixes.json --prefix-index $idx --log-temperature -0.4 0.9 --n 48 --chunk 48 \
        --out $D/prefix/qwen72b-base-p$idx-coarse
    done
    ;;
  prefix72)    # prompt-blocked test at 72B (run PS72; docs/prefix-scale-72b.md): JOBARGS IDX [IDX...]. Qwen2.5-72B at the
               # pinned revision, stage 1 only (48 temperatures over log10 T -0.4..0.9, twins at 1e-5 and 1e-6, 1000 + 2000
               # steps), one line per prefix index of docs/prefix-scale-prefixes.json, layers over all GPUs
    REPO=Qwen/Qwen2.5-72B-Instruct; MREV=495f39366efef23836d0cfae4fbe635880d2be31
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=16)" > "$LOGS/setup-qwen72b.log" 2>&1
    mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
    mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
    for idx in "${JOBARGS[@]}"; do
      TOOL=hidden_chaos.py run ps72-$idx 75 line --model qwen72b --device auto --weights bf16 --dtype float64 \
        --vectors-none --alpha 0 --transient 1000 --record 2000 --eps 1e-5 1e-6 \
        --prefix-file docs/prefix-scale-prefixes.json --prefix-index $idx --log-temperature -0.4 0.9 --n 48 --chunk 48 \
        --out $D/prefix/qwen72b-p$idx-coarse
    done
    ;;
  startstate)  # loop-free cheap predictor (run CP; docs/start-state-predictor.md): JOBARGS MODEL [MODEL...]. For each
               # model: pinned checkpoint (gated ones with the token on stdin), tools/start_state.py on the ten prefixes of
               # docs/prefix-scale-prefixes.json (float64, bf16 storage, layers over all GPUs) into
               # $D/start-state/MODEL (llm-soft-loop/pod64); the checkpoint and the download cache are then deleted
    for model in "${JOBARGS[@]}"; do
      case $model in
        qwen0.5b)     REPO=Qwen/Qwen2.5-0.5B-Instruct;        MREV=7ae557604adf67be50417f59c2c2f167def9a775 ;;
        qwen1.5b)     REPO=Qwen/Qwen2.5-1.5B-Instruct;        MREV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306 ;;
        qwen3b)       REPO=Qwen/Qwen2.5-3B-Instruct;          MREV=aa8e72537993ba99e69dfaafa59ed015b17504d1 ;;
        qwen7b)       REPO=Qwen/Qwen2.5-7B-Instruct;          MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
        qwen14b)      REPO=Qwen/Qwen2.5-14B-Instruct;         MREV=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ;;
        qwen32b)      REPO=Qwen/Qwen2.5-32B-Instruct;         MREV=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd ;;
        qwen72b)      REPO=Qwen/Qwen2.5-72B-Instruct;         MREV=495f39366efef23836d0cfae4fbe635880d2be31 ;;
        llama3.1-8b)  REPO=meta-llama/Llama-3.1-8B-Instruct;  MREV=0e9e39f249a16976918f6564b8830bc894c89659 ;;
        llama3.1-70b) REPO=meta-llama/Llama-3.1-70B-Instruct; MREV=1605565b47bb9346c5515c34102e054115b4f98b ;;
        llama3.3-70b) REPO=meta-llama/Llama-3.3-70B-Instruct; MREV=6f6073b423013f6a7d4d9f39144961bfbfbc386b ;;
        olmo2-1b)     REPO=allenai/OLMo-2-0425-1B-Instruct;   MREV=48d788eca847d4d7548f375ad03d3c9312f6139e ;;
        olmo2-7b)     REPO=allenai/OLMo-2-1124-7B-Instruct;   MREV=470b1fba1ae01581f270116362ee4aa1b97f4c84 ;;
        olmo2-13b)    REPO=allenai/OLMo-2-1124-13B-Instruct;  MREV=3a5c85baefbb1896a54d56fe2e76c0395627ddf4 ;;
        olmo2-32b)    REPO=allenai/OLMo-2-0325-32B-Instruct;  MREV=b96024342a77a69aa0dda815c3454a671f477463 ;;
        *) log "unknown model $model"; continue ;;
      esac
      log "download $REPO"
      HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
      log "download done $(du -sh $NEURAL_SHRIMP_DATA/models/hf/hub | cut -f1), free $(df -h --output=avail / | tail -1)"
      mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
      mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
      TOOL=start_state.py run start-$model 30 --model $model --device auto --weights bf16 --dtype float64 \
        --prefix-file docs/prefix-scale-prefixes.json --directions 8 --eps 1e-5 --out $D/start-state/$model
      if [ "$model" != qwen0.5b ]; then
        case "$mdir" in "$NEURAL_SHRIMP_DATA"/models/hf/hub/models--?*) rm -rf -- "$mdir" ;; esac
      fi
      rm -rf -- "$HOME/.cache/huggingface/xet"
    done
    unset HF_TOKEN
    ;;
  prefixscale) # prompt-blocked size test (run PS; docs/prefix-scale-test.md): JOBARGS MODEL IDX [IDX...]. Pinned checkpoint;
               # for each prefix index of docs/prefix-scale-prefixes.json: stage 1, an unsteered line of 48 temperatures
               # over log10 T -0.4..0.9 (tools/hidden_chaos.py, twins at 1e-5 and 1e-7, 1000 + 2000 steps); stage 2, 24
               # temperatures across the largest entropy increase of stage 1 (tools/prefix_transition.py)
    model=${JOBARGS[0]}
    case $model in
      qwen7b)  REPO=Qwen/Qwen2.5-7B-Instruct;  MREV=a09a35458c702b33eeacc393d103063234e8bc28 ;;
      qwen14b) REPO=Qwen/Qwen2.5-14B-Instruct; MREV=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 ;;
      *) log "unknown model $model"; exit 2 ;;
    esac
    log "download $REPO"
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', revision='$MREV', cache_dir='$NEURAL_SHRIMP_DATA/models/hf/hub',
                  allow_patterns=['*.json', '*.safetensors', '*.txt'], max_workers=16)" > "$LOGS/setup-$model.log" 2>&1
    mdir="$NEURAL_SHRIMP_DATA/models/hf/hub/models--${REPO/\//--}"
    mkdir -p "$mdir/refs" && printf %s "$MREV" > "$mdir/refs/main"
    PSARGS="--model $model --device auto --weights bf16 --dtype float64 --vectors-none --alpha 0 --transient 1000
            --record 2000 --eps 1e-5 1e-7 --prefix-file docs/prefix-scale-prefixes.json"
    for idx in "${JOBARGS[@]:1}"; do
      TOOL=hidden_chaos.py run ps-$model-$idx-coarse 60 line $PSARGS --prefix-index $idx --log-temperature -0.4 0.9 \
        --n 48 --chunk 48 --out $D/prefix/$model-p$idx-coarse
      read -r tlo thi < <($PY tools/prefix_transition.py $D/prefix/$model-p$idx-coarse 2>/dev/null)
      log "prefix $idx transition $tlo $thi"
      TOOL=hidden_chaos.py run ps-$model-$idx-fine 40 line $PSARGS --prefix-index $idx --log-temperature $tlo $thi \
        --n 24 --chunk 24 --out $D/prefix/$model-p$idx-fine
    done
    ;;
  morph)       # morph video frames (run V1): JOBARGS FIRST LAST STEP MINUTES_PER_FRAME COLD_TRANSIENT WARM_TRANSIENT
               # SCAN-ARGS...; float32 scans (tol and twin offset 1e-4) of Qwen2.5-0.5B along the uploaded great-circle
               # directions morph-FIRST..morph-LAST (llm-soft-loop/controls/vectors-morph-r1-r3-240), one frame after
               # another into morph/frame-NNN: the first frame from the fixed start (COLD_TRANSIENT steps), each later
               # frame from the previous frame's final states (WARM_TRANSIENT steps); state files are deleted on the pod
    first=${JOBARGS[0]}; last=${JOBARGS[1]}; fstep=${JOBARGS[2]}; fmin=${JOBARGS[3]}; cold=${JOBARGS[4]}
    warm=${JOBARGS[5]}
    prev=""
    for k in $(seq $first $fstep $last); do
      kk=$(printf %03d $k)
      if [ -z "$prev" ]; then start=(--transient $cold); else start=(--transient $warm --init-states $D/morph/frame-$prev); fi
      run morph-$kk "$fmin" scan --device cuda --dtype float32 --chunk 2048 --tol 1e-4 --eps 1e-4 \
        --model qwen0.5b --layer 12 --vectors llm-soft-loop/controls/vectors-morph-r1-r3-240 --concept morph-$kk \
        "${start[@]}" --save-states "${JOBARGS[@]:6}" --out $D/morph/frame-$kk
      [ -n "$prev" ] && rm -f "$OUT/$D/morph/frame-$prev/states.npz"
      if [ -f "$OUT/$D/morph/frame-$kk/states.npz" ]; then prev=$kk; else prev=""; fi
    done
    rm -f "$OUT"/$D/morph/frame-*/states.npz
    ;;
  scan32)      # one float32 scan with free arguments: scan32 OUT-NAME MINUTES SCAN-ARGS... (coarse maps)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    run scan32-$name "$minutes" scan --device cuda --dtype float32 --chunk 2048 --tol 1e-4 --eps 1e-4 \
      "${JOBARGS[@]:2}" --out $D/scan32/$name
    ;;
  p6z3curves)  # boundary curves of the period-6 shrimp candidate of zoom Z3 ('. -\n\nThe is a') from the p6z3a
               # branches (uploaded with --extra-data): A = upper temperature fold, B = lower temperature fold,
               # C = alpha fold (right side), D = the close fold pair of the alpha-down branch (cusp candidate), E = the
               # flip into the period-12 foot
    P=llm-soft-loop/pod64/p6z3
    C6="$CURVE --steps 60"
    BOUNDS="--log-temperature-bounds -0.12 0.0 --alpha-bounds 0.068 0.09"
    for set_ in "${JOBARGS[@]}"; do         # the orchestrator passes the sets as separate words
      for dir in 1 -1; do n=$([ $dir = 1 ] && echo plus || echo minus)
        case $set_ in
          A) run p6z3-foldA-$n 50 $C6 --start $P/branch-T-up --kind fold --direction $dir $BOUNDS \
               --out $D/p6z3/curve-foldA-$n ;;
          B) run p6z3-foldB-$n 50 $C6 --start $P/branch-T-down --kind fold --direction $dir $BOUNDS \
               --out $D/p6z3/curve-foldB-$n ;;
          C) run p6z3-foldC-$n 50 $C6 --start $P/branch-a-up --kind fold --direction $dir $BOUNDS \
               --out $D/p6z3/curve-foldC-$n ;;
          D) run p6z3-foldD-$n 50 $C6 --start $P/branch-a-down --kind fold --direction $dir $BOUNDS \
               --out $D/p6z3/curve-foldD-$n ;;
          E) run p6z3-flipE-$n 50 $C6 --start $P/branch-a-down --kind flip --direction $dir $BOUNDS \
               --out $D/p6z3/curve-flipE-$n ;;
        esac
      done
    done
    ;;
  fm5)         # fold map of the period-5 window of the two-block plane (run FM5): gratitude@4 + hostility@16 at log10 T
               # -0.1 (vectors uploaded with --extra-data). (1) Four float64 branches of the period-5 orbit from the
               # DSS5 seed cell, alpha1 and alpha2 each way, in parallel. (2) Curves from the boundaries they found,
               # both directions, in parallel: folds of the alpha1-up and alpha2-up branches, the flip of the
               # alpha2-down branch, the first boundary of the alpha1-down branch. JOBARGS: BRANCH-MINUTES
               # CURVE-MINUTES CURVE-STEPS DS DS-MAX
    bmin=${JOBARGS[0]}; cmin=${JOBARGS[1]}; csteps=${JOBARGS[2]}; cds=${JOBARGS[3]}; cdsmax=${JOBARGS[4]}
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2      # eight processes share the pod's CPUs
    TBA="--layer 4 --vectors llm-phase-map/vectors-l4 --concept gratitude --block2 16 --vectors2 llm-phase-map/vectors-l16
         --concept2 hostility --log-temperature -0.1"
    BR5="branch --device cuda --threads 2 --dtype float64 --reference-dtype none --no-exact-jvp --chunk 896 --period 0
         --transient 900 --steps 150 --ds 0.002 --ds-max 0.01 --beyond 0.0005"
    SEED="--alpha -0.118730 --alpha2 0.100693"
    run fm5-branch-a1-up "$bmin" $BR5 $TBA $SEED --free alpha --direction 1 --bounds -0.125 -0.100 \
      --out $D/fm5/branch-a1-up &
    run fm5-branch-a1-down "$bmin" $BR5 $TBA $SEED --free alpha --direction -1 --bounds -0.125 -0.100 \
      --out $D/fm5/branch-a1-down &
    run fm5-branch-a2-up "$bmin" $BR5 $TBA $SEED --free alpha2 --direction 1 --bounds 0.088 0.118 \
      --out $D/fm5/branch-a2-up &
    run fm5-branch-a2-down "$bmin" $BR5 $TBA $SEED --free alpha2 --direction -1 --bounds 0.088 0.118 \
      --out $D/fm5/branch-a2-down &
    wait
    C5="curve --device cuda --threads 2 --dtype float64 --reference-dtype none --no-exact-jvp --gmres-tol 1e-10
        --tol 1e-10 --gtol 1e-9 --max-newton 10 --chunk 448 --steps $csteps --ds $cds --ds-max $cdsmax
        --alpha-bounds -0.125 -0.100 --alpha2-bounds 0.088 0.118"
    for spec in a1-up:fold a2-up:fold a2-down:flip a1-down:auto; do
      IFS=: read -r b kind <<< "$spec"
      for dir in 1 -1; do n=$([ $dir = 1 ] && echo plus || echo minus)
        run fm5-curve-$b-$kind-$n "$cmin" $C5 --start $D/fm5/branch-$b --kind $kind --direction $dir \
          --out $D/fm5/curve-$b-$kind-$n &
      done
    done
    wait
    ;;
  fm5c)        # run FM5c: the two tangent reversals of the FM5 fold curve located and checked as cusps (tools/cusp_locate.py,
               # cusp test, on both traces), and the left fold of the alpha1-down branch continued both ways; all six in
               # parallel (fm5 branches/curves and the vectors uploaded with --extra-data). JOBARGS: LOCATE-MINUTES
               # CURVE-MINUTES CURVE-STEPS
    lmin=${JOBARGS[0]}; cmin=${JOBARGS[1]}; csteps=${JOBARGS[2]}
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    F=$D/fm5
    for spec in curve-a1-up-fold-minus:19 curve-a1-up-fold-minus:30 curve-a2-up-fold-plus:12 curve-a2-up-fold-plus:23; do
      IFS=: read -r cv left <<< "$spec"
      TOOL=cusp_locate.py run fm5c-locate-$cv-$left "$lmin" $F/$cv $left --test cusp --device cuda --threads 2 \
        --out $D/fm5c/locate-$cv-$left &
    done
    C5="curve --device cuda --threads 2 --dtype float64 --reference-dtype none --no-exact-jvp --gmres-tol 1e-10
        --tol 1e-10 --gtol 1e-9 --max-newton 10 --chunk 448 --steps $csteps --ds 0.002 --ds-max 0.02
        --alpha-bounds -0.130 -0.100 --alpha2-bounds 0.085 0.120"
    for dir in 1 -1; do n=$([ $dir = 1 ] && echo plus || echo minus)
      run fm5c-curve-a1-down-fold-$n "$cmin" $C5 --start $F/branch-a1-down --kind fold --direction $dir \
        --out $D/fm5c/curve-a1-down-fold-$n &
    done
    wait
    ;;
  shrimp)      # runs OC and CL2 (consult 10, directions 3 and 2; fm5 branches and curve, vectors uploaded with --extra-data):
               # the orbit census around the swallowtail-shaped junction (tools/window_census.py, docs/oc/oc-points.json)
               # in three parallel parts, then the two-coordinate closure test (tools/window_closure.py,
               # docs/cl/cl-locations.json; the pocket orbits come from the census part holding T-centroid).
               # JOBARGS: CENSUS-MINUTES CLOSURE-MINUTES
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    CEN="census --points docs/oc/oc-points.json --device cuda --threads 4 --max-newton 20"
    TOOL=window_census.py run oc-census-1 "${JOBARGS[0]}" $CEN --only T-centroid T-low T-high T-near-L T-near-U \
      --out $D/oc/census &
    TOOL=window_census.py run oc-census-2 "${JOBARGS[0]}" $CEN --only W-low S-low E-low W-high --out $D/oc/census-2 &
    TOOL=window_census.py run oc-census-3 "${JOBARGS[0]}" $CEN --only N-high E-high below-L above-U --out $D/oc/census-3 &
    wait
    TOOL=window_closure.py run cl2 "${JOBARGS[1]}" run --locations docs/cl/cl-locations.json --device cuda --threads 8 \
      --out $D/cl/run
    ;;
  rr)          # reset test (run RR2; tools/repro_reset.py; consult 10): JOBARGS = specs NAME:MODEL:LAYER:LOG10T:DTYPE:SET:MINUTES,
               # run in order on one GPU; Qwen2.5-7B is downloaded once at its pinned revision when a spec needs it
    have7b=0
    for spec in "${JOBARGS[@]}"; do
      IFS=: read -r name model layer lt dtype set_ minutes <<< "$spec"
      if [ "$model" = qwen7b ] && [ $have7b = 0 ]; then get7b; have7b=1; fi
      TOOL=repro_reset.py run rr-$name "$minutes" run --problems docs/rr-problems.json --set "$set_" --dtype "$dtype" \
        --model "$model" --layer "$layer" --log-temperature "$lt" --device cuda --out $D/rr/$name
    done
    ;;
  fm5d)        # run FM5d: the two tangent reversals of the left fold curve of the period-5 window (FM5c,
               # curve-a1-down-fold-minus, brackets 14-15 and 39-40) located and checked as cusps, in parallel (the fm5c
               # curve, the fm5 branch and the vectors uploaded with --extra-data). JOBARGS: LOCATE-MINUTES
    lmin=${JOBARGS[0]}
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    for left in 14 39; do
      TOOL=cusp_locate.py run fm5d-locate-$left "$lmin" $D/fm5c/curve-a1-down-fold-minus $left --test cusp --device cuda \
        --threads 4 --out $D/fm5d/locate-curve-a1-down-fold-minus-$left &
    done
    wait
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
        Bminus) run p4-foldB-minus 60 $C4 --start $P/branch-T-up --kind fold --start-index 53 --direction -1 \
             --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-foldB-minus ;;
        Dminus) run p4-foldD-minus 60 $C4 --start $P/branch-T-down --kind fold --direction -1 \
             --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-foldD-minus ;;
        Cplus2|Cminus2|Dplus2)       # review follow-up (run P4R): the three curves that set C left unfinished, 70 points,
                                     # new output names (the unfinished folders keep their progress files)
             case $set_ in Cplus2) st=branch-a-up; dir=1; nm=foldC-plus;; Cminus2) st=branch-a-up; dir=-1; nm=foldC-minus;;
                           Dplus2) st=branch-T-down; dir=1; nm=foldD-plus;; esac
             run p4-$nm-2 100 $C4 --start $P/$st --kind fold --direction $dir \
               --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-$nm-2 ;;
        E) for dir in 1 -1; do n=$([ $dir = 1 ] && echo plus || echo minus)
             run p4-flipE-$n 45 $C4 --start $P/branch-a-down --kind flip --direction $dir \
               --log-temperature-bounds -0.1 0.2 --alpha-bounds 0.03 0.1 --out $D/p4/curve-flipE-$n
           done ;;
      esac
    done
    ;;
  scale15)     # scale-up rung 1: Qwen2.5-1.5B soft-token map, coarse 64x64 float64 over the 0.5B scan's range
               # (philosophy steering at layer 14 of 28, vectors uploaded with --extra-data)
    run scale15-coarse 120 scan --device cuda --dtype float64 --chunk 1024 --tol 1e-8 --eps 1e-6 \
      --model qwen1.5b --layer 14 --vectors llm-phase-map/vectors-1.5b-l14 --resolution 64 64 \
      --alpha -0.12 0.12 --log-temperature -1.0 0.3 --out $D/scale15/coarse-64
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
  p6z3a)       # shrimp candidate of zoom Z3: the period-6 window '. -\n\nThe is a' (body, hooked foot with its
               # period-12 doubling, thin leg crossing a period-5 window): float64 orbits, then branches out of the body
    ORB="orbit --device cuda --dtype float64 --tol 1e-12 --max-newton 12"
    for spec in "-0.05994 0.08051 body" "-0.06806 0.08031 body-low" "-0.06312 0.08000 inner-red" \
                "-0.07529 0.07800 leg1" "-0.07388 0.07839 leg2" "-0.07200 0.07960 foot"; do
      set -- $spec
      run orbit-p6z3-$3 15 $ORB --log-temperature $1 --alpha $2 --out $D/p6z3/orbit-$3
    done
    BR="branch --device cuda --dtype float64 --reference-dtype none --no-exact-jvp --steps 150 --chunk 896"
    for dir in 1 -1; do
      name=$([ $dir = 1 ] && echo up || echo down)
      run p6z3-branch-T-$name 30 $BR --free temperature --log-temperature -0.05994 --alpha 0.08051 --direction $dir \
        --ds 0.002 --ds-max 0.01 --beyond 0.003 --bounds -0.2 0.05 --out $D/p6z3/branch-T-$name
      run p6z3-branch-a-$name 30 $BR --free alpha --log-temperature -0.05994 --alpha 0.08051 --direction $dir \
        --ds 0.002 --ds-max 0.01 --beyond 0.0005 --bounds 0.06 0.1 --out $D/p6z3/branch-a-$name
    done
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
