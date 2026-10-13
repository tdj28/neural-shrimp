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
  huginn|huginnA|huginnS)  # Huginn-0125 in its own venv with transformers 4.55, as in the local llm env. huginn: JOBARGS NAME
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
