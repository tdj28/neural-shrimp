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
