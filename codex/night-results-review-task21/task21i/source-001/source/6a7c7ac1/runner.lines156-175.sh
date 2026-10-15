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
