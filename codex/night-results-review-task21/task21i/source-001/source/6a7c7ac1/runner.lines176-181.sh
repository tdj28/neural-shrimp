  bz)          # ordinary steered inference in chaotic and periodic zones (run BZ; tools/behavior_zones.py run, float32 on
               # the GPU; model-written code is executed here, inside the pod): JOBARGS = NAME MINUTES then run arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=behavior_zones.py run bz-$name "$minutes" run --device cuda "${JOBARGS[@]:2}" --out $D/bz/$name
    ;;
