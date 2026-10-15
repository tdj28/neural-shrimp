  xo)          # orbit-preserving readout/network crossover (run XO; tools/crossover.py, float64 on the GPU): JOBARGS =
               # NAME MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=crossover.py run xo-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/xo/$name
    ;;
