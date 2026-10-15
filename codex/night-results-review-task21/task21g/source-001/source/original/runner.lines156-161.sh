  retmap)      # return maps of chaotic attractors (tools/return_map.py, float64 on the GPU): JOBARGS = NAME MINUTES then
               # tool arguments (vectors uploaded with --extra-data)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=return_map.py run retmap-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/retmap/$name
    ;;
