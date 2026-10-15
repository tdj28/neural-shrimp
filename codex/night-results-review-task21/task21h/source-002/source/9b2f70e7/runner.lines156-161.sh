  tb)          # two steering directions at two blocks (run TB; tools/two_block_map.py, float64 on the GPU): JOBARGS = NAME
               # MINUTES then tool arguments (vectors uploaded with --extra-data)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=two_block_map.py run tb-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/tb/$name
    ;;
