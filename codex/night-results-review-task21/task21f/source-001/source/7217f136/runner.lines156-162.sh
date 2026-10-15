  dss)         # superstable curves and doubly superstable points of one orbit (tools/doubly_superstable.py grid, float64
               # on the GPU): JOBARGS = NAME MINUTES then grid arguments (vectors uploaded with --extra-data)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=doubly_superstable.py run dss-$name "$minutes" grid --device cuda --threads 8 "${JOBARGS[@]:2}" \
      --out $D/dss/$name
    ;;
