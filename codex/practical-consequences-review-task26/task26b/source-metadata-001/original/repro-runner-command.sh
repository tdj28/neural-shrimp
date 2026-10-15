  repro)       # incidental numerical differences against the dynamical regime (run REPRO; tools/repro_noise.py, on the
               # GPU): JOBARGS = NAME MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=repro_noise.py run repro-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/repro/$name
    ;;
