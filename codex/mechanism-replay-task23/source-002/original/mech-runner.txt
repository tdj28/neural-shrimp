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
