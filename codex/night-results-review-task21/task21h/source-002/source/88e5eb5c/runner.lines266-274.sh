  tbmulti)     # several two_block_map.py calls in order (run TBC): JOBARGS = NAME MINUTES ARGS [:: NAME MINUTES ARGS ...]
    i=0; n=${#JOBARGS[@]}
    while [ $i -lt $n ]; do
      name=${JOBARGS[$i]}; minutes=${JOBARGS[$((i+1))]}; i=$((i+2)); margs=()
      while [ $i -lt $n ] && [ "${JOBARGS[$i]}" != "::" ]; do margs+=("${JOBARGS[$i]}"); i=$((i+1)); done
      i=$((i+1))
      TOOL=two_block_map.py run tb-$name "$minutes" --device cuda "${margs[@]}" --out $D/tb/$name
    done
    ;;
