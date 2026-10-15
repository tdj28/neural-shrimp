  scan64)      # one float64 scan with free arguments: scan64 OUT-NAME MINUTES SCAN-ARGS... (zooms to be read without
               # the float32 precision caveat; on an H100 float64 costs about the same as float32 for this model)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    run scan64-$name "$minutes" scan --device cuda --dtype float64 --chunk 2048 --tol 1e-8 --eps 1e-6 \
      "${JOBARGS[@]:2}" --out $D/scan64/$name
    ;;
