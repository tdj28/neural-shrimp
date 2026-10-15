  spectrum)    # Lyapunov spectra by QR iteration (tools/lyapunov_spectrum.py, float64 on the GPU): JOBARGS = NAME MINUTES
               # then spectrum arguments (cells file in docs/, vectors uploaded with --extra-data if not the default)
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=lyapunov_spectrum.py run spectrum-$name "$minutes" --device cuda --dtype float64 "${JOBARGS[@]:2}" \
      --out $D/hidden/spectrum-$name
    ;;
