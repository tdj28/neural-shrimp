  tangentcheck) # tangent check TC (review response): JOBARGS CELLS_MINUTES SPECTRUM_MINUTES. (1) the 448 H2 cells of
               # docs/h2-horizon-cells.json exactly as in H2 (float64, 1000 transient + 2000 steps, twins 1e-6 and 1e-9),
               # plus the exact tangent exponent and the running exponents after 250, 500, 1000 and 2000 steps; (2) the
               # four leading exponents by QR iteration (tools/lyapunov_spectrum.py) at the 8 cells listed in
               # docs/tc-spectrum-cells.json, same horizon
    TOOL=hidden_chaos.py run tc-cells "${JOBARGS[0]}" cells --device cuda --dtype float64 \
      --vectors llm-phase-map/vectors-l12 --concept philosophy --cells-file docs/h2-horizon-cells.json \
      --transient 1000 --record 2000 --eps 1e-6 1e-9 --chunk 448 --tangent --checkpoints 250 500 1000 2000 \
      --out $D/hidden/tc-h2-cells-tangent
    TOOL=lyapunov_spectrum.py run tc-spectrum "${JOBARGS[1]}" --device cuda --dtype float64 \
      --vectors llm-phase-map/vectors-l12 --concept philosophy --cells-file docs/h2-horizon-cells.json \
      --cells-index 10 47 56 143 240 293 242 406 --k 4 --transient 1000 --record 2000 \
      --checkpoints 250 500 1000 2000 --out $D/hidden/tc-spectrum
    ;;
