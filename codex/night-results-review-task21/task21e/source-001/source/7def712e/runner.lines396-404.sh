  codim2)      # codimension-two points on uploaded fold and flip curves (run CL, tools/cusp_locate.py, float64 on the GPU):
               # each JOBARGS entry is TEST:CURVE_RUN:LEFT (TEST cusp, crossing or reversal; LEFT the curve index before
               # the sign change), output llm-soft-loop/pod64/codim2/<curve name>-<test>-<left>
    for spec in "${JOBARGS[@]}"; do
      ctest=${spec%%:*}; rest=${spec#*:}; crun=${rest%:*}; left=${rest##*:}
      name=$(basename "$crun")-$ctest-$left
      TOOL=cusp_locate.py run cl-$name 120 "$crun" "$left" --test "$ctest" --device cuda --out $D/codim2/$name
    done
    ;;
