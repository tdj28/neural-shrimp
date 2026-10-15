  reprotask)   # answers of continuous-feedback inference under batch-layout changes (run REPRO-T; tools/repro_task.py;
               # frozen design docs/repro-task-test.md): JOBARGS = NAME MINUTES then tool arguments
    name=${JOBARGS[0]}
    minutes=${JOBARGS[1]}
    TOOL=repro_task.py run reprotask-$name "$minutes" --device cuda "${JOBARGS[@]:2}" --out $D/repro/$name
    ;;
