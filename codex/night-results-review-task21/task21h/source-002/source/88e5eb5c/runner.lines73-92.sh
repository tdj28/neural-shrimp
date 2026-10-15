run() {   # run NAME MINUTES ARGS...: one explore_llm_soft_loop call (or tools/$TOOL) under a wall-time limit,
          # echoing its progress file once a minute so that the streamed log shows the time per point
  local name=$1 minutes=$2
  shift 2
  local outdir
  outdir=$(echo "$@" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
  log "start $name"
  timeout "${minutes}m" $PY tools/${TOOL:-explore_llm_soft_loop.py} "$@" > "$LOGS/$name.log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if [ -f "$OUT/$outdir/progress.jsonl" ]; then
      echo "$(date -u +%T) progress $name $(wc -l < "$OUT/$outdir/progress.jsonl") $(tail -1 "$OUT/$outdir/progress.jsonl" | cut -c1-220)"
    fi
  done
  wait $pid
  local status=$?
  log "end $name status $status"
  grep -v -i -e warn -e Loading -e crop -e jit "$LOGS/$name.log" | tail -1 | cut -c1-700
}
