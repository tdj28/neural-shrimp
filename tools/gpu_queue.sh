#!/bin/bash
# Run queued GPU jobs one at a time, so that concurrent jobs never exhaust GPU memory.
# Usage: tools/gpu_queue.sh QUEUE_DIR [PID ...]
#   Waits for the given PIDs (jobs already on the GPU) to exit, then repeatedly runs the oldest *.sh in
#   QUEUE_DIR (moved to QUEUE_DIR/running/ while it runs, then to QUEUE_DIR/done/ with its exit status),
#   until no job is left. Jobs added while the runner is active are picked up.
queue="$1"; shift
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
mkdir -p "$queue/running" "$queue/done"
while true; do
  job=$(ls "$queue"/*.sh 2>/dev/null | sort | head -n 1)
  [ -z "$job" ] && break
  name=$(basename "$job")
  mv "$job" "$queue/running/$name"
  echo "$(date -u +%FT%TZ) start $name"
  bash "$queue/running/$name" > "$queue/done/$name.log" 2>&1
  status=$?
  mv "$queue/running/$name" "$queue/done/$name"
  echo "$(date -u +%FT%TZ) end $name status $status"
done
echo "$(date -u +%FT%TZ) queue empty"
