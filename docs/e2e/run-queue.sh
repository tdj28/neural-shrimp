#!/bin/bash
# Resumable E2E queue: two lanes of 4-thread scripts; a done-marker per script, so rerunning skips finished ones.
# Usage: bash docs/e2e/run-queue.sh   (logs in ../neural-shrimp-data/explore/e2e/queue/)
cd "$(dirname "$0")/../.."
Q=../neural-shrimp-data/explore/e2e/queue
lane() { for s in "$@"; do
  [ -f $Q/$s.done ] && continue
  echo "start $s $(date -u +%H:%M)" >> $Q/queue.log
  CUDA_VISIBLE_DEVICES= nice -n 10 bash docs/e2e/run-$s.sh > $Q/$s.log 2>&1 && touch $Q/$s.done
  echo "end $s rc=$? $(date -u +%H:%M)" >> $Q/queue.log
done; }
lane q1g q1f q1c q1d &
lane q3e q1e q2b &
wait
[ -f $Q/figure.done ] || { CUDA_VISIBLE_DEVICES= .venv/bin/python tools/e2e_figure.py > $Q/figure.log 2>&1 && touch $Q/figure.done; }
echo "queue done $(date -u +%H:%M)" >> $Q/queue.log
