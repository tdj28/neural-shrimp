#!/bin/bash
# Invoked only after source/card/private-push/current-capacity admission.
# This script neither creates nor approves a card. No retry loop.
set -euo pipefail
renice -n 10 -p "$$" >/dev/null
: "${TASK8_CARD_SHA256:?external exact card hash required}"
: "${TASK8_SOURCE_COMMIT:?external source commit required}"
: "${TASK8_BOOTSTRAP_SHA256:?external reviewed bootstrap hash required}"
TASK8_REPO=/home/d7082791602/.codex/worktrees/llm-p8-certificate/neural-shrimp
TASK8_SUP=/data2/neural-shrimp-data/codex/huginn-independent-replay/part-B-001-supervision
TASK8_BOOT=/tmp/task8-partB-bootstrap-001.py
TASK8_CARD="$TASK8_REPO/codex/huginn-independent-replay/part-B-001-card.json"
TASK8_DRIVER="$TASK8_REPO/codex/huginn-independent-replay/replay.py"
test -d "$TASK8_SUP"
test ! -e "$TASK8_SUP/stdout.log"
test ! -e "$TASK8_SUP/stderr.log"
test ! -e /data2/neural-shrimp-data/codex/huginn-independent-replay/part-B-001
test "$(sha256sum "$TASK8_BOOT" | cut -d' ' -f1)" = "$TASK8_BOOTSTRAP_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
# Bootstrap establishes hard4GiB before replay imports, validates exactnice10.
# timeout owns its fresh child process group and escalates TERM to KILL after5s.
timeout --signal=TERM --kill-after=5s 300s /data2/neural-shrimp/.venv/bin/python -B "$TASK8_BOOT" "$TASK8_DRIVER" --card "$TASK8_CARD" > "$TASK8_SUP/stdout.log" 2> "$TASK8_SUP/stderr.log"
