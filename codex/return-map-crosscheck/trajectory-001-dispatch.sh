#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
if [[ $# != 2 || ! "$1" =~ ^[0-9a-f]{40}$ || ! "$2" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'exact source commit and plan SHA required' >&2
  exit 2
fi
task19_repo=/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp
task19_supervision=/data2/neural-shrimp-data/codex/llm-return-map-crosscheck/trajectory-001-supervision
mkdir "$task19_supervision"
cp -- "$0" "$task19_supervision/executed-dispatch.sh"
printf '%s\n' "$1" "$2" > "$task19_supervision/dispatch-pins.txt"
# The launcher idempotently establishes exact nice10 before Git or child work.
# Do not add10 here: an inherited nice10 must remain10, not become20.
exec setsid --wait /data2/neural-shrimp/.venv/bin/python -I -B -u \
  "$task19_repo/codex/return-map-crosscheck/launch.py" \
  --attempt trajectory-001 --source "$1" --plan trajectory-001-plan.json --plan-sha256 "$2" \
  > "$task19_supervision/launcher.stdout" 2> "$task19_supervision/launcher.stderr"
