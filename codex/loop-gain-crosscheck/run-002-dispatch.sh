#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! "$1" =~ ^[0-9a-f]{40}$ || ! "$2" =~ ^[0-9a-f]{64}$ ]]; then
  echo 'exact source commit and plan SHA required' >&2
  exit 2
fi
task18_repo=/home/d7082791602/.codex/worktrees/llm-p8-certificate/neural-shrimp
task18_supervision=/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002-supervision
mkdir "$task18_supervision"
cp -- "$0" "$task18_supervision/executed-dispatch.sh"
printf '%s\n' "$1" "$2" > "$task18_supervision/dispatch-pins.txt"
exec setsid --wait nice -n 10 /data2/neural-shrimp/.venv/bin/python -I -B -u   "$task18_repo/codex/loop-gain-crosscheck/launch.py"   --attempt run-002 --source "$1" --plan run-002-plan.json --plan-sha256 "$2"   > "$task18_supervision/launcher.stdout" 2> "$task18_supervision/launcher.stderr"
