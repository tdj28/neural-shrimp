"""Append entries to the research log (docs/research-log.md).

Usage:
  python tools/research_log.py run --question "..." --setup "..." \
      --output ../neural-shrimp-data/... --surprise "..." -- <command ...>
  python tools/research_log.py note --title "..." --text "..."
  python tools/research_log.py amend --plan docs/EXP-028-protocol.md --change "..." \
      --why "..." --seen "..." --claims "..."

Each entry records the UTC time, HEAD commit and whether the worktree was dirty.
``run`` with a trailing command executes it after logging and records its exit
status; without a command it only logs the card. Entries are appended, never
rewritten.
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "docs" / "research-log.md"


def git_state():
    head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return head, dirty


def append(lines):
    if not LOG.exists():
        LOG.write_text("# Research log\n\nAppend-only record of runs, decisions and amendments. "
                       "See [the adaptive research protocol](adaptive-research-protocol.md).\n")
    with open(LOG, "a") as handle:
        handle.write("\n" + "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    run = sub.add_parser("run")
    run.add_argument("--question", required=True)
    run.add_argument("--setup", required=True)
    run.add_argument("--output", default="")
    run.add_argument("--surprise", default="")
    run.add_argument("command", nargs=argparse.REMAINDER)
    note = sub.add_parser("note")
    note.add_argument("--title", required=True)
    note.add_argument("--text", required=True)
    amend = sub.add_parser("amend")
    amend.add_argument("--plan", required=True)
    amend.add_argument("--change", required=True)
    amend.add_argument("--why", required=True)
    amend.add_argument("--seen", required=True)
    amend.add_argument("--claims", required=True)
    args = parser.parse_args()
    head, dirty = git_state()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tag = f"commit `{head}`{' (dirty worktree)' if dirty else ''}"
    if args.kind == "note":
        append([f"## {stamp}: {args.title}", "", f"{args.text}", "", f"_{tag}_"])
        return
    if args.kind == "amend":
        append([f"## {stamp}: amendment to `{args.plan}`", "",
                f"- **Change:** {args.change}", f"- **Why:** {args.why}",
                f"- **Outcomes seen:** {args.seen}", f"- **Claim impact:** {args.claims}",
                "", f"_{tag}_"])
        return
    command = [c for c in args.command if c != "--"]
    lines = [f"## {stamp}: run (exploratory)", "", f"- **Question:** {args.question}",
             f"- **Setup:** {args.setup}"]
    if args.output:
        lines.append(f"- **Output:** `{args.output}`")
    if args.surprise:
        lines.append(f"- **Would surprise us:** {args.surprise}")
    if command:
        lines.append(f"- **Command:** `{' '.join(command)}`")
    lines += ["", f"_{tag}_"]
    append(lines)
    if command:
        status = subprocess.run(command, cwd=ROOT).returncode
        append([f"- run started {stamp} finished {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                f" with exit status {status}"])
        sys.exit(status)


if __name__ == "__main__":
    main()
