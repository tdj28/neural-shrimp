"""Render previously scored results into a new directory; preserve original figures."""
import argparse
import json
from pathlib import Path

from analyze import render, sha, write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = json.loads(args.results.read_text())
    render(result, args.out)
    write(args.out / "receipt.json", {
        "purpose": "Figure readability refinement; no numerical scoring repeated or changed",
        "results_path": str(args.results.resolve()), "results_sha256": sha(args.results),
        "renderer_sha256": sha(Path(__file__).with_name("analyze.py")),
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in args.out.iterdir() if p.is_file()}})
    print(args.out)


if __name__ == "__main__":
    main()
