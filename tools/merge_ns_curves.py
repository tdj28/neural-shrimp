"""Merge exploratory Neimark-Sacker curve traces into one ordered curve.

Keeps only Neimark-Sacker rows, sorts them by output rate and, where traces
overlap within --tolerance in log10 output, keeps the row from the trace
listed first. Writes {"sources": [...], "rows": [...]} in the ns-curve format.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, help="ns-curve.json files, highest priority first")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows, sources = [], []
    for path in args.traces:
        data = Path(path).read_bytes()
        sources.append({"path": path, "sha256": hashlib.sha256(data).hexdigest()})
        for r in json.loads(data)["rows"]:
            if r.get("type") != "neimark_sacker":
                continue
            if any(abs(r["log10_output"] - s["log10_output"]) < args.tolerance for s in rows):
                continue
            rows.append({**r, "source": path})
    rows.sort(key=lambda r: r["log10_output"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sources": sources, "rows": rows}, indent=1))
    print(f"{len(rows)} Neimark-Sacker points from {len(sources)} traces; output range "
          f"{rows[0]['log10_output']:.4f} to {rows[-1]['log10_output']:.4f}")


if __name__ == "__main__":
    main()
