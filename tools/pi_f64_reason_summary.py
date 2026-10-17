"""Summary of float64 twin runs of tools/pi_reason_f64.py, with the fields of docs/pi/pm2f-summary.json.

Per set of records.jsonl files (pooled by problem index; an index found twice is an error):
- base accuracy;
- twins whose relative distance at the end of thinking exceeds eps0 (grew) or not (shrank);
- twin answer flips, and the share of problems with any flip;
- twins that end thinking at a different step;
- growth-rate quartiles (where defined);
- the median log10 end distance.
Usage: python tools/pi_f64_reason_summary.py --set NAME:DIR[,DIR...] ... --json OUT
"""
import argparse
import json
from pathlib import Path

import numpy as np

import pi_matrix_analyze as pma


def summarize(dirs, eps0):
    rows = {}
    for d in dirs:
        for line in open(Path(d) / "records.jsonl"):
            r = json.loads(line)
            if r["index"] in rows:
                raise SystemExit(f"problem {r['index']} appears twice")
            rows[r["index"]] = r
    tw = [t for r in rows.values() for t in r["twins"]]
    end = [t["log10_end_think"] for t in tw if t["log10_end_think"] is not None]
    g = [t["growth_rate"] for t in tw if t["growth_rate"] is not None]
    flips = sum(t["flip"] for t in tw)
    k = sum(r["correct"] for r in rows.values())
    anyflip = sum(any(t["flip"] for t in r["twins"]) for r in rows.values())
    return {"problems": len(rows), "twins": len(tw), "accuracy_base": k / len(rows),
            "accuracy_base_wilson95": pma.wilson(k, len(rows)),
            "twins_grew": int(sum(e > np.log10(eps0) for e in end)), "twins_shrank": int(sum(e <= np.log10(eps0) for e in end)),
            "twin_flips": int(flips), "twin_flip_rate": int(flips) / len(tw), "twin_flip_wilson95": pma.wilson(int(flips), len(tw)),
            "problems_any_flip": anyflip / len(rows), "problems_any_flip_wilson95": pma.wilson(anyflip, len(rows)),
            "different_switch": sum(not t["same_switch"] for t in tw),
            "growth_rate_quartiles": [float(q) for q in np.quantile(g, [0.25, 0.5, 0.75])] if g else None,
            "growth_rate_n": len(g), "log10_end_think_median": float(np.median(end)) if end else None,
            "indices": sorted(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--eps0", type=float, default=1e-10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {"eps0": args.eps0, "sets": {}}
    for spec in args.set:
        name, dirs = spec.split(":", 1)
        out["sets"][name] = {"dirs": dirs.split(","), **summarize(dirs.split(","), args.eps0)}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "indices"} for k, v in out["sets"].items()}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
