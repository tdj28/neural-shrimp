"""Run tools/cusp_locate.py (cusp test) on every cusp candidate of a finished fold-curve run, or on the candidates
listed with --only (run P6S-cusp2, 19 Oct 2026: the two tangent reversals of the top period-6 body's fold curve, where
its upper-left leg ends).

A curve point flagged cusp_candidate is the first point after the parameter tangent reversed, so its bracket is
(index - 1, index). Each candidate is located into OUT_PREFIX-<index>; a failure of one does not stop the next.

Usage: python tools/locate_curve_candidates.py CURVE_RUN --out-prefix llm-soft-loop/pod64/p6s/locate-wide2 \
         [--only 64 82] [--near ALPHA LOG10T RADIUS] --device cuda
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("curve", help="explore-relative fold-curve run (curve.json, states.npy)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="candidate indices to locate (default: all)")
    ap.add_argument("--near", type=float, nargs=3, default=None, metavar=("ALPHA", "LOG10T", "RADIUS"),
                    help="only candidates within RADIUS (in alpha units, log10 T scaled by 1/3) of this point")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    c = json.loads((DATA / "explore" / args.curve / "curve.json").read_text())
    cands = [q for q in c["points"] if q.get("cusp_candidate") and not q.get("rejected")]
    if args.only is not None:
        cands = [q for q in cands if q["index"] in set(args.only)]
    if args.near:
        a0, t0, r = args.near
        cands = [q for q in cands if ((q["alpha"] - a0) ** 2 + ((q["log10_temperature"] - t0) / 3) ** 2) ** 0.5 <= r]
    print(json.dumps({"curve": args.curve, "candidates": [(q["index"], q["alpha"], q["log10_temperature"]) for q in cands]}),
          flush=True)
    if not cands:
        raise SystemExit("no cusp candidates selected: nothing to locate")
    for q in cands:
        cmd = [sys.executable, str(ROOT / "tools" / "cusp_locate.py"), args.curve, str(q["index"] - 1), "--test", "cusp",
               "--device", args.device, "--threads", str(args.threads), "--out", f"{args.out_prefix}-{q['index']}"]
        print(" ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        print(json.dumps({"index": q["index"], "returncode": rc}), flush=True)


if __name__ == "__main__":
    main()
