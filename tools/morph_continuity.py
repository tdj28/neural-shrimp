"""Continuity of the rotation morph between consecutive directions: for every pair of neighboring frames (step 2), the
share of cells whose period label differs and the share whose token loop differs, with each frame's chaotic share and
start (cold: the fixed start; warm: the previous frame's final states). Seams (a cold frame after a warm one) and the
frames of a given run (e.g. the V3 fill) are marked, so that a filled frame can be compared with its neighbors.

Usage: python tools/morph_continuity.py --frames llm-soft-loop/pod64/morph --mark 16 18 ... --out docs/morph-continuity.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
CHAOS = 1e-3
LOOP_IDS = {}


def load(d):
    with np.load(d / "scan.npz", allow_pickle=False) as z:
        lyap, per, fam = z["lyapunov"].ravel(), z["period"].ravel(), z["family"].ravel()
    js = json.loads((d / "scan.json").read_text())
    # token loops by their decoded tokens (the V1b records have no token ids); one id per distinct loop in this run
    keys = {f["family"]: tuple(f["tokens"]) for f in js["families"]}
    lut = np.array([LOOP_IDS.setdefault(keys[i], len(LOOP_IDS) + 1) for i in range(len(keys))] or [0], dtype=np.int64)
    loops = np.where(fam >= 0, lut[np.maximum(fam, 0)], 0)
    return {"period": per, "loop": loops, "chaotic": (lyap > CHAOS) & (per == 0),
            "cold": js["args"].get("init_states") is None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="explore-relative folder of frame-NNN/scan.npz")
    ap.add_argument("--mark", type=int, nargs="*", default=[], help="frames to mark (e.g. the V3 fill)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    folder = DATA / "explore" / args.frames
    idx = sorted(int(p.parent.name.split("-")[1]) for p in folder.glob("frame-*/scan.npz"))
    rows, prev = [], None
    for k in idx:
        cur = load(folder / f"frame-{k:03d}")
        row = {"frame": k, "start": "cold" if cur["cold"] else "warm", "marked": k in args.mark,
               "chaotic_share": float(cur["chaotic"].mean())}
        if prev is not None and prev[0] == k - 2:
            p = prev[1]
            row.update(period_changed=float((p["period"] != cur["period"]).mean()),
                       loop_changed=float((p["loop"] != cur["loop"]).mean()),
                       chaotic_label_changed=float((p["chaotic"] != cur["chaotic"]).mean()),
                       seam=bool(cur["cold"]))
        rows.append(row)
        prev = (k, cur)
    pairs = [r for r in rows if "period_changed" in r]
    warm = [r["period_changed"] for r in pairs if not r["seam"]]
    out = {"frames_folder": args.frames, "frames": len(rows), "missing_between": sorted(
               set(range(idx[0], idx[-1] + 1, 2)) - set(idx)),
           "warm_pairs_period_changed": {"median": float(np.median(warm)), "max": float(np.max(warm))},
           "seam_pairs": [{k: r[k] for k in ("frame", "period_changed", "loop_changed")} for r in pairs if r["seam"]],
           "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("frames", "missing_between", "warm_pairs_period_changed")}))


if __name__ == "__main__":
    main()
