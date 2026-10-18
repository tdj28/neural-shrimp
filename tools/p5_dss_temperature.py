"""Period-5 shrimp, step 2 (exploratory; human approval 18 Oct 2026): superstability grid at a lower temperature.

At one temperature (--log-t), in the two-block plane of the period-5 window (Qwen2.5-0.5B-Instruct, gratitude at block
4 = alpha1, hostility at block 16 = alpha2):
1. **Find the window.** Scan a coarse grid (--coarse) over a wide box from the maps' fixed start, and keep the cells
   that reach a period-5 orbit with the window's loop of leading tokens (rotation-invariant; --loop). The default loop is
   the one at log10 T -0.1.
2. **Pick the seed:** the window cell nearest the window's centroid whose 8 neighbours are all in the window.
3. **Run tools/doubly_superstable.py grid** over the window's bounding box (expanded by --pad cells) at --resolution,
   in the same plane and settings as run DSS5. That grid gives the per-step factors of the leading multiplier. Their
   zero curves are the superstable curves, and a crossing of two different factors' curves is a doubly superstable
   point, a shrimp centre.

Output: DATA/explore/OUT/window.json (coarse scan, seed, box) and the grid under OUT/grid.

Usage: python tools/p5_dss_temperature.py --log-t -0.11 --device cuda --out llm-soft-loop/pod64/sheets/p5dss-t-0.110
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402


def canonical(t):
    t = tuple(int(x) for x in t)
    return min(t[i:] + t[:i] for i in range(len(t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--box1", type=float, nargs=2, default=[-0.1240, -0.1080], help="alpha1 range of the coarse scan")
    ap.add_argument("--box2", type=float, nargs=2, default=[0.0940, 0.1080], help="alpha2 range of the coarse scan")
    ap.add_argument("--coarse", type=int, nargs=2, default=[48, 48], help="coarse grid (alpha2 rows, alpha1 columns)")
    ap.add_argument("--loop", type=int, nargs="+", default=[13, 358, 13, 13, 56568])
    ap.add_argument("--transient", type=int, default=900)
    ap.add_argument("--resolution", type=int, nargs=2, default=[31, 31])
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--grid-minutes", type=float, default=95)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = esl.SoftLoop("qwen0.5b", 4, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    v = elpm.load_vectors("llm-phase-map/vectors-l4")["gratitude"]
    loop.sm.add_steer_layer(16)
    w = elpm.load_vectors("llm-phase-map/vectors-l16")["hostility"]
    a1s = np.linspace(*args.box1, args.coarse[1])
    a2s = np.linspace(*args.box2, args.coarse[0])
    cells = [(a1, a2) for a2 in a2s for a1 in a1s]
    T = 10.0 ** args.log_t
    loop.extra = (16, np.array([a2 * w for _, a2 in cells]))
    X = loop.prepare(np.array([a1 * v for a1, _ in cells]))
    temps = np.full(len(cells), T)
    with torch.no_grad():
        for _ in range(args.transient):
            X, _ = loop.step(X, temps)
        hist, toks = [], []
        for _ in range(40):
            X, p = loop.step(X, temps)
            hist.append(X[:, -1, :].double().cpu().numpy())
            toks.append(p.argmax(-1).cpu().numpy())
    hist, toks = np.array(hist), np.array(toks)
    scale = np.maximum(1.0, np.linalg.norm(hist[-1], axis=1))
    per = np.zeros(len(cells), dtype=int)
    for q in range(32, 0, -1):
        per = np.where(np.linalg.norm(hist[-1 - q] - hist[-1], axis=1) < 1e-8 * scale, q, per)
    key = canonical(args.loop)
    inwin = np.array([per[c] == 5 and canonical(toks[-5:, c]) == key for c in range(len(cells))])
    W = inwin.reshape(args.coarse[0], args.coarse[1])
    rep = {"args": vars(args), "coarse_cells": len(cells), "window_cells": int(W.sum()),
           "period_counts": {int(q): int((per == q).sum()) for q in np.unique(per)}}
    if W.sum() == 0:
        rep["status"] = "window not found in the coarse box"
        (out / "window.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep))
        return
    rr, cc = np.nonzero(W)
    interior = [(r, c) for r, c in zip(rr, cc) if 0 < r < W.shape[0] - 1 and 0 < c < W.shape[1] - 1
                and W[r - 1:r + 2, c - 1:c + 2].all()]
    if not interior:
        rep["status"] = "no window cell with all 8 neighbours in the window"
        (out / "window.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep))
        return
    cr, ccen = rr.mean(), cc.mean()
    r0, c0 = min(interior, key=lambda rc: (rc[0] - cr) ** 2 + (rc[1] - ccen) ** 2)
    seed = (float(a1s[c0]), float(a2s[r0]))
    lo1 = a1s[max(cc.min() - args.pad, 0)]
    hi1 = a1s[min(cc.max() + args.pad, len(a1s) - 1)]
    lo2 = a2s[max(rr.min() - args.pad, 0)]
    hi2 = a2s[min(rr.max() + args.pad, len(a2s) - 1)]
    rep.update({"seed": seed, "box_alpha1": [float(lo1), float(hi1)], "box_alpha2": [float(lo2), float(hi2)],
                "window_mask": W.astype(int).tolist(), "alpha1": a1s.tolist(), "alpha2": a2s.tolist(),
                "seconds_scan": round(time.perf_counter() - tic, 1)})
    (out / "window.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("window_cells", "seed", "box_alpha1", "box_alpha2", "seconds_scan")}), flush=True)
    del loop, X
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "doubly_superstable.py"), "grid", "--period", "5",
           "--seed", str(seed[0]), str(seed[1]), "--alpha", str(lo1), str(hi1),
           "--block2", "16", "--vectors2", "llm-phase-map/vectors-l16", "--concept2", "hostility",
           "--alpha2", str(lo2), str(hi2), "--fixed-log-temperature", str(args.log_t),
           "--resolution", str(args.resolution[0]), str(args.resolution[1]), "--layer", "4",
           "--vectors", "llm-phase-map/vectors-l4", "--concept", "gratitude", "--transient", str(args.transient),
           "--k", "6", "--tol", "1e-12", "--device", args.device, "--threads", str(args.threads),
           "--out", f"{args.out}/grid"]
    print(json.dumps({"grid_command": " ".join(cmd[1:])}), flush=True)
    try:
        rc = subprocess.run(cmd, timeout=args.grid_minutes * 60).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    rep["grid_rc"] = rc
    rep["seconds"] = round(time.perf_counter() - tic, 1)
    (out / "window.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"grid_rc": rc, "seconds": rep["seconds"]}))


if __name__ == "__main__":
    main()
