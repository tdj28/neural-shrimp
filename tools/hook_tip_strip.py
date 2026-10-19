"""Short against long horizon near the tip of the hook: is the map's light region near the fold a finite-horizon effect?

Cells on a strip of steering strengths across the predicted tip (tools/hook_tip_check.py), at fixed offsets from the
kite's fold curve (a cubic spline through its Newton-converged points; offsets in rows of the HR12 map, negative =
beyond the fold, where the weaker orbit does not exist). One run per cell from the fixed start: a twin (offset 1e-6,
renormalized every step) starts at step --t0; the exponent is averaged over [t0, t0+m0) (the map's protocol, 300 and
300) and again over [t1, t1+m1) (long horizon). Output: both exponents per cell.

Usage: python tools/hook_tip_strip.py --alpha 0.072958 0.072975 --n 32 --offsets -1 1.5 --device cuda \
    --out llm-soft-loop/kite/hook-tip-strip
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curve", default="llm-soft-loop/pod64/kite/kite-la-curve-plus")
    ap.add_argument("--alpha", type=float, nargs=2, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--offsets", type=float, nargs="+", default=[-1.0, 1.5], help="rows of 1.0955e-6 in log10 T below the fold")
    ap.add_argument("--row", type=float, default=1.095461658828789e-06)
    ap.add_argument("--t0", type=int, default=300)
    ap.add_argument("--m0", type=int, default=300)
    ap.add_argument("--t1", type=int, default=3000)
    ap.add_argument("--m1", type=int, default=400)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scipy.interpolate import CubicSpline

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    cur = json.loads((esl.DATA / "explore" / args.curve / "curve.json").read_text())["points"]
    a_all = np.array([p["alpha"] for p in cur])
    t_all = np.array([p["log10_temperature"] for p in cur])
    br = [i for i in range(13, len(cur)) if i == 13 or a_all[i] < a_all[i - 1]][:12]   # cusp C down the left branch
    fold = CubicSpline(a_all[br][::-1], t_all[br][::-1])
    al = np.linspace(args.alpha[0], args.alpha[1], args.n)
    A = np.concatenate([al for _ in args.offsets])
    LT = np.concatenate([fold(al) - off * args.row for off in args.offsets])
    off = np.concatenate([np.full(args.n, o) for o in args.offsets])
    B = len(A)
    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"], dtype=np.float64)
    torch = loop.torch
    tic = time.perf_counter()
    steer = np.array([x * v for x in A])
    temps2 = np.tile(10.0 ** LT, 2)
    XY = loop.prepare(np.concatenate([steer, steer]))
    lg0, lg1 = np.zeros(B), np.zeros(B)
    rng = np.random.default_rng(0)
    for k in range(args.t1 + args.m1):
        if k == args.t0:
            d = torch.as_tensor(rng.normal(size=tuple(XY[:B].shape)), dtype=loop.state_dtype, device=loop.device)
            d = d / d.flatten(1).norm(dim=1)[:, None, None] * args.eps
            XY = torch.cat([XY[:B], XY[:B] + d])
        XY, _ = loop.step(XY, temps2)
        if k >= args.t0:
            X, Y = XY[:B], XY[B:]
            diff = Y - X
            dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), args.eps * 1e-30)
            g = np.log(dist / args.eps)
            if k < args.t0 + args.m0:
                lg0 += g
            if args.t1 <= k < args.t1 + args.m1:
                lg1 += g
            XY = torch.cat([X, X + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype,
                                                          device=loop.device)[:, None, None]])
        if k % 500 == 0:
            print(json.dumps({"step": k, "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    rec = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1),
           "cells": [{"alpha": float(A[i]), "log10T": float(LT[i]), "rows_below_fold": float(off[i]),
                      "lambda_short": float(lg0[i] / args.m0), "lambda_long": float(lg1[i] / args.m1)} for i in range(B)]}
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hook-tip-strip.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"cells": B, "seconds": rec["seconds"]}))


if __name__ == "__main__":
    main()
