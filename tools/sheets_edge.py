"""Run SH, local part: what a visible edge of the fixed-start map is, at one temperature (exploratory).

Along a line of constant log10 T across an edge where the map switches from family A to family B:
1. **profile:** for each alpha on a fine grid, the family the maps' fixed start reaches and its settle step (as in
   tools/sheets_map.py), with long transients (--transient).
2. **switch:** bisection in alpha for the switch of the fixed start's destination between two given families.
3. **saddle:** just on either side of the switch, the fixed start's trajectory lingers near the orbit whose stable
   manifold separates the two basins. The lingering state (smallest p-step return distance before the trajectory
   leaves) seeds Newton's method for a period-p orbit; its multipliers say whether it is a saddle.

Usage: python tools/sheets_edge.py --log-t -0.148 --alpha 0.0644 0.0648 --n 41 --period 3 --out sheets/edge-t-0.148
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sheets_map as sm  # noqa: E402
from explore_llm_soft_loop import DATA, SoftOrbit  # noqa: E402


def runner(args):
    ns = types.SimpleNamespace(model="qwen0.5b", layer=12, vectors="llm-phase-map/vectors-l12", concept="philosophy",
                               prompt="open", opener="Hello!", dtype="float64", device="cpu", threads=args.threads,
                               samples=args.samples, record=80, max_period=32, tol=1e-8, settle_tol=1e-6, eps=1e-6,
                               seed=0)
    return sm.Runner(ns)


def label(r, out, c):
    inv = {v: k for k, v in r.families.items()}
    f = int(out["family"][c])
    return "".join(r.loop.sm.tok.decode([t]) for t in inv[f]) if f >= 0 else "-"


def trajectory(r, alpha, lt, steps, period):
    """The fixed start's trajectory at one cell: p-step return distances and states (float64, CPU)."""
    loop, torch = r.loop, r.torch
    steer = alpha * r.v
    X = loop.prepare(steer[None])
    T = 10.0 ** lt
    states = [X[0, -1].double().cpu().numpy().copy()]
    for _ in range(steps):
        X, _ = loop.step(X, [T])
        states.append(X[0, -1].double().cpu().numpy().copy())
    S = np.array(states)
    ret = np.linalg.norm(S[period:] - S[:-period], axis=1) / np.maximum(1.0, np.linalg.norm(S[period:], axis=1))
    return S, ret


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--alpha", type=float, nargs=2, required=True)
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--period", type=int, required=True, help="period of the orbit sought near the switch")
    ap.add_argument("--transient", type=int, default=1500)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--bisect", type=int, default=22)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    r = runner(args)
    alphas = np.linspace(*args.alpha, args.n)
    res = r.run_cells([(args.log_t, a) for a in alphas], None, args.transient)
    prof = [{"alpha": float(a), "family": label(r, res, c), "period": int(res["period"][c]),
             "lyapunov": float(res["lyapunov"][c]), "settle": int(res["settle"][c])} for c, a in enumerate(alphas)]
    for p in prof:
        print(json.dumps(p), flush=True)
    fams = [p["family"] for p in prof]
    k = next((i for i in range(len(fams) - 1) if fams[i] != fams[i + 1]), None)
    report = {"args": vars(args), "profile": prof}
    if k is None:
        report["switch"] = None
        report["note"] = "no change of the fixed start's family along the line"
    else:
        lo, hi, fl, fh = alphas[k], alphas[k + 1], fams[k], fams[k + 1]
        for _ in range(args.bisect):
            mid = 0.5 * (lo + hi)
            rm = r.run_cells([(args.log_t, mid)], None, args.transient)
            fm = label(r, rm, 0)
            if fm == fl:
                lo = mid
            elif fm == fh:
                hi = mid
            else:
                report["third_family_at"] = {"alpha": float(mid), "family": fm}
                break
        report["switch"] = {"between": [float(lo), float(hi)], "families": [fl, fh]}
        print(json.dumps(report["switch"]), flush=True)
        sad = []
        for a in (lo, hi):
            S, ret = trajectory(r, a, args.log_t, args.transient, args.period)
            j0 = int(np.argmin(ret[50:len(ret) // 2])) + 50      # lingering before the late convergence
            orb = SoftOrbit(r.loop, a * r.v, 10.0 ** args.log_t, jacobian_method="ad")
            torch = r.torch
            X = torch.as_tensor(S[j0][None, None, :], dtype=torch.float64)
            Xs, hist, vals, pts = orb.newton(X, args.period, tol=1e-11, max_newton=10)
            xs = Xs[0, -1].double().cpu().numpy()
            dist = np.linalg.norm(S - xs, axis=1) / np.linalg.norm(xs)
            sad.append({"alpha": float(a), "linger_step": j0, "linger_return": float(ret[j0]),
                        "newton_residuals": [float(h) for h in hist],
                        "multipliers_abs": [float(abs(v)) for v in vals[:4]],
                        "multipliers": [[float(v.real), float(v.imag)] for v in vals[:4]],
                        "steps_within_1e-4_of_orbit": int((dist < 1e-4).sum()),
                        "closest_approach": float(dist.min()), "closest_step": int(dist.argmin())})
            np.save(out / f"orbit-{a:.10f}.npy", xs)
            print(json.dumps(sad[-1]), flush=True)
        report["lingering_orbits"] = sad
    report["seconds"] = round(time.perf_counter() - tic, 1)
    (out / "edge.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
