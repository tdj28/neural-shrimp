"""Is the hook's lower edge the fixed start crossing the stable manifold of the kite's saddle orbit? (human, 19 Oct
2026: "So we should be able to calculate that manifold fold and it would match the hook?")

At one alpha, between two adjacent cells on either side of the edge (the fixed start reaches the strongly contracting
orbit S below and the weakly contracting orbit W above; both exist on both sides, tools/kite_basin_check.py):
1. bisection in log10 T for the edge: the trajectory from the fixed start is classified by the orbit it ends on;
2. at the bisected edge the trajectory must linger near a third period-6 orbit, the saddle; Newton (exact Jacobians)
   from a lingering state gives the saddle and its multipliers (one outside the unit circle, mu_u);
3. the settling time tau(delta) at distance delta from the edge must grow like (6 / ln mu_u) ln(1/delta): the slope of
   tau against ln(1/delta) is compared with 6 / ln |mu_u|.
Float64, CPU.

Usage: python tools/hook_saddle_check.py --alpha 0.07297495107632094 --log10t -0.1535315068493151 -0.15352876712328767 \
    --out llm-soft-loop/kite/hook-saddle
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
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log10t", type=float, nargs=2, required=True, help="log10 T below (orbit S) and above (orbit W) the edge")
    ap.add_argument("--steps", type=int, default=2400, help="steps per trajectory")
    ap.add_argument("--bisections", type=int, default=34)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"], dtype=np.float64)
    a = args.alpha
    tic = time.perf_counter()

    def orbit_at(t):
        return esl.SoftOrbit(loop, a * v, 10.0 ** t, chunk=args.chunk, jacobian_method="ad")

    def trajectory(t, n):
        orb = orbit_at(t)
        X, out = orb.start, []
        for _ in range(n):
            X = orb.f(X)
            out.append(X.reshape(-1).double().cpu().numpy())
        return np.array(out)

    # reference orbits S (below) and W (above) by Newton from the fixed start
    refs = {}
    for name, t in (("S", args.log10t[0]), ("W", args.log10t[1])):
        orb = orbit_at(t)
        X = orb.iterate(orb.start, 600)
        X, hist, vals, pts = orb.newton(X, 6, 1e-11, 10)
        refs[name] = np.stack([p.reshape(-1).double().cpu().numpy() for p in pts])
        print(json.dumps({"reference": name, "log10T": t, "residual": hist[-1], "mu1": [vals[0].real, vals[0].imag]}), flush=True)
    scale = float(np.linalg.norm(refs["S"][0]))

    def dist(x, R):
        return float(np.min(np.linalg.norm(R - x, axis=1))) / scale

    def classify(traj):
        dS, dW = dist(traj[-1], refs["S"]), dist(traj[-1], refs["W"])
        return ("S" if dS < dW else "W"), min(dS, dW)

    # 1. bisection for the edge
    lo, hi = args.log10t
    log = []
    for i in range(args.bisections):
        mid = 0.5 * (lo + hi)
        lab, d = classify(trajectory(mid, args.steps))
        log.append({"log10T": mid, "ends_on": lab, "final_distance": d})
        if lab == "S":
            lo = mid
        else:
            hi = mid
        print(json.dumps({"bisection": i, "log10T": mid, "ends_on": lab, "width": hi - lo}), flush=True)
    t_edge = 0.5 * (lo + hi)
    # 2. the lingering state at the edge and the saddle
    tr = trajectory(lo, args.steps)
    dS = np.array([dist(x, refs["S"]) for x in tr])
    dW = np.array([dist(x, refs["W"]) for x in tr])
    ret6 = np.array([np.linalg.norm(tr[k + 6] - tr[k]) / scale for k in range(len(tr) - 6)])
    far = (dS[: len(ret6)] > 1e-3) & (dW[: len(ret6)] > 1e-3)        # away from both attractors
    k0 = int(np.argmin(np.where(far, ret6, np.inf)))                  # most nearly 6-periodic state away from S and W
    orb = orbit_at(lo)
    import torch
    X0 = torch.as_tensor(tr[k0].reshape(1, 1, -1), dtype=loop.state_dtype, device=loop.device)
    Xs, hist, vals, pts = orb.newton(X0, 6, 1e-11, 14)
    saddle = np.stack([p.reshape(-1).double().cpu().numpy() for p in pts])
    mods = np.abs(vals)
    rec_saddle = {"seed_step": k0, "seed_return_distance": float(ret6[k0]), "newton_residuals": hist,
                  "converged": bool(hist[-1] < 1e-11), "multipliers": [[float(z.real), float(z.imag)] for z in vals[:4]],
                  "unstable_count": int((mods > 1).sum()), "gap_to_S": dist(saddle[0], refs["S"]) if True else None,
                  "gap_to_W": dist(saddle[0], refs["W"])}
    print(json.dumps({"saddle": rec_saddle}), flush=True)
    # 3. settling time against the distance to the edge, on the S side
    taus = []
    for e in range(4, 12):
        delta = 10.0 ** (-e)
        t = t_edge - delta
        if t < args.log10t[0]:
            continue
        tr = trajectory(t, args.steps)
        lab, _ = classify(tr)
        R = refs[lab]
        d = np.array([dist(x, R) for x in tr])
        settled = np.nonzero(d > 1e-6)[0]
        tau = int(settled[-1] + 1) if len(settled) else 0
        taus.append({"delta": delta, "ends_on": lab, "settling_steps": tau})
        print(json.dumps(taus[-1]), flush=True)
    x = np.log(1.0 / np.array([r["delta"] for r in taus]))
    y = np.array([r["settling_steps"] for r in taus], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(taus) >= 3 else None
    mu_u = float(mods[0])
    rec = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "edge_log10T": t_edge,
           "bisection": log, "saddle": rec_saddle, "settling": taus, "slope_steps_per_e_fold": slope,
           "predicted_slope_6_over_ln_mu_u": 6.0 / np.log(mu_u) if mu_u > 1 else None}
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hook-saddle.json").write_text(json.dumps(rec, indent=1))
    np.save(out / "saddle-orbit.npy", saddle)
    print(json.dumps({k: rec[k] for k in ("edge_log10T", "slope_steps_per_e_fold", "predicted_slope_6_over_ln_mu_u", "seconds")}))


if __name__ == "__main__":
    main()
