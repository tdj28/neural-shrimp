"""Newton-located fold points inside the tip bracket of the hook, and the fixed start's landing coordinate there.

tools/hook_tip_check.py found the landing coordinate u0 negative at fold point 19 of the continued curve and positive
at point 20, 1.65e-5 apart in alpha. Here new fold points are located between them at chosen alphas (Newton on the
minimally augmented system at fixed alpha, bifurcation_curves.trace_curve with fix=1, started from the linear
interpolation of the two stored orbits), and u0 is measured at each (hook_tip_check.landing). The zero of u0 is the
predicted tip.

Usage: python tools/hook_tip_fold_points.py --alphas 0.072964 0.072965 ... --device cuda \
    --out llm-soft-loop/pod64/kite/hook-tip-folds
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
    ap.add_argument("--between", type=int, nargs=2, default=[20, 19])
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--power", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import bifurcation_curves as bc
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    from hook_tip_check import landing

    src = esl.DATA / "explore" / args.curve
    cur = json.loads((src / "curve.json").read_text())["points"]
    states = np.load(src / "states.npy")
    i0, i1 = args.between
    a0, a1 = cur[i0]["alpha"], cur[i1]["alpha"]
    t0, t1 = cur[i0]["log10_temperature"], cur[i1]["log10_temperature"]
    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"], dtype=np.float64)
    fam = esl.soft_family(loop, v, 6, 10.0, 896)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    rows = []
    for a in args.alphas:
        w = (a - a0) / (a1 - a0)
        x0 = (1 - w) * states[i0].ravel() + w * states[i1].ravel()
        tg = (1 - w) * t0 + w * t1
        recs, reason, sts = bc.trace_curve(fam, x0, [tg, 10.0 * a], "fold", fix=1, direction=1, steps=1, ds=1e-5,
                                           ds_min=1e-8, ds_max=1e-4, tol=1e-10, gtol=1e-9, max_newton=10,
                                           gmres_tol=1e-10, gmres_maxiter=1, refine_k=4)
        if not recs:
            rows.append({"alpha": a, "located": False, "reason": reason})
            print(json.dumps(rows[-1]), flush=True)
            continue
        r0 = recs[0]
        lt = float(r0["theta"][0])
        row = landing(loop, v, a, lt, sts[0], args.steps, args.power)
        row.update({"located": True, "fold_residual": r0["residual"], "fold_g": r0["g"],
                    "critical_multiplier_refined": r0.get("critical_multiplier_refined"),
                    "guess_log10T": tg, "seconds": round(time.perf_counter() - tic, 1)})
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("alpha", "log10T", "fold_residual", "fold_g", "orbit_return", "mu_r",
                                               "u0", "final_distance_to_orbit", "b_from_drift", "seconds")}), flush=True)
        (out / "hook-tip-folds.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    ok = sorted([r for r in rows if r.get("located")], key=lambda r: r["alpha"])
    tips = []
    for x, y in zip(ok[:-1], ok[1:]):
        if x["u0"] * y["u0"] < 0:
            f = x["u0"] / (x["u0"] - y["u0"])
            tips.append({"between_alphas": [x["alpha"], y["alpha"]], "alpha_tip": x["alpha"] + f * (y["alpha"] - x["alpha"])})
    rec = {"args": vars(args), "seconds": round(time.perf_counter() - tic, 1), "rows": rows, "predicted_tips": tips}
    (out / "hook-tip-folds.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"predicted_tips": tips, "seconds": rec["seconds"]}))


if __name__ == "__main__":
    main()
