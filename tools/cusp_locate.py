"""Locate a cusp on a continued fold curve (review response, run CL): the point where the oriented quadratic
coefficient b sign(a) of the fold normal form u' = u + a eps + b u^2 changes sign between two curve points.

From the curve point before the sign change, short pseudo-arclength traces (bifurcation_curves.trace_curve, the
corrector of the original curve) with steps of a quarter of the current bracket shrink the bracket round by round;
b sign(a) comes from bifurcation_curves.fold_normal_form at every new point. A last step to the secant root of b in
arclength gives the located point, where bifurcation_curves.fold_unfolding evaluates b, the cubic coefficient c and
the two-parameter unfolding matrix J at the same point. The cusp counts as located when |b| there is below btol times
the largest |b| of the initial bracket, c is nonzero with a finite-difference plateau, and det J is nonzero;
otherwise the record says which condition failed.

Usage: python tools/cusp_locate.py CURVE_RUN LEFT_INDEX --out OUT [--rounds 3] [--per-round 5] [--device cuda]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bifurcation_curves as bc  # noqa: E402


def oriented_b(family, x, theta, param=1, param_scale=10.0, steps=(1e-3, 3e-4, 1e-4)):
    """(b sign(a), a, b, critical multiplier) of the fold normal form at a fold point; b sign(a) does not depend on
    the orientation of the critical eigenvector."""
    nf = bc.fold_normal_form(family, x, theta, param=param, param_scale=param_scale, steps=steps)
    return float(np.sign(nf["a"]) * nf["b"]), float(nf["a"]), float(nf["b"]), nf["multiplier"]


def scaled_gap(xl, thl, xr, thr):
    """Distance between two curve points in the coordinates of trace_curve started at xl: (x / |xl|, theta)."""
    s = float(np.linalg.norm(xl)) or 1.0
    return float(np.sqrt(np.sum(((np.asarray(xr) - np.asarray(xl)) / s) ** 2) + np.sum((np.asarray(thr) -
                                                                                     np.asarray(thl)) ** 2)))


def locate(family, xl, thl, xr, thr, fix, direction, rounds=3, per_round=5, param=1, param_scale=10.0,
           nf_steps=(1e-3, 3e-4, 1e-4), btol=1e-2, trace_kw=None, log=None):
    """Shrink the bracket [left, right] of a sign change of b sign(a) along the fold curve and step to its secant
    root. direction: sign of the curve tangent's theta[1 - fix] component at the left point, pointing toward the
    right point. Returns the record of every evaluated point, the brackets and the final point (state included)."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    trace_kw = dict(trace_kw or {})
    log = log or (lambda rec: None)
    bl = oriented_b(family, xl, thl, param, param_scale, nf_steps)
    br = oriented_b(family, xr, thr, param, param_scale, nf_steps)
    if np.sign(bl[0]) == np.sign(br[0]):
        raise ValueError(f"no sign change of b sign(a) in the bracket ({bl[0]:.4g}, {br[0]:.4g})")
    b0 = max(abs(bl[0]), abs(br[0]))
    points = [{"theta": list(map(float, thl)), "b_oriented": bl[0], "a": bl[1], "b": bl[2], "round": 0},
              {"theta": list(map(float, thr)), "b_oriented": br[0], "a": br[1], "b": br[2], "round": 0}]
    left = (np.asarray(xl, float), np.asarray(thl, float), bl[0])
    gap = scaled_gap(xl, thl, xr, thr)
    brackets = [{"round": 0, "gap": gap, "b": [bl[0], br[0]]}]
    last_arc = None
    for rnd in range(1, rounds + 1):
        ds = gap / 4
        recs, reason, states = bc.trace_curve(family, left[0], left[1], "fold", fix=fix, direction=direction,
                                              steps=per_round + 1, ds=ds, ds_min=ds * 1e-3, ds_max=ds, **trace_kw)
        if len(recs) < 2:
            raise RuntimeError(f"round {rnd}: the trace from the left point stopped: {reason}")
        prev = (np.asarray(states[0], float), np.asarray(recs[0]["theta"], float), left[2], recs[0])
        found = None
        for rec, st in zip(recs[1:], states[1:]):
            bo = oriented_b(family, st, rec["theta"], param, param_scale, nf_steps)
            points.append({"theta": rec["theta"], "b_oriented": bo[0], "a": bo[1], "b": bo[2], "round": rnd,
                           "arclength": rec["arclength"], "multiplier": rec.get("critical_multiplier"),
                           "second_modulus": rec.get("other_spectral_radius")})
            log(points[-1])
            cur = (np.asarray(st, float), np.asarray(rec["theta"], float), bo[0], rec)
            if np.sign(bo[0]) != np.sign(prev[2]):
                found = (prev, cur)
                break
            prev = cur
        if found is None:
            raise RuntimeError(f"round {rnd}: no sign change within {per_round} steps of {ds:.3g} ({reason})")
        (lx, lth, lb, lrec), (rx, rth, rb, rrec) = found
        gap = scaled_gap(lx, lth, rx, rth)
        brackets.append({"round": rnd, "gap": gap, "b": [lb, rb], "theta": [list(map(float, lth)),
                                                                            list(map(float, rth))]})
        left = (lx, lth, lb)
        tp = np.asarray(lrec["parameter_tangent"], float)
        if abs(tp[1 - fix]) < 1e-12:
            raise RuntimeError(f"round {rnd}: the tangent's theta[{1 - fix}] component vanishes at the left point")
        direction = int(np.sign(tp[1 - fix]))
        last_arc = rrec["arclength"] - lrec["arclength"]
    step = last_arc * left[2] / (left[2] - rb)                    # secant root of b sign(a) in arclength
    recs, reason, states = bc.trace_curve(family, left[0], left[1], "fold", fix=fix, direction=direction, steps=2,
                                          ds=step, ds_min=step * 1e-3, ds_max=step, **trace_kw)
    if len(recs) < 2:
        raise RuntimeError(f"final step to the secant root failed: {reason}")
    xf, thf = np.asarray(states[1], float), np.asarray(recs[1]["theta"], float)
    bof = oriented_b(family, xf, thf, param, param_scale, nf_steps)[0]      # same normal form as the bracket
    unf = bc.fold_unfolding(family, xf, thf)
    c_spread = None                                               # flattest adjacent pair, as in fold_cubic
    cs = np.array([t["c"] for t in unf.get("c_steps", []) if t.get("c") is not None], dtype=float)
    if len(cs) >= 2:
        d = np.abs(np.diff(cs))
        k = int(np.argmin(d))
        c_spread = float(d[k] / max(abs(0.5 * (cs[k] + cs[k + 1])), 1e-300))
    checks = {"b_small": abs(bof) <= btol * b0, "c_nonzero": unf["c"] != 0 and (c_spread is None or c_spread < 0.1),
              "unfolding_regular": abs(unf["det_J"]) > 0 and np.isfinite(unf["condition"])}
    final = {"theta": list(map(float, thf)), "b_oriented": bof, "b": float(unf["b"]), "c": float(unf["c"]),
             "c_plateau_relative_spread": c_spread, "J": unf["J"], "det_J": float(unf["det_J"]),
             "singular_values": unf["singular_values"], "condition": float(unf["condition"]),
             "multiplier": unf["multiplier"], "second_modulus": unf["second_modulus"],
             "critical_multiplier": recs[1].get("critical_multiplier"), "secant_step": float(step)}
    return {"points": points, "brackets": brackets, "final": final, "initial_b_max": b0, "btol": btol,
            "checks": checks, "located": all(checks.values()), "state": xf}


def main():
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("curve", help="fold-curve run under explore/ (curve.json, states.npy)")
    ap.add_argument("left", type=int, help="curve index before the sign change of b sign(a); the next index is the "
                                           "right end of the bracket")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=5)
    ap.add_argument("--btol", type=float, default=1e-2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    src = esl.DATA / "explore" / args.curve
    cv = json.loads((src / "curve.json").read_text())
    states = np.load(src / "states.npy")
    tl = json.loads((esl.DATA / "explore" / cv["args"]["start"] / "branch.json").read_text())
    ta = tl["args"]
    fix = 1 if tl.get("free", "temperature") == "temperature" else 0
    ca = cv["args"]
    if ca.get("reference_dtype", "none") != "none" or ca.get("dtype") != "float64":
        raise SystemExit("cusp_locate needs a pure float64 curve run")
    out = esl.DATA / "explore" / args.out
    path = out / "cusp-locate.json"
    if path.exists():
        raise SystemExit(f"{path} exists; outputs are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    loop = esl.SoftLoop(ta["model"], ta["layer"], args.threads, ta["prompt"], ta["opener"], ta["window"], "float64",
                        args.device)
    v = elpm.load_vectors(ta["vectors"])[ta["concept"]]
    sc = float(ca.get("alpha_scale", 10.0))
    fam = esl.soft_family(loop, v, int(cv["period"]), sc, 512)
    pl, pr = cv["points"][args.left], cv["points"][args.left + 1]
    th = lambda p: np.array([p["log10_temperature"], p["alpha"] * sc])        # noqa: E731
    tp = np.asarray(pl["parameter_tangent"], float)
    trace_kw = {k: ca[k] for k in ("tol", "gtol", "max_newton", "gmres_tol", "gmres_maxiter", "refine_k") if k in ca}
    progress = out / "progress.jsonl"
    progress.write_text("")

    def log(rec):
        line = json.dumps({k: rec[k] for k in ("round", "theta", "b_oriented", "a") if k in rec})
        print(line, flush=True)
        with progress.open("a") as fh:
            fh.write(line + "\n")

    tic = time.time()
    res = locate(fam, states[args.left].ravel(), th(pl), states[args.left + 1].ravel(), th(pr), fix,
                 int(np.sign(tp[1 - fix])), rounds=args.rounds, per_round=args.per_round, param=1, param_scale=sc,
                 btol=args.btol, trace_kw=trace_kw, log=log)
    np.save(out / "state.npy", res.pop("state"))
    fin = res["final"]
    fin["log10_temperature"], fin["alpha"] = fin["theta"][0], fin["theta"][1] / sc
    res.update({"curve": args.curve, "left": args.left, "period": int(cv["period"]), "args": vars(args),
                "seconds": round(time.time() - tic, 1)})
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({"located": res["located"], "checks": res["checks"], "log10_T": fin["log10_temperature"],
                      "alpha": fin["alpha"], "b_oriented": fin["b_oriented"], "c": fin["c"], "det_J": fin["det_J"],
                      "seconds": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
