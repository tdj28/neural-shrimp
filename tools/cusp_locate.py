"""Locate codimension-two points on continued fold and flip curves (review response, run CL).

Three tests, each a scalar whose sign changes between two curve points:
- cusp (fold curves): the oriented quadratic coefficient b sign(a) of the fold normal form u' = u + a eps + b u^2
  (bifurcation_curves.fold_normal_form); at the located point bifurcation_curves.fold_unfolding gives b, the cubic
  coefficient c and the two-parameter unfolding matrix J at the same point;
- crossing (fold or flip curves): the largest modulus of the non-critical multipliers minus one, where a second
  multiplier leaves the unit circle;
- reversal (fold or flip curves): the parameter tangent projected on its direction at the left point, where the
  curve's projection on the parameter plane turns back.
At a crossing or a reversal the multipliers of the dense monodromy type the point: a second multiplier at +1 or -1,
real, or a complex pair on the circle; for a double multiplier the angle between the two eigenvectors separates a
Jordan block (strong resonance) from a semisimple pair.

From the curve point before the sign change, short pseudo-arclength traces (bifurcation_curves.trace_curve, the
corrector of the original curve) with steps of a quarter of the current bracket shrink the bracket round by round,
and a last step goes to the secant root in arclength. Each restart holds fixed the parameter whose tangent component is
smaller, so the orientation of the trace is set by the larger one.

Usage: python tools/cusp_locate.py CURVE_RUN LEFT_INDEX --test {cusp,crossing,reversal} --out OUT [--device cuda]
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
    """(b sign(a), a, b) of the fold normal form at a fold point; b sign(a) does not depend on the orientation of the
    critical eigenvector."""
    nf = bc.fold_normal_form(family, x, theta, param=param, param_scale=param_scale, steps=steps)
    return float(np.sign(nf["a"]) * nf["b"]), float(nf["a"]), float(nf["b"])


def scaled_gap(xl, thl, xr, thr):
    """Distance between two curve points in the coordinates of trace_curve started at xl: (x / |xl|, theta)."""
    s = float(np.linalg.norm(xl)) or 1.0
    return float(np.sqrt(np.sum(((np.asarray(xr) - np.asarray(xl)) / s) ** 2) +
                         np.sum((np.asarray(thr) - np.asarray(thl)) ** 2)))


def orientation(tp):
    """(fix, direction) for trace_curve from an oriented parameter tangent: hold the smaller component fixed and
    orient the trace by the sign of the larger one."""
    tp = np.asarray(tp, float)
    fix = int(np.argmin(np.abs(tp)))
    if abs(tp[1 - fix]) == 0:
        raise RuntimeError("the parameter tangent vanishes")
    return fix, int(np.sign(tp[1 - fix]))


def type_point(family, x, theta, kind, tol=5e-3):
    """Multipliers of the dense monodromy at a curve point and the type of the second critical multiplier."""
    M = family.monodromy(np.asarray(x, float), np.asarray(theta, float))
    vals, R = np.linalg.eig(M)
    target = 1.0 if kind == "fold" else -1.0
    i = int(np.argmin(np.abs(vals - target)))
    rest = [j for j in range(len(vals)) if j != i]
    if not rest:
        return {"type": "no other multiplier (one-dimensional map)",
                "critical_multiplier": [float(vals[i].real), float(vals[i].imag)], "second_multiplier": None,
                "second_distance_to_circle": None, "eigenvector_angle_degrees": None,
                "multipliers": [[float(v.real), float(v.imag)] for v in vals]}
    j = min(rest, key=lambda k: abs(abs(vals[k]) - 1.0))
    mu = vals[j]
    dist = float(abs(abs(mu) - 1.0))
    real = abs(mu.imag) <= 1e-7 * max(1.0, abs(mu))
    ri, rj = R[:, i] / np.linalg.norm(R[:, i]), R[:, j] / np.linalg.norm(R[:, j])
    angle = float(np.degrees(np.arccos(min(1.0, abs(np.vdot(ri, rj))))))
    if dist > tol:
        label = "no second multiplier on the unit circle"
    elif real and mu.real > 0:
        label = "double +1 multiplier (1:1 resonance)" if kind == "fold" else "fold-flip (+1 and -1)"
    elif real:
        label = "fold-flip (+1 and -1)" if kind == "fold" else "double -1 multiplier (1:2 resonance)"
    else:
        label = "fold-Neimark-Sacker" if kind == "fold" else "flip-Neimark-Sacker"
    top = vals[np.argsort(-np.abs(vals))][:8]
    return {"type": label, "critical_multiplier": [float(vals[i].real), float(vals[i].imag)],
            "second_multiplier": [float(mu.real), float(mu.imag)], "second_distance_to_circle": dist,
            "eigenvector_angle_degrees": angle, "multipliers": [[float(v.real), float(v.imag)] for v in top]}


def refine(family, kind, xl, thl, xr, thr, tpl, fn, fl, fr, rounds=3, per_round=5, trace_kw=None, log=None):
    """Shrink the bracket of a sign change of fn(x, theta, record) along the curve, starting from the left point with
    oriented parameter tangent tpl (pointing toward the right point), and step to the secant root. Returns the
    evaluated points, the brackets and the final (state, theta, record, step)."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if np.sign(fl) == np.sign(fr):
        raise ValueError(f"no sign change in the bracket ({fl:.4g}, {fr:.4g})")
    trace_kw = dict(trace_kw or {})
    log = log or (lambda rec: None)
    points = [{"theta": list(map(float, thl)), "f": float(fl), "round": 0},
              {"theta": list(map(float, thr)), "f": float(fr), "round": 0}]
    left = (np.asarray(xl, float), np.asarray(thl, float), float(fl), np.asarray(tpl, float))
    gap = scaled_gap(xl, thl, xr, thr)
    brackets = [{"round": 0, "gap": gap, "f": [float(fl), float(fr)]}]
    for rnd in range(1, rounds + 1):
        ds = gap / 4
        fix, direction = orientation(left[3])
        recs, reason, states = bc.trace_curve(family, left[0], left[1], kind, fix=fix, direction=direction,
                                              steps=per_round + 1, ds=ds, ds_min=ds * 1e-3, ds_max=ds, **trace_kw)
        if len(recs) < 2:
            raise RuntimeError(f"round {rnd}: the trace from the left point stopped: {reason}")
        prev = (np.asarray(states[0], float), np.asarray(recs[0]["theta"], float), left[2], recs[0])
        found = None
        for rec, st in zip(recs[1:], states[1:]):
            f = float(fn(np.asarray(st, float), np.asarray(rec["theta"], float), rec))
            points.append({"theta": rec["theta"], "f": f, "round": rnd, "arclength": rec["arclength"],
                           "critical_multiplier": rec.get("critical_multiplier"),
                           "other_spectral_radius": rec.get("other_spectral_radius")})
            log(points[-1])
            cur = (np.asarray(st, float), np.asarray(rec["theta"], float), f, rec)
            if np.sign(f) != np.sign(prev[2]):
                found = (prev, cur)
                break
            prev = cur
        if found is None:
            raise RuntimeError(f"round {rnd}: no sign change within {per_round} steps of {ds:.3g} ({reason})")
        (lx, lth, lf, lrec), (rx, rth, rf, rrec) = found
        gap = scaled_gap(lx, lth, rx, rth)
        brackets.append({"round": rnd, "gap": gap, "f": [lf, rf],
                         "theta": [list(map(float, lth)), list(map(float, rth))]})
        left = (lx, lth, lf, np.asarray(lrec["parameter_tangent"], float))
        last_arc, right_f = rrec["arclength"] - lrec["arclength"], rf
    step = last_arc * left[2] / (left[2] - right_f)
    fix, direction = orientation(left[3])
    recs, reason, states = bc.trace_curve(family, left[0], left[1], kind, fix=fix, direction=direction, steps=2,
                                          ds=step, ds_min=step * 1e-3, ds_max=step, **trace_kw)
    if len(recs) < 2:
        raise RuntimeError(f"final step to the secant root failed: {reason}")
    return points, brackets, (np.asarray(states[1], float), np.asarray(recs[1]["theta"], float), recs[1], step)


def locate(family, xl, thl, xr, thr, tpl, test="cusp", kind="fold", recl=None, recr=None, rounds=3, per_round=5,
           param=1, param_scale=10.0, nf_steps=(1e-3, 3e-4, 1e-4), btol=1e-2, trace_kw=None, log=None):
    """Locate a codimension-two point between two curve points (states x, parameters theta, records rec with the
    fields the test needs; tpl: the oriented parameter tangent at the left point)."""
    tref = np.asarray(tpl, float) / np.linalg.norm(tpl)
    if test == "cusp":
        if kind != "fold":
            raise ValueError("the cusp test needs a fold curve")
        fn = lambda x, th, rec: oriented_b(family, x, th, param, param_scale, nf_steps)[0]   # noqa: E731
    elif test == "crossing":
        fn = lambda x, th, rec: rec["other_spectral_radius"] - 1.0                           # noqa: E731
    elif test == "reversal":
        fn = lambda x, th, rec: float(np.asarray(rec["parameter_tangent"]) @ tref)          # noqa: E731
    else:
        raise ValueError(f"unknown test {test}")
    fl = fn(np.asarray(xl, float), np.asarray(thl, float), recl or {})
    fr = fn(np.asarray(xr, float), np.asarray(thr, float), recr or {})
    f0 = max(abs(fl), abs(fr))
    points, brackets, (xf, thf, recf, step) = refine(family, kind, xl, thl, xr, thr, tpl, fn, fl, fr, rounds,
                                                      per_round, trace_kw, log)
    final = {"theta": list(map(float, thf)), "secant_step": float(step), "f": float(fn(xf, thf, recf)),
             "critical_multiplier": recf.get("critical_multiplier")}
    if test == "cusp":
        unf = bc.fold_unfolding(family, xf, thf)
        cs = np.array([t["c"] for t in unf["c_steps"]], dtype=float)
        d = np.abs(np.diff(cs))
        k = int(np.argmin(d)) if len(d) else 0
        spread = float(d[k] / max(abs(0.5 * (cs[k] + cs[k + 1])), 1e-300)) if len(d) else None
        final.update({"b": float(unf["b"]), "c": float(unf["c"]), "c_plateau_relative_spread": spread,
                      "J": unf["J"], "det_J": float(unf["det_J"]), "singular_values": unf["singular_values"],
                      "condition": float(unf["condition"]), "multiplier": unf["multiplier"],
                      "second_modulus": unf["second_modulus"]})
        checks = {"b_small": bool(abs(final["f"]) <= btol * f0),
                  "c_nonzero": bool(final["c"] != 0 and (spread is None or spread < 0.1)),
                  "unfolding_regular": bool(abs(final["det_J"]) > 0 and np.isfinite(final["condition"]))}
    else:
        final.update(type_point(family, xf, thf, kind))
        checks = {"f_small": bool(abs(final["f"]) <= btol * f0)}
        if test == "crossing":
            d = final["second_distance_to_circle"]
            checks["second_on_circle"] = bool(d is not None and d <= 5e-3)
    return {"test": test, "kind": kind, "points": points, "brackets": brackets, "final": final, "initial_f_max": f0,
            "btol": btol, "checks": checks, "located": bool(all(checks.values())), "state": xf}


def _plain(o):
    """JSON fallback for numpy scalars and arrays."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def main():
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("curve", help="fold- or flip-curve run under explore/ (curve.json, states.npy)")
    ap.add_argument("left", type=int, help="curve index before the sign change; the next index ends the bracket")
    ap.add_argument("--test", required=True, choices=("cusp", "crossing", "reversal"))
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
    ca = cv["args"]
    if ca.get("reference_dtype", "none") != "none" or ca.get("dtype") != "float64":
        raise SystemExit("cusp_locate needs a pure float64 curve run")
    out = esl.DATA / "explore" / args.out
    path = out / "locate.json"
    if path.exists():
        raise SystemExit(f"{path} exists; outputs are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    loop = esl.SoftLoop(ta["model"], ta["layer"], args.threads, ta["prompt"], ta["opener"], ta["window"], "float64",
                        args.device)
    v = elpm.load_vectors(ta["vectors"])[ta["concept"]]
    sc = float(ca.get("alpha_scale", 10.0))
    two = ta.get("block2") is not None                  # a curve of a two-block plane: theta = (alpha, alpha2) * scale
    second = None
    if two:
        loop.sm.add_steer_layer(ta["block2"])
        second = (ta["block2"], elpm.load_vectors(ta["vectors2"])[ta["concept2"]])
    fam = esl.soft_family(loop, v, int(cv["period"]), sc, 512, second=second,
                          log_temperature=ta["log_temperature"] if two else None)
    pl, pr = cv["points"][args.left], cv["points"][args.left + 1]
    if two:   # the normal form's parameter: the one most transverse to the curve (smaller tangent component)
        th = lambda p: np.array([p["alpha"] * sc, p["alpha2"] * sc])         # noqa: E731
        param = int(np.argmin(np.abs(pl["parameter_tangent"])))
    else:
        th = lambda p: np.array([p["log10_temperature"], p["alpha"] * sc])   # noqa: E731
        param = 1
    trace_kw = {k: ca[k] for k in ("tol", "gtol", "max_newton", "gmres_tol", "gmres_maxiter", "refine_k") if k in ca}
    progress = out / "progress.jsonl"
    progress.write_text("")

    def log(rec):
        line = json.dumps({k: rec[k] for k in ("round", "theta", "f", "other_spectral_radius") if k in rec})
        print(line, flush=True)
        with progress.open("a") as fh:
            fh.write(line + "\n")

    tic = time.time()
    res = locate(fam, states[args.left].ravel(), th(pl), states[args.left + 1].ravel(), th(pr),
                 pl["parameter_tangent"], test=args.test, kind=cv["kind"], recl=pl, recr=pr, rounds=args.rounds,
                 per_round=args.per_round, param=param, param_scale=sc, btol=args.btol, trace_kw=trace_kw, log=log)
    np.save(out / "state.npy", res.pop("state"))
    fin = res["final"]
    if two:
        fin["alpha"], fin["alpha2"] = fin["theta"][0] / sc, fin["theta"][1] / sc
        where = {"alpha": fin["alpha"], "alpha2": fin["alpha2"]}
    else:
        fin["log10_temperature"], fin["alpha"] = fin["theta"][0], fin["theta"][1] / sc
        where = {"log10_T": fin["log10_temperature"], "alpha": fin["alpha"]}
    res.update({"curve": args.curve, "left": args.left, "period": int(cv["period"]), "args": vars(args),
                "normal_form_parameter": param, "seconds": round(time.time() - tic, 1)})
    path.write_text(json.dumps(res, indent=1, default=_plain))
    keys = ("type", "second_multiplier", "eigenvector_angle_degrees", "c", "det_J")
    print(json.dumps({"located": res["located"], "checks": res["checks"], **where, "f": fin["f"],
                      **{k: fin[k] for k in keys if k in fin}, "seconds": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
