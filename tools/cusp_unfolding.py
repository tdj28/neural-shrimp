"""Cusp unfolding test at a point of a continued fold curve (consult 6, bounded pilot).

predict: computes the two-parameter unfolding (bifurcation_curves.fold_unfolding: the matrix J of the reduced
equation's derivatives with respect to (log10 T, 10 alpha), b and c) and writes predictions for six frozen parameter
offsets before anything is solved. The offsets are fixed in unfolding coordinates by this rule: with
s = -sign(c), for target root sizes u in (U1, U2), beta_1 = s |c| u^2 (inside the cusp wedge, three roots),
beta_1 = -s |c| u^2 (outside, one root), and (beta_0, beta_1) = (2 beta_0*, s |c| u^2) beyond the wedge edge
beta_0* = 2 |beta_1|^(3/2) / (3 sqrt(3 |c|)) (one root); each is mapped to a parameter offset by J^-1.

solve: at each offset, Newton's method on the p-fold map from x + u r for every predicted root u (and from the
cusp point itself), keeps the distinct converged orbits (relative distance > 1e-6), and records each orbit's
multipliers (critical one, next modulus) and stability. The comparison is the number of orbits and their stability
against the predictions. Orbits are counted only when Newton's residual is below 1e-10 relative.

Usage: python tools/cusp_unfolding.py CURVE_RUN INDEX [DEVICE]   (writes unfolding-INDEX.json next to the curve run)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bifurcation_curves as bc  # noqa: E402

U_TARGETS = (1e-4, 1e-3)


def frozen_offsets(J, c):
    s = -np.sign(c)
    Jinv = np.linalg.inv(np.asarray(J, dtype=np.float64))
    out = []
    for u in U_TARGETS:
        beta1_in = s * abs(c) * u * u
        edge = 2 * abs(beta1_in) ** 1.5 / (3 * np.sqrt(3 * abs(c)))
        for kind, beta in (("inside", (0.0, beta1_in)), ("other side", (0.0, -beta1_in)),
                           ("beyond edge", (2 * edge, beta1_in))):
            out.append({"kind": kind, "u_target": u, "beta": list(beta), "offset": (Jinv @ np.array(beta)).tolist()})
    return out


def newton(family, x0, theta, tol=1e-10, max_iter=12):
    x = np.asarray(x0, dtype=np.float64).copy()
    scale = max(1.0, float(np.linalg.norm(x)))
    for _ in range(max_iter):
        F = family.residual(x, theta)
        if np.linalg.norm(F) < tol * scale:
            return x, float(np.linalg.norm(F) / scale)
        M = family.monodromy(x, theta)
        x = x - np.linalg.solve(M - np.eye(len(x)), F)
    F = family.residual(x, theta)
    return x, float(np.linalg.norm(F) / scale)


def main():
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    run, index = sys.argv[1], int(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    src = esl.DATA / "explore" / run
    out_path = src / f"unfolding-{index}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; outputs are never overwritten")
    cv = json.loads((src / "curve.json").read_text())
    states = np.load(src / "states.npy")
    start = json.loads((esl.DATA / "explore" / cv["args"]["start"] / "branch.json").read_text())["args"]
    loop = esl.SoftLoop(start["model"], start["layer"], 12, start["prompt"], start["opener"], start["window"],
                        "float64", device)
    v = elpm.load_vectors(start["vectors"])[start["concept"]]
    fam = esl.soft_family(loop, v, int(cv["period"]), 10.0, 512)
    p = cv["points"][index]
    theta = np.array([p["log10_temperature"], p["alpha"] * 10.0])
    x = states[index].ravel()
    tic = time.time()
    unf = bc.fold_unfolding(fam, x, theta)
    offsets = frozen_offsets(unf["J"], unf["c"])
    preds = bc.cusp_predictions(unf, [o["offset"] for o in offsets])
    record = {"curve": run, "index": index, "period": int(cv["period"]), "log10_T": p["log10_temperature"],
              "alpha": p["alpha"], "unfolding": {k: unf[k] for k in ("multiplier", "second_modulus", "J", "det_J",
                                                                      "singular_values", "condition", "b", "c")},
              "g_u_theta_steps": unf["g_u_theta_steps"], "c_steps": unf["c_steps"],
              "predictions": [dict(o, **pr) for o, pr in zip(offsets, preds)],
              "predicted_at_seconds": round(time.time() - tic, 1)}
    out_path.write_text(json.dumps(record, indent=1))          # predictions are on disk before anything is solved
    print(json.dumps({"stage": "predicted", "det_J": unf["det_J"], "condition": unf["condition"], "c": unf["c"],
                      "b": unf["b"]}), flush=True)
    r = unf["r"]
    results = []
    for o, pr in zip(offsets, preds):
        th = theta + np.array(o["offset"])
        found = []
        for u0 in list(pr["roots"]) + [0.0]:
            xs, res = newton(fam, x + u0 * r, th)
            if res > 1e-10 or not np.all(np.isfinite(xs)):
                continue
            if any(np.linalg.norm(xs - f["x"]) / max(1.0, np.linalg.norm(xs)) < 1e-6 for f in found):
                continue
            M = fam.monodromy(xs, th)
            mu = np.linalg.eigvals(M)
            k = int(np.argmax(np.abs(mu)))
            found.append({"x": xs, "u": float(r @ (xs - x)), "residual": res,
                          "leading_multiplier": [float(mu[k].real), float(mu[k].imag)],
                          "stable": bool(np.abs(mu).max() < 1)})
        found.sort(key=lambda f: f["u"])
        results.append({"kind": o["kind"], "u_target": o["u_target"], "predicted_roots": len(pr["roots"]),
                        "predicted_stable": pr["stable"], "found": len(found),
                        "found_stable": [f["stable"] for f in found],
                        "found_u": [f["u"] for f in found], "found_leading": [f["leading_multiplier"] for f in found],
                        "match_count": len(found) == len(pr["roots"]),
                        "match_stability": [f["stable"] for f in found] == pr["stable"]})
        print(json.dumps({k: results[-1][k] for k in ("kind", "u_target", "predicted_roots", "found",
                                                       "match_count", "match_stability")}), flush=True)
    record["solved"] = results
    record["seconds"] = round(time.time() - tic, 1)
    out_path.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
