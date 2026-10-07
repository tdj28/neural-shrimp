"""Exploratory minimal-model scan: does an edge-of-stability 2-cycle undergo Neimark-Sacker?

Subcommands (outputs under DATA/explore/minimal-model):
  controls   engineered positive control (Neimark-Sacker typing and criticality
             probe for several kappa) and the two-parameter factorization model
             (D = (tr M)^2 - 4 det M along its 2-cycle branches)
  grid       coupled four-weight model over c x ridge x rate ratio: from the
             minimum through the fixed-point flip, then along the 2-cycle as the
             overall rate grows; boundary type, multiplier angle, dominant-pair
             history, transversality, resonance check, criticality probe
  predict    held-out predictions for c = 0.45 and 0.75 from the grid results
             (written before the held-out cases are run)
  heldout    run the held-out cases and score them against the predictions
  summarize  table and figure

Exploratory under docs/adaptive-research-protocol.md.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.minimal import (EngineeredTwoCycle, TwoModeFactorization, dominant_pair,  # noqa: E402
                                   follow_orbit, iterate)
from neural_shrimp.orbits import certify_orbit  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
OUT = DATA / "explore" / "minimal-model"
C_GRID = [0.0, 0.3, 0.6, 0.9]
C_HELDOUT = [0.45, 0.75]
RIDGES = [0.05, 0.2]
RATIOS = [float(2.0 ** (k / 2)) for k in range(-3, 5)]
STRONG = (0.5, 1.0 / 3.0, 0.25)


def theta_over_2pi(mu):
    return float(abs(np.angle(mu)) / (2 * np.pi))


def history(path):
    """Runs of (complex-dominant-pair flag) along the path as [(flag, s_from, s_to)]."""
    runs = []
    for s, mus, _ in path:
        _, pair = dominant_pair(mus)
        if runs and runs[-1][0] == pair:
            runs[-1][2] = s
        else:
            runs.append([pair, s, s])
    return [{"complex_dominant_pair": bool(f), "s_from": float(a), "s_to": float(b)} for f, a, b in runs]


def transversality(path, s_boundary):
    tail = [(s, abs(dominant_pair(m)[0])) for s, m, _ in path if s_boundary - s < 0.05]
    if len(tail) < 3:
        tail = [(s, abs(dominant_pair(m)[0])) for s, m, _ in path[-4:]]
    if len(tail) < 2:
        return None
    s, r = np.array(tail).T
    return float(np.polyfit(s, np.log(r), 1)[0])


def criticality_probe(step_family, s_boundary, state, period_points=2, deltas=(1e-3, 4e-3), steps=60000):
    """Iterate just past the boundary; small bounded non-periodic attractor near the old orbit
    with amplitude growing like sqrt(delta) suggests a supercritical Neimark-Sacker."""
    out = []
    for delta in deltas:
        step = step_family(s_boundary + delta)
        x = iterate(step, np.asarray(state) + 1e-4, steps)
        if x is None:
            out.append({"delta": delta, "outcome": "diverged"})
            continue
        pts = []
        for _ in range(4000):
            for _ in range(period_points):
                x = step(x)
            pts.append(x.copy())
        pts = np.array(pts)
        distance = float(np.max(np.linalg.norm(pts - np.asarray(state), axis=1)))
        centered = pts - pts.mean(0)
        _, sing, vt = np.linalg.svd(centered, full_matrices=False)
        planar = centered @ vt[:2].T
        angles = np.unwrap(np.arctan2(planar[:, 1], planar[:, 0]))
        rho = float(abs(angles[-1] - angles[0]) / (2 * np.pi * (len(angles) - 1)))
        amplitude = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        recurrence = float(np.min(np.linalg.norm(pts[1:] - pts[0], axis=1)))
        out.append({"delta": delta, "outcome": "bounded", "max_distance_from_old_orbit": distance,
                    "amplitude": amplitude, "rotation": rho, "planar_share": float((sing[:2] ** 2).sum() / max((sing ** 2).sum(), 1e-300)),
                    "min_return_distance": recurrence})
    verdict = "unclear"
    if all(o["outcome"] == "bounded" for o in out):
        a1, a2 = out[0]["amplitude"], out[1]["amplitude"]
        small = all(o["max_distance_from_old_orbit"] < 0.3 for o in out)
        if small and a1 > 1e-7 and 1.3 < a2 / a1 < 3.0:
            verdict = "supercritical_like (amplitude ratio %.2f, sqrt ratio 2)" % (a2 / a1)
        elif not small:
            verdict = "left the neighbourhood (subcritical or other attractor)"
    elif any(o["outcome"] == "diverged" for o in out):
        verdict = "diverged (subcritical or global escape)"
    return {"runs": out, "verdict": verdict}


def run_case(task):
    c, lam, ratio = task
    model = TwoModeFactorization(c=c, lam=lam)
    eta_f = model.flip_rate(ratio)
    direction = model.critical_direction(ratio)
    result = {"c": c, "lam": lam, "ratio": ratio, "eta_flip": eta_f}
    start = None
    for factor in (1.002, 1.01, 1.03, 1.1):
        eta_a = eta_f * factor
        step, jac = model.map_pair(eta_a, ratio * eta_a)
        x = iterate(step, model.theta_star + 1e-3 * direction / np.linalg.norm(direction), 40000)
        if x is None:
            result.setdefault("start_attempts", []).append({"factor": factor, "outcome": "diverged"})
            continue
        cert = certify_orbit(step, jac, x, 2, candidate_tolerance=0.05)
        result.setdefault("start_attempts", []).append({"factor": factor, "outcome": cert["reason"],
                                                        "attracting": cert["attracting"]})
        if cert["verified"] and cert["attracting"]:
            start = (eta_a, np.asarray(cert["orbit"][0]))
            break
    if start is None:
        result["type"] = "no_stable_2cycle_after_flip"
        return result
    eta0, x0 = start
    result["eta_start"] = eta0

    def family(s):
        eta_a = eta0 * 10.0 ** s
        return model.map_pair(eta_a, ratio * eta_a)

    follow = follow_orbit(family, x0, 2, s_max=1.5, h0=0.002)
    result.update({"type": follow["type"], "s_boundary": follow["s_boundary"],
                   "eta_a_boundary": None if follow["s_boundary"] is None else eta0 * 10 ** follow["s_boundary"],
                   "dominant": follow["dominant"]})
    if follow["path"]:
        result["history"] = history(follow["path"])
        result["path_points"] = len(follow["path"])
    if follow["s_boundary"] is not None and follow["dominant"] is not None:
        mu = complex(*follow["dominant"])
        result["theta_over_2pi"] = theta_over_2pi(mu)
        result["modulus"] = abs(mu)
        result["log_modulus_slope"] = transversality(follow["path"], follow["s_boundary"])
        if follow["type"] == "neimark_sacker":
            result["strong_resonance_distance"] = float(min(abs(result["theta_over_2pi"] - q) for q in STRONG))
            result["criticality"] = criticality_probe(lambda s: family(s)[0], follow["s_boundary"],
                                                      follow["last_attracting_state"])
    return result


def cmd_controls(args):
    out = {}
    phi = 2 * np.pi * 0.13

    def family(s, kappa=0.0):
        model = EngineeredTwoCycle(0.95 + s, phi, kappa)
        return model.step, model.jacobian

    follow = follow_orbit(family, EngineeredTwoCycle.p0, 2, s_max=0.1, h0=0.005)
    mu = complex(*follow["dominant"])
    out["engineered"] = {"phi_over_2pi": 0.13, "type": follow["type"], "r_boundary": 0.95 + follow["s_boundary"],
                         "theta_over_2pi": theta_over_2pi(mu), "history": history(follow["path"]),
                         "criticality": {}}
    for kappa in (-1.0, -0.2, 0.0, 0.2, 1.0):
        probe = criticality_probe(lambda s: EngineeredTwoCycle(0.95 + s, phi, kappa).step, 0.05,
                                  EngineeredTwoCycle.p0, deltas=(2e-3, 8e-3))
        out["engineered"]["criticality"][str(kappa)] = probe
    # two-parameter factorization model: D along 2-cycle branches
    from neural_shrimp.gdmaps import GridMap, TwoLayerNet
    net = TwoLayerNet(d=1, width=1, activation="identity", in_scale=1.0, out_scale=1.0, bias=False,
                      loss_scale=1.0, ridge=0.1)
    x, y = np.array([[1.0]]), np.array([1.0])
    saved = np.load(DATA / "explore" / "a3-uv-model" / "map.npz")
    status, period, final = saved["status"], saved["period"], saved["final_state"]
    la, lb = saved["log10_a"], saved["log10_b"]
    branches = []
    for j in np.linspace(0, len(lb) - 1, 8).astype(int):
        rows = np.flatnonzero((status[:, j] == 2) & (period[:, j] == 2))
        if not len(rows):
            continue
        i = rows[len(rows) // 2]

        def uv_family(s, i=i, j=j):
            return GridMap(net, x, y, [10.0 ** (la[i] + s)], [10.0 ** lb[j]]).cell_map(0)

        follow = follow_orbit(uv_family, final[i, j], 2, s_max=0.8, h0=0.002)
        ds = []
        for s, mus, state in follow["path"]:
            map_fn, jac = uv_family(s)
            m = jac(map_fn(state)) @ jac(state)
            ds.append(float(np.trace(m) ** 2 - 4 * np.linalg.det(m)))
        branches.append({"log10_b": float(lb[j]), "type": follow["type"],
                         "min_D": float(min(ds)) if ds else None, "points": len(ds),
                         "any_negative_D": bool(any(d < 0 for d in ds))})
    out["factorization_2param"] = branches
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "controls.json").write_text(json.dumps(out, indent=1, default=str))
    e = out["engineered"]
    print("engineered:", e["type"], "at r =", round(e["r_boundary"], 9), "theta/2pi =", round(e["theta_over_2pi"], 6))
    for kappa, probe in e["criticality"].items():
        print("  kappa", kappa, probe["verdict"], [(r["delta"], r["outcome"], round(r.get("amplitude", float("nan")), 5),
                                                    round(r.get("rotation", float("nan")), 4)) for r in probe["runs"]])
    for b in branches:
        print("factorization branch log10_b", round(b["log10_b"], 3), b["type"], "min D", b["min_D"],
              "any D<0", b["any_negative_D"])


def run_grid(cs, name, workers):
    tasks = [(c, lam, ratio) for c in cs for lam in RIDGES for ratio in RATIOS]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(run_case, tasks, chunksize=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(rows, indent=1, default=str))
    for r in rows:
        hist = "".join("C" if h["complex_dominant_pair"] else "R" for h in r.get("history", []))
        print(f"c={r['c']:.2f} lam={r['lam']:.2f} ratio={r['ratio']:.3f}: {r['type']:<27} "
              f"theta/2pi={r.get('theta_over_2pi', float('nan')):.4f} |mu|={r.get('modulus', float('nan')):.6f} "
              f"slope={r.get('log_modulus_slope') or float('nan'):.3g} history={hist} "
              f"{(r.get('criticality') or {}).get('verdict', '')}")
    return rows


def cmd_predict(args):
    rows = json.loads((OUT / "grid.json").read_text())
    table = {(r["c"], r["lam"], r["ratio"]): r for r in rows}
    predictions = []
    for c, (lo, hi) in ((0.45, (0.3, 0.6)), (0.75, (0.6, 0.9))):
        for lam in RIDGES:
            for ratio in RATIOS:
                a, b = table[(lo, lam, ratio)], table[(hi, lam, ratio)]
                kind = a["type"] if a["type"] == b["type"] else "uncertain"
                pred = {"c": c, "lam": lam, "ratio": ratio, "predicted_type": kind}
                if kind == "neimark_sacker":
                    pred["predicted_theta_over_2pi"] = 0.5 * (a["theta_over_2pi"] + b["theta_over_2pi"])
                predictions.append(pred)
    (OUT / "predictions.json").write_text(json.dumps(predictions, indent=1))
    print(json.dumps({"predictions": len(predictions),
                      "types": {k: sum(p["predicted_type"] == k for p in predictions)
                                for k in sorted({p["predicted_type"] for p in predictions})}}))


def cmd_heldout(args):
    predictions = {(p["c"], p["lam"], p["ratio"]): p for p in json.loads((OUT / "predictions.json").read_text())}
    rows = run_grid(C_HELDOUT, "heldout.json", args.workers)
    hits = scored = 0
    for r in rows:
        p = predictions[(r["c"], r["lam"], r["ratio"])]
        if p["predicted_type"] == "uncertain":
            continue
        scored += 1
        hits += p["predicted_type"] == r["type"]
    print(f"held-out type predictions: {hits}/{scored} correct ({len(rows) - scored} uncertain, not scored)")


def cmd_summarize(args):
    rows = json.loads((OUT / "grid.json").read_text())
    held = json.loads((OUT / "heldout.json").read_text()) if (OUT / "heldout.json").exists() else []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"flip": "#eb6834", "neimark_sacker": "#1baf7a", "fold": "#2a78d6"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    for ax, lam in zip(axes, RIDGES):
        for r in rows + held:
            if r["lam"] != lam:
                continue
            color = colors.get(r["type"], "#a3a29b")
            marker = "o" if r in rows else "s"
            ax.scatter(r["c"], np.log2(r["ratio"]), s=90, color=color, marker=marker, edgecolor="#fcfcfb")
            if r["type"] == "neimark_sacker":
                ax.annotate(f"{r['theta_over_2pi']:.3f}", (r["c"], np.log2(r["ratio"])), textcoords="offset points",
                            xytext=(6, -3), fontsize=7)
        ax.set_title(f"ridge {lam}: stability loss of the edge-of-stability 2-cycle", fontsize=9, loc="left")
        ax.set_xlabel("mode coupling c", fontsize=9)
    axes[0].set_ylabel("log2(eta_b / eta_a)", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=v, label=k.replace("_", "-")) for k, v in colors.items()]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", color="#a3a29b", label="other / none"))
    handles.append(plt.Line2D([], [], marker="s", linestyle="", color="#52514e", label="held-out c"))
    axes[1].legend(handles=handles, fontsize=7, frameon=False, loc="upper right")
    fig.savefig(OUT / "grid.png", dpi=160)
    print("figure", OUT / "grid.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["controls", "grid", "predict", "heldout", "summarize"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "controls":
        cmd_controls(args)
    elif args.command == "grid":
        run_grid(C_GRID, "grid.json", args.workers)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "heldout":
        cmd_heldout(args)
    else:
        cmd_summarize(args)


if __name__ == "__main__":
    main()
