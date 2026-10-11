"""Exploratory: global organization of the S2-28011 period-18 (2/9) window.

The certified period-18 window sits about 0.07 decades above the period-2
Neimark-Sacker curve, whose 2/9 roots lie below it. These helpers test where
the window's boundaries come from, using saved carried boundary nodes
(``explore/s2-period18-boundary-curves``) and rotation maps.

Subcommands:
  circle-check  Does a saved period-18 fold orbit on the window's lower U fold
                lie on the attracting invariant circle of the two-step map just
                outside the fold? Iterates from the fold orbit at rates moved
                by small offsets (decades of the hidden rate) and measures the
                distance of every orbit point to the attractor's point cloud.
  figure        Anatomy figure from a saved rotation map and the carried
                fold/flip curves: rotation field, the 2/9 level set, the
                rotation minimum along each column, and the boundary curves.
  banana        Exact period-18 orbits on the near-curve 2/9 arc: shoot from
                circle points with the hidden rate free, trace the closed
                orbit curve at fixed output rate through its two folds, then
                continue both folds in the rate plane and report where they
                go (toward the 2/9 roots, the window, or elsewhere).
  fold-from-node  Continue a saved carried fold (or flip) node of the window
                in both directions with fast float64 continuation, recording
                every node's rates, multipliers and parameter tangent so cusps
                (tangent reversals), 1:1 points (a second multiplier at +1),
                fold-flip points (a multiplier at -1) and fold-NS points (a
                complex pair on the unit circle) can be located afterwards.
  composite     One figure of the organization: rotation field with the 2/9
                level set, the banana's two folds, continued window folds and
                the carried flip curves, plus a zoom on the banana and its
                locking width along the curve.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.fastcont import (FoldCurve, OrbitCurve, TwoRateMap, arclength,  # noqa: E402
                                    divisor_separations, newton, null_tangent)
from neural_shrimp.regimes import rotation_number  # noqa: E402
from neural_shrimp.trainability import make_s2  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
BOUNDARY = DATA / "explore" / "s2-period18-boundary-curves"
SEED = 28011


def load_node(path):
    node = json.loads(Path(path).read_text())
    point = [float(v) for v in node["point"]]
    orbit = np.array(node["orbit"], dtype=np.float64)
    return {"hidden": point[13], "output": point[14], "orbit": orbit[:18],
            "closure": float(np.max(np.abs(orbit[18] - orbit[0])))}


def make_step(hidden, output):
    net, x, y, _ = make_s2(SEED)
    rates = net.rate_vector([hidden], [output])

    def step(theta):
        return theta - rates * net.gradient(theta, x, y)

    return step


def attractor_cloud(step, start, transient, samples):
    """Every second state after a transient, as (samples, 13)."""
    theta = start[None, :].copy()
    for _ in range(transient):
        theta = step(theta)
    cloud = np.empty((samples, theta.shape[1]))
    for k in range(samples):
        cloud[k] = theta[0]
        theta = step(step(theta))
    return cloud


def circle_check(args):
    node = load_node(args.node)
    result = {"node": str(args.node), "hidden": node["hidden"], "output": node["output"],
              "log10_hidden": math.log10(node["hidden"]), "log10_output": math.log10(node["output"]),
              "closure": node["closure"], "offsets": []}
    orbit = node["orbit"]
    for offset in args.offsets:
        hidden = node["hidden"] * 10.0 ** offset
        step = make_step(hidden, node["output"])
        start = orbit[0] + args.kick
        tic = time.perf_counter()
        cloud = attractor_cloud(step, start, args.transient, args.samples)
        tree = cKDTree(cloud)
        distances, _ = tree.query(orbit)
        centered = cloud - cloud.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        extent = float(2.0 * math.sqrt(2.0) * singular[0] / math.sqrt(len(cloud)))
        rho, rho_a, rho_b, share = rotation_number(cloud[: min(len(cloud), 20000)])
        unique = np.unique(np.round(cloud[-64:], 9), axis=0)
        row = {"offset_decades": offset, "hidden": hidden,
               "distance_even": distances[0::2].tolist(), "distance_odd": distances[1::2].tolist(),
               "cloud_extent": extent, "rotation": rho, "rotation_halves": [rho_a, rho_b],
               "planar_share": share, "distinct_last64": int(len(unique)),
               "seconds": round(time.perf_counter() - tic, 2)}
        result["offsets"].append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()
                          if not k.startswith("distance")}))
        print("   even-point distances", np.round(distances[0::2], 7).tolist())
        print("   odd-point distances ", np.round(distances[1::2], 7).tolist())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "circle-check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out / "circle-check.json")}))


def boundary_curves(summary):
    """Carried boundary nodes grouped into polylines (run, sign) in log10 coordinates."""
    rows = json.loads(Path(summary).read_text())["rows"]
    curves = {}
    for row in rows:
        key = (row["run"], row["sign"], row["sigma"])
        curves.setdefault(key, []).append(row)
    out = []
    for (run, sign, sigma), members in curves.items():
        members.sort(key=lambda r: r["index"])
        out.append({"run": run, "sign": sign, "sigma": sigma,
                    "x": np.log10([r["output"] for r in members]),
                    "y": np.log10([r["hidden"] for r in members]),
                    "attracting": np.array([r["complement_radius"] < 1 for r in members])})
    return out


def figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.rotation)
    lh, lo, rho, status = z["log10_hidden"], z["log10_output"], z["rho"], z["status"]
    target = 2.0 / 9.0
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    extent = [lo[0], lo[-1], lh[0], lh[-1]]
    shown = np.where(np.isfinite(rho) & (rho > 0.05), rho, np.nan)
    image = ax.imshow(shown, origin="lower", extent=extent, aspect="auto", cmap="coolwarm",
                      vmin=target - args.span, vmax=target + args.span, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="rotation number of the two-step map")
    locked = np.where(np.abs(rho - target) < args.lock_tolerance, 1.0, np.nan)
    ax.imshow(locked, origin="lower", extent=extent, aspect="auto", cmap="Greys", vmin=0, vmax=1.2,
              alpha=0.55, interpolation="nearest")
    # rotation minimum along each column, searched between the curve and the window
    # rotation minimum along each column, over quasi-periodic cells (statuses
    # unresolved/drift) between the curve and the window; cells below the curve
    # carry transient spiral angles and chaotic cells noisy ones, so both are excluded
    minima = []
    for j in range(len(lo)):
        column = rho[:, j]
        valid = (np.isfinite(column) & np.isin(status[:, j], (4, 5)) & (column > 0.2)
                 & (lh < args.min_search_top))
        rows = np.where(valid)[0]
        if len(rows) < 5:
            continue
        i = rows[np.argmin(column[rows])]
        if rows[0] < i < rows[-1]:
            minima.append((lo[j], lh[i], column[i]))
    minima = np.array(minima)
    saddle = None
    if len(minima):
        ax.plot(minima[:, 0], minima[:, 1], "k.", ms=2, label="rotation minimum along column")
        k = int(np.argmax(minima[:, 2]))
        saddle = minima[k].tolist()
        ax.plot(saddle[0], saddle[1], "kD", ms=7, mfc="none",
                label=f"highest column minimum {saddle[2]:.5f} (rotation saddle)")
    for curve in boundary_curves(args.summary):
        colour = "tab:green" if curve["sigma"] == 1 else "tab:purple"
        ax.plot(curve["x"], curve["y"], "-", color=colour, lw=1.4)
        bad = ~curve["attracting"]
        ax.plot(curve["x"][bad], curve["y"][bad], "x", color=colour, ms=3)
    for name, (x, y) in {"root A": (-0.0769, 1.7447), "root B": (-0.0681, 1.7369)}.items():
        ax.plot(x, y, "*", ms=13, color="gold", mec="k")
        ax.annotate(name, (x, y), xytext=(4, -12), textcoords="offset points", fontsize=9)
    ax.plot([], [], "-", color="tab:green", label="fold curves (carried, x = complement not attracting)")
    ax.plot([], [], "-", color="tab:purple", label="flip curves (carried)")
    ax.set_xlim(lo[0], lo[-1])
    ax.set_ylim(lh[0], lh[-1])
    ax.set_xlabel("log10 output-layer rate")
    ax.set_ylabel("log10 hidden-layer rate")
    ax.set_title(f"rotation field; grey = |rho - 2/9| < {args.lock_tolerance:g}")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    for x0 in args.columns:
        j = int(np.argmin(np.abs(lo - x0)))
        column = rho[:, j]
        keep = np.isfinite(column) & (column > 0.05)
        ax.plot(lh[keep], column[keep], ".-", ms=2, lw=0.8, label=f"x = {lo[j]:.4f}")
    ax.axhline(target, color="k", lw=0.8)
    ax.set_ylim(target - args.span, target + args.span)
    ax.set_xlabel("log10 hidden-layer rate (column through the map)")
    ax.set_ylabel("rotation number")
    ax.set_title("rotation along columns: fall, minimum below 2/9, rise to the window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "window-anatomy.png", dpi=args.dpi)
    record = {"rotation": str(args.rotation), "summary": str(args.summary), "saddle": saddle,
              "column_minima": minima.tolist() if len(minima) else []}
    (out / "window-anatomy.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"out": str(out / "window-anatomy.png"), "minima_columns": int(len(minima)),
                      "saddle": saddle,
                      "max_column_minimum": float(minima[:, 2].max()) if len(minima) else None,
                      "min_column_minimum": float(minima[:, 2].min()) if len(minima) else None}))


def even_spread(points):
    """Largest distance of the nine even points of a period-18 orbit from their mean."""
    even = points[0:18:2]
    return float(np.max(np.linalg.norm(even - even.mean(axis=0), axis=1)))


def summarize_node(node, period):
    z, m = node["z"], node["monodromy"]
    n = 13
    mult = np.linalg.eigvals(m)
    order = np.argsort(-np.abs(mult))
    return {"lx": float(z[-2]) if len(z) > n + 1 else None, "ly": float(z[-1]),
            "spread": even_spread(node["points"]),
            "min_divisor_separation": min(divisor_separations(node["points"], period).values()),
            "multipliers": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]],
            "newton_last": node["newton"][-1], "h": node["h"]}


def banana(args):
    net, x, y, theta0 = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    period = 18
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"lx": args.lx, "ly_seed": args.ly, "period": period, "transient": args.transient}
    p_seed = [args.lx, args.ly]
    theta = f.iterate(theta0, p_seed, args.transient)
    short = f.orbit(theta, p_seed, 2 * 64)[0::2]
    j = int(np.argmax(short.std(axis=0)))
    target = float(theta[j])
    curve = OrbitCurve(f, period, fixed=args.lx)
    phase_row = np.zeros(f.n + 1)
    phase_row[j] = 1.0
    z0, history = newton(curve, np.concatenate([theta, [args.ly]]), max_iter=30,
                         extra=lambda w: (phase_row, w[j] - target))
    record["seed_newton"] = history
    if z0 is None:
        record["outcome"] = "no period-18 orbit from the circle seed"
        (out / "banana.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({"outcome": record["outcome"], "newton": history[-3:]}))
        return
    _, jac, m, points = curve.residual_jacobian(z0)
    seps = divisor_separations(points, period)
    record["seed_orbit"] = {"ly": float(z0[-1]), "spread": even_spread(points), "divisor_separations": seps,
                            "multipliers_abs": sorted(np.abs(np.linalg.eigvals(m)).tolist(), reverse=True)[:4]}
    print(json.dumps({"seed_ly": z0[-1], "spread": even_spread(points), "min_sep": min(seps.values())}))
    # trace the closed orbit curve at fixed output rate
    scale = np.concatenate([np.ones(f.n), [args.ly_scale]])
    t0 = null_tangent(jac, scale)
    seen = []

    def closed(nd):
        seen.append(1)
        return len(seen) > 10 and np.linalg.norm((nd["z"] - z0) * scale) < 1.5 * nd["h"]

    nodes, why = arclength(curve, z0, t0, scale, steps=args.loop_steps, h0=args.loop_h, h_min=1e-9,
                           h_max=4 * args.loop_h, stop=closed)
    record["loop_stop"] = why
    lys = np.array([nd["z"][-1] for nd in nodes]) if nodes else np.array([])
    record["loop"] = [{"ly": float(nd["z"][-1]), "spread": even_spread(nd["points"]),
                       "theta_j": float(nd["z"][j])} for nd in nodes]
    if len(lys) == 0:
        record["outcome"] = "orbit curve did not advance"
        (out / "banana.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({"outcome": record["outcome"]}))
        return
    record["loop_ly_range"] = [float(lys.min()), float(lys.max())]
    print(json.dumps({"loop_nodes": len(nodes), "why": why, "ly_min": lys.min(), "ly_max": lys.max(),
                      "width_decades": lys.max() - lys.min()}))
    # refine both folds at fixed output rate, then continue them in the rate plane
    folds = []
    for label, k in (("lower", int(np.argmin(lys))), ("upper", int(np.argmax(lys)))):
        nd = nodes[k]
        theta_f, ly_f = nd["z"][:f.n], nd["z"][-1]
        mult, vecs = np.linalg.eig(nd["monodromy"])
        i = int(np.argmin(np.abs(mult - 1.0)))
        q = np.real(vecs[:, i])
        q /= np.linalg.norm(q)
        fold = FoldCurve(f, period, sigma=1.0)
        zf0 = np.concatenate([theta_f, q, [args.lx, ly_f]])
        lx_row = np.zeros(2 * f.n + 2)
        lx_row[-2] = 1.0
        zf, fh = newton(fold, zf0, max_iter=20, extra=lambda w: (lx_row, w[-2] - args.lx))
        entry = {"label": label, "start_ly": float(ly_f), "newton": fh}
        if zf is None:
            entry["outcome"] = "fold refinement failed"
            folds.append(entry)
            continue
        _, fj, fm, fpoints = fold.residual_jacobian(zf)
        entry["fold_ly"] = float(zf[-1])
        fscale = np.concatenate([np.ones(2 * f.n), [args.p_scale, args.p_scale]])
        branches = {}
        for sign in (1, -1):
            tangent = sign * null_tangent(fj, fscale)
            path, reason = arclength(
                fold, zf, tangent, fscale, steps=args.fold_steps, h0=args.fold_h, h_min=1e-7,
                h_max=args.fold_h_max,
                stop=lambda node: (even_spread(node["points"]) < args.min_spread
                                   or not (args.box[0] <= node["z"][-2] <= args.box[1]
                                           and args.box[2] <= node["z"][-1] <= args.box[3])))
            branches[str(sign)] = {"reason": reason,
                                   "nodes": [summarize_node(node, period) | {"lx": float(node["z"][-2])}
                                             for node in path]}
            last = branches[str(sign)]["nodes"][-1] if path else None
            print(json.dumps({"fold": label, "direction": sign, "nodes": len(path), "reason": reason,
                              "end": None if last is None else [last["lx"], last["ly"], last["spread"]]}))
        entry["branches"] = branches
        folds.append(entry)
    record["folds"] = folds
    (out / "banana.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"out": str(out / "banana.json")}))


def node_events(node, sigma):
    """Spectral and tangent data for one fold/flip continuation node."""
    m, z, t = node["monodromy"], node["z"], node["tangent"]
    mult = np.linalg.eigvals(m)
    crit = int(np.argmin(np.abs(mult - sigma)))
    others = np.delete(mult, crit)
    near_plus = others[int(np.argmin(np.abs(others - 1.0)))]
    near_minus = others[int(np.argmin(np.abs(others + 1.0)))]
    complex_others = others[np.abs(others.imag) > 1e-9]
    unit_complex = (complex_others[int(np.argmin(np.abs(np.abs(complex_others) - 1.0)))]
                    if len(complex_others) else None)
    return {"lx": float(z[-2]), "ly": float(z[-1]), "t_lx": float(t[-2]), "t_ly": float(t[-1]),
            "critical": [float(mult[crit].real), float(mult[crit].imag)],
            "complement_radius": float(np.max(np.abs(others))),
            "nearest_plus_one": [float(near_plus.real), float(near_plus.imag)],
            "nearest_minus_one": [float(near_minus.real), float(near_minus.imag)],
            "complex_nearest_unit": None if unit_complex is None else
            [float(unit_complex.real), float(unit_complex.imag)],
            "min_divisor_separation": min(divisor_separations(node["points"], 18).values()),
            "spread": even_spread(node["points"]), "newton_last": node["newton"][-1], "h": node["h"]}


def fold_from_node(args):
    net, x, y, _ = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    node = json.loads(Path(args.node).read_text())
    point = [float(v) for v in node["point"]]
    theta, q = np.array(point[:13]), np.array(point[15:28])
    z0 = np.concatenate([theta, q / np.linalg.norm(q), [math.log10(point[14]), math.log10(point[13])]])
    system = FoldCurve(f, 18, sigma=args.sigma)
    z, history = newton(system, z0, max_iter=15)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"node": str(args.node), "sigma": args.sigma, "start_newton": history}
    if z is None:
        record["outcome"] = "start node did not correct"
        (out / "fold-from-node.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(record))
        return
    _, jac, m, points = system.residual_jacobian(z)
    scale = np.concatenate([np.ones(26), [args.p_scale, args.p_scale]])
    record["start"] = {"lx": float(z[-2]), "ly": float(z[-1])}
    for sign in (1, -1):
        tangent = sign * null_tangent(jac, scale)
        tic = time.perf_counter()
        path, reason = arclength(
            system, z, tangent, scale, steps=args.steps, h0=args.h, h_min=1e-7, h_max=args.h_max,
            stop=lambda nd: (not (args.box[0] <= nd["z"][-2] <= args.box[1]
                                  and args.box[2] <= nd["z"][-1] <= args.box[3])
                             or min(divisor_separations(nd["points"], 18).values()) < 1e-4))
        rows = [node_events(nd, args.sigma) for nd in path]
        record[f"direction_{sign:+d}"] = {"reason": reason, "seconds": round(time.perf_counter() - tic, 1),
                                          "nodes": rows}
        end = rows[-1] if rows else None
        print(json.dumps({"direction": sign, "nodes": len(rows), "reason": reason,
                          "end": None if end is None else [end["lx"], end["ly"], end["complement_radius"]]}))
    (out / "fold-from-node.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"out": str(out / "fold-from-node.json")}))


def composite(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.rotation)
    lh, lo, rho = z["log10_hidden"], z["log10_output"], z["rho"]
    target = 2.0 / 9.0
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(grid[:, 0])
    extent = [lo[0], lo[-1], lh[0], lh[-1]]
    shown = np.where(np.isfinite(rho) & (rho > 0.05), rho, np.nan)
    image = ax.imshow(shown, origin="lower", extent=extent, aspect="auto", cmap="coolwarm",
                      vmin=target - 0.006, vmax=target + 0.006, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="rotation number of the two-step map")
    ax.imshow(np.where(np.abs(rho - target) < 5e-5, 1.0, np.nan), origin="lower", extent=extent,
              aspect="auto", cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
    for curve in boundary_curves(args.summary):
        if curve["sigma"] == -1:
            ax.plot(curve["x"], curve["y"], "-", color="tab:purple", lw=1.0)
    ax.plot([], [], "-", color="tab:purple", label="flip curves (carried)")
    colours = ["k", "tab:green", "tab:olive"]
    for k, path in enumerate(args.folds):
        data = json.loads(Path(path).read_text())
        for key in ("direction_+1", "direction_-1"):
            rows = data.get(key, {}).get("nodes", [])
            if not rows:
                continue
            x = np.array([r["lx"] for r in rows])
            yv = np.array([r["ly"] for r in rows])
            ax.plot(x, yv, "-", color=colours[k % len(colours)], lw=1.4)
        ax.plot([], [], "-", color=colours[k % len(colours)], label=f"continued fold: {Path(path).parent.name}")
    banana_data = json.loads(Path(args.banana).read_text())
    for fold in banana_data["folds"]:
        for branch in fold.get("branches", {}).values():
            x = np.array([r["lx"] for r in branch["nodes"]])
            yv = np.array([r["ly"] for r in branch["nodes"]])
            ax.plot(x, yv, "-", color="gold", lw=1.6)
    ax.plot([], [], "-", color="gold", label="banana folds (exact period-18 orbits)")
    for name, (x, yv) in {"A": (-0.0769, 1.7447), "B": (-0.0681, 1.7369)}.items():
        ax.plot(x, yv, "*", ms=12, color="gold", mec="k")
        ax.annotate(name, (x, yv), xytext=(4, -12), textcoords="offset points")
    ax.set_xlim(args.xlim)
    ax.set_ylim(args.ylim)
    ax.set_xlabel("log10 output-layer rate")
    ax.set_ylabel("log10 hidden-layer rate")
    ax.set_title("2/9 resonance: near-curve banana and the detached window")
    ax.legend(loc="upper right", fontsize=7)
    # zoom on the banana
    zx = fig.add_subplot(grid[0, 1])
    zx.imshow(shown, origin="lower", extent=extent, aspect="auto", cmap="coolwarm",
              vmin=target - 0.006, vmax=target + 0.006, interpolation="nearest")
    widths = []
    lower, upper = None, None
    for fold in banana_data["folds"]:
        xs, ys = [], []
        for branch in fold.get("branches", {}).values():
            xs += [r["lx"] for r in branch["nodes"]]
            ys += [r["ly"] for r in branch["nodes"]]
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        zx.plot(xs, ys, "-", color="gold", lw=1.2)
        if fold["label"] == "lower":
            lower = (xs, ys)
        else:
            upper = (xs, ys)
    zx.set_xlim(-0.0785, -0.0665)
    zx.set_ylim(1.734, 1.752)
    zx.set_title("zoom: banana between roots A and B")
    zx.set_xlabel("log10 output-layer rate")
    wx = fig.add_subplot(grid[1, 1])
    if lower is not None and upper is not None:
        grid_x = np.linspace(max(lower[0].min(), upper[0].min()), min(lower[0].max(), upper[0].max()), 400)
        width = np.interp(grid_x, upper[0], upper[1]) - np.interp(grid_x, lower[0], lower[1])
        wx.semilogy(grid_x, np.abs(width), "-", color="goldenrod")
        widths = list(zip(grid_x.tolist(), width.tolist()))
    wx.set_xlabel("log10 output-layer rate")
    wx.set_ylabel("banana thickness (decades of hidden rate)")
    wx.set_title("locking width along the banana")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "window-organization.png", dpi=args.dpi)
    (out / "window-organization.json").write_text(json.dumps(
        {"rotation": str(args.rotation), "folds": args.folds, "banana": str(args.banana),
         "banana_width": widths}, indent=1))
    print(json.dumps({"out": str(out / "window-organization.png")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("circle-check")
    check.add_argument("--node", default=str(BOUNDARY / "attempt-006-ray32-fold/run/node-+1-010.json"))
    check.add_argument("--offsets", type=float, nargs="+", default=[-0.003, -0.001, -0.0003, 0.0003])
    check.add_argument("--kick", type=float, default=1e-6)
    check.add_argument("--transient", type=int, default=100000)
    check.add_argument("--samples", type=int, default=100000)
    check.add_argument("--out", required=True)
    fig = sub.add_parser("figure")
    fig.add_argument("--rotation", required=True)
    fig.add_argument("--summary", default=str(BOUNDARY / "combined-summary005/summary.json"))
    fig.add_argument("--span", type=float, default=0.006)
    fig.add_argument("--lock-tolerance", type=float, default=5e-5)
    fig.add_argument("--min-search-top", type=float, default=1.84)
    fig.add_argument("--columns", type=float, nargs="+",
                     default=[-0.0830, -0.0805, -0.0790, -0.0769, -0.0745, -0.0727, -0.0700])
    fig.add_argument("--dpi", type=int, default=110)
    fig.add_argument("--out", required=True)
    ban = sub.add_parser("banana")
    ban.add_argument("--lx", type=float, required=True, help="log10 output rate of the section")
    ban.add_argument("--ly", type=float, required=True, help="log10 hidden rate seed on the arc")
    ban.add_argument("--transient", type=int, default=60000)
    ban.add_argument("--ly-scale", type=float, default=100.0)
    ban.add_argument("--loop-h", type=float, default=2e-3)
    ban.add_argument("--loop-steps", type=int, default=400)
    ban.add_argument("--p-scale", type=float, default=100.0)
    ban.add_argument("--fold-h", type=float, default=0.01)
    ban.add_argument("--fold-h-max", type=float, default=0.05)
    ban.add_argument("--fold-steps", type=int, default=300)
    ban.add_argument("--min-spread", type=float, default=2e-3)
    ban.add_argument("--box", type=float, nargs=4, default=[-0.10, -0.06, 1.72, 1.95],
                     help="lx_min lx_max ly_min ly_max")
    ban.add_argument("--out", required=True)
    ffn = sub.add_parser("fold-from-node")
    ffn.add_argument("--node", required=True)
    ffn.add_argument("--sigma", type=float, default=1.0)
    ffn.add_argument("--p-scale", type=float, default=100.0)
    ffn.add_argument("--h", type=float, default=0.02)
    ffn.add_argument("--h-max", type=float, default=0.1)
    ffn.add_argument("--steps", type=int, default=600)
    ffn.add_argument("--box", type=float, nargs=4, default=[-0.13, 0.05, 1.70, 2.00],
                     help="lx_min lx_max ly_min ly_max")
    ffn.add_argument("--out", required=True)
    comp = sub.add_parser("composite")
    comp.add_argument("--rotation", required=True)
    comp.add_argument("--summary", default=str(BOUNDARY / "combined-summary005/summary.json"))
    comp.add_argument("--folds", nargs="+", required=True, help="fold-from-node.json files")
    comp.add_argument("--banana", required=True)
    comp.add_argument("--xlim", type=float, nargs=2, default=[-0.12, -0.064])
    comp.add_argument("--ylim", type=float, nargs=2, default=[1.725, 2.0])
    comp.add_argument("--dpi", type=int, default=110)
    comp.add_argument("--out", required=True)
    args = parser.parse_args()
    {"circle-check": circle_check, "figure": figure, "banana": banana,
     "fold-from-node": fold_from_node, "composite": composite}[args.command](args)


if __name__ == "__main__":
    main()
