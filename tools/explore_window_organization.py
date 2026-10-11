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
    args = parser.parse_args()
    {"circle-check": circle_check, "figure": figure}[args.command](args)


if __name__ == "__main__":
    main()
