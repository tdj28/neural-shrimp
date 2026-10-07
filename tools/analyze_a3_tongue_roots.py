"""Exploratory A3: do locked m/q regions emanate from their predicted roots?

For each rational m/q (q <= --max-den) whose birth rotation arg(mu)/2pi is
crossed by the traced Neimark-Sacker curve of the period-2 orbit, the root is
the interpolated curve point where the birth rotation equals m/q. Locked cells
are periodic cells of period 2q with rotation within 1/samples of m/q (from a
rotation map). The test statistic is the distance, in map cells, from the root
to the nearest locked cell beyond the curve. Null: the same distance from
uniformly random points along the traced curve (2,000 draws per rational), as
a one-sided rank (fraction of random points at least as close).
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--ns", required=True)
    parser.add_argument("--max-den", type=int, default=13)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    m = np.load(args.map)
    lh, lo = m["log10_hidden"], m["log10_output"]
    rho, status, period = m["rho"], m["status"], m["period"]
    samples = int(json.load(open(Path(args.map).with_name("metadata.json")))["args"]["samples"])
    tol = 1.0 / samples + 1e-12
    dh, do = lh[1] - lh[0], lo[1] - lo[0]
    rows = sorted((r for r in json.load(open(args.ns))["rows"] if r.get("type") == "neimark_sacker"),
                  key=lambda r: r["log10_output"])
    cx = np.array([r["log10_output"] for r in rows])
    cy = np.array([r["log10_hidden_boundary"] for r in rows])
    th = np.array([r["theta_over_2pi"] for r in rows])
    seg = np.hypot(np.diff(cx) / do, np.diff(cy) / dh)
    arc = np.r_[0, np.cumsum(seg)]
    rng = np.random.default_rng(args.seed)
    report = []
    for q in range(4, args.max_den + 1):
        for p_ in range(1, q):
            f = Fraction(p_, q)
            if f.denominator != q:
                continue
            target = float(f)
            locked = (status == 2) & (period == 2 * q) & np.isfinite(rho) & (np.abs(rho - target) <= tol)
            if not locked.any():
                continue
            ii, jj = np.nonzero(locked)
            cells = np.stack([lh[ii], lo[jj]], 1)
            for k in range(len(rows) - 1):
                a, b = th[k], th[k + 1]
                if not (min(a, b) < target <= max(a, b)):
                    continue
                t = (target - a) / (b - a)
                root = np.array([cy[k] + t * (cy[k + 1] - cy[k]), cx[k] + t * (cx[k + 1] - cx[k])])
                if not (lh[0] <= root[0] <= lh[-1] and lo[0] <= root[1] <= lo[-1]):
                    report.append({"rotation": f"{p_}/{q}", "period": 2 * q, "outside_map": True,
                                   "root_log10_hidden": float(root[0]), "root_log10_output": float(root[1])})
                    continue

                def distance(point):
                    return float(np.min(np.hypot((cells[:, 0] - point[0]) / dh, (cells[:, 1] - point[1]) / do)))

                d_root = distance(root)
                u = rng.uniform(0, arc[-1], args.draws)
                idx = np.clip(np.searchsorted(arc, u) - 1, 0, len(seg) - 1)
                w = (u - arc[idx]) / np.maximum(seg[idx], 1e-300)
                pts = np.stack([cy[idx] + w * (cy[idx + 1] - cy[idx]), cx[idx] + w * (cx[idx + 1] - cx[idx])], 1)
                null = np.array([distance(p) for p in pts])
                report.append({"rotation": f"{p_}/{q}", "period": 2 * q, "locked_cells": int(locked.sum()),
                               "root_log10_hidden": float(root[0]), "root_log10_output": float(root[1]),
                               "root_distance_cells": d_root, "null_median_cells": float(np.median(null)),
                               "null_fraction_at_least_as_close": float(np.mean(null <= d_root))})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tongue-roots.json").write_text(json.dumps(report, indent=1))
    for r in report:
        if r.get("outside_map"):
            print(f"{r['rotation']:>5} (period {r['period']:>2}): root outside the mapped range, skipped")
            continue
        print(f"{r['rotation']:>5} (period {r['period']:>2}): {r['locked_cells']:6d} locked cells; root distance "
              f"{r['root_distance_cells']:7.1f} cells vs random-point median {r['null_median_cells']:7.1f}; "
              f"null p = {r['null_fraction_at_least_as_close']:.3f}")


if __name__ == "__main__":
    main()
