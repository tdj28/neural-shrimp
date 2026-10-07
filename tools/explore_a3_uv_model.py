"""Exploratory A3: does the minimal two-parameter model show the quasi-periodic route?

Model: L(u, v) = (uv - 1)^2 + 0.05 (u^2 + v^2) (scalar factorization with weight
decay; the analytic control in tests/test_orbits.py), trained by gradient descent
with separate rates (a, b) for u and v. The state space is the (u, v) plane, so
invariant circles of the two-step map can be drawn directly.

Outputs under DATA/explore/a3-uv-model:
  map.npz        status, period, FTLE over a log-rate grid (EXP-028 screening)
  rays.json      period-2 continuation from inside the period-2 region along
                 the a-axis direction at several b, with boundary type and the
                 crossing multiplier (theta / 2 pi)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.continuation import continue_ray  # noqa: E402
from neural_shrimp.gdmaps import GridMap, TwoLayerNet, run_grid  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def model():
    net = TwoLayerNet(d=1, width=1, activation="identity", in_scale=1.0, out_scale=1.0,
                      bias=False, loss_scale=1.0, ridge=0.1)
    return net, np.array([[1.0]]), np.array([1.0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, nargs=2, default=(-0.6, 0.2), help="log10 range of the u rate")
    parser.add_argument("--b", type=float, nargs=2, default=(-0.6, 0.2), help="log10 range of the v rate")
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--rays", type=int, default=24)
    parser.add_argument("--out", default="a3-uv-model")
    args = parser.parse_args()
    net, x, y = model()
    s = np.sqrt(0.95)
    theta0 = np.array([s + 0.05, s - 0.03])
    la = np.linspace(*args.a, args.resolution)
    lb = np.linspace(*args.b, args.resolution)
    r, c = np.divmod(np.arange(args.resolution ** 2), args.resolution)
    # TwoLayerNet orders parameters as (input weight u, output weight v): hidden rate -> u, output rate -> v
    grid = GridMap(net, x, y, 10.0 ** la[r], 10.0 ** lb[c])
    res = run_grid(grid, theta0, 20000, tail_steps=384, ftle_steps=4000, ftle_align=1000, sd_horizons=())
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "map.npz", log10_a=la, log10_b=lb,
             **{k: v.reshape(args.resolution, args.resolution, -1) if k == "final_state"
                else v.reshape(args.resolution, args.resolution) for k, v in res.items()})
    status = res["status"].reshape(args.resolution, args.resolution)
    period = res["period"].reshape(args.resolution, args.resolution)
    counts = {f"{s_}/{p_}": int(n) for (s_, p_), n in
              zip(*np.unique(np.stack([status.ravel(), period.ravel()], 1), axis=0, return_counts=True))}
    print("status/period counts:", dict(sorted(counts.items(), key=lambda kv: -kv[1])[:20]))

    def family(p):
        return GridMap(net, x, y, [10.0 ** p[0]], [10.0 ** p[1]]).cell_map(0)

    rays = []
    for j in np.linspace(0, args.resolution - 1, args.rays).astype(int):
        rows2 = np.flatnonzero((status[:, j] == 2) & (period[:, j] == 2))
        if len(rows2) == 0:
            rays.append({"log10_b": float(lb[j]), "type": "no_period_2"})
            continue
        i = rows2[len(rows2) // 2]
        start = res["final_state"].reshape(args.resolution, args.resolution, -1)[i, j]
        ray = continue_ray(family, start, 2, (la[i], lb[j]), (1.0, 0.0), args.a[1] - la[i] + 0.3, 0.002,
                           bisection_tolerance=1e-11)
        mu = complex(*ray["dominant"]) if ray["dominant"] else None
        rays.append({"log10_b": float(lb[j]), "start_log10_a": float(la[i]), "type": ray["type"],
                     "log10_a_boundary": None if ray["s_boundary"] is None else float(la[i] + ray["s_boundary"]),
                     "multiplier": None if mu is None else [mu.real, mu.imag],
                     "theta_over_2pi": None if mu is None else float(abs(np.angle(mu)) / (2 * np.pi))})
        print(rays[-1])
    (out / "rays.json").write_text(json.dumps(rays, indent=1))


if __name__ == "__main__":
    main()
