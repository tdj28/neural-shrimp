"""Exploratory A3: trace the Neimark-Sacker curve of the period-2 orbit.

For each fixed output rate, start inside the period-2 region at a given hidden
rate, continue the period-2 orbit toward larger hidden rates, and record where
it loses stability, the boundary type and the angle of the crossing multiplier
(theta / 2 pi, the predicted rotation number of the newborn invariant circle).
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.continuation import continue_ray  # noqa: E402
from neural_shrimp.gdmaps import GridMap, run_grid  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402


def one(task):
    system, seed, start, lo, s_max = task
    net, x, y, theta0 = SYSTEMS[system](seed)
    grid = GridMap(net, x, y, [10.0 ** start], [10.0 ** lo])
    res = run_grid(grid, theta0, 40000, tail_steps=384, ftle_steps=1000, ftle_align=200, sd_horizons=())
    if int(res["period"][0]) != 2:
        return {"log10_output": lo, "start_period": int(res["period"][0]), "type": "start_not_period_2"}

    def family(p):
        return GridMap(net, x, y, [10.0 ** p[0]], [10.0 ** p[1]]).cell_map(0)

    r = continue_ray(family, res["final_state"][0], 2, (start, lo), (1.0, 0.0), s_max, 0.002,
                     bisection_tolerance=1e-10)
    mu = complex(*r["dominant"]) if r["dominant"] else None
    return {"log10_output": lo, "type": r["type"],
            "log10_hidden_boundary": None if r["s_boundary"] is None else start + r["s_boundary"],
            "multiplier": None if mu is None else [mu.real, mu.imag],
            "theta_over_2pi": None if mu is None else abs(np.angle(mu)) / (2 * np.pi)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start", type=float, required=True, help="log10 hidden start inside period 2")
    parser.add_argument("--output", type=float, nargs=2, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--s-max", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tasks = [(args.system, args.seed, args.start, float(lo), args.s_max)
             for lo in np.linspace(args.output[0], args.output[1], args.count)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, tasks))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ns-curve.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
