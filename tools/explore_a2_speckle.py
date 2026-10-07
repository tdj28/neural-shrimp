"""Exploratory A2 diagnostic: why do local maps speckle inside verified windows?

Inside a window's ray-continued stability region, cells of the local map that
do not show the window's period (from the shared initial parameters) are
rerun in two ways:

  long    from the same initial parameters with 4x the map's updates, which
          resolves long chaotic transients;
  orbit   from the window's certified orbit point, which shows whether the
          window's orbit (continued) still attracts there, so a different map
          label means a coexisting attractor selected by the initial point.

The stability region of a cell is judged from the 64 ray boundaries,
interpolated in angle in the normalized (center, scale) frame.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import GridMap, iterate_and_screen  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = ROOT / "../neural-shrimp-data/explore/a2-survey"
PLAN = ROOT / "experiments/trainability/EXP-028-plan.json"


def inside_mask(window, lh, lo):
    center, scale = np.array(window["center"]), np.array(window["scale"])
    angles, radii = [], []
    for ray in window["rays"]:
        d = np.array(ray["direction"])
        angles.append(np.arctan2(d[1], d[0]))
        radii.append(ray["s_boundary"] if ray["s_boundary"] is not None else 1.0)
    order = np.argsort(angles)
    angles, radii = np.array(angles)[order], np.array(radii)[order]
    h, o = np.meshgrid((lh - center[0]) / scale[0], (lo - center[1]) / scale[1], indexing="ij")
    r, a = np.hypot(h, o), np.arctan2(o, h)
    limit = np.interp(a, angles, radii, period=2 * np.pi)
    return r < 0.9 * limit


def one(task):
    key, window, per_window, seed = task
    it = json.loads(PLAN.read_text())["iteration"]
    options = {k: it[k] for k in ("tail_steps", "ftle_steps", "ftle_align", "max_period", "min_repeats",
                                  "atol", "rtol", "chaos_threshold", "drift_threshold",
                                  "divergence_threshold", "tangent_seed")}
    options["sd_horizons"] = ()
    saved = np.load(DATA / "maps" / key.replace(":", "-c") / "map.npz")
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    status, period = saved["status"], saved["period"]
    p = window["period"]
    inside = inside_mask(window, lh, lo)
    hit = inside & (status == 2) & (period == p)
    miss = np.argwhere(inside & ~((status == 2) & (period == p)))
    out = {"key": key, "period": p, "inside_cells": int(inside.sum()), "inside_period_p": int(hit.sum()),
           "inside_other": int(len(miss))}
    if len(miss) == 0:
        return out
    rng = np.random.default_rng(seed)
    pick = miss[rng.choice(len(miss), size=min(per_window, len(miss)), replace=False)]
    system, sd = key.split("-")[0], int(key.split("-")[1].split(":")[0])
    net, x, y, theta0 = SYSTEMS[system](sd)
    grid = GridMap(net, x, y, 10.0 ** lh[pick[:, 0]], 10.0 ** lo[pick[:, 1]])

    def step_subset(states, idx):
        grad, loss = grid.net.gradient(states, grid.x, grid.y, return_loss=True)
        return states - grid.rates[idx] * grad, loss

    def jvp_subset(states, idx, vectors):
        return vectors - grid.rates[idx] * grid.net.hvp(states, grid.x, grid.y, vectors)

    certs = json.loads((ROOT / "../neural-shrimp-data/raw/EXP-028/attempt-002/stage3/certificates.json").read_text())
    map_id, comp = key.split(":")
    orbit0 = np.array(next(r for r in certs if r["map_id"] == map_id and r["component"] == int(comp))
                      ["certificate"]["orbit"][0])
    starts = {"long": (np.repeat(theta0[None], len(pick), 0), 4 * it["steps"]),
              "orbit": (np.repeat(orbit0[None], len(pick), 0), it["steps"])}
    for name, (initial, steps) in starts.items():
        res = iterate_and_screen(step_subset, jvp_subset, initial, steps, **options)
        is_p = (res["status"] == 2) & (res["period"] == p)
        labels, counts = np.unique([f"{s}/{q}" for s, q in zip(res["status"], res["period"])], return_counts=True)
        out[name] = {"sampled": int(len(pick)), "period_p": int(is_p.sum()),
                     "labels": dict(zip(labels.tolist(), counts.tolist()))}
    map_labels = [f"{status[r, c]}/{period[r, c]}" for r, c in pick]
    labels, counts = np.unique(map_labels, return_counts=True)
    out["map_labels_of_sampled"] = dict(zip(labels.tolist(), counts.tolist()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-window", type=int, default=120)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    rays = json.loads((DATA / "rays.json").read_text())
    keys = args.only or list(rays)
    tasks = [(k, rays[k], args.per_window, 1000 + i) for i, k in enumerate(keys)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, tasks, chunksize=1))
    (DATA / "speckle.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r["key"], "p", r["period"], "inside", r["inside_cells"], "map p", r["inside_period_p"],
              "| long p", r.get("long", {}).get("period_p"), "/", r.get("long", {}).get("sampled"),
              "| orbit p", r.get("orbit", {}).get("period_p"), "/", r.get("orbit", {}).get("sampled"))


if __name__ == "__main__":
    main()
