"""Exploratory A2: ray continuation around confirmed EXP-028 windows.

For each requested (map_id, component), take the certified orbit and rates
from EXP-028 stage 3, follow the orbit along equally spaced rays in log-rate
coordinates scaled by the component's extent, and type each boundary
(fold / flip / Neimark-Sacker / unresolved). Writes JSON and a figure that
overlays typed boundary points on the source map. Exploratory under the
adaptive research protocol.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.continuation import continue_ray, ray_directions  # noqa: E402
from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

RAW = ROOT / "../neural-shrimp-data/raw/EXP-028/attempt-002"


def map_path(map_id):
    stage = "stage2" if map_id.endswith("-zoom") else "stage1"
    return RAW / stage / f"{map_id}.npz"


def component_geometry(saved, record):
    status, period = saved["status"], saved["period"]
    rows, cols = status.shape
    from scipy import ndimage
    mask = (status == 2) & (period == record["period"])
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    r0, c0 = divmod(record["first_cell"], cols)
    members = np.argwhere(labels == labels[r0, c0])
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    dh, do = lh[1] - lh[0], lo[1] - lo[0]
    extent_h = (members[:, 0].max() - members[:, 0].min() + 1) * dh
    extent_o = (members[:, 1].max() - members[:, 1].min() + 1) * do
    return np.array([max(3 * extent_h, 4 * dh), max(3 * extent_o, 4 * do)]), members


def one_ray(args):
    system, seed, center, scale, direction, orbit0, period = args
    net, x, y, _ = SYSTEMS[system](seed)

    def family(params):
        grid = GridMap(net, x, y, [10.0 ** params[0]], [10.0 ** params[1]])
        return grid.cell_map(0)

    result = continue_ray(family, np.asarray(orbit0), period, center, direction * scale, 1.0, 1.0 / 64,
                          bisection_tolerance=1e-9)
    result.pop("last_attracting_state", None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", required=True, help="map_id:component")
    parser.add_argument("--rays", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    certificates = json.loads((RAW / "stage3" / "certificates.json").read_text())
    report = []
    for target in args.targets:
        map_id, component = target.rsplit(":", 1)
        record = next(r for r in certificates if r["map_id"] == map_id and r["component"] == int(component))
        system, seed = map_id.split("-")[0], int(map_id.split("-")[1])
        saved = np.load(map_path(map_id))
        scale, members = component_geometry(saved, record)
        center = np.log10([record["lr_hidden"], record["lr_output"]])
        orbit0 = record["certificate"]["orbit"][0]
        tasks = [(system, seed, center, scale, d, orbit0, record["period"])
                 for d in ray_directions(args.rays)]
        with ProcessPoolExecutor(args.workers) as pool:
            rays = list(pool.map(one_ray, tasks))
        types = {}
        for ray in rays:
            types[ray["type"]] = types.get(ray["type"], 0) + 1
        report.append({"target": target, "period": record["period"], "center_log10": center.tolist(),
                       "scale_log10": scale.tolist(), "types": types,
                       "rays": [{"direction": d.tolist(), **ray} for d, ray in zip(ray_directions(args.rays), rays)]})
        print(target, "period", record["period"], types, flush=True)
    (out / "rays.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
