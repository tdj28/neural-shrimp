"""POST HOC: type where a window's own period-P branch ends along given directions.

Runs existence continuation from the window's certified orbit towards each
target point and reports whether existence ends at a fold (multiplier near +1)
or otherwise. Directions are towards tracked tongue points (branch.json).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neural_shrimp.ancestry import existence_ray  # noqa: E402
import explore_ancestry as ea  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--map-id", default="S2-28011")
    parser.add_argument("--component", type=int, default=139)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    net, x, y, theta0, family = ea.s2_family("S2", 28011)
    branch = json.loads((DATA / "explore/ancestry" / args.branch_name / "branch.json").read_text())
    certs = json.loads((DATA / "raw/EXP-028/attempt-002/stage3/certificates.json").read_text())
    rec = next(r for r in certs if r["map_id"] == args.map_id and r["component"] == args.component)
    window = np.log10([rec["lr_hidden"], rec["lr_output"]])
    orbit = np.array(rec["certificate"]["orbit"])
    rows = []
    for attempt in branch["attempts"]:
        target = np.array(attempt["tongue_params"])
        direction = (target - window) / np.linalg.norm(target - window)
        ray = existence_ray(family, orbit[0], args.period, window, direction, float(np.linalg.norm(target - window)),
                            0.002, bisection_tolerance=1e-9, cert_options={"newton_tolerance": 1e-8})
        row = {"tongue_step": attempt["tongue_step"], "direction": direction.tolist(), "type": ray["type"],
               "s_boundary": ray["s_boundary"], "distance_to_one": ray.get("distance_to_one"),
               "closest_to_one": ray.get("closest_to_one")}
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    ea.dump(DATA / "explore/ancestry" / args.name / "window-edges.json", {"window": window.tolist(), "rows": rows})


if __name__ == "__main__":
    main()
