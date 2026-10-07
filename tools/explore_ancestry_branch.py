"""POST HOC: is a window's certified orbit on the same period-P branch as a tracked tongue?

Rebuilds the tongue orbit along a recorded centre-line track (connect.json),
starting from the top approach level's certified orbit (levels.json), then
continues the window's certified orbit along straight segments towards tracked
centres (nearest first) and tests whether the two orbits coincide up to cyclic
shift. This is branch connectivity through a region where the orbit may be
unstable; it is not the frozen two-fold-edge ancestry criterion.
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

from neural_shrimp.ancestry import continue_segment, orbit_jump, same_cycle  # noqa: E402
from neural_shrimp.orbits import certify_orbit  # noqa: E402
import explore_ancestry as ea  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels-name", required=True)
    parser.add_argument("--connect-name", required=True)
    parser.add_argument("--map-id", default="S2-28011")
    parser.add_argument("--component", type=int, default=139)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--targets", type=int, default=4, help="nearest tracked centres to try")
    parser.add_argument("--newton-tolerance", type=float, default=1e-8)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    out = DATA / "explore/ancestry" / args.name
    options = {"newton_tolerance": args.newton_tolerance}
    net, x, y, theta0, family = ea.s2_family("S2", 28011)
    levels = json.loads((DATA / "explore/ancestry" / args.levels_name / "levels.json").read_text())
    track = json.loads((DATA / "explore/ancestry" / args.connect_name / "connect.json").read_text())["track"]
    top = levels["levels"][0]
    params, orbit = np.array(top["params"]), np.array(top["orbit"])
    tongue = []
    for entry in track:
        if "center" not in entry:
            break
        target = np.array(entry["center"])
        seg = continue_segment(family, orbit[0], args.period, params, target, h0=0.05, cert_options=options)
        if not seg["ok"]:
            tongue.append({"step": entry["step"], "params": target.tolist(), "ok": False, "reason": seg["reason"]})
            break
        params, orbit = target, np.array(seg["orbit"])
        tongue.append({"step": entry["step"], "params": target.tolist(), "ok": True, "orbit": orbit.tolist(),
                       "stability": seg["certificate"]["stability"]})
    print("tongue orbit rebuilt along", sum(t["ok"] for t in tongue), "of", len(track), "tracked centres", flush=True)
    certs = json.loads((DATA / "raw/EXP-028/attempt-002/stage3/certificates.json").read_text())
    rec = next(r for r in certs if r["map_id"] == args.map_id and r["component"] == args.component)
    window = np.log10([rec["lr_hidden"], rec["lr_output"]])
    window_orbit = np.array(rec["certificate"]["orbit"])
    good = [t for t in tongue if t["ok"]]
    good.sort(key=lambda t: float(np.linalg.norm(np.array(t["params"]) - window)))
    attempts = []
    outcome = "not_connected"
    for t in good[:args.targets]:
        seg = continue_segment(family, window_orbit[0], args.period, window, np.array(t["params"]), h0=0.02,
                               cert_options=options)
        entry = {"tongue_step": t["step"], "tongue_params": t["params"],
                 "distance": float(np.linalg.norm(np.array(t["params"]) - window)),
                 "reached": bool(seg["ok"]), "t_reached": seg["t_reached"], "reason": seg["reason"]}
        if seg["ok"]:
            a, b = np.array(seg["orbit"]), np.array(t["orbit"])
            entry["same_cycle"] = bool(same_cycle(a, b, 1e-6))
            entry["orbit_distance"] = orbit_jump(a, b)
            if entry["same_cycle"]:
                outcome = "same_branch"
        attempts.append(entry)
        print(json.dumps(entry, default=float), flush=True)
        if outcome == "same_branch":
            break
    ea.dump(out / "branch.json", {"outcome": outcome, "window": window.tolist(), "attempts": attempts,
                                  "tongue": tongue})
    print("POST HOC branch connectivity:", outcome)


if __name__ == "__main__":
    main()
