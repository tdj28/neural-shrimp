"""Exploratory A3: rotation numbers of certified EXP-028 orbits (phase-locking test).

For each certified periodic orbit, use the two-step map's points (every other
orbit point) for even periods and the map's own points for odd periods,
project them on their first two principal axes, and count net turns around
the centroid over one cycle. A locked orbit of an invariant circle gives an
integer number of turns m over q points: rotation number m/q.
"""

import json
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "../neural-shrimp-data/raw/EXP-028/attempt-002"
OUT = ROOT / "../neural-shrimp-data/explore/a3-rotation"


def turns(points):
    centered = points - points.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ vt[:2].T
    angles = np.arctan2(planar[:, 1], planar[:, 0])
    steps = np.diff(np.r_[angles, angles[0]])
    steps = (steps + np.pi) % (2 * np.pi) - np.pi
    share = float((singular[:2] ** 2).sum() / max((singular ** 2).sum(), 1e-300))
    return float(steps.sum() / (2 * np.pi)), share


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in json.loads((RAW / "stage3" / "certificates.json").read_text()):
        cert = record["certificate"] or {}
        p = record["period"]
        if not cert.get("verified") or p < 3:
            continue
        orbit = np.array(cert["orbit"])
        points = orbit[0::2] if p % 2 == 0 else orbit
        q = len(points)
        m, share = turns(points)
        m_int = int(round(abs(m)))
        rows.append({"map_id": record["map_id"], "component": record["component"], "kind": record["kind"],
                     "period": p, "map": "F^2" if p % 2 == 0 else "F", "points": q,
                     "net_turns": m, "integer": abs(abs(m) - m_int) < 1e-6, "m": m_int,
                     "reduced": f"{m_int // gcd(m_int, q)}/{q // gcd(m_int, q)}" if m_int else "0",
                     "primitive_locking": m_int > 0 and gcd(m_int, q) == 1, "planar_share": share})
    (OUT / "rotation.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    print("orbits (period >= 3):", n)
    print("integer net turns:", sum(r["integer"] for r in rows), "| primitive locking m/q with gcd 1:",
          sum(r["primitive_locking"] for r in rows))
    for kind in ("embedded", "other"):
        sub = [r for r in rows if r["kind"] == kind]
        print(kind, len(sub), "primitive:", sum(r["primitive_locking"] for r in sub))
    for r in sorted(rows, key=lambda r: (r["map_id"], r["period"])):
        print(f"{r['map_id']:15s} c{r['component']:<5d} {r['kind']:8s} p={r['period']:<3d} {r['map']:3s} "
              f"turns={r['net_turns']:+.3f} rho={r['reduced']:>6s} {'LOCK' if r['primitive_locking'] else ''} "
              f"planar={r['planar_share']:.2f}")


if __name__ == "__main__":
    main()
