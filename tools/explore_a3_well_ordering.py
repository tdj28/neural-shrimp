"""Exploratory A3: are certified window orbits Birkhoff (well-ordered) circle orbits?

For each certified EXP-028 periodic orbit of period p >= 3, take the points
of the two-step map (every other orbit point) when p is even (q = p / 2
points) or the map's own points when p is odd (q = p), project them on their
first two principal axes and sort them by angle about the centroid. The orbit
is well-ordered (a Birkhoff orbit of rotation number m / q) when every step
advances the angular rank by the same shift m modulo q, with gcd(m, q) = 1.
Integer net winding is automatic for any closed cycle and is not used.

Null: the same test applied to uniformly random angles (10,000 draws per
point count); its pass rate is compared with the observed one.
Writes DATA/explore/a3-well-ordering/well-ordering.json.
"""

import json
import os
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
RAW = DATA / "raw/EXP-028/attempt-002"
SUMMARY = ROOT / "docs/EXP-028-summary.json"
AUDIT = DATA / "audits/EXP-028/validation-001/audit.json"


def shift(angles):
    """Return m if the cyclic sequence is a rigid rotation of its angular order, else None."""
    q = len(angles)
    rank = np.empty(q, dtype=int)
    rank[np.argsort(angles)] = np.arange(q)
    steps = (rank[np.r_[1:q, 0]] - rank) % q
    m = int(steps[0])
    return m if np.all(steps == m) and m != 0 and gcd(m, q) == 1 else None


def planar_angles(points):
    centered = points - points.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ vt[:2].T
    share = float((singular[:2] ** 2).sum() / max((singular ** 2).sum(), 1e-300))
    return np.arctan2(planar[:, 1], planar[:, 0]), share


def main():
    summary = json.loads(SUMMARY.read_text())
    audit = json.loads(AUDIT.read_text())
    confirmed = ({(q["map_id"], q["component"]) for q in summary["gate"]["qualifying"]}
                 & {(q["map_id"], q["component"]) for q in audit["audit_gate"]["qualifying"]})
    rng = np.random.default_rng(20261007)
    null_rate = {}
    rows = []
    for record in json.loads((RAW / "stage3" / "certificates.json").read_text()):
        cert = record["certificate"] or {}
        p = record["period"]
        if not cert.get("verified") or p < 3:
            continue
        orbit = np.array(cert["orbit"])
        points = orbit[0::2] if p % 2 == 0 else orbit
        q = len(points)
        angles, share = planar_angles(points)
        m = shift(angles)
        if q not in null_rate:
            null_rate[q] = float(np.mean([shift(rng.uniform(-np.pi, np.pi, q)) is not None
                                          for _ in range(10000)]))
        rows.append({"map_id": record["map_id"], "component": record["component"], "kind": record["kind"],
                     "confirmed": (record["map_id"], record["component"]) in confirmed, "period": p,
                     "map": "F^2" if p % 2 == 0 else "F", "points": q, "well_ordered": m is not None,
                     "rotation": None if m is None else f"{min(m, q - m)}/{q}", "planar_share": share,
                     "null_pass_rate": null_rate[q]})
    out = DATA / "explore/a3-well-ordering"
    out.mkdir(parents=True, exist_ok=True)
    (out / "well-ordering.json").write_text(json.dumps(rows, indent=1))
    for label, sub in (("all certified, >= 5 points", [r for r in rows if r["points"] >= 5]),
                       ("confirmed windows", [r for r in rows if r["confirmed"]]),
                       ("confirmed, even period", [r for r in rows if r["confirmed"] and r["period"] % 2 == 0]),
                       ("confirmed, odd period", [r for r in rows if r["confirmed"] and r["period"] % 2 == 1])):
        if not sub:
            continue
        observed = sum(r["well_ordered"] for r in sub)
        expected = sum(r["null_pass_rate"] for r in sub)
        print(f"{label}: {observed}/{len(sub)} well-ordered; null expectation {expected:.2f}")
    for r in sorted((r for r in rows if r["confirmed"]), key=lambda r: (r["map_id"], r["component"])):
        print(f"  {r['map_id']}:{r['component']} p={r['period']} {r['map']} q={r['points']} "
              f"{'rho=' + r['rotation'] if r['well_ordered'] else 'not well-ordered'} share={r['planar_share']:.3f}")


if __name__ == "__main__":
    main()
