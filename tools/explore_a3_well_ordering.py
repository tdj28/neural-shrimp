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

Refinements (``--refined``), each scored against a multiplicity-corrected
null (the sum of per-test null rates, capped at one per orbit):
  tour     cyclic order from the shortest closed tour (nearest neighbour plus
           2-opt) through the points in the space of their top six principal
           components, for circles that are curved out of any single plane;
  doubled  for even periods divisible by 2^(k+1), the points of F^(2^(k+1))
           (q = p / 2^(k+1) points, k = 1, 2, 3): an orbit born by period
           doubling inside an m/q tongue is well-ordered at its base level.
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


def tour_order(points):
    """Positions of the points along the shortest closed tour (greedy start, 2-opt)."""
    centered = points - points.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    z = centered @ vt[:6].T
    d = np.linalg.norm(z[:, None] - z[None], axis=2)
    q = len(z)
    tour, left = [0], set(range(1, q))
    while left:
        nxt = min(left, key=lambda j: d[tour[-1], j])
        tour.append(nxt)
        left.remove(nxt)
    improved = True
    while improved:
        improved = False
        for i in range(1, q - 1):
            for j in range(i + 1, q):
                a, b, c, e = tour[i - 1], tour[i], tour[j], tour[(j + 1) % q]
                if d[a, c] + d[b, e] < d[a, b] + d[c, e] - 1e-15:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    position = np.empty(q)
    position[tour] = np.arange(q)
    return 2 * np.pi * position / q


def refined_tests(orbit, p, null):
    """All (level, method, q, rotation) tests; returns passes and the summed null rate."""
    tests, levels = [], [0]
    if p % 2 == 0:
        levels += [k for k in (1, 2, 3) if p % (2 ** (k + 1)) == 0 and p // 2 ** (k + 1) >= 3]
    for k in levels:
        stride = 2 ** (k + 1) if p % 2 == 0 else 1
        points = orbit[0::stride]
        q = len(points)
        for method in ("plane", "tour"):
            angles = planar_angles(points)[0] if method == "plane" else tour_order(points)
            m = shift(angles)
            tests.append({"level": k, "method": method, "q": q,
                          "rotation": None if m is None else f"{min(m, q - m)}/{q}", "null": null(q)})
    return tests


def main():
    refined = "--refined" in sys.argv
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
        row = {"map_id": record["map_id"], "component": record["component"], "kind": record["kind"],
               "confirmed": (record["map_id"], record["component"]) in confirmed, "period": p,
               "map": "F^2" if p % 2 == 0 else "F", "points": q, "well_ordered": m is not None,
               "rotation": None if m is None else f"{min(m, q - m)}/{q}", "planar_share": share,
               "null_pass_rate": null_rate[q]}
        if refined:
            def null(n):
                if n not in null_rate:
                    null_rate[n] = float(np.mean([shift(rng.uniform(-np.pi, np.pi, n)) is not None
                                                  for _ in range(10000)]))
                return null_rate[n]
            tests = refined_tests(orbit, p, null)
            passed = [t for t in tests if t["rotation"] is not None and t["q"] >= 5]
            row["refined"] = {"tests": tests, "passed_q5": bool(passed),
                              "base": None if not passed else min(passed, key=lambda t: (t["level"], t["method"])),
                              "null_q5": min(1.0, sum(t["null"] for t in tests if t["q"] >= 5))}
        rows.append(row)
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
    if refined:
        for label, sub in (("confirmed windows", [r for r in rows if r["confirmed"]]),
                           ("all certified", rows)):
            testable = [r for r in sub if any(t["q"] >= 5 for t in r["refined"]["tests"])]
            observed = sum(r["refined"]["passed_q5"] for r in testable)
            expected = sum(r["refined"]["null_q5"] for r in testable)
            print(f"refined, {label}: {observed}/{len(testable)} with a test on >= 5 points passing; "
                  f"multiplicity-corrected null expectation {expected:.2f}")
        for r in sorted((r for r in rows if r["confirmed"]), key=lambda r: (r["map_id"], r["component"])):
            b = r["refined"]["base"]
            print(f"  {r['map_id']}:{r['component']} p={r['period']} " + (
                "no test passes" if b is None else f"level {b['level']} {b['method']} q={b['q']} rho={b['rotation']}"))
        return
    for r in sorted((r for r in rows if r["confirmed"]), key=lambda r: (r["map_id"], r["component"])):
        print(f"  {r['map_id']}:{r['component']} p={r['period']} {r['map']} q={r['points']} "
              f"{'rho=' + r['rotation'] if r['well_ordered'] else 'not well-ordered'} share={r['planar_share']:.3f}")


if __name__ == "__main__":
    main()
