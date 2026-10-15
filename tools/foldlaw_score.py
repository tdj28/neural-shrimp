"""Score runs FL15 and FL7 by their run card (docs/research-log.md, 15 Oct 2026 02:24 UTC): do the windows beyond the
fold of a short cycle follow d_N = C / (N + delta)^2, and what effective passage fraction f = sqrt(C ab) / pi does C
give with the fold's normal form?

Reading of the card, fixed before the line results (the card leaves these choices open):
- A window is a maximal run of consecutive periodic cells of one period p on the far side of the fold (where the base
  cycle of period q no longer exists). A run whose neighbouring run has period p / 2 is the period doubling of that
  window and is not a window. A run of one cell is unresolved and is not a window. Among several runs of one period
  the one nearest the fold counts.
- The windows of the fan are those whose periods form the progression p = q (N + k); laps are read as x = p / q, and
  the offset delta absorbs k, so the law is d (x + delta)^2 = C. Windows are taken in order from the fold outward
  while consecutive in the progression.
- The distance d is that of the window's edge nearest the fold (the edge resolved at the grid spacing), from the fold
  point of the normal form. Window centres are reported as a check.
- d (x + delta)^2 is constant within 10 % when (max - min) / mean <= 0.10 for the delta that minimizes this spread.
  The law holds when the windows closest to the fold meet this over at least four consecutive windows.
- Refinement, added before the line results (15 Oct 2026): with a free offset, other powers pass over a short range
  of laps (d = 0.004 / x passes over x = 3-8 with delta = 4.9). Since the laps N = x - k with k >= 0 and the lap
  offset of the law is below one lap, x + delta can exceed the period's laps by at most one: delta <= 1. The free fit
  (the card's criterion) and the bounded fit are both reported; a free fit that needs delta > 1 is flagged.

Usage: python tools/foldlaw_score.py LINE_JSON NORMAL_FORM_JSON BASE_PERIOD [--out JSON]
"""
import argparse
import json
from pathlib import Path

import numpy as np

SPREAD = 0.10


def runs(cells, coordinate="log10_temperature"):
    """Maximal runs of equal period along the line: (period, first position, last position, cell count)."""
    pts = sorted(((c[coordinate], c["period"] if c["class"] == "periodic" else 0) for c in cells))
    out, i = [], 0
    while i < len(pts):
        k = i
        while k + 1 < len(pts) and pts[k + 1][1] == pts[i][1]:
            k += 1
        out.append((pts[i][1], pts[i][0], pts[k][0], k - i + 1))
        i = k + 1
    return out


def windows(cells, tc, q, coordinate="log10_temperature", min_cells=2):
    """Windows on the far side of the fold at tc: dicts with period, laps x = p / q, the edge nearest the fold, its
    distance, the centre's distance and the cell count, ordered from the fold outward."""
    rs = runs(cells, coordinate)
    base = [r for r in rs if r[0] > 0 and q % r[0] == 0]       # the base cycle (or a divisor of its period)
    if not base:
        raise ValueError(f"no run of the base period {q} on the line")
    nearest = min(base, key=lambda r: min(abs(r[1] - tc), abs(r[2] - tc)))
    side = 1.0 if (nearest[1] + nearest[2]) / 2 < tc else -1.0   # windows lie beyond the fold, away from the base
    best = {}
    for i, (p, lo, hi, n) in enumerate(rs):
        if p <= q or n < min_cells or p % 2 == 0 and any(0 <= j < len(rs) and rs[j][0] == p // 2 for j in (i - 1, i + 1)):
            continue
        if side * ((lo + hi) / 2 - tc) <= 0:
            continue
        edge = lo if abs(lo - tc) < abs(hi - tc) else hi
        w = {"period": int(p), "laps": p / q, "edge": float(edge), "d": abs(float(edge) - tc),
             "d_centre": abs((lo + hi) / 2 - tc), "cells": int(n)}
        if p not in best or w["d"] < best[p]["d"]:
            best[p] = w
    return sorted(best.values(), key=lambda w: w["d"]), side


def fan(ws, q):
    """The windows from the fold outward while their periods step by q (consecutive laps): from the window nearest the
    fold, step to period p - q as long as that window exists and lies farther out. Windows outside the progression
    (other periods between the fan windows) are skipped. (Correction of 15 Oct 2026, after the first FL scoring: the
    first version stopped at the first window outside the progression, against the rule above.)"""
    if not ws:
        return []
    by = {w["period"]: w for w in ws}
    out = [ws[0]]
    while out[-1]["period"] - q in by and by[out[-1]["period"] - q]["d"] > out[-1]["d"]:
        out.append(by[out[-1]["period"] - q])
    return out


def fit(ws, key="d", delta_max=10.0):
    """The offset delta <= delta_max minimizing the relative spread of d (x + delta)^2, with C its mean there."""
    x = np.array([w["laps"] for w in ws], dtype=float)
    d = np.array([w[key] for w in ws], dtype=float)
    lo = -x.min() + 1e-3
    best = None
    for delta in np.concatenate([np.linspace(lo, delta_max, 20001), np.linspace(lo, lo + 1.0, 2001)]):
        v = d * (x + delta) ** 2
        s = float((v.max() - v.min()) / v.mean())
        if best is None or s < best["spread"]:
            best = {"C": float(v.mean()), "delta": float(delta), "spread": s}
    return best


def score(cells, nf, q, coordinate="log10_temperature"):
    tc = nf["source"]["log10_temperature"] if coordinate == "log10_temperature" else nf["source"]["alpha"]
    ws, side = windows(cells, tc, q, coordinate)
    chain = fan(ws, q)
    out = {"fold": tc, "side": side, "base_period": q, "windows": ws, "fan": [w["period"] for w in chain],
           "ab": abs(nf["a"] * nf["b"]), "a": nf["a"], "b": nf["b"]}
    blocks = []
    for m in range(4, len(chain) + 1):
        f = fit(chain[:m])
        f["windows"] = m
        f["passage_fraction"] = float(np.sqrt(f["C"] * out["ab"]) / np.pi)
        f["offset_physical"] = bool(f["delta"] <= 1.0)
        fb = fit(chain[:m], delta_max=1.0)
        f["bounded"] = {"C": fb["C"], "delta": fb["delta"], "spread": fb["spread"],
                        "passage_fraction": float(np.sqrt(fb["C"] * out["ab"]) / np.pi)}
        fc = fit(chain[:m], "d_centre")
        f["centre"] = {"C": fc["C"], "delta": fc["delta"], "spread": fc["spread"],
                       "passage_fraction": float(np.sqrt(fc["C"] * out["ab"]) / np.pi)}
        blocks.append(f)
    out["fits"] = blocks
    held = [b for b in blocks if b["spread"] <= SPREAD]
    out["law_holds"] = bool(blocks and blocks[0]["spread"] <= SPREAD)
    out["law_holds_bounded"] = bool(blocks and blocks[0]["bounded"]["spread"] <= SPREAD)
    out["windows_meeting_law"] = max((b["windows"] for b in held if all(
        c["spread"] <= SPREAD for c in blocks if c["windows"] <= b["windows"])), default=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("line")
    ap.add_argument("normal_form")
    ap.add_argument("base_period", type=int)
    ap.add_argument("--coordinate", default="log10_temperature", choices=["log10_temperature", "alpha"])
    ap.add_argument("--out")
    args = ap.parse_args()
    cells = json.loads(Path(args.line).read_text())["cells"]
    nf = json.loads(Path(args.normal_form).read_text())
    res = score(cells, nf, args.base_period, args.coordinate)
    for w in res["windows"]:
        print("P%-4d laps %-6.2f edge %.6f  d %.4e  centre d %.4e  (%d cells)%s" % (
            w["period"], w["laps"], w["edge"], w["d"], w["d_centre"], w["cells"],
            "  fan" if w["period"] in res["fan"] else ""))
    for b in res["fits"]:
        print("closest %d: C %.5g  delta %.3f  spread %.1f %%  f %.3f%s | delta <= 1: C %.5g delta %.3f spread %.1f %%"
              " f %.3f | centres: spread %.1f %%  f %.3f" % (
                  b["windows"], b["C"], b["delta"], 100 * b["spread"], b["passage_fraction"],
                  "" if b["offset_physical"] else " (delta > 1)", b["bounded"]["C"], b["bounded"]["delta"],
                  100 * b["bounded"]["spread"], b["bounded"]["passage_fraction"], 100 * b["centre"]["spread"],
                  b["centre"]["passage_fraction"]))
    print("law holds: %s (over %d windows), with delta <= 1: %s; ab %.4g" % (
        res["law_holds"], res["windows_meeting_law"], res["law_holds_bounded"], res["ab"]))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
