"""Summarize run QF (consult 8 follow-up, Qwen2.5-72B with the float64 hook fix) by the rules of its run card
(docs/research-log.md, 14 Oct 2026 16:05 UTC), written before the data:
- a twin exponent counts only where the one-step central-difference ratio of the fd probe at that h is within 10 % of
  its value at h = 1e-4 (resolved scale);
- chaotic: the tangent exponent and every counted twin exponent above 1e-3, and no period up to 64;
- unresolved infinitesimal growth: tangent above 1e-3 but some counted twin exponent at or below 1e-3;
- finite-size growth only: tangent at or below 1e-3 but every counted twin exponent above 1e-3;
- not chaotic: otherwise (tangent and counted twins at or below 1e-3, or a period).
Also reports the wander (attractor diameter) at period-1 cells, which the fix should bring to about 1e-15.

Usage: python tools/q72fix_summary.py OUT.json [RUN_ROOT]   (default llm-soft-loop/pod64/big)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
LEVEL = 1e-3
RUNS = ("qwen72b-fix-cells", "qwen72b-fix-p2", "qwen72b-fix-p3", "qwen72b-fix-p4", "qwen72b-fix-p5")


def resolved(cell, eps, tol=0.1):
    """True where the fd probe's central-difference ratio at h = eps is within tol of its value at h = 1e-4."""
    fd = cell.get("fd_probe")
    if fd is None:
        return False
    hs, ratio = fd["h"], fd["central_ratio"]
    ref = ratio[min(range(len(hs)), key=lambda k: abs(hs[k] - 1e-4))]
    k = min(range(len(hs)), key=lambda k: abs(hs[k] / eps - 1))
    return abs(hs[k] / eps - 1) < 1e-9 and ref > 0 and abs(ratio[k] / ref - 1) < tol


def verdict(cell):
    eps = cell["eps"]
    counted = [l for l, e in zip(cell["lyapunov"], eps) if resolved(cell, e)]
    tang = cell.get("lyapunov_tangent")
    if tang is None or not counted:
        return "invalid", counted
    if cell["period"] > 0:
        return "not chaotic", counted
    pos_t, pos_c = tang > LEVEL, all(l > LEVEL for l in counted)
    if pos_t and pos_c:
        return "chaotic", counted
    if pos_t:
        return "unresolved infinitesimal growth", counted
    if pos_c:
        return "finite-size growth only", counted
    return "not chaotic", counted


def summarize(cells, meta=None):
    rows, counts = [], {}
    for i, c in enumerate(cells):
        v, counted = verdict(c)
        counts[v] = counts.get(v, 0) + 1
        rows.append({"log10_temperature": c["log10_temperature"], "alpha": c["alpha"], "group": (meta or [None] * len(cells))[i],
                     "verdict": v, "tangent": c.get("lyapunov_tangent"), "twins": dict(zip(map(str, c["eps"]), c["lyapunov"])),
                     "twins_counted": len(counted), "period": c["period"], "entropy_mean": c["entropy_mean"],
                     "diameter_rel": c["diameter_rel"], "repeat_diff": (c.get("fd_probe") or {}).get("repeat_diff")})
    wander = [r["diameter_rel"] for r in rows if r["period"] == 1]
    return {"counts": counts, "period1_wander_max": max(wander) if wander else None, "cells": rows}


def main():
    import explore_llm_attractors as ela

    out = Path(sys.argv[1])
    root = sys.argv[2] if len(sys.argv) > 2 else "llm-soft-loop/pod64/big"
    meta = json.loads((ROOT / "docs" / "q72fix-cells.json").read_text())["meta"]
    res = {}
    for name in RUNS:
        p = ela.DATA / "explore" / root / name / "hidden-chaos.json"
        if not p.exists():
            res[name] = None
            continue
        cells = json.loads(p.read_text())["cells"]
        res[name] = summarize(cells, meta if name.endswith("cells") else None)
        print(name, json.dumps(res[name]["counts"]), "period-1 wander max", res[name]["period1_wander_max"])
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
