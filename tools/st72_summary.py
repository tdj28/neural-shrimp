"""Score run ST72 (growing-context soft thinking twins on Qwen2.5-72B-Instruct) by the rule of its run card
(docs/research-log.md, 14 Oct 2026 21:08 UTC), written before the data: a cell grows when at least two of its three
twins reach a distance of 1e-3 (from 1e-10) within the run; rates and final distances are reported, and a twin that
passed 1e-3 but ended below it is flagged as a transient burst. ST1 (Qwen2.5-7B and -0.5B, same grid and settings) is
scored by the same rule for comparison.

Usage: python tools/st72_summary.py OUT.json [RUN_ROOT]   (default llm-soft-loop/pod64/big)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DMAX = 1e-3
RUNS = {"qwen72b-softthink-p0": "72B, paper's prompt, unsteered", "qwen72b-softthink-steer": "72B, paper's prompt, QFa cells",
        "qwen72b-softthink-p3": "72B, prompt 3, unsteered", "qwen7b-softthink": "7B (ST1)", "qwen0.5b-softthink": "0.5B (ST1)"}


def score_cell(cell, need=2):
    """Growth verdict of one cell: grows when at least `need` twins passed DMAX; bursts are twins that passed DMAX but
    ended below it."""
    first = cell["first_above_dmax"]
    passed = [f is not None for f in first]
    burst = [p and d < DMAX for p, d in zip(passed, cell["final_distance"])]
    return {"log10_temperature": cell["log10_temperature"], "alpha": cell.get("alpha", 0.0), "grows": sum(passed) >= need,
            "twins_passed": int(sum(passed)), "bursts": int(sum(burst)), "first_above_dmax": first,
            "growth_rates": cell["growth_rates"], "final_distance": cell["final_distance"],
            "entropy_mean": cell["entropy_mean"]}


def summarize(cells):
    rows = [score_cell(c) for c in cells]
    grow = [r for r in rows if r["grows"]]
    return {"cells": rows, "n": len(rows), "growing": len(grow),
            "growing_log10_temperatures": [round(r["log10_temperature"], 4) for r in grow],
            "max_rate": max((max(r["growth_rates"]) for r in grow), default=None)}


def main():
    import explore_llm_attractors as ela

    out = Path(sys.argv[1])
    root = sys.argv[2] if len(sys.argv) > 2 else "llm-soft-loop/pod64/big"
    res = {}
    for name, label in RUNS.items():
        p = ela.DATA / "explore" / root / name / "soft-thinking-twins.json"
        if not p.exists():
            res[name] = None
            print(name, "missing")
            continue
        data = json.loads(p.read_text())
        res[name] = {"label": label, "complete": data.get("complete", True), **summarize(data["cells"])}
        print(name, f"{res[name]['growing']}/{res[name]['n']} growing", res[name]["growing_log10_temperatures"],
              "complete" if res[name]["complete"] else "INCOMPLETE")
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
