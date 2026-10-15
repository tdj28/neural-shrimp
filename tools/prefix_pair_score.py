"""Score the stage-1 prompt-blocked comparisons frozen in docs/prefix-scale-72b.md (run PS72),
docs/prefix-scale-olmo.md (run PSO) and docs/prefix-scale-72b-base.md (run PB72), written before their data.

For a pair of models and the prefixes 2, 3, 4, 5: C1 = chaotic temperatures of stage 1 (48 over log10 T -0.4..0.9;
chaotic = "hidden chaos" or "visible chaos" of tools/hidden_chaos.py, i.e. positive at both offsets and no period),
M = the largest min-offset exponent, and the chaotic temperatures. A prefix missing either model counts against the
prediction and is reported missing.
- PS72: C1(qwen72b) < C1(qwen14b) at >= 3 of the 4 prefixes.
- PSO:  C1(olmo2-32b) > C1(olmo2-7b) at >= 3 of the 4 prefixes.
- PB72: C1(qwen72b-base) >= C1(qwen72b) + 3 at >= 3 of the 4 prefixes.

Usage: python tools/prefix_pair_score.py ps72|pso|pb72 OUT.json [RUN_ROOT]   (default llm-soft-loop/pod64/prefix)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
PREFIXES = (2, 3, 4, 5)
TESTS = {"ps72": ("qwen72b", "qwen14b", "less", 0), "pso": ("olmo2-32b", "olmo2-7b", "more", 0),
         "pb72": ("qwen72b-base", "qwen72b", "more", 3)}
CHAOTIC = ("hidden chaos", "visible chaos")


def stage1(cells):
    import hidden_chaos as hc

    lab = [hc.classify(c) for c in cells]
    chaotic = [c["log10_temperature"] for c, k in zip(cells, lab) if k in CHAOTIC]
    return {"C1": len(chaotic), "M": max(min(c["lyapunov"]) for c in cells), "cells": len(cells),
            "chaotic_log10T": [round(float(t), 4) for t in chaotic]}


def score(table, model, ref, direction, margin=0):
    """table[model][idx] = stage1 result or None; returns per-prefix outcomes and the verdict. With a margin,
    "more" means C1(model) >= C1(ref) + margin."""
    wins, missing, per = 0, [], {}
    for i in PREFIXES:
        a, b = table[model].get(i), table[ref].get(i)
        if a is None or b is None:
            missing.append(i)
            per[i] = None
            continue
        if direction == "less":
            ok = a["C1"] < b["C1"]
        else:
            ok = a["C1"] >= b["C1"] + margin if margin else a["C1"] > b["C1"]
        per[i] = bool(ok)
        wins += ok
    return {"successes": wins, "missing": missing, "per_prefix": per, "holds": wins >= 3}


def main():
    import explore_llm_attractors as ela

    test, out = sys.argv[1], Path(sys.argv[2])
    root = sys.argv[3] if len(sys.argv) > 3 else "llm-soft-loop/pod64/prefix"
    model, ref, direction, margin = TESTS[test]
    base = ela.DATA / "explore" / root
    table = {}
    for m in (model, ref):
        table[m] = {}
        for i in PREFIXES:
            p = base / f"{m}-p{i}-coarse" / "hidden-chaos.json"
            if p.exists():
                cells = json.loads(p.read_text())["cells"]
                if len(cells) != 48:
                    raise SystemExit(f"{p}: {len(cells)} cells, expected 48")
                table[m][i] = stage1(cells)
            else:
                table[m][i] = None
    res = score(table, model, ref, direction, margin)
    out.write_text(json.dumps({"test": test, "model": model, "reference": ref, "direction": direction, "margin": margin,
                               "score": res,
                               "table": {m: {str(i): table[m][i] for i in PREFIXES} for m in table}}, indent=1))
    for i in PREFIXES:
        f = lambda e: "missing" if e is None else f"C1 {e['C1']:2d}  M {e['M']:+.3f}"
        print(f"prefix {i}: {model} {f(table[model][i])}   {ref} {f(table[ref][i])}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
