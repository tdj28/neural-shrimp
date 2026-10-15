"""Score the prompt-blocked size test (run PS) exactly as frozen in docs/prefix-scale-test.md, and draw its figure.

Per model (qwen7b, qwen14b) and prefix index (0-9): C1 = chaotic temperatures of stage 1 (of 48), C2 = of stage 2
(of 24), M = the largest min(exponent at 1e-5, exponent at 1e-7) over both stages. Chaotic = exponent above 1e-3 at
both offsets and no period up to 64 (tools/hidden_chaos.py classify: "hidden chaos" or "visible chaos").
Held-out prefixes are 2-9; 0 and 1 (development) are reported separately. A prefix missing either model's result is
reported missing and never imputed; a prediction that a missing prefix could still decide is "undetermined".

Usage: python tools/prefix_scale_score.py OUT.json OUT.png [RUN_ROOT]   (default llm-soft-loop/pod64/prefix)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
MODELS = ("qwen7b", "qwen14b")
HELD_OUT = tuple(range(2, 10))
DEVELOPMENT = (0, 1)
CHAOTIC = ("hidden chaos", "visible chaos")


def endpoints(coarse, fine):
    """C1, C2 and M from the cell lists of the two stages (None when a stage is missing)."""
    import hidden_chaos as hc

    if coarse is None or fine is None:
        return None
    c1 = sum(hc.classify(c) in CHAOTIC for c in coarse)
    c2 = sum(hc.classify(c) in CHAOTIC for c in fine)
    m = max(min(c["lyapunov"]) for c in coarse + fine)
    return {"C1": int(c1), "C2": int(c2), "M": float(m), "stage1_cells": len(coarse), "stage2_cells": len(fine),
            "stage2_log10T": [float(fine[0]["log10_temperature"]), float(fine[-1]["log10_temperature"])],
            "chaotic_log10T": [round(float(c["log10_temperature"]), 4) for c in coarse + fine
                               if hc.classify(c) in CHAOTIC]}


def score(table):
    """table[model][idx] = endpoints or None. Returns PS1, PS2 (holds / fails / undetermined) and PS3."""
    pairs = {i: (table["qwen7b"].get(i), table["qwen14b"].get(i)) for i in HELD_OUT}
    known = {i: p for i, p in pairs.items() if p[0] is not None and p[1] is not None}
    missing = len(HELD_OUT) - len(known)
    s1 = sum(b["C1"] < a["C1"] for a, b in known.values())
    v1 = [i for i, (a, b) in known.items() if b["C1"] > a["C1"] + 2]
    if v1 or s1 + missing < 6:
        ps1 = "fails"
    elif s1 >= 6 and missing == 0:
        ps1 = "holds"
    else:
        ps1 = "undetermined"
    s2 = sum(b["M"] < a["M"] for a, b in known.values())
    ps2 = "holds" if s2 >= 6 else ("fails" if s2 + missing < 6 else "undetermined")
    ps3 = {m: sum(1 for i in HELD_OUT if table[m].get(i) is not None and table[m][i]["C2"] > 0) for m in MODELS}
    return {"PS1": ps1, "PS1_successes": int(s1), "PS1_violations": v1, "PS2": ps2, "PS2_successes": int(s2),
            "PS3_prefixes_with_stage2_chaos": ps3, "held_out_missing": int(missing)}


def main():
    import explore_llm_attractors as ela

    out_json, out_png = sys.argv[1], sys.argv[2]
    run_root = sys.argv[3] if len(sys.argv) > 3 else "llm-soft-loop/pod64/prefix"
    base = ela.DATA / "explore" / run_root

    def cells(model, idx, stage):
        p = base / f"{model}-p{idx}-{stage}" / "hidden-chaos.json"
        return json.loads(p.read_text())["cells"] if p.exists() else None

    table = {m: {i: endpoints(cells(m, i, "coarse"), cells(m, i, "fine")) for i in range(10)} for m in MODELS}
    result = {"score": score(table), "table": {m: {str(i): table[m][i] for i in range(10)} for m in MODELS},
              "held_out": list(HELD_OUT), "development": list(DEVELOPMENT)}
    Path(out_json).write_text(json.dumps(result, indent=1))
    for i in range(10):
        a, b = table["qwen7b"][i], table["qwen14b"][i]
        f = lambda e: "missing" if e is None else f"C1 {e['C1']:2d}  C2 {e['C2']:2d}  M {e['M']:+.3f}"
        print(f"prefix {i} ({'held out' if i in HELD_OUT else 'development'}):  7B {f(a)}   14B {f(b)}")
    print(json.dumps(result["score"], indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.0))
    styles = (("qwen7b", "tab:red", "Qwen2.5-7B", -0.19), ("qwen14b", "tab:purple", "Qwen2.5-14B", 0.19))
    for m, colour, label, dx in styles:
        xs = [i for i in range(10) if table[m][i] is not None]
        ax.bar([x + dx for x in xs], [table[m][i]["C1"] for i in xs], 0.38, color=colour, label=label)
        for i in xs:
            e = table[m][i]
            lo, hi = e["stage2_log10T"]
            bx.add_patch(plt.Rectangle((i + dx - 0.17, lo), 0.34, hi - lo, color=colour, alpha=0.18, lw=0))
            pts = e["chaotic_log10T"]
            bx.plot([i + dx] * len(pts), pts, "_", color=colour, ms=9, mew=1.6)
    for a in (ax, bx):
        a.axvspan(-0.5, 1.5, color="0.93", zorder=0)
        a.set_xticks(range(10))
        a.set_xlim(-0.5, 9.5)
        a.set_xlabel("prefix (gray: development, 0-1; held out: 2-9)")
    ax.set_ylabel("chaotic temperatures in stage 1 (of 48)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    bx.set_ylabel("log10 T of the chaotic temperatures")
    bx.axhline(0.3, color="0.5", lw=0.7, ls="--")
    bx.text(9.45, 0.31, "end of the fixed window", fontsize=7, ha="right", va="bottom", color="0.4")
    bx.set_title("dashes: chaotic temperatures; shading: entropy transition (stage-2 interval)",
                 fontsize=8)
    sc = result["score"]
    fig.suptitle(f"Prompt-blocked size test (frozen): PS1 {sc['PS1']} ({sc['PS1_successes']}/8), "
                 f"PS2 {sc['PS2']} ({sc['PS2_successes']}/8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)


if __name__ == "__main__":
    main()
