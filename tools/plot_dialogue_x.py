"""Figure and statistics of run DLGx: two different models (Qwen2.5-0.5B and 1.5B, unsteered) in one soft-token
conversation, against each model talking with itself, along log10 T (tools/soft_dialogue.py; CrossPairLoop).

Per frame (dialogue, cowrite): the three lines' classes and smaller twin exponents per exchange against log10 T, and
per temperature the comparison of the pair with the two models alone: entrained (pair periodic, both alone chaotic),
made chaotic (pair chaotic, both alone periodic), with the counts. Labels: hidden_chaos.classify.

Usage: python tools/plot_dialogue_x.py [--out PNG] [--stats JSON]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/dialogue"
CHAOS = ("visible chaos", "hidden chaos")


def line(run):
    d = json.loads((DATA / run / "dialogue.json").read_text())
    by_t = {round(c["log10_temperature"], 6): c for c in d["cells"]}
    return by_t


def cls(c):
    return "chaotic" if c["class"] in CHAOS else ("periodic" if c["class"] == "periodic" else "other")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs/dlg/dlgx-lines.png"))
    ap.add_argument("--stats", default=str(ROOT / "docs/dlg/dlgx-stats.json"))
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), dpi=150, sharex=True)
    stats = {}
    for ax, frame, tag in zip(axes, ("dialogue", "cowrite"), ("dl", "cw")):
        A, B, P = line(f"x-{tag}-05"), line(f"x-{tag}-15"), line(f"x-{tag}-pair")
        ts = sorted(set(A) & set(B) & set(P))
        rec = {"temperatures": len(ts), "classes": {}, "entrained": [], "made_chaotic": []}
        for name, L in (("0.5B alone", A), ("1.5B alone", B), ("pair", P)):
            rec["classes"][name] = {k: sum(1 for t in ts if cls(L[t]) == k) for k in ("chaotic", "periodic", "other")}
        for t in ts:
            a, b, p = cls(A[t]), cls(B[t]), cls(P[t])
            if a == b == "chaotic" and p == "periodic":
                rec["entrained"].append([t, P[t]["period"]])
            if a == b == "periodic" and p == "chaotic":
                rec["made_chaotic"].append(t)
        rec["both_chaotic_alone"] = sum(1 for t in ts if cls(A[t]) == cls(B[t]) == "chaotic")
        rec["both_periodic_alone"] = sum(1 for t in ts if cls(A[t]) == cls(B[t]) == "periodic")
        stats[frame] = rec
        for name, L, col, off in (("0.5B alone", A, "tab:blue", 0.0), ("1.5B alone", B, "tab:orange", 0.0),
                                  ("0.5B <-> 1.5B", P, "k", 0.0)):
            x = np.array(ts)
            y = np.array([min(L[t]["lyapunov"]) for t in ts])
            ax.plot(x, y, "-", lw=1.0 if name != "0.5B <-> 1.5B" else 1.4, color=col, label=name)
            per = np.array([cls(L[t]) == "periodic" for t in ts])
            ax.plot(x[per], y[per], "o", ms=2.2, color=col)
        for t, q in rec["entrained"]:
            ax.axvline(t, color="tab:green", lw=0.6, alpha=0.5)
        ax.axhline(0, color="0.5", lw=0.6, ls=":")
        ax.set_ylabel("smaller twin exponent per exchange", fontsize=8)
        ax.set_title(f"{frame} frame: dots = periodic; green lines = pair periodic while both models alone are chaotic "
                     f"({len(rec['entrained'])} of {rec['both_chaotic_alone']} temperatures)", fontsize=8)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel(r"$\log_{10}T$", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    Path(args.stats).write_text(json.dumps(stats, indent=1))
    print(json.dumps({f: {k: v for k, v in r.items() if k not in ("entrained", "made_chaotic")} | {
        "entrained_n": len(r["entrained"]), "made_chaotic_n": len(r["made_chaotic"]),
        "entrained_first": r["entrained"][:12]} for f, r in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
