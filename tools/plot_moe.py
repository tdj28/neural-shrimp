"""Figure fig-moe: mixtures of experts against same-family dense models (run MOE).

(a) Network factor log g_net against the mean next-token entropy on the orbit (run D, tools/damping_locate.py;
    unsteered loop, 24 temperatures, float64): Qwen3.5-4B and -9B (dense) and Qwen3.5-35B-A3B (mixture of experts,
    about 3B of 35B parameters active). OLMoE-1B-7B and the dense OLMo-2-1B and -7B are summarized in the JSON only:
    OLMoE also differs from OLMo-2 in where it normalizes, so that pair is confounded.
(b) Share of chaotic cells (no period, exponent above 1e-3) for the three Qwen3.5 models: the unsteered sweep
    (128 temperatures over log10 T in [-1, 0.3]) and the 32x32 maps along philosophy and hostility-vs-kindness at
    the middle layer. The mixture of experts' 52 hostility cells are called irregular in the paper, since routing
    jumps were not excluded when that map was computed.
(c) Routing margins along the mixture-of-experts orbits of run D: per temperature, the median over steps (filled)
    and the minimum (open) of the per-step smallest gap between the k-th and (k+1)-th router logits over all layers
    (k = 8). A perturbation must move the router logits by about this gap to switch an expert.

Usage: python tools/plot_moe.py [OUT.png] [OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
QWEN = [  # key, label, color, line style
    ("qwen3.5-4b", "Qwen3.5-4B (dense)", "#9ad29a", "-"), ("qwen3.5-9b", "Qwen3.5-9B (dense)", "#3a9a4a", "-"),
    ("qwen3.5-35b-a3b", "Qwen3.5-35B-A3B (experts)", "#0b4d1e", "--")]
OLMO = [("olmo2-1b", "OLMo-2-1B (dense)", "#b0b0b0", "-"), ("olmo2-7b", "OLMo-2-7B (dense)", "#707070", "-"),
        ("olmoe-1b-7b", "OLMoE-1B-7B (experts)", "#d07020", "--")]
MARKERS = {"qwen3.5-35b-a3b": "o", "olmoe-1b-7b": "s"}
SCANS = {  # model -> (unsteered sweep, philosophy map, hostility map), relative to explore/llm-soft-loop/pod64
    "qwen3.5-4b": ("moe/qwen3.5-4b-alpha0", "moe/qwen3.5-4b-phil", "moe/qwen3.5-4b-host"),
    "qwen3.5-9b": ("moe/qwen3.5-9b-alpha0", "moe/qwen3.5-9b-phil", "moe/qwen3.5-9b-host"),
    "qwen3.5-35b-a3b": ("controls/m1t-alpha0/scan", "controls/m1t-phil/scan", "controls/m1t-host/scan")}


def chaotic_share(path):
    z = np.load(path / "scan.npz")
    lam, per = z["lyapunov"].ravel(), z["period"].ravel()
    n = int(((per == 0) & (lam > 1e-3)).sum())
    return n, int(lam.size), float(lam.max())


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    out_png = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-moe.png")
    out_json = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "paper/llm/figures/fig-moe.json")
    D = ela.DATA / "explore" / "damping" / "d1"
    P = ela.DATA / "explore" / "llm-soft-loop" / "pod64"
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.75), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.12, 1]})
    summary = {"damping": {}, "chaotic": {}, "routing": {}}
    ax = axes[0]
    for key, label, col, ls in QWEN + OLMO:
        s = json.loads((D / key / "damping.json").read_text())
        H = np.array([c["mean_entropy"] for c in s["cells"]])
        g = np.array([c["mean_log_g_net"] for c in s["cells"]])
        o = np.argsort(H)
        if key.startswith("qwen"):
            ax.plot(H[o], g[o], ls, color=col, lw=1.1, label=label)
        sel = (H >= 0.3) & (H < 6)
        summary["damping"][key] = {"median_g_net_0.3_6": float(np.median(g[sel])), "cells_0.3_6": int(sel.sum()),
                                   "max_closure": float(max(c["max_closure"] for c in s["cells"]))}
        if s.get("routers"):
            cells = s["cells"]
            Hr = np.array([c["mean_entropy"] for c in cells])
            med = np.array([c["margin_median"] for c in cells])
            mn = np.array([c["margin_min_over_steps"] for c in cells])
            frozen = max(abs(c["mean_log_g_net_frozen"] - c["mean_log_g_net"]) for c in cells)
            switches = float(sum(c["mean_switches"] for c in cells))
            mk = MARKERS[key]
            axes[2].plot(Hr, med, mk, color=col, ms=3.5, label=label.split(" (")[0] + ", median")
            axes[2].plot(Hr, mn, mk, color=col, ms=3.5, mfc="white", label=label.split(" (")[0] + ", minimum")
            summary["routing"][key] = {"routers": int(s["routers"]), "median_gap_range": [float(med.min()),
                                       float(med.max())], "min_gap_range": [float(mn.min()), float(mn.max())],
                                       "max_frozen_minus_free": float(frozen), "total_mean_switches": switches}
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("mean next-token entropy (nats)", fontsize=8)
    ax.set_ylabel(r"network factor $\log g_{\mathrm{net}}$", fontsize=8)
    ax.set_title("(a) network factor", fontsize=9, loc="left")
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax = axes[1]
    labels = ["unsteered", "philosophy", "hostility"]
    ticks = ["unsteered\nsweep", "philosophy\nmap", "hostility\nmap"]
    w = 0.26
    for i, (key, label, col, _) in enumerate(QWEN):
        vals = []
        for j, rel in enumerate(SCANS[key]):
            n, tot, top = chaotic_share(P / rel)
            vals.append(n / tot)
            summary["chaotic"].setdefault(key, {})[labels[j]] = {"chaotic": n, "cells": tot, "largest_exponent": top}
        bars = ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=col, edgecolor="black", lw=0.4,
                      label=label.split(" (")[0])
        if key == "qwen3.5-35b-a3b":
            bars[2].set_hatch("////")
            bars[2].set_edgecolor("white")
            for j, v in enumerate(vals):
                if v == 0:
                    ax.text(j + (i - 1) * w, 0.004, "0", ha="center", va="bottom", fontsize=6, color=col)
            ax.text(2 + (i - 1) * w, vals[2] + 0.006, "irreg.", ha="center", va="bottom", fontsize=5.5, color=col)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(ticks, fontsize=6)
    ax.set_ylabel("share of chaotic cells", fontsize=8)
    ax.set_title("(b) chaos, Qwen3.5", fontsize=9, loc="left")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax = axes[2]
    ax.set_yscale("log")
    ax.set_xlabel("mean next-token entropy (nats)", fontsize=8)
    ax.set_ylabel("smallest router-logit gap", fontsize=8)
    ax.set_title("(c) routing margins", fontsize=9, loc="left")
    ax.legend(fontsize=5.2, frameon=False, loc="lower left", ncol=1)
    ax.set_ylim(1e-8, 1e-1)
    for a in axes:
        a.tick_params(labelsize=7)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
