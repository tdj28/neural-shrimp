"""Figure fig-damping: where the loop damps a change of its fed-back token, across model sizes (run D,
tools/damping_locate.py).

Every model runs the unsteered loop on the paper's prompt at 24 temperatures (log10 T -0.3 to 0.9), float64.
The network factor log g_net (the stretching of the change of the fed-back token x, made relative, through the whole
stack to the final normalized hidden state) splits exactly into growth of the change through the stack, A_L, minus
dilution by the residual stream's own growth, N_L, plus the final normalization.
(a) Network factor against the mean next-token entropy on the orbit, per model (Qwen2.5 blue, Llama-3 red, OLMo-2
    gray; darker for larger models).
(b) Median network factor between 0.3 and 6 nats, where the loop's chaos occurs at low temperature, against the
    number of parameters, per family; the bars give the range over the entropy bins 0-1, 1-3 and 3-6 nats.
(c) The relative change through depth, A_l - N_l = log((|dh_l| / |h_l|) / (|dx| / |x|)), against relative depth
    l / L, at each model's temperature nearest 1 nat, for Qwen2.5 0.5B, 3B, 14B, 72B and Llama-3.1 8B, 70B.
(d) The two parts of the final relative change at that temperature: the gap after the first block, A_1 - N_1, and
    the deficit through the remaining blocks, (N_L - N_2) - (A_L - A_2), against the number of parameters.
OLMo-2 normalizes each sublayer's output before adding it to the residual stream, so its per-sublayer decomposition is
not used; the quantities drawn here use only block inputs, x and the final state and are valid for it.

Usage: python tools/plot_damping.py [OUT.png] [OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
MODELS = [  # key, label, family, parameters (billions)
    ("qwen0.5b", "Qwen2.5-0.5B", "qwen", 0.49), ("qwen1.5b", "Qwen2.5-1.5B", "qwen", 1.54),
    ("qwen3b", "Qwen2.5-3B", "qwen", 3.09), ("qwen7b", "Qwen2.5-7B", "qwen", 7.62),
    ("qwen14b", "Qwen2.5-14B", "qwen", 14.7), ("qwen32b", "Qwen2.5-32B", "qwen", 32.8),
    ("qwen72b", "Qwen2.5-72B", "qwen", 72.7), ("llama3.1-8b", "Llama-3.1-8B", "llama", 8.03),
    ("llama3.1-70b", "Llama-3.1-70B", "llama", 70.6), ("llama3.3-70b", "Llama-3.3-70B", "llama", 70.6),
    ("olmo2-1b", "OLMo-2-1B", "olmo", 1.48), ("olmo2-7b", "OLMo-2-7B", "olmo", 7.30),
    ("olmo2-13b", "OLMo-2-13B", "olmo", 13.7), ("olmo2-32b", "OLMo-2-32B", "olmo", 32.2)]
CMAPS = {"qwen": "Blues", "llama": "Reds", "olmo": "Greys"}
PROFILE = ["qwen0.5b", "qwen3b", "qwen14b", "qwen72b", "llama3.1-8b", "llama3.1-70b"]
BINS = ((0, 1), (1, 3), (3, 6))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    out_png = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-damping.png")
    out_json = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "paper/llm/figures/fig-damping.json")
    D = ela.DATA / "explore" / "damping" / "d1"
    fam_members = {f: [m for m in MODELS if m[2] == f] for f in CMAPS}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), gridspec_kw={"hspace": 0.48, "wspace": 0.3})
    summary = {"models": {}}
    series = {f: [] for f in CMAPS}
    for key, label, fam, params in MODELS:
        s = json.loads((D / key / "damping.json").read_text())
        z = np.load(D / key / "layers.npz")
        members = fam_members[fam]
        shade = 0.35 + 0.6 * members.index((key, label, fam, params)) / max(1, len(members) - 1)
        col = plt.get_cmap(CMAPS[fam])(shade)
        cells = s["cells"]
        H = np.array([c["mean_entropy"] for c in cells])
        g = np.array([c["mean_log_g_net"] for c in cells])
        o = np.argsort(H)
        axes[0, 0].plot(H[o], g[o], "-", color=col, lw=0.9)
        sel = (H >= 0.3) & (H < 6)
        bins = {f"{lo}-{hi}": (float(np.median(g[(H >= lo) & (H < hi)])) if ((H >= lo) & (H < hi)).any() else None)
                for lo, hi in BINS}
        vals = [v for v in bins.values() if v is not None]
        med = float(np.median(g[sel])) if sel.any() else None
        c1 = int(np.argmin(np.abs(H - 1.0)))
        A, N = z["mean_A"][:, c1], z["mean_N"][:, c1]
        L = len(A)
        gap1 = float(A[1] - N[1])
        deficit = float((N[-1] - N[2]) - (A[-1] - A[2]))
        if key in PROFILE:
            axes[1, 0].plot(np.arange(L) / L, A - N, "-", color=col, lw=1.1, label=f"{label} ({H[c1]:.1f} nats)")
        series[fam].append((params, med, min(vals) if vals else None, max(vals) if vals else None, gap1, deficit,
                            label))
        summary["models"][key] = {"label": label, "family": fam, "params_b": params, "layers": s["layers"],
                                  "hidden": s["hidden"], "median_g_net_0.3_6": med, "bin_medians": bins,
                                  "chaotic_temperatures": int(sum(c["chaotic"] for c in cells)),
                                  "profile_entropy": float(H[c1]), "first_block_gap": gap1,
                                  "depth_deficit": deficit, "final_relative_change": float(A[-1] - N[-1]),
                                  "max_closure": float(max(c["max_closure"] for c in cells)),
                                  "decomposition_valid": bool(max(c["max_closure"] for c in cells) < 1e-5)}
    for fam, pts in series.items():
        pts = sorted(pts)
        col = plt.get_cmap(CMAPS[fam])(0.8)
        x = [p[0] for p in pts]
        ax = axes[0, 1]
        ax.plot(x, [p[1] for p in pts], "o-", color=col, ms=4, lw=1.0, label={"qwen": "Qwen2.5", "llama": "Llama-3",
                                                                             "olmo": "OLMo-2"}[fam])
        for p in pts:
            if p[2] is not None:
                ax.plot([p[0], p[0]], [p[2], p[3]], "-", color=col, lw=0.6, alpha=0.7)
        ax = axes[1, 1]
        ax.plot(x, [p[4] for p in pts], "o-", color=col, ms=3.5, lw=1.0)
        ax.plot(x, [-p[5] for p in pts], "s--", color=col, ms=3.5, lw=1.0, mfc="white")
    ax = axes[0, 0]
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("mean next-token entropy (nats)", fontsize=8)
    ax.set_ylabel(r"network factor $\log g_{\mathrm{net}}$ (per step)", fontsize=8)
    ax.set_title("(a) network factor, 14 models", fontsize=9, loc="left")
    ax.text(0.03, 0.95, "Qwen2.5", color=plt.get_cmap("Blues")(0.8), fontsize=7, transform=ax.transAxes, va="top")
    ax.text(0.03, 0.87, "Llama-3", color=plt.get_cmap("Reds")(0.8), fontsize=7, transform=ax.transAxes, va="top")
    ax.text(0.03, 0.79, "OLMo-2", color=plt.get_cmap("Greys")(0.8), fontsize=7, transform=ax.transAxes, va="top")
    ax = axes[0, 1]
    ax.set_xscale("log")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("parameters (billions)", fontsize=8)
    ax.set_ylabel("median network factor, 0.3-6 nats", fontsize=8)
    ax.set_title("(b) damping grows with size", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[1, 0]
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("relative depth $l/L$", fontsize=8)
    ax.set_ylabel(r"$A_l-N_l$ (log relative change)", fontsize=8)
    ax.set_title("(c) the change through depth", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax = axes[1, 1]
    ax.set_xscale("log")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("parameters (billions)", fontsize=8)
    ax.set_ylabel("nats", fontsize=8)
    ax.set_title("(d) first-block gap (solid), minus depth deficit (dashed)", fontsize=8, loc="left")
    for a in axes.flat:
        a.tick_params(labelsize=7)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    for k, v in summary["models"].items():
        print(f"{v['label']:14s} median {v['median_g_net_0.3_6']:6.2f} gap1 {v['first_block_gap']:6.2f} "
              f"deficit {v['depth_deficit']:5.2f} final {v['final_relative_change']:6.2f} valid {v['decomposition_valid']}")


if __name__ == "__main__":
    main()
