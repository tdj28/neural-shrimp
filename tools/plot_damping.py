"""Figure fig-damping: where the loop damps a change of its fed-back token, across model sizes (run D,
tools/damping_locate.py).

Every model runs the unsteered loop on the paper's prompt at 24 temperatures (log10 T -0.3 to 0.9), float64.
(a) Network factor log g_net against the mean next-token entropy on the orbit, per model.
(b) The exact split of the network factor at matched entropy: growth of the change through the stack, dilution by the
    residual stream's own growth, and the final normalization, per model (median over the temperatures in the bin).
(c) The relative change through depth, A_l - N_l = log((|dh_l| / |h_l|) / (|dx| / |x|)), against relative depth
    l / L, at each model's temperature nearest the target entropy.
(d) Knockout effects against relative depth (MLP solid, attention dashed) at the knockout temperature nearest the
    target entropy: positive values amplify the change, negative ones damp it.

Usage: python tools/plot_damping.py OUT.png OUT.json LABEL=RUN [LABEL=RUN ...]   (RUN explore-relative)
       options via environment: DAMP_ENTROPY (target nats, default 5), DAMP_BIN (lo,hi nats, default 3,8)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def load(run):
    import explore_llm_attractors as ela

    d = ela.DATA / "explore" / run
    s = json.loads((d / "damping.json").read_text())
    z = np.load(d / "layers.npz")
    return s, z


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png, out_json = sys.argv[1], sys.argv[2]
    specs = [a.split("=", 1) for a in sys.argv[3:]]
    target = float(os.environ.get("DAMP_ENTROPY", "5"))
    lo, hi = (float(x) for x in os.environ.get("DAMP_BIN", "3,8").split(","))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), gridspec_kw={"hspace": 0.5, "wspace": 0.32})
    summary = {"target_entropy": target, "bin": [lo, hi], "models": {}}
    for m, (label, run) in enumerate(specs):
        s, z = load(run)
        col = cmap(m / max(1, len(specs) - 1))
        cells = s["cells"]
        H = np.array([c["mean_entropy"] for c in cells])
        g = np.array([c["mean_log_g_net"] for c in cells])
        stack = np.array([c["mean_stack"] for c in cells])
        dil = np.array([c["mean_dilution"] for c in cells])
        fn = np.array([c["mean_final_norm"] for c in cells])
        chaotic = np.array([c["chaotic"] for c in cells])
        L = s["layers"]
        ax = axes[0, 0]
        o = np.argsort(H)
        ax.plot(H[o], g[o], "-", color=col, lw=1.0, label=label)
        ax.plot(H[chaotic], g[chaotic], "o", mfc="white", color=col, ms=3.5)
        inbin = (H >= lo) & (H <= hi)
        med = {k: (float(np.median(v[inbin])) if inbin.any() else None)
               for k, v in (("log_g_net", g), ("stack", stack), ("dilution", dil), ("final_norm", fn))}
        ax = axes[0, 1]
        if inbin.any():
            x = m
            ax.bar(x - 0.27, med["stack"], 0.25, color="#4d9221")
            ax.bar(x, -med["dilution"], 0.25, color="#c51b7d")
            ax.bar(x + 0.27, med["log_g_net"], 0.25, color="black")
        c_t = int(np.argmin(np.abs(H - target)))
        rel = z["mean_A"][:, c_t] - z["mean_N"][:, c_t]
        depth = np.arange(L) / L
        axes[1, 0].plot(depth, rel, "-", color=col, lw=1.0, label=f"{label} ({H[c_t]:.1f} nats)")
        entry = {"run": run, "layers": L, "hidden": s["hidden"], "embedding_norm_median": s["embedding_norm_median"],
                 "chaotic_temperatures": int(chaotic.sum()), "bin_medians": med, "bin_cells": int(inbin.sum()),
                 "profile_entropy": float(H[c_t]), "relative_change_profile": rel.tolist()}
        ko = s.get("knockouts")
        if ko:
            kc = ko["cells"]
            Hk = H[kc]
            j = int(np.argmin(np.abs(Hk - target)))
            dm, da = np.array(ko["delta_mlp"])[:, j], np.array(ko["delta_attn"])[:, j]
            axes[1, 1].plot(depth, dm, "-", color=col, lw=1.0)
            axes[1, 1].plot(depth, da, "--", color=col, lw=0.8)
            entry.update({"knockout_entropy": float(Hk[j]), "delta_mlp": dm.tolist(), "delta_attn": da.tolist(),
                          "mlp0": float(dm[0]), "attn0": float(da[0]),
                          "sum_mlp_negative": float(dm[dm < 0].sum()), "sum_attn_negative": float(da[da < 0].sum()),
                          "plain_twin_check_max": float(np.max(np.abs(ko["plain_twin_check"])))})
        summary["models"][label] = entry
    ax = axes[0, 0]
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("mean next-token entropy (nats)", fontsize=8)
    ax.set_ylabel(r"network factor $\log g_{\mathrm{net}}$", fontsize=8)
    ax.set_title("(a) network factor", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[0, 1]
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels([sp[0] for sp in specs], fontsize=6.5, rotation=30)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"(b) split, {lo:g}-{hi:g} nats", fontsize=9, loc="left")
    ax.text(0.02, 0.95, "growth through stack", color="#4d9221", fontsize=6.5, transform=ax.transAxes, va="top")
    ax.text(0.02, 0.86, "minus dilution", color="#c51b7d", fontsize=6.5, transform=ax.transAxes, va="top")
    ax.text(0.02, 0.77, "net", color="black", fontsize=6.5, transform=ax.transAxes, va="top")
    ax = axes[1, 0]
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("relative depth $l/L$", fontsize=8)
    ax.set_ylabel("log relative change", fontsize=8)
    ax.set_title("(c) the change through depth", fontsize=9, loc="left")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 1]
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("relative depth $l/L$", fontsize=8)
    ax.set_ylabel("knockout effect (nats)", fontsize=8)
    ax.set_title("(d) knockouts: MLP solid, attention dashed", fontsize=9, loc="left")
    for a in axes.flat:
        a.tick_params(labelsize=7)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("bin_medians", "mlp0", "sum_mlp_negative") if kk in v}
                      for k, v in summary["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
