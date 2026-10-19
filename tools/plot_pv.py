"""Figure of run PV: the chaos of the 0.5B soft loop under 22 prompt variants (human, 19 Oct 2026: "Did we see how the
chaos changes when we change the prompt? ... both together and both separately"; "the burden is on us to show").

(a) The chaotic share of the standard plane (128 x 128 float64 cells, alpha -0.12 to 0.12, log10 T -1.0 to 0.3,
    philosophy at layer 12) for every condition, grouped by the kind of change, with the base prompt's share as a line.
(b) Eight of the maps side by side, colored as the paper's maps (chaotic cells by their largest exponent, periodic
    cells in gray by contraction).
Every map passes the gates of tools/pv_summary.py (same grid as the base map; the record names the condition's prefix
or start vector; all exponents finite); a map that fails is reported and left out.

Usage: python tools/plot_pv.py --out paper/llm/figures/fig-prompt-variants.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pv_summary as pvs  # noqa: E402

GROUPS = [("base", ["p00"]), ("each part alone", ["p01", "p02", "p03"]), ("shuffled", ["p04", "p05", "p06", "p07"]),
          ("other content", ["p08", "p09", "p10", "p11", "p12"]), ("prompt shape", ["p13", "p14", "p15"]),
          ("no text: one random vector", ["p16", "p17", "p18"]), ("base, random start", ["s1", "s2", "s3"])]
SHOWN = ["p00", "p01", "p02", "p03", "p07", "p08", "p17", "s2"]
SHORT = {"p00": "base prompt", "p01": "system prompt only", "p02": '"Hello!" only', "p03": "neither",
         "p07": "both shuffled", "p08": "opposite system", "p17": "random-token prompt",
         "s2": "random start"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chaos-level", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    base, _ = pvs.load("p00")
    a, t = base["alpha"], base["log10_temperature"]
    maps, shares, failed = {}, {}, []
    for _, conds in GROUPS:
        for c in conds:
            try:
                z, rec = pvs.load(c)
            except FileNotFoundError:
                failed.append(c)
                continue
            lam = z["lyapunov"]
            if not (np.array_equal(z["alpha"], a) and np.array_equal(z["log10_temperature"], t)) or \
                    not np.all(np.isfinite(lam)):
                failed.append(c)
                continue
            maps[c] = (lam, z["period"])
            shares[c] = float(((lam > args.chaos_level)).mean())
    plt.rcParams.update({"font.size": 7.5})
    fig = plt.figure(figsize=(7.2, 5.6))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 2.3], wspace=0.42)
    gs = outer[0, 1].subgridspec(2, 4, hspace=0.32, wspace=0.08)
    axb = fig.add_subplot(outer[0, 0])
    ys, labels, colors = [], [], []
    y = 0
    palette = ["k", "tab:blue", "tab:purple", "tab:orange", "tab:green", "tab:red", "0.45"]
    for (gname, conds), col in zip(GROUPS, palette):
        for c in conds:
            if c in shares:
                ys.append(y); labels.append(pvs.LABELS[c]); colors.append(col)
                axb.barh(y, 100 * shares[c], color=col, height=0.75)
            y -= 1
        y -= 0.6
    axb.axvline(100 * shares["p00"], color="k", lw=0.6, ls=":")
    axb.set_yticks(ys)
    axb.set_yticklabels(labels, fontsize=5.6)
    axb.set_xlabel("chaotic share of the plane (%)")
    axb.set_title("(a) all 22 prefixes", fontsize=8)
    axb.tick_params(axis="x", labelsize=6.5)
    ext = [a[0], a[-1], t[0], t[-1]]
    vmax = max(float(np.max(np.where(m[0] > args.chaos_level, m[0], 0))) for m in maps.values())
    for k, c in enumerate(SHOWN):
        if c not in maps:
            continue
        ax = fig.add_subplot(gs[k // 4, k % 4])
        lam, per = maps[c]
        chaos = lam > args.chaos_level              # colored as the paper's maps: periodic gray by contraction
        neg = np.clip(-lam, 0, None)
        cap = max(float(np.percentile(neg[~chaos], 99)) if (~chaos).any() else 1.0, 1e-12)
        grey = 1.0 - 0.9 * np.clip(neg / cap, 0, 1) ** 0.5
        img = np.stack([grey, grey, grey, np.ones_like(grey)], axis=-1)
        img[chaos] = plt.get_cmap("plasma")(np.clip(lam[chaos] / vmax, 0, 1))
        ax.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
        im = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, vmax))
        ax.set_title(f"{SHORT[c]}\n{100 * shares[c]:.1f} % chaotic", fontsize=5.6, linespacing=1.1)
        ax.tick_params(labelsize=5)
        if k % 4:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$\log_{10}T$", fontsize=6, labelpad=1)
        if k // 4 == 1:
            ax.set_xlabel(r"$\alpha$", fontsize=6)
        else:
            ax.set_xticklabels([])
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("largest exponent (chaotic cells)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.62, 0.005, "(b) eight of the maps, with their chaotic shares (gray: periodic, darker = more contracting)",
             ha="center", fontsize=7.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    rec = {"shares": shares, "failed": failed, "shown": SHOWN}
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
