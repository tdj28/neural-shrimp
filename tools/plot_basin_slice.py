"""Figure: basins of coexisting attractors on a plane of starting states (tools/basin_slice.py output).

Each grid start x0 + |x0| (s u + t w) is coloured by the attractor it settles on (the census's periodic attractors by
their token loops; grey: no period found within the transient; black: periodic but not a census attractor). The fixed
start (s, t) = (0, 0) is marked, and so are the directions of the two attractors when the plane was spanned by them.
A second panel zooms into the stretch of boundary nearest the fixed start.

Usage: python tools/plot_basin_slice.py --run llm-soft-loop/portraits/pair-basin-f32 --out FIG.png [--json J]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from transformers import AutoTokenizer

    z = np.load(DATA / "explore" / args.run / "basin.npz")
    rec = json.loads((DATA / "explore" / args.run / "basin.json").read_text())
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                        cache_dir=str(DATA / "models/hf/hub"))
    s, lab = z["s"], z["labels"]
    na = len(rec["attractors"])
    # colour index: 0 black (periodic, other), 1 grey (no period), 2.. attractors
    idx = np.where(lab == -1, 0, np.where(lab == 0, 1, lab + 1))
    cmap = ListedColormap(["k", "0.75"] + COLOURS[:na])
    ext = [s[0], s[-1], s[0], s[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), dpi=200)
    handles = []
    for k, a in enumerate(rec["attractors"]):
        text = tok.decode(a["token_ids"]).replace("\n", "\\n")
        share = rec["label_counts"].get(str(a["label"]), 0) / lab.size
        handles.append(Patch(color=COLOURS[k], label=f"period {a['period']}: '{text}' ({100 * share:.1f} %)"))
    for key, colour, name in (("0", "0.75", "no period within the transient"), ("-1", "k", "other periodic orbit")):
        if rec["label_counts"].get(key, 0):
            handles.append(Patch(color=colour, label=f"{name} ({100 * rec['label_counts'][key] / lab.size:.1f} %)"))
    # boundary cells: a neighbour with another label
    edge = np.zeros_like(lab, dtype=bool)
    edge[:-1, :] |= lab[:-1, :] != lab[1:, :]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    for ax, title in zip(axes, ("(a) the plane of starting states", "(b) near the fixed start")):
        ax.imshow(idx, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=na + 1.5, interpolation="nearest")
        ax.plot(0, 0, "*", color="w", mec="k", ms=14, mew=1.0, zorder=5)
        ax.set_xlabel(r"$s$ (towards the first attractor), in units of $|x_0|$")
        ax.set_ylabel(r"$t$ (towards the second, orthogonalized)")
        ax.set_title(title, fontsize=10)
    h = (s[-1] - s[0]) / 6
    axes[1].set_xlim(-h, h)
    axes[1].set_ylim(-h, h)
    axes[0].add_patch(plt.Rectangle((-h, -h), 2 * h, 2 * h, fill=False, ec="w", lw=1.2))
    axes[0].legend(handles=handles + [plt.Line2D([], [], marker="*", color="w", mec="k", ls="none", ms=12,
                                                 label="the fixed start")],
                   fontsize=7.5, loc="upper left", framealpha=0.9)
    a = rec["args"]
    fig.suptitle(f"Basins of the coexisting attractors at $\\alpha={a['alpha']}$, $\\log_{{10}}T={a['log_t']}$ "
                 f"({lab.shape[0]}$\\times${lab.shape[1]} starts, {a.get('dtype', 'float64')}, {a['transient']} steps each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    summary = {"run": args.run, "grid": list(lab.shape), "label_counts": rec["label_counts"],
               "boundary_share": float(edge.mean()), "attractors": rec["attractors"],
               "plane_angle_deg": rec.get("plane_angle_deg")}
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
