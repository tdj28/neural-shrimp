"""Runs RK-P5 and RK-G figure: the rank of the period-5 shrimp's return map near its head, and the zero set of the
singular reduction lambda_red = s1 (v1 . u1) around it (tools/monodromy_svd.py).

(a) d1 = s2/s1 at three starting phases for the body, the head, four points at 5e-5 and one at 2e-4 (RK-P5), with the
largest singular value s1 and the leading eigenvalue modulus |mu1| for scale. (b) sign and log10|lambda_red| on the
11 x 11 grid of spacing 1e-5 around the head (RK-G), with the zero contour and the head.

Usage: python tools/plot_head_rank.py --rk llm-soft-loop/rk/p5-head --grid llm-soft-loop/pod64/rk/g-p5 \
         --out paper/llm/figures/fig-head-rank.png [--json docs/rk/rk-summary.json]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def records(path):
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "label" in r and "phases" in r:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rk", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = records(DATA / "explore" / args.rk / "progress.jsonl")
    grid = records(DATA / "explore" / args.grid / "progress.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    ax = axes[0]
    names = [r["label"] for r in pts]
    for k, ph in enumerate(sorted({p["phase"] for r in pts for p in r["phases"]})):
        ys = [next(p["d1"] for p in r["phases"] if p["phase"] == ph) for r in pts]
        ax.semilogy(np.arange(len(pts)) + 0.12 * (k - 1), ys, "o", label=f"$\\sigma_2/\\sigma_1$, phase {ph}")
    ax.semilogy(np.arange(len(pts)), [r["moduli"][0] for r in pts], "k^", ms=6, label=r"$|\mu_1|$ (eigenvalue)")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(pts)), [r["phases"][0]["sigma"][0] for r in pts], "s--", color="0.5", ms=4,
             label=r"$\sigma_1$")
    ax2.set_ylabel(r"largest singular value $\sigma_1$")
    ax.set_xticks(np.arange(len(pts)), names, rotation=30, fontsize=8)
    ax.set_ylabel("ratio or modulus")
    ax.legend(fontsize=7, loc="lower left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.set_title("(a) the return map near the head is nearly of rank one", fontsize=9)
    ax = axes[1]
    G = {}
    for r in grid:
        if r["label"].startswith("g"):
            i, j = map(int, r["label"][1:].split("_"))
            G[(i, j)] = r
    n = 1 + max(max(i, j) for i, j in G)
    centre = next(r for r in grid if r["label"] == "centre")
    c1, c2 = centre["point"]
    lam = np.full((n, n), np.nan)
    A1 = np.full((n, n), np.nan)
    A2 = np.full((n, n), np.nan)
    for (i, j), r in G.items():
        lam[i, j] = r["phases"][0]["lambda_red"]
        A1[i, j], A2[i, j] = r["point"]
    a1 = np.nanmean(A1, axis=0)
    a2 = np.nanmean(A2, axis=1)
    val = np.sign(lam) * np.log10(1 + np.abs(lam) / 1e-4)
    vmax = float(np.nanmax(np.abs(val)))
    im = ax.pcolormesh(a1, a2, val, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    ax.contour(a1, a2, lam, levels=[0], colors="k", linewidths=1.2)
    ax.plot(c1, c2, "o", color="w", mec="k", ms=7, zorder=5)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"sign$(\lambda)\,\log_{10}(1+|\lambda|/10^{-4})$")
    ax.set_xlabel(r"$\alpha_1$ (gratitude, block 4)")
    ax.set_ylabel(r"$\alpha_2$ (hostility, block 16)")
    ax.ticklabel_format(useOffset=False)
    ax.set_title(r"(b) the singular reduction $\lambda=\sigma_1\,(v_1\cdot u_1)$: one zero curve through the head",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    signs = np.sign(lam)
    changes = int(np.nansum(np.abs(np.diff(signs, axis=1)) > 0))
    summary = {"rk_points": [{"label": r["label"], "point": r["point"], "moduli": r["moduli"][:4],
                              "sigma1_phase0": r["phases"][0]["sigma"][0],
                              "d1": {p["phase"]: p["d1"] for p in r["phases"]},
                              "d3": {p["phase"]: p["d3"] for p in r["phases"]}} for r in pts],
               "grid_points": len(G), "grid_n": n, "centre_lambda_red": centre["phases"][0]["lambda_red"],
               "row_sign_changes_total": changes,
               "rows_with_exactly_one_sign_change": int(sum(
                   1 for i in range(n) if np.nansum(np.abs(np.diff(signs[i][~np.isnan(signs[i])])) > 0) == 1))}
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
