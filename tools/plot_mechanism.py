"""Figure fig-mechanism and its statistics: why steering makes the soft-token loop chaotic (runs MECH and XO).

Reads (DATA = ../neural-shrimp-data/explore/llm-soft-loop/pod64):
  mech/amap-a, amap-b     the block-12 alpha map (alpha -0.12..0.12 x log10 T -1..0.3), gain split per cell;
  mech/bmap-a, bmap-b     the logit-bias map at alpha = 0 (beta -8..8, same temperatures);
  mech/b4, b8, b12, b16, b20   alpha lines at five steering blocks (log10 T -0.6, -0.3, 0.0, 0.15, 0.25);
  xo/cells                the orbit-preserving four-arm crossover.
Panels (each drawn when its data exist):
  (a) readout and network factors against alpha on the block-12 lines, chaotic cells marked;
  (b) chaotic minus periodic, row-matched, on the alpha map: factors, entropy, gap;
  (c) crossover arms per held-out chaotic cell and periodic control;
  (d) chaotic share per temperature: residual steering (alpha map) against the logit bias (beta map);
  (e) the two factors of chaotic cells by steering block.
Statistics go to paper/llm/figures/fig-mechanism.json: per-row differences with bootstrap intervals over rows,
row-stratified permutation tests, crossover contrasts with bootstrap intervals over held-out cells.

Usage: python tools/plot_mechanism.py [--out PNG]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"
FIELDS = ("mean_log_g_net", "mean_log_g_read", "mean_entropy", "mean_gap", "mean_log_r1_rel", "mean_stable_rank")


def load_cells(*runs):
    cells = []
    for r in runs:
        f = DATA / "mech" / r / "gain.json"
        if f.exists():
            cells += json.loads(f.read_text())["cells"]
    return cells


def arrays(cells):
    A = {k: np.array([c[k] for c in cells], dtype=float) for k in ("alpha", "beta", "log10_temperature") + FIELDS}
    A["lyap"] = np.array([c["lyapunov"][0] for c in cells], dtype=float)
    A["chaotic"] = np.array([c["chaotic"] for c in cells])
    A["periodic"] = np.array([c["period"] > 0 for c in cells])
    return A


def row_stats(A, x, group_a, group_b, rng, n_boot=10000, n_perm=10000, min_n=3):
    """Mean over temperature rows of (mean x in group a - mean x in group b), rows weighted by the group-a count,
    finite values only; bootstrap interval over rows; row-stratified permutation test of the group labels."""
    rows = np.unique(np.round(A["log10_temperature"], 6))
    diffs, w, members = [], [], []
    for t in rows:
        m = np.isclose(A["log10_temperature"], t) & np.isfinite(x)
        a, b = m & group_a, m & group_b
        if a.sum() >= min_n and b.sum() >= min_n:
            diffs.append(x[a].mean() - x[b].mean())
            w.append(a.sum())
            members.append((x[a], x[b]))
    if not diffs:
        return None
    diffs, w = np.array(diffs), np.array(w, dtype=float)
    est = float(np.average(diffs, weights=w))
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(diffs), len(diffs))
        boots.append(np.average(diffs[k], weights=w[k]))
    perm = 0
    for _ in range(n_perm):
        d = []
        for xa, xb in members:
            pool = np.concatenate([xa, xb])
            rng.shuffle(pool)
            d.append(pool[:len(xa)].mean() - pool[len(xa):].mean())
        perm += abs(np.average(d, weights=w)) >= abs(est)
    return {"estimate": est, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_permutation": (perm + 1) / (n_perm + 1), "rows": int(len(diffs)), "rows_larger_than_zero": int((diffs > 0).sum()),
            "group_a_cells": int(sum(len(a) for a, _ in members)), "group_b_cells": int(sum(len(b) for _, b in members))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-mechanism.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    stats = {}
    amap = load_cells("amap-a", "amap-b")
    bmap = load_cells("bmap-a", "bmap-b")
    lines = {b: load_cells(f"b{b}") for b in (4, 8, 12, 16, 20)}
    xo_f = DATA / "xo" / "cells" / "crossover.json"
    xo = json.loads(xo_f.read_text()) if xo_f.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=150)
    axes = axes.ravel()

    # (a) factors against alpha on the block-12 lines
    ax = axes[0]
    if lines[12]:
        L = arrays(lines[12])
        for t, col in zip((-0.6, -0.3, 0.0, 0.15, 0.25), ("tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:red")):
            m = np.isclose(L["log10_temperature"], t)
            o = np.argsort(L["alpha"][m])
            a = L["alpha"][m][o]
            ax.plot(a, L["mean_log_g_read"][m][o], "-", color=col, lw=1.2, label=f"readout, log10 T {t:+.2f}")
            ax.plot(a, L["mean_log_g_net"][m][o], ":", color=col, lw=1.0)
            c = L["chaotic"][m][o]
            ax.plot(a[c], L["mean_log_g_read"][m][o][c], "o", color=col, ms=2.5)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylim(-8, 3)
        ax.set_xlabel("steering strength alpha (block 12)")
        ax.set_ylabel("mean log factor per step (nats)")
        ax.set_title("(a) readout (solid) and network (dotted) factors;\ndots: chaotic cells", fontsize=9)
        ax.legend(fontsize=6, loc="lower left")
    # (b) chaotic minus periodic on the alpha map
    ax = axes[1]
    if amap:
        A = arrays(amap)
        near0 = np.abs(A["alpha"]) <= 0.01
        stats["R1_chaotic_minus_near_zero"] = {k: row_stats(A, A[k], A["chaotic"], near0, rng)
                                              for k in ("mean_log_g_net", "mean_log_g_read")}
        stats["R2_chaotic_minus_periodic"] = {k: row_stats(A, A[k], A["chaotic"], A["periodic"], rng) for k in FIELDS}
        names = [("mean_log_g_read", "log readout factor"), ("mean_log_g_net", "log network factor"),
                 ("mean_entropy", "entropy (nats)"), ("mean_gap", "top-2 gap / T")]
        for k, (f, lab) in enumerate(names):
            s = stats["R2_chaotic_minus_periodic"][f]
            if s:
                ax.errorbar([k], [s["estimate"]], yerr=[[s["estimate"] - s["ci95"][0]], [s["ci95"][1] - s["estimate"]]],
                            fmt="o", color="k", capsize=4)
                ax.text(k, s["ci95"][1] + 0.1, "p=%.4f" % s["p_permutation"], ha="center", fontsize=7)
        s1 = stats["R1_chaotic_minus_near_zero"]
        for k, f, mk in ((0, "mean_log_g_read", "s"), (1, "mean_log_g_net", "s")):
            if s1[f]:
                ax.errorbar([k + 0.25], [s1[f]["estimate"]], yerr=[[s1[f]["estimate"] - s1[f]["ci95"][0]],
                                                                    [s1[f]["ci95"][1] - s1[f]["estimate"]]],
                            fmt=mk, color="tab:red", capsize=4, label="chaotic - |alpha|<=0.01" if k == 0 else None)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[1] for n in names], fontsize=7, rotation=12)
        ax.set_ylabel("difference, row-matched (95 % CI over rows)")
        ax.set_title("(b) chaotic minus periodic cells (black)\nand minus unsteered cells (red), alpha map", fontsize=9)
        ax.legend(fontsize=7)
    # (c) crossover arms
    ax = axes[2]
    if xo:
        labels = xo.get("labels") or ["?"] * len(xo["cells"])
        arms = ["arm00", "arm10", "arm01", "arm11"]
        X = np.array([[c[a] for a in arms] for c in xo["cells"]])
        held = np.array([i >= 2 or lab != "chaotic" for i, lab in enumerate(labels)])
        chao = np.array([lab == "chaotic" for lab in labels])
        for i in range(len(X)):
            ax.plot(range(4), X[i], "-", color="tab:red" if chao[i] else "tab:blue", alpha=0.35, lw=0.8)
        sel = chao & held
        if sel.sum():
            m = X[sel].mean(0)
            ax.plot(range(4), m, "o-", color="darkred", lw=2, label="chaotic cells, mean")
            contr = {"readout_effect (10-00)": X[sel, 1] - X[sel, 0], "network_effect (01-00)": X[sel, 2] - X[sel, 0],
                     "interaction (11-10-01+00)": X[sel, 3] - X[sel, 1] - X[sel, 2] + X[sel, 0],
                     "full_minus_readout_only (11-10)": X[sel, 3] - X[sel, 1]}
            stats["XO"] = {}
            for name, v in contr.items():
                b = [rng.choice(v, len(v)).mean() for _ in range(10000)]
                stats["XO"][name] = {"mean": float(v.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                                     "cells": int(len(v))}
            stats["XO"]["validation_max_abs(arm11 - twin)"] = float(max(abs(c["arm11"] - c["twin_exponent"]) for c in xo["cells"]))
            dev = np.array([abs(c["arm11"] - c["twin_exponent"]) for c in xo["cells"]])
            stats["XO"]["validation_cells_within_0.02"] = int((dev <= 0.02).sum())
            stats["XO"]["arm10_minus_arm11_mean"] = float((X[sel, 1] - X[sel, 3]).mean())
            lt = np.array([c["log10_temperature"] for c in xo["cells"]])
            for tag, part in (("log10T<0", sel & (lt < 0)), ("log10T>=0", sel & (lt >= 0))):
                if part.sum():
                    stats["XO"][tag] = {"cells": int(part.sum()), "arm_means": dict(zip(arms, X[part].mean(0).round(4).tolist())),
                                        "readout_effect": float((X[part, 1] - X[part, 0]).mean()),
                                        "network_effect": float((X[part, 2] - X[part, 0]).mean()),
                                        "interaction": float((X[part, 3] - X[part, 1] - X[part, 2] + X[part, 0]).mean())}
        ax.set_xticks(range(4))
        ax.set_xticklabels(["00\nunsteered\nboth", "10\nsteered\nreadout", "01\nsteered\nnetwork", "11\nsteered\nboth"], fontsize=7)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel("finite-time growth along the steered orbit (nats/step)")
        ax.set_title("(c) crossover: which factor's steering makes the orbit expand\n(red: chaotic cells, blue: periodic)", fontsize=9)
        ax.legend(fontsize=7)
    # (d) chaotic share per temperature: alpha map against logit-bias map
    ax = axes[3]
    for cells, lab, col in ((amap, "residual steering, block 12 (alpha map)", "k"), (bmap, "logit bias only (beta map)", "tab:purple")):
        if cells:
            A = arrays(cells)
            rows = np.unique(np.round(A["log10_temperature"], 6))
            share = [A["chaotic"][np.isclose(A["log10_temperature"], t)].mean() for t in rows]
            ax.plot(rows, np.array(share) * 100, "-", color=col, label=f"{lab}: {A['chaotic'].mean() * 100:.1f} % of cells")
            stats.setdefault("chaotic_share", {})[lab] = {"share": float(A["chaotic"].mean()), "cells": int(len(cells)),
                                                          "chaotic": int(A["chaotic"].sum())}
    ax.set_xlabel("log10 temperature")
    ax.set_ylabel("chaotic cells in the row (%)")
    ax.set_title("(d) does token competition alone give chaos?\nresidual steering against a pure logit bias", fontsize=9)
    ax.legend(fontsize=7)
    # (e) factors of chaotic cells by block
    ax = axes[4]
    blk, net, read = [], [], []
    for b, cells in lines.items():
        if cells:
            L = arrays(cells)
            c = L["chaotic"]
            if c.sum():
                blk.append(b)
                net.append(L["mean_log_g_net"][c])
                read.append(L["mean_log_g_read"][c])
                stats.setdefault("blocks", {})[str(b)] = {"chaotic": int(c.sum()), "cells": int(len(c)),
                                                          "median_log_g_net": float(np.median(L["mean_log_g_net"][c])),
                                                          "median_log_g_read": float(np.median(L["mean_log_g_read"][c])),
                                                          "chaotic_log10T_min": float(L["log10_temperature"][c].min())}
    if blk:
        pos = np.arange(len(blk))
        ax.boxplot(read, positions=pos - 0.18, widths=0.3, patch_artist=True, boxprops=dict(facecolor="tab:orange", alpha=0.6))
        ax.boxplot(net, positions=pos + 0.18, widths=0.3, patch_artist=True, boxprops=dict(facecolor="tab:gray", alpha=0.6))
        ax.set_xticks(pos)
        ax.set_xticklabels([f"block {b}\n(n={len(r)})" for b, r in zip(blk, read)], fontsize=7)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel("mean log factor per step (nats)")
        ax.set_title("(e) chaotic cells by steering block:\nreadout (orange) and network (grey) factors", fontsize=9)
    axes[5].axis("off")
    fig.tight_layout()
    fig.savefig(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1)[:3000])


if __name__ == "__main__":
    main()
