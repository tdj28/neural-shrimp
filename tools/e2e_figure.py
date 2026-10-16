"""Run E2E: the composite figure paper/llm/figures/fig-e2e.png and its summary fig-e2e.json.

(a) the 2-cycle ' is it' at alpha 0.070 continued in temperature: its leading multiplier from the fold at the top,
    through its stable range, to the flip below and the unstable orbit pinned on the ' is' / "'s" tie
(b) beyond the fold (log10 T = -0.115): the largest Lyapunov exponent along the row of the 256^2 float64 map, with
    the stripe windows and the fold alpha_c; inset, the window law 1/sqrt(alpha_N - alpha_c) against the period
(c) the 512^2 float64 map with the 2-cycle's continued fold curve, the band's and the crescent's located
    boundaries and the coexistence census cells
(d) the readout's rank along orbits (Sigma_p and DF ratios) and the scalar return map of the crescent: the full
    map's section pairs, the invariant-curve map R1 and the two-token model R2

Every panel reads raw outputs under neural-shrimp-data/explore (e2e and llm-soft-loop); a missing input leaves its
panel with a note rather than failing. Usage: python tools/e2e_figure.py [--out paper/llm/figures/fig-e2e.png]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e2e_model as em  # noqa: E402

SOFT = em.DATA / "explore" / "llm-soft-loop"
E2E = em.OUT


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def records(name):
    j = load(E2E / name / "branch.json")
    return j["records"] if j else []


def panel_a(ax, summary):
    recs = records("p2-tcont-a0.070-up") + records("p2-tcont-a0.070-down") + records("p2-tcont-a0.070-through")
    if not recs:
        ax.text(0.5, 0.5, "no branch data", transform=ax.transAxes, ha="center")
        return
    pts = sorted({(round(r["lambda"], 12), r["leading"][0]) for r in recs})
    lt = np.array([p[0] for p in pts])
    mu = np.array([p[1] for p in pts])
    stable = np.abs(mu) < 1
    ax.axhspan(-1, 1, color="0.92", zorder=0)
    ax.plot(lt, mu, "-", color="0.4", lw=0.8, zorder=1)
    ax.plot(lt[stable], mu[stable], "o", ms=3.2, color="tab:blue", label="stable", zorder=2)
    ax.plot(lt[~stable], mu[~stable], "o", ms=3.2, mfc="none", color="tab:red", label="unstable", zorder=2)
    ax.set_yscale("symlog", linthresh=1.0)
    fold = load(E2E / "p2-tcont-a0.070-up" / "branch.json")["fold"]["fold"]
    ax.axvline(fold, color="tab:cyan", ls="--", lw=1)
    ax.text(fold, 30, " fold", color="tab:cyan", fontsize=7, va="top")
    flip = summary.get("q1", {}).get("p2_flip_a0.070")
    if flip:
        ax.axvline(flip, color="tab:purple", ls=":", lw=1)
        ax.text(flip, 30, " flip", color="tab:purple", fontsize=7, va="top")
    ax.set_xlabel(r"$\log_{10}T$ ($\alpha=0.070$)")
    ax.set_ylabel(r"leading multiplier of the 2-cycle")
    ax.set_ylim(-40, 3)
    # what competes at the two phases, at three temperatures
    for target in (-0.19, -0.5325, -1.1004):
        r = min(recs, key=lambda q: abs(q["lambda"] - target))
        ph0, ph1 = r["phases"]
        txt = (" / ".join(f"{t.strip() or repr(t)} {p:.2f}" for t, p in ph0["top"][:3]) + "\n" +
               " / ".join(f"{t.strip() or repr(t)} {p:.2f}" for t, p in ph1["top"][:2]))
        ax.annotate(txt, (r["lambda"], r["leading"][0]), fontsize=5.5, xytext=(0, -26 if r["leading"][0] > -2 else 14),
                    textcoords="offset points", ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
    ax.legend(fontsize=6, loc="lower left", frameon=False)
    ax.set_title("(a) the 2-cycle exists in a band of temperature", fontsize=8, loc="left")


def panel_b(ax, summary):
    d = np.load(SOFT / "pod64" / "map-stripes-256" / "scan.npz")
    lt, al, L, P = d["log10_temperature"], d["alpha"], d["lyapunov"], d["period"]
    i = int(np.argmin(np.abs(lt + 0.115)))
    row, per = L[i], P[i]
    chaos = (row > 1e-3) & (per <= 0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.plot(al[~chaos], row[~chaos], ".", ms=2.5, color="0.45", label="periodic")
    ax.plot(al[chaos], row[chaos], ".", ms=2.5, color="tab:red", label="chaotic")
    ac = 0.06279827
    ax.axvline(ac, color="tab:cyan", ls="--", lw=1)
    ax.text(ac, 0.45, r" $\alpha_c$", color="tab:cyan", fontsize=7)
    edges = load(SOFT / "pod64" / "edges" / "stripes-t-0.115" / "edges.json")
    lab = []
    if edges:
        for w in edges["windows"]:
            c = w["alpha_centre"]
            ax.text(c, -0.95, str(w["period"]), fontsize=5.5, ha="center", color="0.2")
            folds = [e["lambda"] for e in w["edges"] if e["kind"] == "fold"]
            if folds:
                lab.append((w["period"], folds[0]))
    ax.set_xlim(al[0], al[-1])
    ax.set_ylim(-1.05, 0.6)
    ax.set_xlabel(r"$\alpha$ ($\log_{10}T=%.3f$, row of the $256^2$ map)" % lt[i])
    ax.set_ylabel("largest Lyapunov exponent")
    ax.legend(fontsize=6, loc="upper right", frameon=False)
    ax.set_title("(b) beyond the fold: the ghost's stripes", fontsize=8, loc="left")
    if lab:
        ins = ax.inset_axes([0.55, 0.08, 0.4, 0.3])
        n = np.array([p for p, _ in lab])
        y = 1 / np.sqrt(np.array([a for _, a in lab]) - ac)
        ins.plot(n, y, "o", ms=2.5, color="k")
        ins.set_xlabel("period", fontsize=5)
        ins.set_ylabel(r"$1/\sqrt{\alpha_N-\alpha_c}$", fontsize=5)
        ins.tick_params(labelsize=5)
        summary.setdefault("q1", {})["stripe_fold_edges_t-0.115"] = lab


def panel_c(ax, summary):
    d = np.load(SOFT / "pod64" / "cross-512" / "scan.npz")
    lt, al, L, P = d["log10_temperature"], d["alpha"], d["lyapunov"], d["period"]
    chaos = (L > 1e-3) & (P <= 0)
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    import matplotlib.pyplot as plt
    cmap = ListedColormap(np.vstack([plt.get_cmap("Greys_r")(np.linspace(0.0, 1.0, 256)),
                                     plt.get_cmap("plasma")(np.linspace(0.15, 0.85, 256))]))
    val = np.where(chaos, np.clip(L, 1e-9, 0.5), np.clip(L, -1.0, 0.0))
    ax.imshow(val, origin="lower", extent=[al[0], al[-1], lt[0], lt[-1]], aspect="auto", interpolation="nearest",
              cmap=cmap, norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.5))
    c = load(SOFT / "pod64" / "curve-p2-fold-plus" / "curve.json")
    if c:
        ax.plot([p["alpha"] for p in c["points"]], [p["log10_temperature"] for p in c["points"]], color="tab:cyan",
                lw=1.2, label="2-cycle fold")
    marks = {"flip of the period-3 orbit": ("m", "^", "period-3 flip"), "flip": ("tab:orange", "v", "period-6 flip"),
             "fold": ("lime", "s", "period-6 fold")}
    seen = set()
    rows = []
    for cut in sorted(E2E.glob("band-cut-a*")) + sorted(E2E.glob("crescent-cut-t*")):
        j = load(cut / "cut.json")
        if not j:
            continue
        a = j["args"]
        for dname, v in j["directions"].items():
            b = v["boundary"]
            if b.get("lambda") is None:
                continue
            kind = b["type"]
            if j["args"]["period"] == 3 and kind == "flip":
                kind = "flip of the period-3 orbit"
            colour, m, label = marks.get(kind, ("w", "x", kind))
            x, y = (a["value"], b["lambda"]) if a["fixed"] == "alpha" else (b["lambda"], a["value"])
            ax.plot(x, y, m, color=colour, ms=4, mec="k", mew=0.4, label=None if label in seen else label)
            seen.add(label)
            rows.append({"cut": cut.name, "direction": dname, "type": b["type"], "alpha": x, "log10_temperature": y})
    cen = load(E2E / "census" / "census.json")
    if cen:
        cells = sorted({(r["alpha"], r["log10_temperature"]) for r in cen["rows"]})
        ax.plot([c[0] for c in cells], [c[1] for c in cells], "o", mfc="none", mec="w", ms=6, mew=1.0,
                label="census cells")
    ax.set_xlim(al[0], al[-1])
    ax.set_ylim(lt[0], lt[-1])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\log_{10}T$")
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.7)
    ax.set_title("(c) the band crosses the stripes", fontsize=8, loc="left")
    summary.setdefault("q2", {})["boundary_points"] = rows


def panel_d(ax1, ax2, summary):
    sp = load(E2E / "rank-spectra" / "spectra.json")
    if sp:
        names = []
        for k, cell in enumerate(sp["cells"]):
            r1 = [s["sigma_ratio_2_1"] for s in cell["steps"]]
            r2 = [s["DF_ratio_2_1"] for s in cell["steps"]]
            ax1.plot(np.full(len(r1), k) - 0.12, r1, "o", ms=2.5, color="tab:blue",
                     label=r"$\lambda_2/\lambda_1$ of $\Sigma_p$" if k == 0 else None)
            ax1.plot(np.full(len(r2), k) + 0.12, r2, "s", ms=2.5, color="tab:orange",
                     label=r"$\sigma_2/\sigma_1$ of $DF$" if k == 0 else None)
            ax1.plot(k, max(cell["composed_ratio_2_1"], 1e-12), "*", ms=6, color="k",
                     label="composed map" if k == 0 else None)
            names.append(cell["cell"])
        ax1.set_yscale("log")
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=35, fontsize=5.5, ha="right")
        ax1.set_ylabel("second / first", fontsize=7)
        ax1.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax1.set_title("(d) rank of the readout and of the loop", fontsize=8, loc="left")
    sec = load(E2E / "red-sec-crescent" / "section.json")
    if sec:
        pr = np.array(sec["pairs"])
        ax2.plot(pr[:, 0], pr[:, 1], ".", ms=1.5, color="k", label="full map (section)")
        lo, hi = pr.min(), pr.max()
        ax2.plot([lo, hi], [lo, hi], color="0.6", lw=0.6)
        a0, t0 = sec["args"]["alpha"], sec["args"]["log_t"]
        for tag, style, label in (("r1", "-", "R1: invariant curve"), ("r2", "--", "R2: two tokens")):
            f = E2E / f"red-{tag}-crescent" / f"{tag}-a{a0:.7f}-t{t0:.5f}.npz"
            if f.exists():
                m = np.load(f)
                s, s2 = m["s"], m["s2"]
                keep = (s > lo - 0.5) & (s < hi + 0.5)
                ax2.plot(s[keep], s2[keep], style, lw=1, label=label)
        ax2.set_xlim(lo - 0.3, hi + 0.3)
        ax2.set_ylim(lo - 0.3, hi + 0.3)
        ax2.set_xlabel(r"$s_n=(z_{\rm existence}-z_{\rm question})/T$", fontsize=7)
        ax2.set_ylabel(r"$s_{n+1}$", fontsize=7)
        ax2.legend(fontsize=5.5, frameon=False)
    ax2.set_title("(e) the crescent's return map", fontsize=8, loc="left")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "llm" / "figures" / "fig-e2e.png"))
    ap.add_argument("--summary", default=str(ROOT / "docs" / "e2e" / "e2e-summary.json"))
    args = ap.parse_args()
    summary = load(args.summary) or {}
    fig = plt.figure(figsize=(10.5, 7.6))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=0.9)
    panel_a(fig.add_subplot(gs[0, 0:3]), summary)
    panel_b(fig.add_subplot(gs[0, 3:6]), summary)
    panel_c(fig.add_subplot(gs[1, 0:2]), summary)
    panel_d(fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6]), summary)
    fig.savefig(args.out, dpi=250, bbox_inches="tight")
    side = Path(args.out).with_suffix(".json")
    side.write_text(json.dumps({"figure": Path(args.out).name, "inputs_root": str(em.DATA / "explore"),
                                "summary": summary}, indent=1, default=float))
    print("wrote", args.out, side)


if __name__ == "__main__":
    main()
