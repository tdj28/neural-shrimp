"""Figure fig-fold: the fold of the 2-cycle "is it" at log10 T = -0.115 (Qwen2.5-0.5B-Instruct, philosophy direction at
layer 12), the saddle-node bifurcation behind the stripes (human request: "show us the fold").

(a) The 2-cycle and its saddle meet at the fold: one point of each orbit along the continued branch
    (branch-p2-t-0.115-alpha-up), in the fold's critical coordinate u = l.(x - x_c) (normal-form-p2-t-0.115, l.r = 1),
    with the normal form's fixed points u* = +-sqrt(|a| (alpha_c - alpha) / |b|).
(b) The largest multiplier of the two orbits passes through +1 at the fold; normal form 1 -+ 2 sqrt(ab (alpha_c - alpha)).
(c) Beyond the fold the orbit is gone but its ghost remains: controlled passages through the bottleneck from the
    channel's entrance (passage-p2-t-0.115), oriented coordinate v = sign(b) u in units of U = 0.1/|b|, measured on the
    network (one marker shape per eps) and iterated from the normal form v' = v + |a| eps + |b| v^2 (lines).
(d) Free trajectories alternate between laminar phases in the bottleneck (|u| < U, shaded) and bursts
    (passage-natural-p2-t-0.115, one trajectory at eps = 1e-5).

Usage: python tools/plot_fold.py [OUT.png] [OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
ALPHA_C = 0.06279827217667086          # float64 fold (normal-form-p2-t-0.115 source)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    out_png = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-fold.png")
    out_json = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "paper/llm/figures/fig-fold.json")
    D = ela.DATA / "explore" / "llm-soft-loop"
    nf = json.loads((D / "normal-form-p2-t-0.115" / "normal-form.json").read_text())
    a, b = float(nf["a"]), float(nf["b"])
    vec = np.load(D / "normal-form-p2-t-0.115" / "vectors.npz")
    x_c, l, r = vec["x_c"], vec["l"], vec["r"]
    br = json.loads((D / "branch-p2-t-0.115-alpha-up" / "branch.json").read_text())
    states = np.load(D / "branch-p2-t-0.115-alpha-up" / "states.npy")[:, 0, :].astype(np.float64)
    alpha = np.array([p["lambda"] for p in br["points"]])
    u = (states - x_c) @ l / float(l @ r)
    turn = next(i for i, p in enumerate(br["points"]) if p["turned_since_previous"])
    stable = np.arange(len(alpha)) < turn
    lead = [p["multipliers"][0] for p in br["points"]]
    mu = np.array([np.hypot(*m) if isinstance(m, list) else abs(m) for m in lead])
    cplx = np.array([isinstance(m, list) and abs(m[1]) > 0 for m in lead])
    summary = {"a": a, "b": b, "alpha_c": ALPHA_C, "branch_points": int(len(alpha)), "turn_index": turn}

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    # (a) branch diagram
    ax = axes[0, 0]
    k = 1e3
    ax.plot(alpha[stable], k * u[stable], "o-", color="black", ms=3, lw=1.0, label="2-cycle (attracting)")
    ax.plot(alpha[~stable], k * u[~stable], "o--", color="black", mfc="white", ms=3, lw=1.0,
            label="saddle 2-cycle")
    ag = np.linspace(ALPHA_C - 1.2e-3, ALPHA_C, 300)                 # the normal form is local: drawn near the fold
    ust = np.sqrt(np.abs(a) * (ALPHA_C - ag) / np.abs(b))
    ax.plot(ag, k * ust, "-", color="#d95f02", lw=0.9, alpha=0.9, label="normal form")
    ax.plot(ag, -k * ust, "-", color="#d95f02", lw=0.9, alpha=0.9)
    ax.axvline(ALPHA_C, color="0.4", lw=0.7, ls=":")
    ax.axvspan(ALPHA_C, 0.0632, color="0.92", lw=0)
    ax.text(0.06301, -5.5, "no 2-cycle: bottleneck", fontsize=6.3, ha="center", va="center", color="0.3", rotation=90)
    ax.set_xlim(0.0600, 0.0632)
    ax.set_ylim(-27, 16)
    ax.set_xlabel(r"steering strength $\alpha$", fontsize=7.5)
    ax.set_ylabel(r"critical coordinate $u$ ($10^{-3}$)", fontsize=7.5)
    ax.set_title("(a) the 2-cycle and its saddle meet at the fold", fontsize=8)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.45), frameon=False)
    # (b) multipliers
    ax = axes[0, 1]
    ax.plot(alpha[stable & ~cplx], mu[stable & ~cplx], "o", ms=3, color="black", label="attracting")
    ax.plot(alpha[stable & cplx], mu[stable & cplx], "o", ms=3, color="black", mfc="white")     # complex leading pair
    ax.plot(alpha[~stable], mu[~stable], "s", ms=3, color="black", mfc="white", label="saddle")
    ag = np.linspace(ALPHA_C - 1.2e-3, ALPHA_C, 300)
    d = 2 * np.sqrt(a * b * (ALPHA_C - ag))
    ax.plot(ag, 1 - d, "-", color="#d95f02", lw=0.9, label="normal form")
    ax.plot(ag, 1 + d, "-", color="#d95f02", lw=0.9)
    ax.axhline(1.0, color="0.4", lw=0.7, ls=":")
    ax.axvline(ALPHA_C, color="0.4", lw=0.7, ls=":")
    ax.set_xlim(0.0600, 0.0632)
    ax.set_ylim(0, 2.6)
    ax.set_xlabel(r"steering strength $\alpha$", fontsize=7.5)
    ax.set_ylabel(r"largest multiplier $|\mu_1|$ of $F^2$", fontsize=7.5)
    ax.set_title(r"(b) the multiplier passes through $+1$", fontsize=8)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False)
    # (c) controlled passages through the ghost
    ax = axes[1, 0]
    pas = json.loads((D / "passage-p2-t-0.115" / "passage.json").read_text())
    V = np.load(D / "passage-p2-t-0.115" / "u.npy")
    U = float(pas["U"])
    colors = {3e-6: "#1b9e77", 1e-5: "#d95f02", 3e-5: "#7570b3", 1e-4: "#e7298a"}
    markers = {3e-6: "o", 1e-5: "s", 3e-5: "^", 1e-4: "D"}       # one marker per eps: told apart without colour
    summary["passages"] = {}
    for i, row in enumerate(pas["rows"]):
        e = float(row["eps"])
        if row["entrance"] != "full" or e not in colors:
            continue
        v = np.concatenate([[row["u_in"]], V[:, i]])
        n_exit = int(np.argmax(v > U))
        vv = [row["u_in"]]                                      # normal form from the same entrance
        while vv[-1] <= U and len(vv) < 600:
            vv.append(vv[-1] + abs(a) * e + abs(b) * vv[-1] ** 2)
        ax.plot(np.arange(len(vv)), np.array(vv) / U, "-", color=colors[e], lw=0.9)
        ax.plot(np.arange(n_exit + 1), v[:n_exit + 1] / U, markers[e], color=colors[e], ms=3.0, mew=0, ls="none",
                label=rf"$\varepsilon={e:.0e}$: {n_exit} iterates".replace("e-0", r"\times10^{-").replace("$:", "}$:"))
        summary["passages"][str(e)] = {"measured": n_exit, "normal_form": len(vv) - 1}
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylim(-1.05, 1.1)
    ax.set_xlim(-3, 150)
    ax.set_xlabel(r"two-step iterates $n$ (of $G=F^2$)", fontsize=7.5)
    ax.set_ylabel(r"$v_n/U$ (oriented $u$)", fontsize=7.5)
    ax.set_title(r"(c) beyond the fold ($\varepsilon=\alpha-\alpha_c>0$): the ghost", fontsize=8)
    ax.legend(fontsize=5.6, loc="lower right", frameon=False, markerscale=2.5)
    # (d) a free trajectory: laminar phases and bursts
    ax = axes[1, 1]
    nat = json.loads((D / "passage-natural-p2-t-0.115" / "passage.json").read_text())
    UN = np.load(D / "passage-natural-p2-t-0.115" / "u.npy")
    pick = next(i for i, row in enumerate(nat["rows"]) if float(row["eps"]) == 1e-5 and row["entrance"] == "natural-3")
    un = UN[:600, pick]
    ax.semilogy(np.arange(len(un)), np.maximum(np.abs(un), 1e-6), "-", color="black", lw=0.6)
    ax.axhspan(1e-6, U, color="#fde0c5", lw=0)
    # label in the empty stretch of the band between the spike near n = 120 and the descent near n = 237
    ax.text(180, U * 0.45, r"bottleneck" + "\n" + r"$|u|<U$", fontsize=5.8, ha="center", va="top", color="#a6430a")
    inside = np.abs(UN[:, pick]) < U
    runs, n = [], 0
    for x in inside:
        if x:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    summary["natural"] = {"eps": 1e-5, "row": pick, "laminar_runs": runs, "fraction_in_bottleneck": float(inside.mean())}
    ax.set_ylim(1e-5, 3)
    ax.set_xlim(0, 600)
    ax.set_xlabel(r"two-step iterates $n$", fontsize=7.5)
    ax.set_ylabel(r"$|u_n|$", fontsize=7.5)
    ax.set_title(r"(d) a free orbit at $\varepsilon=10^{-5}$: laminar phases and bursts", fontsize=8)
    for axx in axes.ravel():
        axx.tick_params(labelsize=6.5)
    for axx in axes[0]:
        axx.ticklabel_format(axis="x", useOffset=False, style="plain")
        axx.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "natural"}, indent=1), summary["natural"]["laminar_runs"])


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    main()
