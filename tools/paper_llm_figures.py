"""Quantitative figures for the LLM-dynamics paper draft (paper/llm/figures), from saved run outputs.

fig-passages.png: bottleneck passage times through the 2-cycle's fold, measured (controlled entrances and
    natural laminar phases) against the float64 normal-form predictions, versus eps = alpha - alpha_c.
fig-window-law.png: stripe-centre distances d_N from the fold against the lap count N, with the full- and
    half-passage predictions from the normal form.
fig-bridge.png: mean laminar length against eps for the exact map and for m sampled embeddings.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bifurcation_curves as bc  # noqa: E402
import explore_llm_attractors as ela  # noqa: E402

SOFT = ela.DATA / "explore" / "llm-soft-loop"
OUT = ROOT / "paper" / "llm" / "figures"


def passages(plt):
    rec = json.loads((SOFT / "passage-p2-t-0.115" / "passage.json").read_text())
    nf = rec["normal_form"]
    a, b = abs(nf["a"]), abs(nf["b"])
    U = rec["U"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    eps = np.logspace(np.log10(2e-6), np.log10(6e-4), 100)
    ax.plot(eps, [bc.passage_count(-U, a, b, e, U) for e in eps], "-", color="0.3", lw=1,
            label="normal form, full channel")
    ax.plot(eps, [bc.passage_count(0.0, a, b, e, U) for e in eps], "--", color="0.5", lw=1,
            label="normal form, from the midpoint")
    marks = {"full": ("o", "C0"), "midpoint": ("s", "C1"), "full+transverse": ("^", "C2")}
    for kind, (m, c) in marks.items():
        rows = [r for r in rec["rows"] if r["entrance"] == kind and r["measured_G"]]
        ax.plot([r["eps"] for r in rows], [r["measured_G"] for r in rows], m, color=c, ms=6, mfc="none",
                label={"full": "measured, full entrance", "midpoint": "measured, midpoint entrance",
                       "full+transverse": "measured, full + transverse kick"}[kind])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"distance beyond the fold $\varepsilon=\alpha-\alpha_c$")
    ax.set_ylabel("two-step iterates to cross the bottleneck")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig-passages.png", dpi=200)


def window_law(plt):
    nf = json.loads((SOFT / "normal-form-p2-t-0.115" / "normal-form.json").read_text())
    ab = abs(nf["ab"])
    alpha_c = 0.06279827
    centres = {6: 0.077872, 8: 0.071674, 10: 0.067766, 12: 0.065550, 14: 0.064443, 16: 0.063857, 18: 0.063534,
               20: 0.063335}                                      # stripe centres of the float32 cut (06e24)
    N = np.array([p // 2 - 3 for p in centres])                   # laps of the 'is it' channel: k - 2, period 2k+2
    d = np.array([centres[p] - alpha_c for p in centres])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(N, 1 / np.sqrt(d), "o", color="C0", label="stripe centers (fine cut)")
    n = np.linspace(1, 8, 50)
    ax.plot(n, np.sqrt(ab) * n / np.pi, "-", color="0.3", lw=1, label=r"full passage $\sqrt{ab}\,N/\pi$")
    ax.plot(n, 2 * np.sqrt(ab) * n / np.pi, "--", color="0.5", lw=1, label=r"half passage $2\sqrt{ab}\,N/\pi$")
    fit = np.polyfit(N[N >= 4], 1 / np.sqrt(d[N >= 4]), 1)
    ax.plot(n, np.polyval(fit, n), ":", color="C3", lw=1,
            label=f"fit N>=4: slope {fit[0]:.2f}, intercept {fit[1]:.2f}")
    ax.set_xlabel(r"laps $N$ of the 'is it' channel (period $2N+6$)")
    ax.set_ylabel(r"$1/\sqrt{\alpha_N-\alpha_c}$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig-window-law.png", dpi=200)


def bridge(plt):
    alpha_c = 0.0627983
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for name, label, mark in (("exact", "exact expected embedding", "o"), ("m64", "mean of 64 samples", "s"),
                              ("m8", "mean of 8 samples", "^"), ("m1", "single samples (m = 1)", "v")):
        w = np.load(SOFT / f"bridge-{name}-t-0.115" / "walk.npz")
        P, al, toks = w["probabilities"], w["alphas"], list(w["tokens"])
        i_it, i_a, i_there, i_is = (toks.index(t) for t in (" it", " a", " there", " is"))
        res = {}
        for b_, a in enumerate(al):
            p = P[:, b_, :]
            lam = (p[:, i_is] > 0.5) | ((p[:, i_it] > p[:, i_a]) & (p[:, i_it] > p[:, i_there]) & (p[:, i_it] > 0.1))
            runs, n = [], 0
            for v in lam:
                if v:
                    n += 1
                else:
                    if n >= 6:
                        runs.append(n)
                    n = 0
            res.setdefault(round(a - alpha_c, 7), []).extend(runs)
        e = [k for k in sorted(res) if res[k]]
        ax.plot(e, [np.mean(res[k]) for k in e], mark + "-", ms=5, mfc="none", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon=\alpha-\alpha_c$")
    ax.set_ylabel("mean laminar length (steps, phases of 6 or more)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig-bridge.png", dpi=200)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    passages(plt)
    window_law(plt)
    bridge(plt)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
