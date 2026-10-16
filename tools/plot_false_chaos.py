"""Figure fig-false-chaos: false chaos and real chaos in a twin test (run FC, tools/false_chaos_demo.py).

Qwen2.5-0.5B-Instruct in float64, paper's prompt, philosophy steering at block 12: a chaotic cell and two contracting
cells (a weakly contracting period-3 loop, a strongly contracting orbit). The twin exponent is the mean log growth of
a twin renormalized to a relative offset eps after every step.
(a) Chaotic cell: clean float64 and Gaussian noise of relative size sigma in both rows, against the offset; dashed,
    the one-leading-direction prediction max(lambda, log(sqrt(2) sigma / eps)), with lambda the clean exponent.
(b) The two contracting cells, same conditions: noise turns a contracting orbit "chaotic" once eps < sqrt(2) sigma.
(c) The float32 final normalization of the early two-GPU runs, no added noise: false exponents at offsets near
    float32 resolution, exact merging of twin and base below it (open markers: more than half the steps merged),
    and the clean values from 1e-7 up.
(d) A free twin started at 1e-12 and never renormalized: real chaos grows exponentially until it saturates;
    a contracting orbit decays; under noise of 1e-8 the same contracting orbit's twin jumps to the noise level and
    stays there instead of growing.

Usage: python tools/plot_false_chaos.py [RUN] [OUT.png] [OUT.json]   (RUN explore-relative, default false-chaos/fc1)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
NOISE = ["noise-1e-10", "noise-1e-08", "noise-1e-06"]
NOISE_COLORS = {"noise-1e-10": "#2c7fb8", "noise-1e-08": "#d95f02", "noise-1e-06": "#7570b3"}
CELL_COLORS = ["#c0392b", "#1b7837", "#2166ac"]


def prediction(lam, sigma, eps):
    return np.maximum(lam, np.log(np.sqrt(2) * sigma / np.asarray(eps)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    run = sys.argv[1] if len(sys.argv) > 1 else "false-chaos/fc1"
    out_png = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "paper/llm/figures/fig-false-chaos.png")
    out_json = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "paper/llm/figures/fig-false-chaos.json")
    d = json.loads((ela.DATA / "explore" / run / "false-chaos.json").read_text())
    eps = np.array(d["eps"])
    C = d["conditions"]
    missing = [c for c in ["clean", *NOISE, "float32-final-norm"] if c not in C]
    if missing:
        raise SystemExit(f"conditions missing from {run}: {missing}")
    clean = np.array(C["clean"]["exponent"])
    lam = clean.mean(axis=1)                                     # clean exponent per cell, mean over the offsets
    labels = [c["label"] for c in d["cells"]]
    summary = {"run": run, "clean_exponent_per_cell": lam.tolist(),
               "clean_spread_over_offsets": (clean.max(axis=1) - clean.min(axis=1)).tolist(), "prediction": {}}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    # (a) chaotic cell under noise
    ax = axes[0, 0]
    ax.semilogx(eps, clean[0], "o-", color="black", ms=3.5, lw=1.2, label="clean float64")
    for cond in NOISE:
        s = C[cond]["sigma"]
        y = np.array(C[cond]["exponent"])[0]
        ax.semilogx(eps, y, "o", color=NOISE_COLORS[cond], ms=3.5, label=rf"noise $\sigma=10^{{{int(np.log10(s))}}}$")
        ax.semilogx(eps, prediction(lam[0], s, eps), "--", color=NOISE_COLORS[cond], lw=0.9)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("(a) chaotic orbit", fontsize=9, loc="left")
    ax.set_xlabel(r"twin offset $\varepsilon$", fontsize=8)
    ax.set_ylabel("measured exponent (per step)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    # (b) contracting cells under noise
    ax = axes[0, 1]
    for k in (1, 2):
        ax.semilogx(eps, clean[k], "o-", color="black", ms=3.0, lw=1.0)
        ax.text(eps[-1] * 1.4, clean[k][-1], labels[k].split(",")[0], fontsize=6.5, va="center")
        for cond in NOISE:
            s = C[cond]["sigma"]
            ax.semilogx(eps, np.array(C[cond]["exponent"])[k], "o", color=NOISE_COLORS[cond], ms=3.0)
            ax.semilogx(eps, prediction(lam[k], s, eps), "--", color=NOISE_COLORS[cond], lw=0.8)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlim(eps[0] / 2, eps[-1] * 30)
    ax.set_title("(b) contracting orbits, same noise", fontsize=9, loc="left")
    ax.set_xlabel(r"twin offset $\varepsilon$", fontsize=8)
    # (c) float32 final normalization
    ax = axes[1, 0]
    f32 = C["float32-final-norm"]
    y32 = np.array(f32["exponent"], dtype=float)
    merged = np.array(f32["merges"]) / np.maximum(np.array(f32["merges"]) + np.array(f32["counted"]), 1)
    for k in range(len(labels)):
        ok = merged[k] <= 0.5
        ax.semilogx(eps[ok], y32[k][ok], "o-", color=CELL_COLORS[k], ms=3.5, lw=1.0, label=labels[k])
        if (~ok).any():
            ax.semilogx(eps[~ok], np.nan_to_num(y32[k][~ok], nan=lam[k]), "o", mfc="white", color=CELL_COLORS[k],
                        ms=3.5)
        ax.axhline(lam[k], color=CELL_COLORS[k], lw=0.6, ls=":")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("(c) float32 final normalization", fontsize=9, loc="left")
    ax.set_xlabel(r"twin offset $\varepsilon$", fontsize=8)
    ax.set_ylabel("measured exponent (per step)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    # (d) free twins
    ax = axes[1, 1]
    steps = np.arange(1, len(C["clean"]["free_log10_distance"][0]) + 1)
    series = [("clean", 0, CELL_COLORS[0], "-", "chaotic, clean"),
              ("clean", 1, CELL_COLORS[1], "-", "period 3, clean"),
              ("noise-1e-08", 1, NOISE_COLORS["noise-1e-08"], "-", r"period 3, noise $10^{-8}$")]
    for cond, k, col, ls, lab in series:
        y = np.array([np.nan if v is None else v for v in C[cond]["free_log10_distance"][k]], dtype=float)
        ax.plot(steps, y, ls, color=col, lw=1.0, label=lab)
    ax.set_ylim(-17, 1)
    ax.set_title(r"(d) free twin from $10^{-12}$", fontsize=9, loc="left")
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel(r"$\log_{10}$ relative distance", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    for a in axes.flat:
        a.tick_params(labelsize=7)

    for cond in NOISE:
        s = C[cond]["sigma"]
        y = np.array(C[cond]["exponent"])
        pred = np.array([prediction(lam[k], s, eps) for k in range(len(labels))])
        summary["prediction"][cond] = {"sigma": s, "max_abs_error": float(np.nanmax(np.abs(y - pred))),
                                       "max_abs_error_per_cell": np.nanmax(np.abs(y - pred), axis=1).tolist()}
    summary["float32_final_norm"] = {"exponent": y32.tolist(), "merged_fraction": merged.tolist()}
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["prediction"], indent=1))


if __name__ == "__main__":
    main()
