"""Figure: the derivative factorization (paper eq. 3) against finite differences, at one state of the period-8 orbit
(written by tools/softmax_gain.py convergence). (a) Central differences (F(x + h u) - F(x - h u)) / (2h) converge to
the forward-mode derivative like h^2 until rounding takes over near h = 3e-6, where they bottom out at a few 1e-9.
(b) The factorization (1/T) Sigma_p J_h u has no step size: each direction's error against forward mode (near 1e-13)
is drawn beside that direction's best central difference, four orders of magnitude above it.

Usage: python tools/plot_derivative_check.py RESULT.json OUT.png
"""
import json
import sys

import numpy as np


def sci(x, sig, rounding):
    """x in LaTeX scientific notation with `sig` significant digits, rounded down, up (bounds) or to nearest (range ends)."""
    e = int(np.floor(np.log10(x)))
    m = x / 10.0 ** e * 10 ** (sig - 1)
    m = {"down": np.floor, "up": np.ceil, "nearest": np.round}[rounding](m) / 10 ** (sig - 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.{sig - 1}f}\times10^{{{e}}}$"


def plot_convergence(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.array(result["fd_steps"])
    fd = np.array(result["fd_relative_errors"])
    errs = np.array(result["relative_errors"])
    med = np.median(fd, axis=0)
    best = fd.min(axis=1)                                       # each direction's best central difference
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.4, 2.6), sharey=True,
                                 gridspec_kw={"width_ratios": [3.1, 1.25], "wspace": 0.06})
    for row in fd:
        ax.loglog(h, row, "-", color="0.55", lw=0.7)
    i3 = int(np.argmin(np.abs(h - 1e-3)))
    seg = h >= 2e-6                                             # truncation regime: error falls like h^2
    ax.loglog(h[seg], med[i3] * (h[seg] / h[i3]) ** 2, "--", color="black", lw=0.9)
    i8 = int(np.argmin(np.abs(h - 1e-8)))
    seg = h <= 5e-6                                             # rounding regime: error grows like 1/h
    ax.loglog(h[seg], med[i8] * (h[i8] / h[seg]), ":", color="black", lw=1.1)
    ax.text(2.2e-3, 2.2e-5, r"$\propto h^2$", fontsize=7, ha="left")
    ax.text(2.5e-10, 1.2e-8, r"$\propto 1/h$", fontsize=7, ha="left")
    ax.set_xlabel(r"finite-difference step $h$", fontsize=8)
    ax.set_ylabel(r"relative error against $DF(x)\,u$" "\n(forward mode)", fontsize=7.5)
    ax.set_xlim(h.min() / 1.6, h.max() * 1.6)                  # step increases to the right
    ax.set_ylim(errs.min() / 4, 40)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(a) central differences", fontsize=7.5, loc="left")
    # (b): no step axis; one point per direction for the best central difference and for the formula
    jitter = np.linspace(-0.12, 0.12, len(errs))
    bx.semilogy(0.0 + jitter, best, "o", ms=3.2, mfc="none", mec="0.35", mew=0.8)
    bx.semilogy(1.0 + jitter, errs, "o", ms=3.2, color="#d95f02")
    bx.annotate("", xy=(0.5, errs.max() * 1.3), xytext=(0.5, best.min() * 0.75),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.25"))
    bx.text(0.57, np.sqrt(best.min() * errs.max()), r"$\times 10^{4}$", fontsize=6.5, va="center", color="0.25")
    bx.set_xlim(-0.55, 1.55)
    bx.set_xticks([0.0, 1.0])
    bx.set_xticklabels(["best\ncentral\ndifference", "formula\n(3)"], fontsize=6.2)
    bx.tick_params(axis="y", labelsize=6.5)
    bx.set_title("(b) per direction", fontsize=7.5, loc="left")
    result["band_label"] = f"{sci(errs.min(), 2, 'down')} to {sci(errs.max(), 3, 'up')}"     # for the caption
    result["best_fd_label"] = f"{sci(best.min(), 2, 'nearest')} to {sci(best.max(), 2, 'nearest')}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    res = json.loads(open(sys.argv[1]).read())
    plot_convergence(res, sys.argv[2])
    print("formula:", res["band_label"], "best central differences:", res["best_fd_label"])
