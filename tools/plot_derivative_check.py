"""Figure: the derivative factorization (paper eq. 3) against finite differences, at one state of the period-8 orbit
(written by tools/softmax_gain.py convergence). Central differences (F(x + h u) - F(x - h u)) / (2h) converge to the
forward-mode derivative like h^2 until rounding takes over near h = 3e-6, where they bottom out at a few 1e-9; the
factorization (1/T) Sigma_p J_h u agrees with forward mode near 1e-13 in every direction, four orders of magnitude
below the best finite difference.

Usage: python tools/plot_derivative_check.py RESULT.json OUT.png
"""
import json
import sys

import numpy as np


def sci(x, sig, rounding):
    """x in LaTeX scientific notation with `sig` significant digits, rounded down or up (range ends, bounds)."""
    e = int(np.floor(np.log10(x)))
    m = x / 10.0 ** e * 10 ** (sig - 1)
    m = (np.floor(m) if rounding == "down" else np.ceil(m)) / 10 ** (sig - 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return rf"${m:.{sig - 1}f}\times10^{{{e}}}$"


def plot_convergence(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = np.array(result["fd_steps"])
    fd = np.array(result["fd_relative_errors"])
    errs = np.array(result["relative_errors"])
    med = np.median(fd, axis=0)
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    for row in fd:
        ax.loglog(h, row, "-", color="0.6", lw=0.7)
    i3 = int(np.argmin(np.abs(h - 1e-3)))
    seg = h >= 2e-6                                             # truncation regime: error falls like h^2
    ax.loglog(h[seg], med[i3] * (h[seg] / h[i3]) ** 2, "--", color="black", lw=0.9)
    i8 = int(np.argmin(np.abs(h - 1e-8)))
    seg = h <= 5e-6                                             # rounding regime: error grows like 1/h
    ax.loglog(h[seg], med[i8] * (h[i8] / h[seg]), ":", color="black", lw=1.1)
    ax.fill_between([h.min(), h.max()], errs.min(), errs.max(), color="#d95f02", alpha=0.45, lw=0)
    result["band_label"] = f"{sci(errs.min(), 2, 'down')} to {sci(errs.max(), 3, 'up')}"     # for the caption
    # the gap between the best finite difference and the formula
    j = int(np.argmin(med))
    best = fd[:, j].min()
    ax.annotate("", xy=(h[j], errs.max() * 1.15), xytext=(h[j], best * 0.85),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.25"))
    ax.text(h[j] * 1.8, np.sqrt(best * errs.max()), "four orders\nof magnitude", fontsize=6.3, va="center",
            ha="left", color="0.25")
    ax.set_xlabel(r"finite-difference step $h$", fontsize=8)
    ax.set_ylabel("relative error against the exact\n(forward-mode) derivative", fontsize=7.5)
    ax.set_ylim(errs.min() / 4, 40)
    ax.set_xlim(h.min() / 1.6, h.max() * 1.6)                  # step increases to the right
    handles = [Line2D([], [], color="0.6", lw=0.9, label="central differences, 8 random directions"),
               Line2D([], [], color="black", ls="--", lw=0.9, label=r"truncation error $\propto h^2$ (slope 2)"),
               Line2D([], [], color="black", ls=":", lw=1.1, label=r"rounding error $\propto 1/h$ (slope $-1$)"),
               Patch(color="#d95f02", alpha=0.45, lw=0, label="formula (3), same 8 directions")]
    ax.legend(handles=handles, fontsize=5.6, loc="upper center", frameon=False, handlelength=2.2,
              borderaxespad=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    res = json.loads(open(sys.argv[1]).read())
    plot_convergence(res, sys.argv[2])
    print("band:", res["band_label"])
