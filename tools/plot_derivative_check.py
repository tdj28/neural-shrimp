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
    h = np.array(result["fd_steps"])
    fd = np.array(result["fd_relative_errors"])
    errs = np.array(result["relative_errors"])
    med = np.median(fd, axis=0)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k, row in enumerate(fd):
        ax.loglog(h, row, "-", color="0.6", lw=0.7)
    i3 = int(np.argmin(np.abs(h - 1e-3)))
    seg = h >= 2e-6                                             # truncation regime
    ax.loglog(h[seg], med[i3] * (h[seg] / h[i3]) ** 2, "--", color="black", lw=0.8)
    i8 = int(np.argmin(np.abs(h - 1e-8)))
    seg = h <= 5e-6                                             # rounding regime
    ax.loglog(h[seg], med[i8] * (h[i8] / h[seg]), ":", color="black", lw=0.9)
    ax.fill_between([h.min(), h.max()], errs.min(), errs.max(), color="#d95f02", alpha=0.45, lw=0)
    ax.text(np.sqrt(h.min() * h.max()), errs.max() * 2.2,
            f"formula (3), {len(errs)} directions", fontsize=6.5, ha="center", va="bottom", color="#a6430a")
    result["band_label"] = f"{sci(errs.min(), 2, 'down')} to {sci(errs.max(), 3, 'up')}"     # for the caption
    ax.set_xlabel(r"finite-difference step $h$", fontsize=8)
    ax.set_ylabel("relative error against\nforward-mode derivative", fontsize=8)
    ax.set_ylim(errs.min() / 4, 4)
    ax.set_xlim(h.max() * 1.6, h.min() / 1.6)                  # decreasing step to the right
    # direct labels instead of a legend (gray lines: central differences, one per direction; see the caption)
    ax.text(1.2e-3, 2e-7, "truncation\n" + r"$\propto h^2$", fontsize=6.5, ha="center", va="top")
    ax.text(3e-9, 4e-10, "rounding\n" + r"$\propto 1/h$", fontsize=6.5, ha="center", va="top")
    ax.text(6e-5, 3e-2, "central differences", fontsize=6.5, ha="center", va="center", color="0.4")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    res = json.loads(open(sys.argv[1]).read())
    plot_convergence(res, sys.argv[2])
    print("band:", res["band_label"])
