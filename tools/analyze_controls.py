"""Steering controls (run C1): chaotic share of soft-token maps along concept, norm-matched random and SAE feature
directions on the same grid, overall, by |alpha| and at the smallest |alpha|, with a side-by-side figure of the
largest Lyapunov exponent. A map given as BASE+TOP is merged from a timed-out scan's checkpoint (rows done, flat and
temperature-major in BASE/scan-partial.npz) and a completing scan of the remaining rows (TOP/scan.npz); the merge
checks that the grids join. Unfinished cells stay NaN and are reported, never counted as periodic or chaotic.

Usage: python tools/analyze_controls.py OUT_PREFIX LABEL=DIR[+TOPDIR] [LABEL=DIR[+TOPDIR] ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
BINS = ((0.0, 0.01), (0.01, 0.03), (0.03, 0.06), (0.06, 0.09), (0.09, 0.121))


def load(spec):
    import explore_llm_attractors as ela

    base = ela.DATA / "explore"
    if "+" not in spec:
        z = np.load(base / spec / "scan.npz")
        return z["lyapunov"].astype(float), z["period"], z["alpha"], z["log10_temperature"]
    a, b = spec.split("+")
    zp = np.load(base / a / "scan-partial.npz")
    lt, al = zp["log10_temperature"], zp["alpha"]
    n_t, n_a = len(lt), len(al)
    done = int(zp["done"])
    L = zp["lyapunov"].astype(float).reshape(n_t, n_a)
    P = zp["period"].reshape(n_t, n_a)
    rows_done = done // n_a
    L[rows_done:] = np.nan
    P[rows_done:] = 0
    zt = np.load(base / b / "scan.npz")
    lt_t, al_t = zt["log10_temperature"], zt["alpha"]
    if not (np.allclose(al_t, al, rtol=0, atol=1e-12) and np.allclose(lt_t, lt[n_t - len(lt_t):], rtol=0, atol=1e-12)):
        raise SystemExit(f"{spec}: the completing scan's grid does not match the missing rows")
    if n_t - len(lt_t) > rows_done:
        raise SystemExit(f"{spec}: rows {rows_done}..{n_t - len(lt_t) - 1} missing in both scans")
    L[n_t - len(lt_t):] = zt["lyapunov"]
    P[n_t - len(lt_t):] = zt["period"]
    return L, P, al, lt


def summarize(L, P, al):
    done = ~np.isnan(L)
    ch = (L > 1e-3) & (P <= 0) & done
    absa = np.abs(al)
    by = {f"{lo:.2f}-{hi:.2f}": float(ch[:, (absa >= lo) & (absa < hi)].sum() / max(done[:, (absa >= lo) &
                                                                                       (absa < hi)].sum(), 1))
          for lo, hi in BINS}
    j0 = int(np.argmin(absa))
    per = P[(P > 0) & done]
    return {"cells": int(L.size), "unfinished": int((~done).sum()), "chaotic_share": float(ch.sum() / done.sum()),
            "by_abs_alpha": by, "column_nearest_zero": {"alpha": float(al[j0]),
                                                       "chaotic": int(ch[:, j0].sum()), "of": int(done[:, j0].sum())},
            "periodic_share": float(((P > 0) & done).sum() / done.sum()),
            "median_period": int(np.median(per)) if per.size else None}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    res = {}
    fig, axes = plt.subplots(1, len(specs), figsize=(3.3 * len(specs), 3.4), squeeze=False)
    for ax, (label, spec) in zip(axes[0], specs):
        L, P, al, lt = load(spec)
        res[label] = dict(summarize(L, P, al), source=spec)
        da, dt = al[1] - al[0], lt[1] - lt[0]
        im = ax.imshow(np.clip(L, -0.5, 0.5), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                       extent=[al[0] - da / 2, al[-1] + da / 2, lt[0] - dt / 2, lt[-1] + dt / 2],
                       interpolation="nearest")
        ax.set_title(f"{label}\nchaotic {res[label]['chaotic_share']:.3f}", fontsize=9)
        ax.set_xlabel("alpha")
    axes[0, 0].set_ylabel("log10 temperature")
    fig.tight_layout()
    cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.02, pad=0.01, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    cb.ax.set_yticklabels([r"$\leq -0.5$", "$-0.25$", "0", "0.25", r"$\geq 0.5$"], fontsize=7)
    cb.set_label("largest Lyapunov exponent per step", fontsize=8)
    fig.savefig(out + ".png", dpi=250, bbox_inches="tight", pad_inches=0.05)    # keeps the colour bar's labels
    Path(out + ".json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:<14} chaotic {v['chaotic_share']:.3f} unfinished {v['unfinished']} by|a| "
              f"{[round(x, 3) for x in v['by_abs_alpha'].values()]} col0 {v['column_nearest_zero']['chaotic']}/"
              f"{v['column_nearest_zero']['of']}")


if __name__ == "__main__":
    main()
