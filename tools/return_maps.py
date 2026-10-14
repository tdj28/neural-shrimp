"""One-dimensional return maps of chaotic cells whose leading-token text repeats exactly (run H3).

For every cell of a tools/hidden_chaos.py run saved with --save-sequences that is chaotic (positive exponent at both
twin offsets, no state period) and has a periodic leading-token sequence (text period q), the probability of the
leading token at each phase k of the text is followed over the repetitions, p_k(n) -> p_k(n+1), and fitted by a
polynomial (degree 8, lower when there are few repetitions). The phase with the smallest residual is the best
one-dimensional coordinate. Its fitted map gives a Lyapunov exponent per repetition (mean log |f'| along the data),
which divided by q is compared with the full map's twin exponent per step (mean of the two offsets).

Usage: python tools/return_maps.py RUN_DIR OUT.png      (RUN_DIR explore-relative; writes return-maps.json there)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MIN_REPETITIONS = 30


def fit(pk, degree):
    x, y = pk[:-1], pk[1:]
    lo, rng = x.min(), x.max() - x.min()
    if rng <= 0 or y.max() - y.min() <= 0:
        return None
    u = (x - lo) / rng
    c = np.polynomial.polynomial.polyfit(u, y, degree)
    res = y - np.polynomial.polynomial.polyval(u, c)
    dc = np.polynomial.polynomial.polyder(c)
    fprime = np.polynomial.polynomial.polyval(u, dc) / rng
    crit = [r.real for r in np.polynomial.polynomial.polyroots(dc) if abs(r.imag) < 1e-9 and 0 < r.real < 1]
    return {"residual_rel": float(np.sqrt(np.mean(res ** 2)) / (y.max() - y.min())),
            "exponent_per_repetition": float(np.mean(np.log(np.abs(fprime) + 1e-300))),
            "critical_points_inside": len(crit), "degree": degree, "points": int(len(x))}


def main():
    import hidden_chaos as hc
    import explore_llm_attractors as ela

    run, out_png = sys.argv[1], sys.argv[2]
    base = ela.DATA / "explore" / run
    cells = json.loads((base / "hidden-chaos.json").read_text())["cells"]
    seq = np.load(base / "sequences.npz")
    lead, pmax = seq["lead"], seq["pmax"]
    if lead.shape[1] != len(cells):
        raise SystemExit("sequences and cells differ in number")
    rows = []
    for c, info in enumerate(cells):
        cls = hc.classify(info)
        q = hc.symbolic_period(lead[:, c].tolist())
        row = {"log10_temperature": info["log10_temperature"], "alpha": info["alpha"], "class": cls,
               "lyapunov": info["lyapunov"], "text_period": q}
        if cls == "hidden chaos" and q > 0:
            phase = np.arange(lead.shape[0]) % q
            reps = int((phase == 0).sum())
            row["repetitions"] = reps
            if reps >= MIN_REPETITIONS:
                degree = int(min(8, max(2, (reps - 1) // 6)))
                fits = [(k, fit(pmax[phase == k, c], degree)) for k in range(q)]
                fits = [(k, f) for k, f in fits if f is not None]
                if fits:
                    k, best = min(fits, key=lambda kf: kf[1]["residual_rel"])
                    full = float(np.mean(info["lyapunov"]))
                    one = best["exponent_per_repetition"] / q
                    row.update({"best_phase": k, **best, "exponent_1d_per_step": one, "exponent_full_per_step": full,
                                "relative_discrepancy": (one - full) / full})
        rows.append(row)
    fitted = [r for r in rows if "relative_discrepancy" in r]
    single = [r for r in fitted if r["residual_rel"] < 1e-2]
    summary = {"cells": len(rows), "classes": {k: sum(r["class"] == k for r in rows) for k in
                                               sorted(set(r["class"] for r in rows))},
               "repeating_text_chaotic": sum(r["class"] == "hidden chaos" for r in rows),
               "fitted (>= %d repetitions)" % MIN_REPETITIONS: len(fitted),
               "single_valued (residual < 1e-2)": len(single),
               "single_valued_with_one_critical_point": sum(r["critical_points_inside"] == 1 for r in single),
               "median_abs_discrepancy_single_valued": float(np.median([abs(r["relative_discrepancy"]) for r in single]))
               if single else None,
               "within_10_percent_single_valued": sum(abs(r["relative_discrepancy"]) <= 0.10 for r in single),
               "text_periods_fitted": sorted(set(r["text_period"] for r in fitted))}
    (base / "return-maps.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    if fitted:
        full = np.array([r["exponent_full_per_step"] for r in fitted])
        one = np.array([r["exponent_1d_per_step"] for r in fitted])
        good = np.array([r["residual_rel"] < 1e-2 for r in fitted])
        qq = np.array([r["text_period"] for r in fitted])
        sc = ax.scatter(full[good], one[good], c=np.log2(qq[good] / 7), cmap="viridis", s=18,
                        label="single-valued return map")
        ax.scatter(full[~good], one[~good], facecolors="none", edgecolors="0.5", s=18,
                   label="no single-valued phase")
        lim = [0, max(full.max(), one.max()) * 1.08]
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046)
        cb.set_label("log2(text period / 7)")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_xlabel("largest exponent of the full map (per step)")
    ax.set_ylabel("exponent of the fitted 1D return map (per step)")
    ax.set_title("Repeating-text chaos: one-dimensional reduction", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)


if __name__ == "__main__":
    main()
