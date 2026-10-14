"""One-dimensional return maps of chaotic cells whose leading-token text repeats exactly (run H3).

For every cell of a tools/hidden_chaos.py run saved with --save-sequences that is chaotic (positive exponent at both
twin offsets, no state period) and has a periodic leading-token sequence (text period q), the probability of the
leading token at each phase k of the text is followed over the repetitions, p_k(n) -> p_k(n+1), and fitted by a
polynomial (degree 8, lower when there are few repetitions). The phase with the smallest residual is the best
one-dimensional coordinate. Its fitted map gives a Lyapunov exponent per repetition (mean log |f'| along the data),
which divided by q is compared with the full map's twin exponent per step (mean of the two offsets).

Amended method (14 Oct, research log): Chebyshev fits whose degree is chosen by 5-fold cross-validation from
{4, 6, ..., 24} (capped at (repetitions - 1)/6); a phase is single valued when its cross-validated residual is below
1e-2 of the range; the cell's exponent is the median over its single-valued phases (exponents are invariant under
smooth conjugacy, so all single-valued phases must agree), with their spread. The pre-specified analysis (degree 8,
minimum-residual phase) is kept in the output as "prespecified".

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


DEGREES = tuple(range(4, 25, 2))
CV_FOLDS = 5
SINGLE_VALUED = 1e-2


def cv_fit(pk, degrees=DEGREES, folds=CV_FOLDS, seed=0):
    """Chebyshev fit of the return map p(n) -> p(n+1) with the degree chosen by k-fold cross-validation; returns the
    degree, the cross-validated residual relative to the range, the exponent per repetition (mean log |f'| along the
    data, refitted on all points) and the number of interior critical points."""
    from numpy.polynomial import Chebyshev

    x, y = pk[:-1], pk[1:]
    span = y.max() - y.min()
    if x.max() - x.min() <= 0 or span <= 0:
        return None
    cap = max(degrees[0], (len(x) - 1) // 6)
    cands = [g for g in degrees if g <= cap] or [degrees[0]]
    order = np.random.default_rng(seed).permutation(len(x))
    parts = np.array_split(order, folds)
    best = None
    for g in cands:
        err = []
        for k in range(folds):
            test = parts[k]
            train = np.concatenate([parts[j] for j in range(folds) if j != k])
            c = Chebyshev.fit(x[train], y[train], g)
            err.append(np.mean((c(x[test]) - y[test]) ** 2))
        cv = float(np.sqrt(np.mean(err)) / span)
        if best is None or cv < best[1]:
            best = (g, cv)
    g, cv = best
    c = Chebyshev.fit(x, y, g)
    d = c.deriv()
    lo, hi = x.min(), x.max()
    crit = [r.real for r in d.roots() if abs(r.imag) < 1e-9 and lo < r.real < hi]
    return {"degree": g, "cv_residual_rel": cv, "exponent_per_repetition": float(np.mean(np.log(np.abs(d(x)) + 1e-300))),
            "critical_points_inside": len(crit), "points": int(len(x))}


def main():
    import hidden_chaos as hc
    import explore_llm_attractors as ela

    run, out_png = sys.argv[1], sys.argv[2]
    base = ela.DATA / "explore" / run
    js = json.loads((base / "hidden-chaos.json").read_text())
    cells, run_args = js["cells"], js["args"]
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
                amended = [(k, cv_fit(pmax[phase == k, c])) for k in range(q)]
                kept = [(k, f) for k, f in amended if f is not None and f["cv_residual_rel"] < SINGLE_VALUED]
                if kept:
                    full = float(np.mean(info["lyapunov"]))
                    ex = np.array([f["exponent_per_repetition"] / q for _, f in kept])
                    row["amended"] = {"single_valued_phases": [k for k, _ in kept], "degrees": [f["degree"] for _, f in kept],
                                      "cv_residuals": [f["cv_residual_rel"] for _, f in kept],
                                      "exponents_per_step": ex.tolist(), "exponent_1d_per_step": float(np.median(ex)),
                                      "spread_per_step": float(ex.max() - ex.min()),
                                      "unimodal_phases": sum(f["critical_points_inside"] == 1 for _, f in kept),
                                      "exponent_full_per_step": full,
                                      "relative_discrepancy": float((np.median(ex) - full) / full)}
        rows.append(row)
    fitted = [r for r in rows if "relative_discrepancy" in r]
    single = [r for r in fitted if r["residual_rel"] < 1e-2]
    amended = [r for r in rows if "amended" in r]
    disc = np.array([abs(r["amended"]["relative_discrepancy"]) for r in amended])
    summary = {"cells": len(rows), "classes": {k: sum(r["class"] == k for r in rows) for k in
                                               sorted(set(r["class"] for r in rows))},
               "repeating_text_chaotic": sum(r["class"] == "hidden chaos" for r in rows),
               "amended": {"cells_with_single_valued_phase": len(amended),
                           "median_abs_discrepancy": float(np.median(disc)) if len(disc) else None,
                           "within_10_percent": int((disc <= 0.10).sum()),
                           "within_5_percent": int((disc <= 0.05).sum()),
                           "median_single_valued_phases": float(np.median([len(r["amended"]["single_valued_phases"])
                                                                          for r in amended])) if amended else None,
                           "median_relative_spread_across_phases": float(np.median(
                               [r["amended"]["spread_per_step"] / r["amended"]["exponent_full_per_step"] for r in amended]))
                           if amended else None,
                           "cells_with_a_unimodal_phase": sum(r["amended"]["unimodal_phases"] > 0 for r in amended)},
               "prespecified": {"fitted (>= %d repetitions)" % MIN_REPETITIONS: len(fitted),
                                "single_valued (residual < 1e-2)": len(single),
                                "single_valued_with_one_critical_point": sum(r["critical_points_inside"] == 1
                                                                             for r in single),
                                "median_abs_discrepancy_single_valued":
                                    float(np.median([abs(r["relative_discrepancy"]) for r in single])) if single else None,
                                "within_10_percent_single_valued": sum(abs(r["relative_discrepancy"]) <= 0.10
                                                                       for r in single)},
               "text_periods_fitted": sorted(set(r["text_period"] for r in fitted))}
    (base / "return-maps.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    if amended:
        full = np.array([r["amended"]["exponent_full_per_step"] for r in amended])
        one = np.array([r["amended"]["exponent_1d_per_step"] for r in amended])
        lo = np.array([min(r["amended"]["exponents_per_step"]) for r in amended])
        hi = np.array([max(r["amended"]["exponents_per_step"]) for r in amended])
        qq = np.array([r["text_period"] for r in amended])
        ax.errorbar(full, one, yerr=[one - lo, hi - one], fmt="none", ecolor="0.75", lw=0.8, zorder=1)
        sc = ax.scatter(full, one, c=np.log2(qq / 7), cmap="viridis", s=18, zorder=2,
                        label="median over single-valued phases\n(bars: range over phases)")
        lim = [0, max(full.max(), hi.max()) * 1.08]
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046)
        cb.set_label("log2(text period / 7)")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_xlabel("largest exponent of the full map (per step)")
    ax.set_ylabel("exponent of the fitted 1D return maps (per step)")
    model = ela.MODELS[run_args["model"]].split("/")[-1]
    ax.set_title(f"{model}, {run_args['concept']} (layer {run_args['layer']}): {len(amended)} chaotic cells\n"
                 "with repeating text, one-dimensional reduction", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)


if __name__ == "__main__":
    main()
