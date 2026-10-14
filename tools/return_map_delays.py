"""Is the scalar return map dynamically sufficient? (consult 7, recommendation 2, first part)

For each repeating-text chaotic cell (tools/hidden_chaos.py --save-sequences) and each phase of its text whose return
map p(n) -> p(n+1) is single valued, compare held-out prediction of p(n+1) by (a) a scalar polynomial in p(n),
(b) a polynomial in (p(n), p(n-1)) and (c) in (p(n), p(n-1), p(n-2)). Repetitions are split in time: training on the
first 60 %, a gap of 10 %, testing on the last 30 %. Degrees chosen on the training part by 5-fold cross-validation
(scalar: 2..12; two delays: total degree 2..5; three delays: 2..4). Error: held-out mean squared error over the
held-out variance. Scalar sufficiency is rejected for a cell when the best delay predictor improves the held-out error
of its best scalar phase by 20 % or more.

Usage: python tools/return_map_delays.py RUN_DIR OUT.json      (RUN_DIR explore-relative, from H3)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def design(X, degree):
    """Monomials of total degree <= degree in the columns of X (scaled to [-1, 1] by the caller)."""
    n, k = X.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(k), deg):
            cols.append(np.prod(X[:, combo], axis=1))
    return np.stack(cols, axis=1)


def fit_predict(Xtr, ytr, Xte, degrees, folds=5, seed=0):
    lo, hi = Xtr.min(0), Xtr.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    sc = lambda X: 2 * (X - lo) / span - 1
    Atr = sc(Xtr)
    order = np.random.default_rng(seed).permutation(len(ytr))
    parts = np.array_split(order, folds)
    best = None
    for g in degrees:
        if design(Atr[:2], g).shape[1] >= len(ytr) * 0.8:
            continue
        err = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            c, *_ = np.linalg.lstsq(design(Atr[tr], g), ytr[tr], rcond=None)
            err.append(np.mean((design(Atr[te], g) @ c - ytr[te]) ** 2))
        if best is None or np.mean(err) < best[1]:
            best = (g, np.mean(err))
    g = best[0]
    c, *_ = np.linalg.lstsq(design(Atr, g), ytr, rcond=None)
    return design(sc(Xte), g) @ c, g


def main():
    import hidden_chaos as hc
    import explore_llm_attractors as ela

    run, out = sys.argv[1], sys.argv[2]
    base = ela.DATA / "explore" / run
    cells = json.loads((base / "hidden-chaos.json").read_text())["cells"]
    seq = np.load(base / "sequences.npz")
    rows = []
    for c, info in enumerate(cells):
        if hc.classify(info) != "hidden chaos":
            continue
        lead, pmax = seq["lead"][:, c], seq["pmax"][:, c]
        q = hc.symbolic_period(lead.tolist())
        if q == 0:
            continue
        phase = np.arange(len(lead)) % q
        best = None
        for k in range(q):
            p = pmax[phase == k]
            n = len(p)
            if n < 40 or p.max() - p.min() <= 0:
                continue
            idx = np.arange(2, n - 1)                     # targets p[i+1] with p[i], p[i-1], p[i-2] available
            ntr, nte = int(0.6 * len(idx)), int(0.3 * len(idx))
            tr, te = idx[:ntr], idx[len(idx) - nte:]
            y = p[1:]
            res = {}
            for name, lags, degs in (("scalar", 1, range(2, 13)), ("two", 2, range(2, 6)), ("three", 3, range(2, 5))):
                X = np.stack([p[idx - l] for l in range(lags)], axis=1)
                Xtr, Xte = X[:ntr], X[len(idx) - nte:]
                ytr, yte = p[tr + 1], p[te + 1]
                pred, g = fit_predict(Xtr, ytr, Xte, list(degs))
                res[name] = float(np.mean((pred - yte) ** 2) / max(np.var(yte), 1e-300))
            if best is None or res["scalar"] < best[1]["scalar"]:
                best = (k, res)
        if best is None:
            continue
        k, res = best
        gain = 1 - min(res["two"], res["three"]) / res["scalar"] if res["scalar"] > 0 else 0.0
        rows.append({"log10_temperature": info["log10_temperature"], "text_period": q, "phase": k,
                     "held_out_error": res, "delay_improvement": float(gain), "scalar_rejected": bool(gain >= 0.2)})
    errs = np.array([r["held_out_error"]["scalar"] for r in rows])
    summary = {"cells": len(rows), "scalar_rejected": int(sum(r["scalar_rejected"] for r in rows)),
               "median_scalar_held_out_error": float(np.median(errs)),
               "median_delay_improvement": float(np.median([r["delay_improvement"] for r in rows]))}
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["scalar_rejected"]:
            print(" rejected:", round(r["log10_temperature"], 5), r["text_period"], {k: round(v, 4) for k, v in r["held_out_error"].items()})


if __name__ == "__main__":
    main()
