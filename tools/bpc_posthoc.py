"""Post-hoc diagnostics of the frozen BPC result (docs/bpc-test.md; research log, 15 Oct, BPC result entry). Not part
of the frozen analysis: these describe why the primary coefficient is null and what shapes the regime gradient.

1. Which frozen covariates carry the raw chaotic-minus-periodic gap: the chaotic coefficient with no covariates, with
   each covariate alone, with the three first-token measures (entropy, largest probability, log-probability of the
   correct first token) and with the four structural ones (greedy correct, digits, carry, a + b).
2. How the loop exponent relates to the first-token entropy (Spearman correlation over all labelled prompts).
3. The regime gradient in finer bins: covariate-adjusted accuracy (residual of the covariate-only fit, as in the
   frozen secondary e) in ten equal-count bins of the exponent, with the exponent's quantiles, to show which part of
   the range drives the frozen quadratic coefficient.

Usage: python tools/bpc_posthoc.py --labels RUN [RUN ...] --gen RUN --cov RUN --out-json PATH
"""
import argparse
import json
from types import SimpleNamespace

import numpy as np

import bpc_analyze as ba

NAMES = ["greedy_correct", "first_token_entropy", "first_token_p_top", "logp_correct_first", "digits", "carry", "a_plus_b"]


def chaotic_coef(y, chaotic, X):
    D = np.column_stack([np.ones(len(y)), chaotic.astype(float)] +
                        ([((X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0))] if X.shape[1] else []))
    return float(ba.ols_coef(y, D)[1])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", default="docs/bpc-tasks.json")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    rows, _, allrows = ba.load(SimpleNamespace(tasks=args.tasks, labels=args.labels, gen=args.gen, cov=args.cov))
    y = np.array([r["succ"] / r["trials"] for r in rows])
    ch = np.array([r["chaotic"] for r in rows])
    X = np.array([r["X"] for r in rows])
    res = {"carriers": {"none": chaotic_coef(y, ch, X[:, :0]), "all_frozen": chaotic_coef(y, ch, X)}}
    for j, n in enumerate(NAMES):
        res["carriers"][n] = chaotic_coef(y, ch, X[:, [j]])
    res["carriers"]["first_token_three"] = chaotic_coef(y, ch, X[:, [1, 2, 3]])
    res["carriers"]["structural_four"] = chaotic_coef(y, ch, X[:, [0, 4, 5, 6]])
    ya = np.array([r["y"] for r in allrows])
    Xa = np.array([r["X"] for r in allrows])
    ex = np.array([r["exponent"] for r in allrows])
    ok = np.isfinite(ex)
    ya, Xa, ex = ya[ok], Xa[ok], ex[ok]
    res["spearman_exponent_vs_entropy"] = spearman(ex, Xa[:, 1])
    res["spearman_exponent_vs_p_top"] = spearman(ex, Xa[:, 2])
    res["exponent_quantiles"] = {str(q): float(np.quantile(ex, q)) for q in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)}
    Dc = np.column_stack([np.ones(len(ya)), (Xa - Xa.mean(0)) / np.where(Xa.std(0) > 0, Xa.std(0), 1.0)])
    r = ya - Dc @ ba.ols_coef(ya, Dc)
    edges = np.quantile(ex, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, ex, side="right") - 1, 0, 9)
    rng = np.random.default_rng(0)
    res["deciles"] = []
    for k in range(10):
        v = r[idx == k]
        b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        res["deciles"].append({"exponent_range": [float(edges[k]), float(edges[k + 1])], "n": int(len(v)),
                               "mean_adjusted": float(v.mean()),
                               "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]})
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
