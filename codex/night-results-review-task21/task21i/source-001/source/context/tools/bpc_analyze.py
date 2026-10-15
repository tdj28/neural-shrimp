"""Frozen analysis of run BPC (docs/bpc-test.md): does a prompt's loop regime predict the accuracy of its ordinary
sampled answers beyond difficulty measures fixed before sampling?

Per prompt i (labelled chaotic or periodic at log10 T 0.40; other labels excluded): y_i = fraction of its k sampled
answers that are correct; covariates X_i, all computed before any sampled outcome: greedy answer correct, entropy and
largest probability of the reply's first-token distribution at T, log-probability of the correct first token at T,
number of digits of the product, a carry from the units digits (a % 10 * b % 10 >= 10), and a + b.

Primary: the coefficient of the chaotic indicator in the least-squares fit y ~ 1 + chaotic + X (covariates
standardised), with a 95 % bootstrap interval over prompts (10,000) and a two-sided permutation test that permutes the
labels among the included prompts (10,000). Positive finding: coefficient below zero with p < 0.05.
Secondary: (a) the raw difference of mean accuracy, chaotic minus periodic, with its bootstrap interval; (b) the
same difference among greedy-correct prompts; (c) held-out predictive value: logistic regressions of per-sample
correctness on X, and on X plus the indicator, fitted on even-numbered prompts and scored on odd ones; the mean
log-loss improvement per sample, with a bootstrap interval over test prompts; (d) the primary fit with the continuous
twin exponent (offset 1e-6) in place of the indicator; (e) the regime gradient and the edge-of-chaos hypothesis (human,
15 Oct: "test deeply periodic regions and deeply chaotic regions and regions that are on the edge"): accuracy adjusted
for X (residuals of the covariate-only least-squares fit) in three bins of the exponent fixed in advance, deep periodic
(lambda <= -0.3), edge (|lambda| <= 0.1) and deep chaotic (lambda >= 0.3); the edge contrast (edge minus the two deep
bins pooled) and the quadratic coefficient of the residual on lambda, each with a bootstrap interval over prompts. An
edge optimum predicts a positive contrast and a negative quadratic coefficient. All prompts with a finite exponent
enter (e), whatever their label.

Usage: python tools/bpc_analyze.py --labels RUN [RUN ...] --gen RUN --cov RUN --out-json PATH
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore"
LT = 0.4


def ols_coef(y, D):
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return beta


def design(chaotic, X):
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return np.column_stack([np.ones(len(chaotic)), chaotic.astype(float), Xs])


def primary(y, chaotic, X, n_boot=10000, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    D = design(chaotic, X)
    est = float(ols_coef(y, D)[1])
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(0, n, n)
        boots[b] = ols_coef(y[k], design(chaotic[k], X[k]))[1]
    perm = 0
    for _ in range(n_perm):
        c = rng.permutation(chaotic)
        perm += abs(ols_coef(y, design(c, X))[1]) >= abs(est)
    return {"coefficient": est, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_permutation": (perm + 1) / (n_perm + 1), "n": int(n), "n_chaotic": int(chaotic.sum())}


def logistic_fit(D, succ, trials, iters=50, ridge=1e-6):
    """Binomial logistic regression by Newton's method (succ out of trials per row)."""
    w = np.zeros(D.shape[1])
    for _ in range(iters):
        eta = np.clip(D @ w, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        g = D.T @ (succ - trials * p) - ridge * w
        H = (D * (trials * p * (1 - p))[:, None]).T @ D + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def logloss_rows(D, w, succ, trials):
    p = np.clip(1 / (1 + np.exp(-np.clip(D @ w, -30, 30))), 1e-12, 1 - 1e-12)
    return -(succ * np.log(p) + (trials - succ) * np.log(1 - p))          # summed over a prompt's samples


def heldout(succ, trials, chaotic, X, train, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    test = ~train
    Da = design(np.zeros_like(chaotic), X)[:, [0] + list(range(2, 2 + X.shape[1]))]
    Db = design(chaotic, X)
    wa = logistic_fit(Da[train], succ[train], trials[train])
    wb = logistic_fit(Db[train], succ[train], trials[train])
    la = logloss_rows(Da[test], wa, succ[test], trials[test])
    lb = logloss_rows(Db[test], wb, succ[test], trials[test])
    t = trials[test]
    imp = (la - lb)
    est = float(imp.sum() / t.sum())
    m = len(imp)
    boots = [imp[k].sum() / t[k].sum() for k in (rng.integers(0, m, m) for _ in range(n_boot))]
    return {"logloss_improvement_per_sample": est, "ci95": [float(np.quantile(boots, 0.025)),
                                                             float(np.quantile(boots, 0.975))],
            "test_prompts": int(m)}


def diff_ci(y, chaotic, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = y[chaotic], y[~chaotic]
    est = float(a.mean() - b.mean())
    boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return {"difference": est, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "n_chaotic": int(len(a)), "n_periodic": int(len(b))}


def edge(y, X, expo, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(expo)
    y, X, expo = y[ok], X[ok], expo[ok]
    Dc = np.column_stack([np.ones(len(y)), (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)])
    r = y - Dc @ ols_coef(y, Dc)
    bins = {"deep_periodic": expo <= -0.3, "edge": np.abs(expo) <= 0.1, "deep_chaotic": expo >= 0.3}
    out = {"bins": {}}
    for name, m in bins.items():
        v = r[m]
        b = [rng.choice(v, len(v)).mean() for _ in range(2000)] if len(v) > 1 else [np.nan]
        out["bins"][name] = {"n": int(m.sum()), "mean_adjusted": float(v.mean()) if len(v) else None,
                             "ci95": [float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))] if len(v) > 1 else None}
    deep = bins["deep_periodic"] | bins["deep_chaotic"]
    if bins["edge"].sum() > 1 and deep.sum() > 1:
        est = float(r[bins["edge"]].mean() - r[deep].mean())
        e, d = r[bins["edge"]], r[deep]
        boots = [rng.choice(e, len(e)).mean() - rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        out["edge_contrast"] = {"estimate": est, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    Q = np.column_stack([np.ones(len(r)), expo, expo ** 2])
    est = float(ols_coef(r, Q)[2])
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(r), len(r))
        boots.append(ols_coef(r[k], Q[k])[2])
    out["quadratic_coefficient"] = {"estimate": est, "ci95": [float(np.quantile(boots, 0.025)),
                                                              float(np.quantile(boots, 0.975))]}
    out["n"] = int(len(r))
    return out


def load(args):
    items = {it["prompt"]: it for it in json.loads((ROOT / args.tasks).read_text())["items"]}
    lab = {}
    for run in args.labels:
        for line in open(DATA / run / "labels.jsonl"):
            r = json.loads(line)
            cell = next(c for c in r["cells"] if abs(c["log10_temperature"] - LT) < 1e-9)
            lab[r["prompt"]] = cell
    gen = json.loads((DATA / args.gen / "generations.json").read_text())
    cov = json.loads((DATA / args.cov / "covariates.json").read_text())[str(LT)]
    rows, allrows = [], []
    for p, cell in sorted(lab.items()):
        cls = cell["class"]
        it, c = items[p], cov[str(p)]
        samples = gen["sampled"][str(LT)][str(p)]
        Xp = [float(gen["greedy"][str(p)]["correct"]), c["entropy"], c["p_top"], c["logp_correct_first"],
              float(len(str(it["answer"]))), float((it["a"] % 10) * (it["b"] % 10) >= 10), float(it["a"] + it["b"])]
        allrows.append({"y": sum(s["correct"] for s in samples) / len(samples), "X": Xp,
                        "exponent": float(cell["lyapunov"][0])})
        if cls not in ("periodic", "visible chaos", "hidden chaos"):
            continue
        samples = gen["sampled"][str(LT)][str(p)]
        it = items[p]
        c = cov[str(p)]
        rows.append({"prompt": p, "chaotic": cls != "periodic", "succ": sum(s["correct"] for s in samples),
                     "trials": len(samples), "greedy": bool(gen["greedy"][str(p)]["correct"]),
                     "X": [float(gen["greedy"][str(p)]["correct"]), c["entropy"], c["p_top"], c["logp_correct_first"],
                           float(len(str(it["answer"]))), float((it["a"] % 10) * (it["b"] % 10) >= 10),
                           float(it["a"] + it["b"])],
                     "exponent": float(cell["lyapunov"][0])})
    return rows, len(lab), allrows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", default="docs/bpc-tasks.json")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--cov", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    rows, n_labelled, allrows = load(args)
    y = np.array([r["succ"] / r["trials"] for r in rows])
    succ = np.array([r["succ"] for r in rows], dtype=float)
    trials = np.array([r["trials"] for r in rows], dtype=float)
    chaotic = np.array([r["chaotic"] for r in rows])
    X = np.array([r["X"] for r in rows])
    greedy = np.array([r["greedy"] for r in rows])
    prompts = np.array([r["prompt"] for r in rows])
    res = {"n_labelled": n_labelled, "n_included": len(rows), "n_chaotic": int(chaotic.sum()),
           "primary": primary(y, chaotic, X),
           "raw_difference": diff_ci(y, chaotic),
           "greedy_correct_difference": diff_ci(y[greedy], chaotic[greedy]) if greedy.sum() > 10 else None,
           "heldout": heldout(succ, trials, chaotic, X, prompts % 2 == 0)}
    expo = np.array([r["exponent"] for r in rows])
    D = np.column_stack([np.ones(len(y)), expo, (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)])
    res["exponent_coefficient"] = float(ols_coef(y, D)[1])
    res["edge"] = edge(np.array([r["y"] for r in allrows]), np.array([r["X"] for r in allrows]),
                       np.array([r["exponent"] for r in allrows]))
    res["primary_positive"] = bool(res["primary"]["coefficient"] < 0 and res["primary"]["p_permutation"] < 0.05)
    Path(args.out_json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
