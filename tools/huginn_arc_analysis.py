"""Analysis for the Huginn relaxation test: spectral predictions from fixed-point spectra, and paired accuracy tests
on rows written by tools/huginn_relax_eval.py arc.

Spectral predictions. The relaxed recurrence s <- (1 - lam) s + lam F(s) keeps the fixed points of F and maps each
multiplier mu to 1 - lam + lam mu. For a prompt with computed leading eigenvalues {mu} (largest modulus, smallest and
largest real part), the predicted asymptotic rate is rho(lam) = max |1 - lam + lam mu|; lam* minimises it; lam_c is the
smallest lam > 1 with rho(lam) >= 1, where a real negative eigenvalue reaches -1 (flip) or a complex pair the unit
circle (Neimark-Sacker).

Accuracy tests (paired over questions; every arm sees the same questions and the same initial latent states):
exact McNemar tests, a paired equivalence test (two one-sided tests on the mean difference, question-level
bootstrap), and a breakpoint fit of accuracy against lam (flat, then linear decline) with a bootstrap interval.

Usage:
  python tools/huginn_arc_analysis.py spectra huginn-relax/arc-dev-spectrum-{0..4}
  python tools/huginn_arc_analysis.py accuracy huginn-relax/arc-dev-arms --metric acc_norm --steps 32
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def eigenvalues(spectrum):
    """All computed eigenvalues (LM, SR, LR) of a fixed-spectrum result, duplicates removed."""
    mus = []
    for key in ("LM", "SR", "LR"):
        mus += [complex(a, b) for a, b in spectrum["spectra_of_F"].get(key, [])]
    return np.unique(np.round(np.array(mus), 10))


def rho(mus, lam):
    return float(np.max(np.abs(1.0 - lam + lam * np.asarray(mus))))


def lam_critical(mus, upper=3.0):
    """Smallest lam > 1 with rho(lam) = 1, by bisection on each eigenvalue's own crossing; returns (lam_c, mu)."""
    best = (math.inf, None)
    for mu in np.asarray(mus):
        # |1 - lam (1 - mu)|^2 = 1  <=>  lam = 2 Re(1 - mu) / |1 - mu|^2 (the nonzero root)
        d = 1.0 - mu
        if abs(d) == 0:
            continue
        lam = 2.0 * d.real / abs(d) ** 2
        if 1.0 < lam < upper and lam < best[0]:
            best = (float(lam), complex(mu))
    return best


def lam_star(mus, lo=0.3, hi=1.6, n=13001):
    grid = np.linspace(lo, hi, n)
    vals = np.array([rho(mus, g) for g in grid])
    return float(grid[vals.argmin()]), float(vals.min())


def load_rows(path, metric="acc_norm"):
    """{arm: {steps: correctness array ordered by question index}} and the sorted question indices."""
    table, indices = {}, set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            indices.add(row["index"])
            for r, rec in row["by_steps"].items():
                table.setdefault(row["arm"], {}).setdefault(int(r), {})[row["index"]] = bool(rec[metric])
    order = sorted(indices)
    out = {}
    for arm, by in table.items():
        for r, d in by.items():
            if sorted(d) != order:
                raise ValueError(f"arm {arm} at {r} steps does not cover the same questions")
            out.setdefault(arm, {})[r] = np.array([d[i] for i in order])
    return out, order


def mcnemar_less(base, other):
    """Exact one-sided McNemar p-value for 'other is less accurate than base' (discordant pairs only)."""
    from scipy.stats import binom

    b = int(np.sum(base & ~other))       # base right, other wrong
    c = int(np.sum(~base & other))
    n = b + c
    return (float(binom.sf(b - 1, n, 0.5)) if n else 1.0), b, c


def paired_bootstrap(base, other, n_boot=10000, seed=0):
    """Mean paired difference (other - base) and its bootstrap distribution over questions."""
    rng = np.random.default_rng(seed)
    d = other.astype(float) - base.astype(float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return float(d.mean()), d[idx].mean(axis=1)


def equivalent(base, other, margin, alpha=0.05, n_boot=10000, seed=0):
    """Two one-sided tests: the (1 - 2 alpha) bootstrap interval of the paired difference lies inside +-margin."""
    mean, boots = paired_bootstrap(base, other, n_boot, seed)
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return bool(lo > -margin and hi < margin), mean, float(lo), float(hi)


def breakpoint_fit(lams, correct, grid=None):
    """Least-squares fit of mean correctness against lam: flat level c up to lam_b, then c - s (lam - lam_b), s >= 0.
    `correct` is (n_lams, n_questions). Returns (lam_b, c, s, sse)."""
    lams = np.asarray(lams, float)
    y = np.asarray(correct, float).mean(axis=1)
    grid = np.linspace(lams.min(), lams.max(), 401) if grid is None else grid
    best = None
    for lb in grid:
        x = np.maximum(lams - lb, 0.0)
        a = np.vstack([np.ones_like(x), -x]).T
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        if coef[1] < 0:                                   # a rising tail is not a cliff: fit flat
            coef = np.array([y.mean(), 0.0])
        sse = float(np.sum((a @ coef - y) ** 2))
        if best is None or sse < best[3] - 1e-15:
            best = (float(lb), float(coef[0]), float(coef[1]), sse)
    return best


def breakpoint_interval(lams, correct, n_boot=2000, seed=0, grid=None):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct)
    n = correct.shape[1]
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        draws.append(breakpoint_fit(lams, correct[:, idx], grid)[0])
    return np.percentile(draws, [2.5, 97.5])


def load_gold_loglik(path, steps):
    """{arm: {index: (lam, gold-choice log-likelihood at `steps`)}} from arc rows."""
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rec = row["by_steps"][str(steps)]
            out.setdefault(row["arm"], {})[row["index"]] = (row["lam"], rec["loglik"][row["gold"]])
    return out


def per_item_effect(gold, lam_c, base="1.0", n_boot=4000, seed=0):
    """Does a question lose gold-choice log-likelihood when lam exceeds its own spectral lam_c?

    For every relaxation arm (numeric lam, not the base) and every question with a predicted lam_c, the outcome is
    delta = loglik(lam) - loglik(base) and the predictor u = 1[lam > lam_c]. beta is the pooled within-arm regression
    slope of delta on u (arm fixed effects); its bootstrap distribution resamples questions. Returns beta, the one-sided
    bootstrap p-value for beta >= 0, the bootstrap 95 % interval and the number of (arm, question) pairs per group."""
    arms = [a for a in gold if a != base and a.replace(".", "", 1).isdigit()]
    qs = sorted(set(lam_c) & set(gold[base]))
    lam = {a: float(a) for a in arms}

    def beta(sample):
        num = den = 0.0
        for a in arms:
            d = np.array([gold[a][q][1] - gold[base][q][1] for q in sample])
            u = np.array([1.0 if lam[a] > lam_c[q] else 0.0 for q in sample])
            if u.min() == u.max():
                continue
            num += float(np.sum((u - u.mean()) * (d - d.mean())))
            den += float(np.sum((u - u.mean()) ** 2))
        return num / den if den > 0 else float("nan")

    est = beta(qs)
    rng = np.random.default_rng(seed)
    boots = np.array([beta([qs[i] for i in rng.integers(0, len(qs), len(qs))]) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    counts = {a: int(sum(lam[a] > lam_c[q] for q in qs)) for a in arms}
    return est, float(np.mean(boots >= 0)), np.percentile(boots, [2.5, 97.5]), counts, len(qs)


def stability_interval(mus, lo=0.02, hi=2.0, n=19801):
    """Interval of lam (containing the small-lam end, where the relaxed map always contracts) on which
    rho(lam) < 1: (lam_lo, lam_up); lam_up < 1 means the native loop (lam = 1) is unstable at this fixed point."""
    grid = np.linspace(lo, hi, n)
    ok = np.array([rho(mus, g) < 1.0 for g in grid])
    if not ok.any():
        return float("nan"), float("nan")
    first = int(np.argmax(ok))
    last = first
    while last + 1 < n and ok[last + 1]:
        last += 1
    return float(grid[first]), float(grid[last])


def two_way_effect(outcome, unstable, n_boot=4000, seed=0):
    """Effect of being predicted unstable on an outcome, with question and arm fixed effects.

    outcome, unstable: {arm: {question: value}} over the same questions and arms. beta is the coefficient of the
    unstable indicator after removing question and arm means (two-way within transformation, exact for a balanced
    panel); its bootstrap resamples questions. Returns beta, the one-sided p-value for beta >= 0, the 95 % interval,
    and the number of unstable (arm, question) cells."""
    arms = sorted(outcome)
    qs = sorted(set.intersection(*[set(outcome[a]) for a in arms]) & set.intersection(*[set(unstable[a]) for a in arms]))

    def beta(sample):
        Y = np.array([[outcome[a][q] for a in arms] for q in sample], float)
        U = np.array([[float(unstable[a][q]) for a in arms] for q in sample])
        Yw = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
        Uw = U - U.mean(1, keepdims=True) - U.mean(0, keepdims=True) + U.mean()
        den = float(np.sum(Uw ** 2))
        return float(np.sum(Uw * Yw)) / den if den > 0 else float("nan")

    est = beta(qs)
    rng = np.random.default_rng(seed)
    boots = np.array([beta([qs[i] for i in rng.integers(0, len(qs), len(qs))]) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    cells = int(sum(bool(unstable[a][q]) for a in arms for q in qs))
    return est, float(np.mean(boots >= 0)), np.percentile(boots, [2.5, 97.5]), cells, len(qs)


def spectra_cmd(args):
    import explore_llm_attractors as ela

    rows = []
    for folder in args.folders:
        spec = json.load(open(ela.DATA / "explore" / folder / "fixed-spectrum.json"))
        if spec.get("fixed_point_residual_F", 0.0) > args.fixed_tol:      # gate: no fixed point, no multipliers
            print(json.dumps({"folder": folder, "residual": spec.get("fixed_point_residual_F"),
                              "status": "not a fixed point: spectrum not used (recompute with a relaxed transient)"}))
            continue
        mus = eigenvalues(spec)
        lc, mu_c = lam_critical(mus)
        ls, rs = lam_star(mus)
        rows.append({"folder": folder, "residual": spec.get("fixed_point_residual_F"),
                     "transient_lams": spec.get("transient_lams", [spec["args"].get("lam")]),
                     "native_unstable": bool(rho(mus, 1.0) >= 1.0), "max_modulus": float(np.abs(mus).max()),
                     "lam_star": ls, "rho_star": rs, "rho_native": rho(mus, 1.0), "lam_c": lc,
                     "critical_mu": [mu_c.real, mu_c.imag] if mu_c is not None else None,
                     "rho": {str(l): rho(mus, l) for l in args.lams}})
        print(json.dumps(rows[-1]))
    lcs = np.array([r["lam_c"] for r in rows])
    print(json.dumps({"lam_c_median": float(np.median(lcs)), "lam_c_min": float(lcs.min()), "lam_c_max": float(lcs.max())}))


def accuracy_cmd(args):
    import explore_llm_attractors as ela

    table, order = load_rows(ela.DATA / "explore" / args.folder / "rows.jsonl", args.metric)
    base = table[args.base]
    for r in args.steps:
        print(f"steps {r}: n = {len(order)}")
        for arm in sorted(table, key=lambda a: (not a.replace('.', '', 1).isdigit(), a)):
            acc = table[arm][r].mean()
            p, b, c = mcnemar_less(base[r], table[arm][r])
            mean, boots = paired_bootstrap(base[r], table[arm][r])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  {arm:>14}: {acc:.4f}  diff {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  McNemar(less) p {p:.4g} ({b} vs {c})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("spectra")
    sp.add_argument("folders", nargs="+")
    sp.add_argument("--lams", type=float, nargs="+", default=[0.55, 0.75, 1.0, 1.06, 1.1, 1.15, 1.2, 1.25])
    sp.add_argument("--fixed-tol", type=float, default=1e-4)
    ac = sub.add_parser("accuracy")
    ac.add_argument("folder")
    ac.add_argument("--metric", default="acc_norm", choices=("acc", "acc_norm"))
    ac.add_argument("--steps", type=int, nargs="+", default=[32])
    ac.add_argument("--base", default="1.0")
    args = parser.parse_args()
    {"spectra": spectra_cmd, "accuracy": accuracy_cmd}[args.command](args)


if __name__ == "__main__":
    main()
