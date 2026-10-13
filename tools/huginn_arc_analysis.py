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


def spectra_cmd(args):
    import explore_llm_attractors as ela

    rows = []
    for folder in args.folders:
        spec = json.load(open(ela.DATA / "explore" / folder / "fixed-spectrum.json"))
        mus = eigenvalues(spec)
        lc, mu_c = lam_critical(mus)
        ls, rs = lam_star(mus)
        rows.append({"folder": folder, "residual": spec.get("fixed_point_residual_F"), "max_modulus": float(np.abs(mus).max()),
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
    ac = sub.add_parser("accuracy")
    ac.add_argument("folder")
    ac.add_argument("--metric", default="acc_norm", choices=("acc", "acc_norm"))
    ac.add_argument("--steps", type=int, nargs="+", default=[32])
    ac.add_argument("--base", default="1.0")
    args = parser.parse_args()
    {"spectra": spectra_cmd, "accuracy": accuracy_cmd}[args.command](args)


if __name__ == "__main__":
    main()
