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


def item_table(spectra, rows, lams):
    """One record per (question, lam) with a spectrum: rho_q(lam), lam_up(q), the log native residual at the last
    recurrence step, and whether the chosen answer (argmax of the log-likelihood) changes across the read-out steps."""
    out = []
    lams = [float(v) for v in lams]
    for r in rows:
        q, lam = r["index"], float(r["lam"])
        if q not in spectra or not any(abs(lam - v) < 1e-12 for v in lams):
            continue
        mus = spectra[q]
        steps = sorted(r["by_steps"], key=int)
        choice = np.array([int(np.argmax(r["by_steps"][k]["loglik"])) for k in steps])
        out.append({"q": q, "lam": lam, "rho": rho(mus, lam), "lam_up": stability_interval(mus)[1],
                    "log_residual": math.log(max(r["native_residual"][-1], 1e-30)),
                    "changes": bool((np.diff(choice) != 0).any())})
    return out


def item_statistic(records, lams, questions):
    """Mean over lam arms of the Spearman correlation, across questions, of rho_q(lam) and the log residual."""
    from scipy.stats import spearmanr

    vals = []
    for lam in lams:
        by_q = {r["q"]: r for r in records if abs(r["lam"] - lam) < 1e-12}
        x = [by_q[q]["rho"] for q in questions if q in by_q]
        y = [by_q[q]["log_residual"] for q in questions if q in by_q]
        if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            vals.append(spearmanr(x, y).statistic)
    return float(np.mean(vals)) if vals else float("nan"), vals


def items_cmd(args):
    """Part A of the frozen Huginn test (docs/huginn-causal-test-design.md): does each question's own fixed-point
    spectrum predict how its recurrence responds to relaxation?"""
    import explore_llm_attractors as ela
    from scipy.stats import fisher_exact

    spectra, excluded = {}, []
    for folder in args.spectra:
        root = ela.DATA / "explore" / folder
        for f in sorted(root.glob("q[0-9][0-9][0-9][0-9]/fixed-spectrum.json")):
            spec = json.load(open(f))
            q = int(f.parent.name[1:])
            res_f = spec.get("fixed_point_residual_F")
            if res_f is None or not math.isfinite(res_f) or res_f > args.fixed_tol:
                excluded.append({"q": q, "residual": res_f})
                continue
            spectra[q] = eigenvalues(spec)
    rows = [json.loads(line) for line in open(ela.DATA / "explore" / args.rows / "rows.jsonl")]
    expected = set(range(args.start, args.start + args.count))
    got = {r["index"] for r in rows}
    if got != expected:
        raise SystemExit(f"rows cover {len(got)} questions, expected {len(expected)}")
    arms = {float(r["lam"]) for r in rows}
    if not all(any(abs(a - float(l)) < 1e-12 for a in arms) for l in args.lams):
        raise SystemExit(f"arms in rows {sorted(arms)} do not include {args.lams}")
    lams = [float(v) for v in args.lams]
    recs = item_table(spectra, rows, lams)
    questions = sorted(spectra)
    share_excluded = 1 - len(questions) / len(expected)
    stat, per_arm = item_statistic(recs, lams, questions)
    rng = np.random.default_rng(0)
    boots = [item_statistic(recs, lams, list(rng.choice(questions, len(questions))))[0] for _ in range(args.boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    unstable = np.array([r["rho"] >= 1 for r in recs])
    changes = np.array([r["changes"] for r in recs])
    table = [[int((unstable & changes).sum()), int((unstable & ~changes).sum())],
             [int((~unstable & changes).sum()), int((~unstable & ~changes).sum())]]
    p_fisher = float(fisher_exact(table, alternative="greater").pvalue)
    result = {"args": vars(args), "questions_with_spectra": len(questions), "excluded": excluded,
              "setup_failed": bool(share_excluded > 0.25),
              "primary": {"statistic": stat, "per_arm": dict(zip([str(l) for l in lams], per_arm)),
                          "ci95": [float(lo), float(hi)], "success": bool(lo > 0.3 and share_excluded <= 0.25)},
              "answer_changes": {"unstable_changed": table[0][0], "unstable_total": int(unstable.sum()),
                                 "stable_changed": table[1][0], "stable_total": int((~unstable).sum()),
                                 "fisher_one_sided_p": p_fisher},
              "share_unstable_by_arm": {str(l): float(np.mean([r["rho"] >= 1 for r in recs if abs(r["lam"] - l) < 1e-12]))
                                        for l in lams}}
    print(json.dumps(result["primary"]))
    print(json.dumps(result["answer_changes"]))
    dest = ela.DATA / "explore" / args.out
    if (dest / "items.json").exists():
        raise SystemExit(f"{dest / 'items.json'} exists; outputs are never overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "items.json").write_text(json.dumps(result, indent=1))


def generation_cmd(args):
    """Part B of the frozen Huginn test: GSM8K generation accuracy against lam (rows of huginn_relax_eval accuracy)."""
    import explore_llm_attractors as ela

    rows = []
    for folder in args.rows:
        rows += [json.loads(line) for line in open(ela.DATA / "explore" / folder / "rows.jsonl")]
    by = {}
    for r in rows:
        key = (str(r["arm"]), int(r["index"]))
        if key in by:
            raise SystemExit(f"duplicate row {key}")
        by[key] = r
    expected = set(range(args.start, args.start + args.count))
    arms = [str(a) for a in args.arms]
    for a in arms:
        got = {i for (arm, i) in by if arm == a}
        if got != expected:
            raise SystemExit(f"arm {a}: {len(got)} problems, expected {len(expected)} ({args.start}..)")
    order = sorted(expected)
    result = {"args": vars(args), "n": len(order), "metrics": {}}
    for metric in ("correct", "correct_strict"):
        acc = {a: np.array([bool(by[(a, i)][metric]) for i in order]) for a in arms}
        base = acc[args.base]
        out = {"accuracy": {a: float(v.mean()) for a, v in acc.items()}}
        for a in arms:
            if a == args.base:
                continue
            p, b, c = mcnemar_less(base, acc[a])
            mean, boots = paired_bootstrap(base, acc[a])
            out[a] = {"difference": mean, "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
                      "lower90": float(np.percentile(boots, 5)), "mcnemar_less_p": p, "lost": b, "gained": c}
        result["metrics"][metric] = out
    m = result["metrics"]["correct"]
    result["B1_collapse"] = {"arm": args.collapse_arm, "p": m[args.collapse_arm]["mcnemar_less_p"],
                             "success": bool(m[args.collapse_arm]["mcnemar_less_p"] < 0.05)}
    result["B3_no_loss"] = {"arm": args.safe_arm, "lower90": m[args.safe_arm]["lower90"], "margin": -args.margin,
                            "success": bool(m[args.safe_arm]["lower90"] > -args.margin)}
    result["primary_success"] = bool(result["B1_collapse"]["success"] and result["B3_no_loss"]["success"])
    result["B2_break"] = {"arm": args.break_arm, "p": m[args.break_arm]["mcnemar_less_p"],
                          "success": bool(m[args.break_arm]["mcnemar_less_p"] < 0.05)}
    result["tokens_median"] = {a: float(np.median([by[(a, i)]["new_tokens"] for i in order])) for a in arms}
    result["strict_format_share"] = {a: float(np.mean([by[(a, i)]["strict_pred"] is not None for i in order]))
                                     for a in arms}
    print(json.dumps({k: result[k] for k in ("n", "B1_collapse", "B3_no_loss", "primary_success", "B2_break")}))
    dest = ela.DATA / "explore" / args.out
    if (dest / "generation.json").exists():
        raise SystemExit(f"{dest / 'generation.json'} exists; outputs are never overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "generation.json").write_text(json.dumps(result, indent=1))


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
    it = sub.add_parser("items")
    it.add_argument("--spectra", nargs="+", required=True, help="folders holding qNNNN/fixed-spectrum.json")
    it.add_argument("--rows", required=True, help="folder of the arc read-out rows.jsonl")
    it.add_argument("--start", type=int, required=True)
    it.add_argument("--count", type=int, required=True)
    it.add_argument("--lams", type=float, nargs="+", default=[1.0, 1.05, 1.08, 1.10])
    it.add_argument("--fixed-tol", type=float, default=1e-4)
    it.add_argument("--boot", type=int, default=2000)
    it.add_argument("--out", required=True)
    ge = sub.add_parser("generation")
    ge.add_argument("--rows", nargs="+", required=True, help="folders of huginn_relax_eval accuracy rows.jsonl")
    ge.add_argument("--start", type=int, required=True)
    ge.add_argument("--count", type=int, required=True)
    ge.add_argument("--arms", nargs="+", default=["1.0", "1.05", "1.15", "1.20"])
    ge.add_argument("--base", default="1.0")
    ge.add_argument("--collapse-arm", default="1.20")
    ge.add_argument("--break-arm", default="1.15")
    ge.add_argument("--safe-arm", default="1.05")
    ge.add_argument("--margin", type=float, default=0.04, help="non-inferiority margin for the safe arm")
    ge.add_argument("--out", required=True)
    args = parser.parse_args()
    {"spectra": spectra_cmd, "accuracy": accuracy_cmd, "items": items_cmd, "generation": generation_cmd}[args.command](args)


if __name__ == "__main__":
    main()
