import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_arc_analysis as ha  # noqa: E402


def test_lam_critical_and_star():
    mus = np.array([0.9136, -0.775 + 0.062j, -0.775 - 0.062j, 0.246 + 0.834j, 0.246 - 0.834j])
    lc, mu = ha.lam_critical(mus)
    assert abs(ha.rho(mus, lc) - 1.0) < 1e-9 and mu.real < 0          # the negative pair sets lam_c
    assert 1.12 < lc < 1.13
    ls, rs = ha.lam_star(mus)
    assert 1.0 < ls < lc and rs < ha.rho(mus, 1.0)


def test_mcnemar_and_equivalence():
    base = np.array([True] * 60 + [False] * 40)
    other = base.copy()
    other[:15] = False                                              # 15 losses, no gains
    p, b, c = ha.mcnemar_less(base, other)
    assert (b, c) == (15, 0) and p < 1e-4
    ok, mean, lo, hi = ha.equivalent(base, base.copy(), margin=0.015)
    assert ok and mean == 0.0


def test_breakpoint_fit_recovers_a_cliff():
    rng = np.random.default_rng(1)
    lams = np.array([1.0, 1.05, 1.10, 1.13, 1.16, 1.20, 1.25, 1.30])
    p = np.where(lams <= 1.13, 0.40, 0.40 - 1.5 * (lams - 1.13))
    correct = rng.random((len(lams), 4000)) < p[:, None]
    lb, c, s, _ = ha.breakpoint_fit(lams, correct)
    assert abs(lb - 1.13) < 0.03 and abs(c - 0.40) < 0.02 and s > 0.5


def test_per_item_effect_detects_own_threshold_losses(tmp_path):
    rng = np.random.default_rng(2)
    lam_c = {q: float(rng.uniform(1.06, 1.16)) for q in range(80)}
    gold = {"1.0": {q: (1.0, -10.0 + rng.normal(0, 0.2)) for q in lam_c}}
    for lam in (1.05, 1.10, 1.15, 1.20):
        gold[str(lam)] = {q: (lam, gold["1.0"][q][1] - (3.0 if lam > lam_c[q] else 0.0) + rng.normal(0, 0.5))
                          for q in lam_c}
    beta, p, ci, counts, n = ha.per_item_effect(gold, lam_c, n_boot=500)
    assert beta < -2.0 and p < 0.01 and ci[1] < 0 and n == 80
    # no effect when losses ignore the question's own threshold
    for lam in (1.05, 1.10, 1.15, 1.20):
        gold[str(lam)] = {q: (lam, gold["1.0"][q][1] - 1.0 + rng.normal(0, 0.5)) for q in lam_c}
    beta, p, ci, counts, n = ha.per_item_effect(gold, lam_c, n_boot=500)
    assert ci[0] < 0 < ci[1]
