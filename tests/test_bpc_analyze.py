"""The frozen BPC analysis detects a planted direct effect of the label and not a purely confounded one."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bpc_analyze as ba  # noqa: E402


def synthetic(direct, n=1500, seed=1):
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n)
    chaotic = (difficulty + rng.normal(size=n)) > 0.5            # the label depends on difficulty (a confound)
    X = np.column_stack([difficulty, rng.normal(size=n)])           # the covariates measure difficulty
    p = 1 / (1 + np.exp(-(0.5 - 1.2 * difficulty + direct * chaotic)))
    succ = rng.binomial(10, p).astype(float)
    return succ / 10, succ, np.full(n, 10.0), chaotic, X


def test_detects_a_planted_direct_effect():
    y, succ, trials, chaotic, X = synthetic(direct=-0.6)
    r = ba.primary(y, chaotic, X, n_boot=300, n_perm=300)
    assert r["coefficient"] < 0 and r["p_permutation"] < 0.05 and r["ci95"][1] < 0
    h = ba.heldout(succ, trials, chaotic, X, np.arange(len(y)) % 2 == 0, n_boot=300)
    assert h["logloss_improvement_per_sample"] > 0


def test_confounded_label_without_direct_effect_is_not_positive():
    y, succ, trials, chaotic, X = synthetic(direct=0.0)
    raw = ba.diff_ci(y, chaotic, n_boot=300)
    assert raw["difference"] < -0.05                                 # the raw contrast is confounded
    r = ba.primary(y, chaotic, X, n_boot=300, n_perm=300)
    assert r["ci95"][0] < 0 < r["ci95"][1]                           # the adjusted coefficient is not


def test_edge_analysis_finds_a_planted_inverted_u():
    rng = np.random.default_rng(2)
    n = 3000
    expo = rng.uniform(-0.8, 0.8, n)
    X = rng.normal(size=(n, 2))
    y = 0.6 + 0.1 * X[:, 0] - 0.4 * expo ** 2 + rng.normal(scale=0.1, size=n)
    r = ba.edge(y, X, expo, n_boot=300)
    assert r["edge_contrast"]["ci95"][0] > 0
    assert r["quadratic_coefficient"]["ci95"][1] < 0
