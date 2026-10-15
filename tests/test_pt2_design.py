import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codex" / "confirm-fold-2"))
import build_design as bd  # noqa: E402

A, B = 7.054924948575865, 24.92734030789243        # Task 5 target A (codex/confirm-fold/arctangent-scope.md)


def test_predictors_reproduce_task5_target_a():
    U = 0.1 / B
    mid = bd.predictions(0.0, A, B, 3e-6, U)
    full = bd.predictions(-U, A, B, 3e-6, U)
    assert mid["quadratic_G"] == 61 and abs(mid["arctan_G"] - 58.55773561712685) < 1e-9
    assert full["quadratic_G"] == 118 and abs(full["arctan_G"] - 117.1154712342537) < 1e-9


def test_distance_rule_holds_for_q_ladder():
    U = 0.1 / B
    for q in bd.Q:
        eps = q * q / (A * B)
        assert bd.predictions(-U, A, B, eps, U)["arctan_G"] >= 16
        assert bd.predictions(0.0, A, B, eps, U)["arctan_G"] >= 8


def test_transverse_kick_is_in_the_kernel_of_l():
    rng = np.random.default_rng(0)
    x = rng.normal(size=12)
    r = rng.normal(size=12); r /= np.linalg.norm(r)
    l = rng.normal(size=12); l /= l @ r
    w, starts = bd.entrances(x, r, l, 0.01, 7)
    assert abs(l @ w) < 1e-12 and abs(np.linalg.norm(w) - 1) < 1e-12
    assert np.allclose(starts[1], x) and math.isclose((starts[0] - x) @ l, -0.01, rel_tol=1e-12)
