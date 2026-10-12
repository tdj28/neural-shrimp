import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bifurcation_curves as bc  # noqa: E402

torch = pytest.importorskip("torch")


def henon(x, theta):
    a, b = theta[0], theta[1]
    return torch.stack([1 - a * x[0] ** 2 + x[1], b * x[0]])


def henon_fixed_point(a, b, larger=False):
    root = np.sqrt((1 - b) ** 2 + 4 * a)
    x = (-(1 - b) + (-root if larger else root)) / (2 * a)
    return np.array([x, b * x])


def family(phi, n):
    return bc.Family(phi, (n,), torch.float64)


def test_henon_flip_curve_matches_the_analytic_curve():
    """The Henon fixed point flips on a = 3 (1 - b)^2 / 4."""
    b0 = 0.3
    x0 = henon_fixed_point(0.37, b0)
    records, reason, states = bc.trace_curve(family(henon, 2), x0, [0.37, b0], "flip", fix=1, direction=-1,
                                             steps=40, ds=0.05, ds_max=0.1, bounds=((0.0, 3.0), (-0.6, 0.6)))
    assert len(records) > 10, reason
    for r in records:
        a, b = r["theta"]
        assert abs(a - 0.75 * (1 - b) ** 2) < 1e-8
        assert abs(r["critical_multiplier"][0] + 1) < 1e-8 and abs(r["critical_multiplier"][1]) < 1e-12
    assert not any(r["cusp_candidate"] for r in records)


def test_henon_fold_curve_matches_the_analytic_curve():
    """The two Henon fixed points are born in a saddle-node on a = -(1 - b)^2 / 4."""
    b0 = 0.3
    x0 = henon_fixed_point(-0.12, b0, larger=True)
    records, reason, _ = bc.trace_curve(family(henon, 2), x0, [-0.12, b0], "fold", fix=1, direction=1,
                                        steps=30, ds=0.05, ds_max=0.1, bounds=((-1.0, 0.0), (-0.6, 0.6)))
    assert len(records) > 10, reason
    for r in records:
        a, b = r["theta"]
        assert abs(a + 0.25 * (1 - b) ** 2) < 1e-8
        assert abs(r["critical_multiplier"][0] - 1) < 1e-8


def test_fold_curve_passes_through_a_cusp_and_flags_it():
    """x -> x + t1 + t2 x - x^3 has the fold curve (t1, t2) = (-2 x^3, 3 x^2) with a cusp at the origin."""
    def cubic(x, theta):
        return x + theta[0] + theta[1] * x - x ** 3

    x0 = np.array([0.52])                                   # near the fold point x = 0.5 at t2 = 0.75
    records, reason, states = bc.trace_curve(family(cubic, 1), x0, [-0.25, 0.75], "fold", fix=1, direction=1,
                                             steps=200, ds=0.02, ds_max=0.05, bounds=((-1.0, 1.0), (-1.0, 1.0)))
    assert abs(records[0]["theta"][0] + 0.25) < 1e-10 and abs(states[0][0] - 0.5) < 1e-10
    for r in records:
        t1, t2 = r["theta"]
        assert abs(27 * t1 ** 2 - 4 * t2 ** 3) < 1e-9
    cusps = [r for r in records if r["cusp_candidate"]]
    assert len(cusps) == 1, reason
    assert max(abs(v) for v in cusps[0]["theta"]) < 0.05
    assert states[-1][0] < 0                                # continued through the cusp onto the other side


def test_branch_turns_at_the_logistic_period3_saddle_node():
    """Continuing the stable logistic 3-cycle toward smaller r, the branch turns at r = 1 + sqrt(8) and the
    crossing there is typed as a real +1."""
    def logistic3(x, theta):
        r = theta[0]
        for _ in range(3):
            x = r * x * (1 - x)
        return x

    r0 = 3.835
    x = 0.5
    for _ in range(3000):
        x = r0 * x * (1 - x)
    records, reason, _ = bc.trace_branch(family(logistic3, 1), np.array([x]), [r0, 0.0], free=0, direction=-1,
                                         steps=200, ds=0.002, ds_max=0.01, tol=1e-12, beyond=0.003)
    turns = [r for r in records if "turn_lambda_estimate" in r]
    assert turns, reason
    assert abs(turns[0]["turn_lambda_estimate"] - (1 + np.sqrt(8))) < 1e-4
    crossings = bc.classify_crossings(records)
    assert [c["type"] for c in crossings] == ["real +1 (fold)"]
    re_, im_ = crossings[0]["crossing_multiplier"][:2]
    assert im_ == 0 and 1 < re_ < 1.5                       # just past the turn, on the saddle branch


def test_branch_types_the_henon_flip():
    """Along a at b = 0.3 the Henon fixed point loses stability at a = 0.3675 with a multiplier -1."""
    x0 = henon_fixed_point(0.2, 0.3)
    records, reason, _ = bc.trace_branch(family(henon, 2), x0, [0.2, 0.3], free=0, direction=1, steps=40,
                                         ds=0.02, ds_max=0.05, tol=1e-12, bounds=(0.0, 0.6))
    crossings = bc.classify_crossings(records)
    assert [c["type"] for c in crossings] == ["real -1 (flip)"], reason
    a, b = crossings[0]["between_log10_T"]
    assert min(a, b) < 0.3675 < max(a, b)
    assert not any(r["turned_since_previous"] for r in records)


def test_gmres_tolerance_below_the_jvp_accuracy_is_never_met():
    """With JVPs rounded to float32, GMRES cannot reach a relative residual of 1e-10: every g evaluation then
    runs to its full budget and is counted as unconverged. At 1e-6 it converges, and the curve is still found."""
    class Rounded(bc.Family):
        def jvp_x(self, x, theta, v):
            return super().jvp_x(x, theta, v).astype(np.float32).astype(np.float64)

    b0 = 0.3
    x0 = henon_fixed_point(0.37, b0)
    runs = {}
    for gm in (1e-10, 1e-6):
        records, reason, _ = bc.trace_curve(Rounded(henon, (2,), torch.float64), x0, [0.37, b0], "flip", fix=1,
                                            direction=-1, steps=12, ds=0.05, ds_max=0.1, gtol=1e-6, tol=1e-9,
                                            gmres_tol=gm, bounds=((0.0, 3.0), (-0.6, 0.6)))
        assert len(records) == 12, reason
        for r in records:
            a, b = r["theta"]
            assert abs(a - 0.75 * (1 - b) ** 2) < 1e-6
        runs[gm] = records[1:]
    assert all(r["gmres_unconverged"] > 0 for r in runs[1e-10])
    assert all(r["gmres_unconverged"] == 0 for r in runs[1e-6])
    assert sum(r["jvps"] for r in runs[1e-10]) > 2 * sum(r["jvps"] for r in runs[1e-6])
