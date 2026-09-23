"""Pre-outcome synthetic arithmetic and artificial-model timing checks."""
import sys
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import boundary_behavior as b
import run_boundary_behavior as run
from hopfield_drift_probe import build_trace


def test_balanced_factorial_has_no_linear_parity():
    x = b.SIGNS
    y = x.prod(axis=1)
    w, intercept = b.fit_readout(x, y, 1e-4)
    np.testing.assert_allclose(x@w+intercept, 0, atol=1e-14)
    q = np.column_stack((x, y))
    w, intercept = b.fit_readout(q, y, 1e-4)
    assert np.mean((q@w+intercept-y)**2) < 1e-7


def test_heldout_scoring_retains_failure_and_recovers_known_signal():
    y = b.labels(8)
    states = np.tile((.2*y).reshape(8,4,1,3), (1,1,61,1))
    good = b.score(states[:4], states[4:], [20,25], 1e-4)
    assert min(good['r2']) > .999
    bad = b.score(states[:4], -states[4:], [20,25], 1e-4)
    assert max(bad['r2']) < -2.9


def test_no_cross_transect_pairs_and_decision_symmetry():
    points = [dict(alpha=a,gamma=g) for a in [1.,2.] for g in [0.,1.]]
    classes = ['periodic','irregular-supported']*2
    assert b.crossing_pairs(points, classes) == [(0,1),(2,3)]
    losses = [np.full(128,x) for x in [.2,.4,.2,.4]]
    result = b.compare_boundaries(points, classes, losses, .05)
    assert result['decision'].startswith('periodic advantage')
    result = b.compare_boundaries(points, classes, losses[::-1], .05)
    assert result['decision'].startswith('irregular advantage')
    assert b.compare_boundaries(points, ['unresolved']*4, losses,.05)['pairs'] == []


def test_context_split_is_disjoint_shared_reproducible():
    p = run.validate(__import__('json').loads(run.PLAN.read_text()))
    a = run.context_indices(p)
    np.testing.assert_array_equal(a, run.context_indices(p))
    for row in a:
        assert len(set(row)) == 96
        assert not set(row[:32]) & set(row[32:])


def test_two_pulse_matches_independent_solver_artificial_fixture(tmp_path):
    fn, info = build_trace(tmp_path)
    started = time.monotonic()
    for signs in b.SIGNS:
        native = run.two_pulse(fn, 0., 0., [.1,.2,.3], .1, signs, .0025)
        independent = run.independent_pulse(0., 0., [.1,.2,.3], .1, signs)
        assert native.shape == (61,3)
        np.testing.assert_allclose(native, independent, atol=2e-8, rtol=2e-8)
    # Artificial fixture only. Never time a target trajectory before its freeze.
    t = time.monotonic()
    for _ in range(16):
        run.two_pulse(fn, 0.,0.,[.1,.2,.3],.1,[-1.,1.],.00125)
    print({'artificial_fine_trajectory_seconds':(time.monotonic()-t)/16,
           'fixture_total_seconds':time.monotonic()-started})
