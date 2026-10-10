import copy
import numpy as np
import pytest
from neural_shrimp.tiny_lm_response import run_case, profile_forecast

def exercise(gradient, hvp, ce, theta=np.array([0.8, 0.5]), break_at=None):
    saved = {}
    calls = dict(gradient=0, hvp=0, ce=0)
    checks = 0

    def wrap(name, fun):

        def f(*args):
            calls[name] += 1
            return fun(*args)
        return f

    def check():
        nonlocal checks
        checks += 1
        if break_at is not None and checks == break_at:
            raise TimeoutError('synthetic interruption')
    try:
        out = run_case(theta, 0.1, wrap('gradient', gradient), wrap('hvp', hvp), wrap('ce', ce), 0.0, np.array([0.3, -0.7]), lambda n, v: saved.__setitem__(n, v.copy()), check)
    except TimeoutError:
        return (saved, calls, None)
    return (saved, calls, out)

def test_quadratic_response_exact_and_operations_complete():
    H = np.array([[2.0, 0.3], [0.3, 1.0]])
    saved, calls, result = exercise(lambda x: H @ x, lambda x, v: H @ v, lambda x: 0.5 * x @ H @ x)
    assert calls == dict(gradient=145, hvp=64, ce=24)
    assert all((r['mechanism_qualified'] and r['frozen_vector_error'] < 1e-09 for r in result['rows']))
    assert all((r['training_ce_slope_resolved'] for r in result['rows']))
    assert saved['reference'].shape == (17, 2) and saved['direction-1-epsilon-1-minus'].shape == (17, 2)

def test_changing_curvature_resolves_transport_but_falsifies_frozen():
    _, _, result = exercise(lambda x: x ** 3, lambda x, v: 3 * x * x * v, lambda x: float(np.sum(x ** 4)) / 4)
    row = next((r for r in result['rows'] if r['direction'] == 'gradient' and r['horizon'] == 16))
    assert row['mechanism_qualified'] and row['frozen_vector_error'] > 0.2

def test_wrong_native_derivative_blocks_later_mechanism():
    _, _, result = exercise(lambda x: x ** 3, lambda x, v: 0.5 * 3 * x * x * v, lambda x: float(np.sum(x ** 4)) / 4)
    assert not any((r['mechanism_qualified'] for r in result['rows']))

def test_interrupted_reference_retains_unique_latest_state():
    saved, calls, result = exercise(lambda x: x, lambda x, v: v, lambda x: float(x @ x) / 2, break_at=6)
    assert result is None and calls['gradient'] == 5 and (saved['reference'].shape == (6, 2))
    np.testing.assert_allclose(saved['reference'][-1], 0.9 ** 5 * np.array([0.8, 0.5]))


def test_forecast_prices_both_arms_and_rejects_missing_or_bad_measurements():
    rows = [dict(arm=arm, times=dict(gradient=[0.01, 0.02], hvp=[0.03, 0.04], ce=[0.001, 0.002])) for arm in (8, 256)]
    assert profile_forecast(rows) == pytest.approx(4 * 2 * (145 * 0.02 + 64 * 0.04 + 24 * 0.002) + 20)
    bad = copy.deepcopy(rows)
    bad[1]['times']['hvp'][0] = float('nan')
    with pytest.raises(ValueError):
        profile_forecast(bad)
    with pytest.raises(ValueError):
        profile_forecast(rows[:1])
    bad = copy.deepcopy(rows)
    bad[1]['arm'] = 8
    with pytest.raises(ValueError):
        profile_forecast(bad)
    bad = copy.deepcopy(rows)
    bad[0]['times']['ce'] = [0.1]
    with pytest.raises(ValueError):
        profile_forecast(bad)


def test_interrupted_branch_retains_completed_training_ce():
    saved, calls, result = exercise(lambda x: x, lambda x, v: v, lambda x: float(x @ x) / 2, break_at=51)
    assert result is None and calls == dict(gradient=18, hvp=64, ce=1)
    acquired = saved['direction-0-epsilon-0-plus-ce']
    assert acquired.shape == (1, 2) and acquired[0, 0] == 1
    state = saved['direction-0-epsilon-0-plus'][-1]
    assert acquired[0, 1] == pytest.approx(float(state @ state) / 2)
