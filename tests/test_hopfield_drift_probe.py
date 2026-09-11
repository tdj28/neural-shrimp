"""Non-target numerical fixtures for trace and local sensitivity probes."""
import ctypes
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hopfield_drift_probe import (ProbeFailure, build_trace, integrate_trace,
                                 local_parameter_sensitivity, phase_projected_gram)


def independent_rhs(_time, state, alpha=0., gamma=0.):
    weights = np.array([[1.5, 2.9, alpha], [-1.5, 1.18, 0.], [gamma, -22., .47]])
    return -state + weights @ np.tanh(state)


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    if shutil.which("c++") is None:
        pytest.skip("C++ compiler required")
    fn, info = build_trace(tmp_path_factory.mktemp("drift-trace"))
    assert len(info["included_model_sha256"]) == 64
    return fn


def test_trace_agrees_with_independent_dop853_and_retains_endpoints(trace):
    initial = np.array([.13, -.09, .2])
    times, states = integrate_trace(trace, 0., 0., initial, .01, duration=1.2, sample_dt=.1)
    expected = solve_ivp(independent_rhs, (0., 1.2), initial, t_eval=times,
                         method="DOP853", rtol=1e-12, atol=1e-14).y.T
    np.testing.assert_allclose(states, expected, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(states[0], initial)
    assert times[0] == 0. and times[-1] == pytest.approx(1.2)
    assert times.dtype == states.dtype == np.float64


def test_trace_stride_and_split_preserve_same_rk4_flow(trace):
    initial = [.13, -.09, .2]
    _, dense = integrate_trace(trace, 0., 0., initial, .01, duration=1., sample_dt=.01)
    _, sparse = integrate_trace(trace, 0., 0., initial, .01, duration=1., sample_dt=.1)
    _, first = integrate_trace(trace, 0., 0., initial, .01, duration=.5, sample_dt=.1)
    _, second = integrate_trace(trace, 0., 0., first[-1], .01, duration=.5, sample_dt=.1)
    np.testing.assert_array_equal(sparse, dense[::10])
    np.testing.assert_array_equal(np.vstack((first, second[1:])), sparse)


def test_zero_state_trace_and_invalid_arguments(trace):
    _, states = integrate_trace(trace, 0., 0., [0., 0., 0.], .01, duration=1.)
    np.testing.assert_array_equal(states, 0.)
    for changes in [dict(dt=0.), dict(sample_dt=.015), dict(duration=.95),
                    dict(initial=[np.nan, 0., 0.]), dict(alpha=np.inf)]:
        args = dict(fn=trace, alpha=0., gamma=0., initial=[0., 0., 0.], dt=.01, duration=1.)
        args.update(changes)
        with pytest.raises(ValueError):
            integrate_trace(**args)


def test_failure_keeps_partial_trace_payload(tmp_path):
    def failing_fn(alpha, gamma, initial, dt, steps, stride, count, states, final, progress):
        output = np.ctypeslib.as_array(states, shape=(count * 3,)).reshape(count, 3)
        output[0] = [.1, .2, .3]
        np.ctypeslib.as_array(progress, shape=(2,))[:] = [2, 1]
        return -2
    path = tmp_path / "failure.npz"
    with pytest.raises(ProbeFailure) as caught:
        integrate_trace(failing_fn, 0., 0., [.1, .2, .3], .01, duration=1., failure_path=path)
    assert caught.value.payload["native_status"] == -2
    with np.load(path, allow_pickle=False) as saved:
        assert saved["native_status"] == -2
        np.testing.assert_array_equal(saved["states"][0], [.1, .2, .3])
        assert np.isnan(saved["states"][1:]).all()


def test_native_invalid_status(trace):
    initial = np.zeros(3)
    states, final, progress = np.empty((2, 3)), np.empty(3), np.empty(2)
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    status = trace(0., 0., ptr(initial), -.01, 10, 10, 2, ptr(states), ptr(final), ptr(progress))
    assert status == -1


def test_native_nonfinite_step_retains_initial_row_and_progress(trace):
    # Artificially enormous step forces overflow at the non-target fixture.
    with pytest.raises(ProbeFailure) as caught:
        integrate_trace(trace, 0., 0., [.13, -.09, .2], 1e308,
                        duration=1e308, sample_dt=1e308)
    payload = caught.value.payload
    assert payload["native_status"] == -2
    np.testing.assert_array_equal(payload["states"][0], [.13, -.09, .2])
    np.testing.assert_array_equal(payload["progress"], [1., 1.])
    assert np.isnan(payload["states"][1:]).all()


def test_local_parameter_sensitivity_matches_independent_central_differences():
    initial = np.array([.13, -.09, .2])
    times = np.linspace(0., 1.2, 25)
    result = local_parameter_sensitivity(0., 0., initial, times)
    columns = []
    eps = 1e-6
    for direction in np.eye(2):
        solutions = []
        for sign in [-1., 1.]:
            a, g = sign * eps * direction
            solution = solve_ivp(lambda t, y: independent_rhs(t, y, a, g),
                                 (0., times[-1]), initial, t_eval=times,
                                 method="DOP853", rtol=1e-12, atol=1e-14, max_step=.05)
            solutions.append(np.tanh(solution.y.T))
        columns.append((solutions[1] - solutions[0]) / (2 * eps))
    numerical = np.stack(columns, axis=-1)
    np.testing.assert_allclose(result["activation_sensitivity"], numerical, rtol=2e-6, atol=2e-9)
    np.testing.assert_array_equal(result["activation_sensitivity"][0], 0.)
    assert np.linalg.norm(result["minimum_eigenvector"]) == pytest.approx(1.)
    assert result["minimum_eigenvector"][0] >= 0.
    np.testing.assert_allclose(result["gram"] @ result["minimum_eigenvector"],
                               result["eigenvalues"][0] * result["minimum_eigenvector"], atol=1e-14)


def test_zero_state_parameter_sensitivity_is_zero():
    result = local_parameter_sensitivity(0., 0., np.zeros(3), np.linspace(0., 1., 11))
    np.testing.assert_array_equal(result["states"], 0.)
    np.testing.assert_array_equal(result["activation_sensitivity"], 0.)
    np.testing.assert_array_equal(result["gram"], 0.)
    assert not result["minimum_eigenvector_unique"]


def test_single_phase_projection_removes_only_constant_time_shift():
    velocity = np.arange(1., 31.).reshape(10, 3) / 30.
    sensitivity = velocity[:, :, None] * np.array([2., -3.])[None, None, :]
    result = phase_projected_gram(sensitivity, velocity)
    np.testing.assert_allclose(result["gram"], 0., atol=1e-29)
    np.testing.assert_allclose(result["phase_coefficients"], [2., -3.])
    sensitivity *= np.arange(10.)[:, None, None]
    varied = phase_projected_gram(sensitivity, velocity)
    assert np.trace(varied["gram"]) > 1.
    flat, phase = sensitivity.reshape(-1, 2), velocity.ravel()
    expected = (flat.T @ flat - np.outer(flat.T @ phase, flat.T @ phase) / (phase @ phase)) / len(phase)
    np.testing.assert_allclose(varied["gram"], expected, rtol=1e-12, atol=1e-12)
