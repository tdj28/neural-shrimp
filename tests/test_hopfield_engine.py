"""Non-outcome numerical checks for the native continuous-flow integrator.

Target parameter-plane scans are run only by the prospectively frozen driver.
These tests use the artificial alpha=gamma=0 fixture and algebraic identities.
"""
from __future__ import annotations

import ctypes
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the native engine checks")
    root = pathlib.Path(__file__).resolve().parents[1]
    dest = tmp_path_factory.mktemp("hopfield") / "libhopfield.so"
    subprocess.run([compiler, "-O3", "-std=c++17", "-dynamiclib" if sys.platform == "darwin" else "-shared",
                    "-fPIC", str(root / "native/hopfield.cpp"), "-o", str(dest)], check=True)
    lib = ctypes.CDLL(str(dest))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.hopfield_rhs.argtypes = [ctypes.c_double, ctypes.c_double, pointer, pointer]
    lib.hopfield_jacobian.argtypes = [ctypes.c_double, ctypes.c_double, pointer, pointer]
    lib.hopfield_integrate.argtypes = [ctypes.c_double, ctypes.c_double, pointer,
                                      ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, pointer, pointer, pointer]
    lib.hopfield_integrate.restype = ctypes.c_int
    return lib


def python_rhs(_time, state, alpha=0., gamma=0.):
    weights = np.array([[1.5, 2.9, alpha], [-1.5, 1.18, 0.], [gamma, -22., .47]])
    return -state + weights @ np.tanh(state)


def integrate(engine, state, dt, burn_steps=0, sample_steps=0, capacity=32):
    final = np.empty(3)
    events = np.empty((capacity, 4))
    stats = np.empty(20)
    count = engine.hopfield_integrate(0., 0., np.asarray(state, dtype=np.float64), dt,
                                     burn_steps, sample_steps, capacity, final, events, stats)
    return count, final, events[:max(count, 0)], stats


@pytest.mark.parametrize("alpha,gamma", [(0., 0.), (.123, -.456)])
def test_rhs_and_jacobian_against_independent_formula(engine, alpha, gamma):
    state = np.array([.13, -.09, .2])
    rhs, jac = np.empty(3), np.empty((3, 3))
    engine.hopfield_rhs(alpha, gamma, state, rhs)
    engine.hopfield_jacobian(alpha, gamma, state, jac)
    np.testing.assert_allclose(rhs, python_rhs(0., state, alpha, gamma), rtol=2e-15, atol=2e-15)
    eps = 1e-6
    numerical = np.column_stack([(python_rhs(0., state + eps*e, alpha, gamma) - python_rhs(0., state - eps*e, alpha, gamma))/(2*eps)
                                 for e in np.eye(3)])
    np.testing.assert_allclose(jac, numerical, rtol=2e-9, atol=2e-9)


def test_rk4_order_and_independent_flow(engine):
    initial = np.array([.13, -.09, .2])
    target = solve_ivp(python_rhs, (0., .8), initial, method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
    errors = []
    for dt in [.1, .05, .025, .0125]:
        count, final, _, stats = integrate(engine, initial, dt, sample_steps=round(.8/dt))
        assert count >= 0
        assert stats[3] == pytest.approx(.8)
        assert np.isnan(stats[1])  # No exponent before tangent alignment.
        errors.append(np.linalg.norm(final-target))
    assert 12 < errors[0]/errors[1] < 20
    assert 12 < errors[1]/errors[2] < 20
    assert errors[-1] < 2e-8


def test_burn_sampling_boundary_preserves_flow(engine):
    initial = np.array([.13, -.09, .2])
    _, a, _, _ = integrate(engine, initial, .01, burn_steps=80)
    _, b, _, _ = integrate(engine, initial, .01, burn_steps=40, sample_steps=40)
    _, c, _, _ = integrate(engine, initial, .01, sample_steps=80)
    np.testing.assert_allclose(a, b, rtol=0., atol=1e-14)
    np.testing.assert_allclose(a, c, rtol=0., atol=1e-14)


def test_equilibrium_moments_and_tangent_duration(engine):
    count, final, events, stats = integrate(engine, np.zeros(3), .1, sample_steps=1100)
    assert count == 0 and len(events) == 0
    np.testing.assert_array_equal(final, 0.)
    np.testing.assert_array_equal(stats[4:18], 0.)
    assert stats[2] == pytest.approx(10.)
    assert np.isfinite(stats[1])
    assert stats[18] == 0 and stats[19] == 1100


def test_invalid_parameters_return_error(engine):
    assert integrate(engine, [0., 0., 0.], 0.)[0] == -1
    assert integrate(engine, [np.nan, 0., 0.], .01)[0] == -1


def test_peak_interpolation_and_chronological_ring_buffer(engine):
    initial = np.array([.13, -.09, .2])
    def maximum(_time, state):
        return python_rhs(_time, state)[2]
    maximum.direction = -1
    reference = solve_ivp(python_rhs, (0., 20.), initial, method="DOP853",
                          events=maximum, rtol=1e-12, atol=1e-14)
    count, _, events, stats = integrate(engine, initial, .01, sample_steps=2000)
    assert count == len(reference.t_events[0]) and count > 3
    np.testing.assert_allclose(events[:, 0], reference.t_events[0], rtol=0., atol=2e-5)
    np.testing.assert_allclose(events[:, 1:], reference.y_events[0], rtol=0., atol=2e-5)
    short_count, _, short_events, short_stats = integrate(engine, initial, .01, sample_steps=2000, capacity=3)
    assert short_count == 3 and short_stats[0] == stats[0]
    np.testing.assert_array_equal(short_events, events[-3:])


def test_variational_equation_against_matrix_exponential(engine):
    # At x=0 the variational equation is exactly the constant J=W-I ODE.
    jac = np.array([[.5, 2.9, 0.], [-1.5, .18, 0.], [0., -22., -.53]])
    tangent = np.array([1., np.sqrt(2.), np.sqrt(3.)])/np.sqrt(6.)
    aligned = expm(100.*jac) @ tangent
    aligned /= np.linalg.norm(aligned)
    expected = np.log(np.linalg.norm(expm(10.*jac) @ aligned))/10.
    _, _, _, stats = integrate(engine, np.zeros(3), .005, sample_steps=22000)
    assert stats[1] == pytest.approx(expected, rel=2e-6, abs=2e-7)
