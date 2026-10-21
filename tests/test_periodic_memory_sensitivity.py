import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import periodic_memory_sensitivity as sensitivity


def nonlinear_cycle(beta=0.):
    """A stable primitive logistic two-cycle driving a contracting coordinate."""
    r, c, q, d = 3.2, .2, .13, .3

    def f(x):
        original = np.array([r*x[0]*(1-x[0]), c*x[1]+q*x[0]**2+d*x[1]**2])
        return (1-beta)*original+beta*x

    def jacobian(x):
        return (1-beta)*np.array([[r*(1-2*x[0]), 0.], [2*q*x[0], c+2*d*x[1]]])+beta*np.eye(2)

    solved = root(lambda x: f(f(x))-x, [.51, .09], tol=1e-12)
    x = solved.x
    states = np.stack([x, f(x), f(f(x))])
    assert np.linalg.norm(states[-1]-states[0]) < 1e-12
    assert np.linalg.norm(states[1]-states[0]) > .1
    matrices = [jacobian(point) for point in states[:-1]]
    monodromy = matrices[1] @ matrices[0]
    assert max(abs(np.linalg.eigvals(monodromy))) < 1
    callbacks = [lambda v, matrix=matrix: matrix @ v for matrix in matrices]

    def hessian(eta, v):
        return (1-beta)*np.vstack([-2*r*eta[0]*v[0], 2*q*eta[0]*v[0]+2*d*eta[1]*v[1]])

    return states, monodromy, callbacks, [hessian, hessian]


def test_total_sensitivity_of_nonlinear_primitive_cycle_matches_resolved_orbits():
    states, matrix, callbacks, hessians = nonlinear_cycle()
    result = sensitivity.periodic_memory_trace_sensitivity(matrix, states, callbacks, hessians)
    assert result["passed"]
    assert result["orbit_tangent"]["tangent_relative_closure"] < 1e-12
    assert np.linalg.norm(result["orbit_tangent"]["tangents"][0]) > .01
    slopes, eta_errors = [], []
    for step in [1e-3, 3e-4, 1e-4]:
        minus = nonlinear_cycle(-step)
        plus = nonlinear_cycle(step)
        slopes.append((np.trace(plus[1])-np.trace(minus[1]))/(2*step))
        fd_eta = (plus[0]-minus[0])/(2*step)
        eta_errors.append(np.linalg.norm(fd_eta-result["orbit_tangent"]["tangents"]))
    errors = np.abs(np.array(slopes)-result["trace_derivatives"]["total"])
    assert errors[-1] < 1e-6
    assert errors[-1] < errors[0]/50
    assert eta_errors[-1] < eta_errors[0]/50
    assert result["trace_derivatives"]["orbit_mediated"] != pytest.approx(0., abs=1e-3)
    assert result["trace_derivatives"]["total"] == pytest.approx(
        result["trace_derivatives"]["explicit"]+result["trace_derivatives"]["orbit_mediated"], abs=1e-14)


def test_nonlinear_total_response_is_covariant_under_constant_coordinates():
    states, matrix, callbacks, hessians = nonlinear_cycle()
    original = sensitivity.periodic_memory_trace_sensitivity(matrix, states, callbacks, hessians)
    transform = np.array([[1., 2.], [-.2, 1.]])
    inverse = np.linalg.inv(transform)
    changed_j = [lambda v, callback=callback: transform @ callback(inverse @ v) for callback in callbacks]
    changed_h = [lambda eta, v, callback=callback: transform @ callback(inverse @ eta, inverse @ v)
                 for callback in hessians]
    changed = sensitivity.periodic_memory_trace_sensitivity(
        transform @ matrix @ inverse, states @ transform.T, changed_j, changed_h)
    for name in ("explicit", "orbit_mediated", "total"):
        assert changed["trace_derivatives"][name] == pytest.approx(original["trace_derivatives"][name], rel=1e-10)
    np.testing.assert_allclose(changed["orbit_tangent"]["tangents"],
                               original["orbit_tangent"]["tangents"] @ transform.T, atol=1e-12)


def test_cyclic_phase_keeps_both_derivative_contributions():
    states, matrix, callbacks, hessians = nonlinear_cycle()
    original = sensitivity.periodic_memory_trace_sensitivity(matrix, states, callbacks, hessians)
    rotated = np.stack([states[1], states[2], states[1]])
    rotated_matrix = callbacks[0](callbacks[1](np.eye(2)))
    changed = sensitivity.periodic_memory_trace_sensitivity(
        rotated_matrix, rotated, callbacks[::-1], hessians[::-1])
    for name in ("explicit", "orbit_mediated", "total"):
        assert changed["trace_derivatives"][name] == pytest.approx(original["trace_derivatives"][name], rel=1e-11)
    np.testing.assert_allclose(changed["orbit_tangent"]["tangents"][0],
                               original["orbit_tangent"]["tangents"][1], atol=1e-12)


def test_memory_does_not_move_a_fixed_point_but_changes_its_derivative():
    matrix = np.array([[.3, .4], [0., -.2]])
    state = np.array([.2, -.1])
    result = sensitivity.periodic_memory_trace_sensitivity(
        matrix, np.stack([state, state]), [lambda v: matrix @ v],
        [lambda eta, v: np.vstack([2*eta[0]*v[0], -3*eta[1]*v[1]])])
    np.testing.assert_array_equal(result["orbit_tangent"]["tangents"], np.zeros((2, 2)))
    assert result["trace_derivatives"]["orbit_mediated"] == 0
    assert result["trace_derivatives"]["total"] == pytest.approx(2-np.trace(matrix))


def test_nonclosed_orbit_and_inconsistent_monodromy_fail():
    states, matrix, callbacks, _ = nonlinear_cycle()
    wrong = states.copy()
    wrong[-1, 0] += 1e-5
    with pytest.raises(ValueError, match="orbit failed the closure"):
        sensitivity.periodic_memory_tangent(matrix, wrong, callbacks)
    with pytest.raises(ValueError, match="tangent failed closure"):
        sensitivity.periodic_memory_tangent(matrix+.03*np.eye(2), states, callbacks)
    with pytest.raises(ValueError, match="resolvent is ill-conditioned"):
        sensitivity.periodic_memory_tangent(np.eye(2), states, callbacks)


def test_bad_hessian_or_callback_shape_is_not_a_zero_contribution():
    states, matrix, callbacks, hessians = nonlinear_cycle()
    tangent = sensitivity.periodic_memory_tangent(matrix, states, callbacks)
    with pytest.raises(ValueError):
        sensitivity.memory_product_derivatives(callbacks, [lambda eta, v: np.nan*v]*2,
                                               tangent["tangents"], np.eye(2))
    with pytest.raises(ValueError):
        sensitivity.memory_product_derivatives(callbacks, hessians[:1], tangent["tangents"], np.eye(2))
    with pytest.raises(ValueError):
        sensitivity.periodic_memory_tangent(matrix, states, [lambda v: np.zeros(2)]*2)
