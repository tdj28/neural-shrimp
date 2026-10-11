import numpy as np
import pytest

from neural_shrimp.gdmaps import STATUS_FIXED, GridMap, TwoLayerNet, run_grid
from neural_shrimp.orbits import certify_orbit


def _henon(a, b):
    def map_fn(state):
        x, y = state
        return np.array([1.0 - a * x * x + y, b * x])

    def jacobian(state):
        return np.array([[-2.0 * a * state[0], 1.0], [b, 0.0]])

    return map_fn, jacobian


def test_henon_two_cycle_certificate_matches_analytic_multipliers():
    a, b = 0.6, 0.3
    root = np.sqrt(4 * a - 3 * (1 - b) ** 2)
    x1 = ((1 - b) + root) / (2 * a)
    x2 = ((1 - b) - root) / (2 * a)
    map_fn, jacobian = _henon(a, b)
    seed = np.array([x1, b * x2]) + 1e-6
    result = certify_orbit(map_fn, jacobian, seed, 2)
    assert result["verified"] and result["primitive_period"] == 2
    trace = 4 * a * a * x1 * x2 + 2 * b
    det = b * b
    disc = np.sqrt(complex(trace * trace - 4 * det))
    expected = max(abs((trace + disc) / 2), abs((trace - disc) / 2))
    assert result["spectral_radius"] == pytest.approx(expected, rel=1e-10)
    assert result["attracting"] == (expected < 1 - 1e-7)
    assert result["newton_converged"] and result["final_relative_correction"] <= 1e-10


def test_requesting_a_multiple_of_the_period_is_rejected_as_nonprimitive():
    a, b = 0.6, 0.3
    map_fn, jacobian = _henon(a, b)
    state = np.zeros(2)
    for _ in range(5000):
        state = map_fn(state)
    result = certify_orbit(map_fn, jacobian, state, 4)
    assert not result["verified"]
    assert result["reason"] == "nonprimitive_or_unseparated_period"
    assert result["min_divisor_separation"] < 1e-6


def test_a_remote_root_is_not_attributed_to_the_candidate():
    a, b = 0.6, 0.3
    map_fn, jacobian = _henon(a, b)
    result = certify_orbit(map_fn, jacobian, np.array([0.9, 0.2]), 1, candidate_tolerance=1e-3)
    assert not result["verified"]
    assert result["reason"] in {"root_remote_from_candidate", "nonprimitive_or_unseparated_period"}


def _uv_model(ridge=0.1):
    net = TwoLayerNet(d=1, width=1, activation="identity", in_scale=1.0, out_scale=1.0,
                      bias=False, loss_scale=1.0, ridge=ridge)
    return net, np.array([[1.0]]), np.array([1.0])


def test_analytic_gradient_descent_flip_boundary_through_the_batched_engine():
    # L(u, v) = (uv - 1)^2 + 0.05 (u^2 + v^2): balanced minimum u = v = s with
    # s^2 = 0.95; Hessian eigenvalues 3.8 and 0.2, so equal rates flip at 2 / 3.8.
    net, x, y = _uv_model()
    s = np.sqrt(0.95)
    hessian = net.hessian(np.array([s, s]), x, y)
    np.testing.assert_allclose(np.linalg.eigvalsh(hessian), [0.2, 3.8], atol=1e-12)
    edge = 2.0 / 3.8
    rates = np.array([edge - 0.02, edge + 0.02])
    grid = GridMap(net, x, y, rates, rates)
    result = run_grid(grid, np.array([s + 0.01, s - 0.01]), 20000, tail_steps=384,
                      ftle_steps=1000, ftle_align=100, sd_horizons=())
    assert result["status"][0] == STATUS_FIXED
    assert result["status"][1] != STATUS_FIXED
    map_fn, jacobian = grid.cell_map(0)
    certificate = certify_orbit(map_fn, jacobian, result["final_state"][0], 1)
    assert certificate["verified"] and certificate["attracting"]
    moduli = sorted(m["modulus"] for m in certificate["multipliers"])
    expected = sorted([abs(1 - rates[0] * 3.8), abs(1 - rates[0] * 0.2)])
    np.testing.assert_allclose(moduli, expected, rtol=1e-9)


def test_unequal_rate_flip_curve_is_reproduced():
    # Flip when det(2I - P H) = 0 with P = diag(a, b) and H = [[2, 1.8], [1.8, 2]].
    net, x, y = _uv_model()
    s = np.sqrt(0.95)
    a = 0.4
    h11 = h22 = 2.0
    h12 = 1.8
    det_h = h11 * h22 - h12 * h12
    # 4 - 2 (a h11 + b h22) + a b det_h = 0  =>  b = (4 - 2 a h11) / (2 h22 - a det_h)
    b_edge = (4 - 2 * a * h11) / (2 * h22 - a * det_h)
    grid = GridMap(net, x, y, [a, a], [0.97 * b_edge, 1.03 * b_edge])
    result = run_grid(grid, np.array([s + 0.01, s - 0.01]), 20000, tail_steps=384,
                      ftle_steps=1000, ftle_align=100, sd_horizons=())
    assert result["status"][0] == STATUS_FIXED
    assert result["status"][1] != STATUS_FIXED


def test_complex_dominant_condition_uses_adjoint_pairing_for_nonnormal_real_matrix():
    from scipy.linalg import eig
    # Similarity by diag(s,1) of a contraction/rotation. For either simple
    # eigenvalue, analytic condition = (s + 1/s)/2 = 2.6, not ~1/epsilon.
    s = 5.
    matrix = .8 * np.array([[.6, -s*.8], [.8/s, .6]])
    result = certify_orbit(lambda x: matrix @ x, lambda x: matrix, np.zeros(2), 1)
    assert result['verified'] and result['attracting'] and result['primitive_period'] == 1
    assert result['spectral_radius'] == pytest.approx(.8, abs=1e-14)
    assert result['dominant_eigenvalue_condition'] == pytest.approx((s+1/s)/2, rel=1e-13)
    values, left, right = eig(matrix, left=True, right=True)
    dominant = complex(result['multipliers'][0]['real'], result['multipliers'][0]['imag'])
    index = int(np.argmin(np.abs(values-dominant)))
    l, r = left[:, index], right[:, index]
    np.testing.assert_allclose(matrix.T @ l, np.conj(values[index])*l, rtol=1e-13, atol=1e-14)
    independent = np.linalg.norm(l)*np.linalg.norm(r)/abs(np.vdot(l, r))
    assert result['dominant_eigenvalue_condition'] == pytest.approx(independent, rel=1e-13)


def test_criticality_adjoint_pairing_is_already_correct_on_nonnormal_control():
    from neural_shrimp.criticality import critical_pair
    from scipy.linalg import eig
    matrix = .8 * np.array([[.6, -4.], [.16, .6]])
    mu, q, p, gap = critical_pair(matrix)
    np.testing.assert_allclose(matrix @ q, mu*q, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(matrix.T @ p, np.conj(mu)*p, rtol=1e-13, atol=1e-14)
    assert np.vdot(p, q) == pytest.approx(1., abs=1e-14)
    values, left, right = eig(matrix, left=True, right=True)
    j = int(np.argmin(np.abs(values-mu)))
    independent = np.linalg.norm(left[:, j])*np.linalg.norm(right[:, j])/abs(np.vdot(left[:, j], right[:, j]))
    assert np.linalg.norm(p)*np.linalg.norm(q) == pytest.approx(independent, rel=1e-13)
