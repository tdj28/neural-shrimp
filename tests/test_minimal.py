import numpy as np
import pytest

from neural_shrimp.minimal import (EngineeredTwoCycle, TwoModeFactorization, dominant_pair, follow_orbit,
                                   iterate)


def _fd_gradient(f, x, h=1e-6):
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])


@pytest.mark.parametrize("c", [0.0, 0.3, 0.9])
def test_two_mode_derivatives_match_finite_differences(c):
    model = TwoModeFactorization(c=c, lam=0.05)
    rng = np.random.default_rng(1)
    for _ in range(3):
        theta = rng.normal(size=4)
        np.testing.assert_allclose(model.gradient(theta), _fd_gradient(model.loss, theta), rtol=1e-6, atol=1e-8)
        fd_hess = np.array([_fd_gradient(lambda x: model.gradient(x)[i], theta) for i in range(4)])
        np.testing.assert_allclose(model.hessian(theta), fd_hess, rtol=1e-6, atol=1e-7)
        rates = (0.3, 0.7)
        fd_jac = np.array([_fd_gradient(lambda x: model.step(x, *rates)[i], theta) for i in range(4)])
        np.testing.assert_allclose(model.jacobian(theta, *rates), fd_jac, rtol=1e-6, atol=1e-7)


def test_batched_gradient_matches_rows():
    model = TwoModeFactorization(c=0.6, lam=0.2)
    thetas = np.random.default_rng(2).normal(size=(5, 4))
    np.testing.assert_allclose(model.gradient(thetas), np.array([model.gradient(t) for t in thetas]))
    np.testing.assert_allclose(model.loss(thetas), np.array([model.loss(t) for t in thetas]))


@pytest.mark.parametrize("c,lam", [(0.0, 0.05), (0.6, 0.05), (0.9, 0.2)])
def test_balanced_point_is_a_strict_minimum(c, lam):
    model = TwoModeFactorization(c=c, lam=lam)
    np.testing.assert_allclose(model.gradient(model.theta_star), 0.0, atol=1e-12)
    eig = np.linalg.eigvalsh(model.hessian(model.theta_star))
    assert eig.min() > 0
    # imbalance modes (a - b) have curvature 2 lam; balanced modes 2 eig(diag(s) C diag(s))
    assert np.sum(np.isclose(eig, 2 * lam, rtol=1e-10)) == 2
    s = np.sqrt(np.asarray(model.z_star))
    np.testing.assert_allclose(np.sort(eig[~np.isclose(eig, 2 * lam, rtol=1e-10)]),
                               np.sort(2 * np.linalg.eigvalsh(np.diag(s) @ model.C @ np.diag(s))), rtol=1e-10)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_flip_rate_separates_attracting_and_unstable_minimum(ratio):
    model = TwoModeFactorization(c=0.3, lam=0.05)
    eta = model.flip_rate(ratio)

    def radius(scale):
        jac = model.jacobian(model.theta_star, scale * eta, scale * ratio * eta)
        return np.max(np.abs(np.linalg.eigvals(jac)))

    assert radius(0.999) < 1.0 < radius(1.001)
    np.testing.assert_allclose(radius(1.0), 1.0, atol=1e-9)


@pytest.mark.parametrize("kappa", [0.0, 0.3])
def test_engineered_cycle_has_prescribed_two_step_derivative(kappa):
    r, phi = 0.97, 2 * np.pi * 0.13
    model = EngineeredTwoCycle(r, phi, kappa)
    np.testing.assert_allclose(model.step(model.p0), model.p1, atol=1e-12)
    np.testing.assert_allclose(model.step(model.p1), model.p0, atol=1e-12)
    s0, s1 = model.jacobian(model.p0), model.jacobian(model.p1)
    np.testing.assert_allclose(s0, s0.T, atol=1e-12)
    np.testing.assert_allclose(s1, s1.T, atol=1e-12)
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    np.testing.assert_allclose(s1 @ s0, r * rotation, atol=1e-10)
    x = np.array([0.3, -0.4])
    fd_hess = np.array([_fd_gradient(lambda p: model.gradient(p)[i], x) for i in range(2)])
    np.testing.assert_allclose(model.hessian(x), fd_hess, rtol=1e-6, atol=1e-6)


def test_follow_orbit_types_the_engineered_neimark_sacker():
    phi = 2 * np.pi * 0.13

    def family(s):
        model = EngineeredTwoCycle(0.95 + s, phi)
        return model.step, model.jacobian

    result = follow_orbit(family, EngineeredTwoCycle.p0, 2, s_max=0.1, h0=0.005)
    assert result["type"] == "neimark_sacker"
    np.testing.assert_allclose(0.95 + result["s_boundary"], 1.0, atol=1e-7)
    mu = complex(*result["dominant"])
    np.testing.assert_allclose(abs(np.angle(mu)) / (2 * np.pi), 0.13, atol=1e-6)
    _, pair = dominant_pair(result["last_multipliers"])
    assert pair


def test_iterate_reports_divergence():
    assert iterate(lambda x: 2.0 * x + 1.0, np.array([1.0]), 100) is None
    np.testing.assert_allclose(iterate(lambda x: 0.5 * x, np.array([1.0]), 3), [0.125])
