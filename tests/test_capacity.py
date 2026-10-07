import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neural_shrimp.capacity import (
    consistency, degree2_targets, hopfield_weights, memory_targets, ridge_capacities, simulate,
)


def test_simulate_matches_adaptive_reference_for_driven_hopfield():
    weights = hopfield_weights([1.07], [10.4])
    x0 = np.array([[0.1, -0.2, 0.3]])
    inputs = np.array([0.3, -0.7, 0.5, 0.1])
    eps, dt, hold = 0.2, 0.005, 200
    _, final, _ = simulate(weights, x0, inputs, eps, dt, hold, 4, feature="state")
    w = weights[0]

    state = x0[0].copy()
    for k, u in enumerate(inputs):
        def rhs(_t, x, u=u):
            return -x + w @ np.tanh(x) + eps * u * np.array([1.0, 0.0, 0.0])
        sol = solve_ivp(rhs, (0.0, dt * hold), state, method="DOP853", rtol=1e-12, atol=1e-12)
        state = sol.y[:, -1]
    np.testing.assert_allclose(final[0], state, rtol=0, atol=1e-8)


def test_leaky_integrator_memory_function_matches_analytic_capacity():
    # With W = 0 each coordinate is an exact leaky integrator of the held input.
    rng = np.random.default_rng(1)
    dt, hold, eps = 0.01, 50, 0.5
    a = np.exp(-dt * hold)
    weights = np.zeros((1, 3, 3))
    train, test = rng.uniform(-1, 1, 6000), rng.uniform(-1, 1, 6000)
    feats = []
    for stream in (train, test):
        f, _, _ = simulate(weights, np.zeros((1, 3)), stream, eps, dt, hold, 1, feature="state")
        feats.append(f[:, :, :1])  # neuron 1 only, sampled at the end of each interval
    targets_tr, targets_te = memory_targets(train, 8), memory_targets(test, 8)
    capacities = ridge_capacities(feats[0], targets_tr, feats[1], targets_te, ridge=1e-10)[0]
    expected = np.array([a ** (2 * d) * (1 - a * a) for d in range(1, 9)])
    np.testing.assert_allclose(capacities, expected, atol=0.02)


def test_exact_delay_line_has_unit_capacity_up_to_its_length():
    rng = np.random.default_rng(2)
    length = 5

    def delay_features(u):
        cols = [np.r_[np.zeros(d), u[:-d]] for d in range(1, length + 1)]
        return np.column_stack(cols)[None]

    train, test = rng.uniform(-1, 1, 4000), rng.uniform(-1, 1, 4000)
    caps = ridge_capacities(delay_features(train), memory_targets(train, 8),
                            delay_features(test), memory_targets(test, 8), ridge=1e-10)[0]
    np.testing.assert_allclose(caps[:length], 1.0, atol=1e-9)
    assert np.all(np.abs(caps[length:]) < 0.01)


def test_degree2_targets_are_recovered_from_quadratic_features():
    rng = np.random.default_rng(3)

    def quad(u):
        u1, u2 = np.r_[0.0, u[:-1]], np.r_[0.0, 0.0, u[:-2]]
        return np.column_stack([u1 * u1, u1 * u2, u2 * u2])[None]

    train, test = rng.uniform(-1, 1, 5000), rng.uniform(-1, 1, 5000)
    t_tr, labels = degree2_targets(train, 3, 3)
    t_te, _ = degree2_targets(test, 3, 3)
    caps = ridge_capacities(quad(train), t_tr, quad(test), t_te, ridge=1e-10)[0]
    lookup = dict(zip(labels, caps))
    assert lookup[("P2", 1)] == pytest.approx(1.0, abs=1e-9)
    assert lookup[("P2", 2)] == pytest.approx(1.0, abs=1e-9)
    assert lookup[("cross", 1, 2)] == pytest.approx(1.0, abs=1e-9)
    assert abs(lookup[("P2", 3)]) < 0.01 and abs(lookup[("cross", 2, 3)]) < 0.01


def test_total_capacity_respects_the_feature_bound_for_a_fading_memory_system():
    rng = np.random.default_rng(4)
    weights = np.zeros((1, 3, 3))
    weights[0] = 0.4 * rng.normal(size=(3, 3))
    train, test = rng.uniform(-1, 1, 5000), rng.uniform(-1, 1, 5000)
    feats = [simulate(weights, np.zeros((1, 3)), s, 1.0, 0.02, 20, 1)[0] for s in (train, test)]
    lin = ridge_capacities(feats[0], memory_targets(train, 30), feats[1], memory_targets(test, 30), 1e-8)
    t_tr, _ = degree2_targets(train, 10, 6)
    t_te, _ = degree2_targets(test, 10, 6)
    quad = ridge_capacities(feats[0], t_tr, feats[1], t_te, 1e-8)
    total = np.clip(lin, 0, None).sum() + np.clip(quad, 0, None).sum()
    assert total <= 3.0 + 0.1


def test_conditional_exponent_of_linear_decay_is_minus_one():
    weights = np.zeros((2, 3, 3))
    rng = np.random.default_rng(5)
    tangent = rng.normal(size=(2, 3))
    inputs = rng.uniform(-1, 1, 200)
    _, _, growth = simulate(weights, np.zeros((2, 3)), inputs, 0.3, 0.01, 10, 1, tangent=tangent)
    np.testing.assert_allclose(growth / (len(inputs) * 0.1), -1.0, atol=1e-6)


def test_consistency_is_one_for_identical_responses_and_low_for_independent_ones():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(2, 500, 4))
    b = a.copy()
    b[1] = rng.normal(size=(500, 4))
    values = consistency(a, b)
    assert values[0] == pytest.approx(1.0)
    assert abs(values[1]) < 0.15
