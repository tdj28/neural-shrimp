import numpy as np
import pytest

from neural_shrimp.diagnostics import certify_cycle
from neural_shrimp.gdmaps import (
    STATUS_CHAOTIC, STATUS_DIVERGENT, STATUS_FIXED, STATUS_PERIODIC, GridMap,
    HenonGrid, TwoLayerNet, connected_components, iterate_and_screen, run_grid,
    screen_tails, sohl_dickstein_measure,
)
from neural_shrimp.models import TanhRegressor


def _data(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=n)


def _sd_net(d=4, width=3, activation="tanh"):
    return TwoLayerNet(d=d, width=width, activation=activation, in_scale=np.sqrt(2.0 / d),
                       out_scale=1.0 / width, bias=False, loss_scale=1.0)


def _central_gradient(net, theta, x, y, step=1e-6):
    grad = np.empty_like(theta)
    for i in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        grad[i] = (net.loss(plus[None], x, y)[0] - net.loss(minus[None], x, y)[0]) / (2 * step)
    return grad


def test_two_layer_net_matches_scalar_reference_exactly():
    reference = TanhRegressor(d=2, width=3, ridge=1e-3)
    net = TwoLayerNet(d=2, width=3, ridge=1e-3)
    x, y = _data(24, 2, 7)
    rng = np.random.default_rng(3)
    thetas = rng.normal(size=(5, net.n_parameters))
    batched_grad, batched_loss = net.gradient(thetas, x, y, return_loss=True)
    for theta, grad, loss in zip(thetas, batched_grad, batched_loss):
        np.testing.assert_allclose(grad, reference.gradient(theta, x, y), rtol=0, atol=1e-14)
        assert loss == pytest.approx(reference.loss(theta, x, y), abs=1e-14)
        np.testing.assert_allclose(net.hessian(theta, x, y), reference.hessian(theta, x, y),
                                   rtol=0, atol=1e-13)
    grid = GridMap(net, x, y, [0.1, 0.7], [0.3, 1.9])
    stepped = grid.step(thetas[:2])
    for i, (lh, lo) in enumerate([(0.1, 0.3), (0.7, 1.9)]):
        scalar = reference.training_map(x, y, lh, lo)
        np.testing.assert_allclose(stepped[i], scalar(thetas[i]), rtol=0, atol=1e-14)
        np.testing.assert_allclose(grid.rates[i], scalar.rates, rtol=0, atol=0)
        map_fn, jacobian = grid.cell_map(i)
        np.testing.assert_allclose(map_fn(thetas[i]), scalar(thetas[i]), rtol=0, atol=1e-14)
        np.testing.assert_allclose(jacobian(thetas[i]), scalar.jacobian(thetas[i]),
                                   rtol=0, atol=1e-12)


@pytest.mark.parametrize("activation", ["tanh", "identity"])
def test_gradient_and_hvp_match_finite_differences(activation):
    net = _sd_net(activation=activation)
    x, y = _data(net.n_parameters, net.d, 11)
    rng = np.random.default_rng(5)
    theta = rng.normal(size=net.n_parameters)
    grad = net.gradient(theta[None], x, y)[0]
    np.testing.assert_allclose(grad, _central_gradient(net, theta, x, y), rtol=1e-6, atol=1e-8)
    vector = rng.normal(size=net.n_parameters)
    hv = net.hvp(theta[None], x, y, vector[None])[0]
    h = 1e-6
    numeric = (net.gradient((theta + h * vector)[None], x, y)[0]
               - net.gradient((theta - h * vector)[None], x, y)[0]) / (2 * h)
    np.testing.assert_allclose(hv, numeric, rtol=1e-6, atol=1e-8)
    hessian = net.hessian(theta, x, y)
    np.testing.assert_allclose(hessian @ vector, hv, rtol=1e-10, atol=1e-12)


def test_gauss_newton_layer_maximum_matches_hessian_at_zero_residual():
    net = TwoLayerNet(d=2, width=3, ridge=0.0)
    x, _ = _data(24, 2, 13)
    theta = np.random.default_rng(2).normal(size=net.n_parameters)
    y = net.predict(theta[None], x)[:, 0]
    hidden_max, output_max = net.gauss_newton_layer_max(theta, x, y)
    hessian = net.hessian(theta, x, y)
    hid = net.hidden_parameter_count
    assert hidden_max == pytest.approx(np.linalg.eigvalsh(hessian[:hid, :hid])[-1], rel=1e-10)
    assert output_max == pytest.approx(np.linalg.eigvalsh(hessian[hid:, hid:])[-1], rel=1e-10)


def test_rate_vector_rejects_invalid_rates():
    net = TwoLayerNet(d=2, width=3)
    with pytest.raises(ValueError):
        net.rate_vector([0.1, -1.0], [0.1, 0.1])
    with pytest.raises(ValueError):
        net.rate_vector([0.1], [0.1, 0.2])


def test_screen_tails_reports_smallest_full_state_period():
    steps, dim = 384, 3
    t = np.arange(steps)
    fixed = np.ones((steps, dim))
    pattern3 = np.array([[0.0, 1, 2], [3, 4, 5], [6, 7, 8]])
    period3 = pattern3[t % 3]
    pattern64 = np.random.default_rng(1).normal(size=(64, dim))
    period64 = pattern64[t % 64]
    drift = np.column_stack([1e-3 * t, np.zeros(steps), np.zeros(steps)])
    nan = fixed.copy()
    nan[-1, 0] = np.nan
    only_last_pair = drift.copy()
    only_last_pair[-2:] = 5.0
    tail = np.stack([fixed, period3, period64, drift, nan, only_last_pair])
    period, residual = screen_tails(tail, max_period=64, min_repeats=6)
    np.testing.assert_array_equal(period, [1, 3, 64, 0, -1, 0])
    assert residual[0] == 0.0 and residual[1] == 0.0


def test_sohl_dickstein_measure_matches_published_rule():
    steps = 100
    decreasing = np.linspace(1.0, 0.1, steps)
    exploding = np.r_[np.geomspace(1.0, 1e9, steps - 1), np.inf]
    convergent, score = sohl_dickstein_measure(np.column_stack([decreasing, exploding]))
    np.testing.assert_array_equal(convergent, [True, False])
    assert score[0] == pytest.approx(-np.sum(decreasing))
    normalized = np.minimum(exploding.copy(), 1e6)
    normalized[~np.isfinite(exploding)] = 1e6
    assert score[1] == pytest.approx(np.sum(1.0 / normalized))


def test_henon_positive_control_reproduces_analytic_boundaries_and_exponent():
    b = 0.3
    first_flip = 3 * (1 - b) ** 2 / 4
    second_flip = ((1 + b) ** 2 + 4 * (1 - b) ** 2) / 4
    a = np.array([first_flip - 0.02, first_flip + 0.02, second_flip - 0.02,
                  second_flip + 0.02, 1.4])
    grid = HenonGrid(a, np.full_like(a, b))
    initial = np.zeros((len(a), 2))
    result = iterate_and_screen(grid.step_subset, grid.jvp_subset, initial, 30000,
                                tail_steps=512, ftle_steps=20000, ftle_align=1000,
                                sd_horizons=(), atol=1e-10, rtol=1e-8)
    np.testing.assert_array_equal(result["period"][:4], [1, 2, 2, 4])
    np.testing.assert_array_equal(result["status"][:4],
                                  [STATUS_FIXED, STATUS_PERIODIC, STATUS_PERIODIC, STATUS_PERIODIC])
    assert result["status"][4] == STATUS_CHAOTIC
    # Published largest Lyapunov exponent of the classical attractor: about 0.419.
    assert result["ftle"][4] == pytest.approx(0.419, abs=0.01)
    assert np.all(result["ftle"][:4] < 0)


def test_run_grid_classifies_small_and_divergent_rates_and_certifies_fixed_point():
    net = TwoLayerNet(d=2, width=3, ridge=1e-3)
    x, y = _data(24, 2, 21)
    theta0 = TanhRegressor(2, 3).initialize(seed=4)
    hidden_max, output_max = net.gauss_newton_layer_max(theta0, x, y)
    lr_hidden = np.array([0.5, 1.0, 0.5])
    lr_output = np.array([0.5, 1.0, 400.0 * 2.0 / output_max])
    grid = GridMap(net, x, y, lr_hidden, lr_output)
    result = run_grid(grid, theta0, 20000, tail_steps=384, ftle_steps=1000, ftle_align=200,
                      sd_horizons=(100,))
    assert result["status"][2] == STATUS_DIVERGENT
    assert result["diverged_at"][2] >= 0
    fixed = np.flatnonzero(result["status"] == STATUS_FIXED)
    assert len(fixed) >= 1
    index = int(fixed[0])
    map_fn, jacobian = grid.cell_map(index)
    certificate = certify_cycle(map_fn, result["final_state"][index], 1, jacobian=jacobian)
    assert certificate["verified"] and certificate["attracting"]
    assert result["sd_convergent_100"][index]
    for key in ("status", "period", "ftle", "final_loss", "tail_mean_loss", "sd_score_100"):
        assert result[key].shape == (3,)
    assert result["final_state"].shape == (3, net.n_parameters)


def test_connected_components_uses_eight_connectivity_and_equal_labels():
    labels = np.array([[3, 3, 0, 5],
                       [0, 3, 0, 5],
                       [3, 0, 5, 5],
                       [0, 0, 0, 3]])
    mask = labels > 0
    components = connected_components(labels, mask)
    assert components[0, 0] == components[0, 1] == components[1, 1] == components[2, 0]
    assert components[0, 3] == components[2, 2] == components[2, 3]
    assert components[3, 3] != components[0, 0]
    assert np.all(components[~mask] == -1)
    assert len(np.unique(components[mask])) == 3
