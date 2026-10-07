import numpy as np
import pytest

from neural_shrimp.gdmaps import GridMap
from neural_shrimp.realdata import (
    DIGITS_SHA256,
    data_root,
    digits_task,
    load_digits_array,
    make_digits,
    torch_default_init,
)


def synthetic(rows=60, seed=0):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 17, size=(rows, 64)).astype(np.float64)
    pixels[:, 0] = 0.0  # a constant pixel, as in the real data
    labels = rng.integers(0, 10, size=rows)
    return pixels, labels


def test_task_standardizes_and_uses_signed_targets():
    pixels, labels = synthetic()
    x, y = digits_task(5, 40, task="halves", pixels=pixels, labels=labels)
    assert x.shape == (40, 64) and y.shape == (40,)
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(x[:, 1:].std(axis=0), 1.0)
    assert np.all(x[:, 0] == 0.0)
    assert set(np.unique(y)) <= {-1.0, 1.0}
    again, _ = digits_task(5, 40, task="halves", pixels=pixels, labels=labels)
    assert np.array_equal(x, again)
    _, parity = digits_task(5, 40, task="parity", pixels=pixels, labels=labels)
    assert not np.array_equal(y, parity) or np.all(y == parity)


def test_reserved_seeds_are_refused():
    pixels, labels = synthetic()
    with pytest.raises(ValueError):
        digits_task(29000, 10, pixels=pixels, labels=labels)
    with pytest.raises(ValueError):
        make_digits(29999, n=10, pixels=pixels, labels=labels)


def test_torch_default_bounds():
    pixels, labels = synthetic()
    net, _, _, theta = make_digits(3, width=5, n=30, pixels=pixels, labels=labels)
    s0, sb0, s1, sb1 = net.layer_slices()
    assert np.max(np.abs(theta[s0])) <= 1 / 8 and np.max(np.abs(theta[sb0])) <= 1 / 8
    assert np.max(np.abs(theta[s1])) <= 1 / np.sqrt(5) and np.max(np.abs(theta[sb1])) <= 1 / np.sqrt(5)
    assert np.array_equal(theta, torch_default_init(net, 3))


def test_gradient_hvp_and_step_jacobian_match_finite_differences():
    pixels, labels = synthetic()
    net, x, y, theta = make_digits(7, width=3, n=24, ridge=1e-3, pixels=pixels, labels=labels)
    theta = theta + 0.3 * np.random.default_rng(1).normal(size=theta.shape)
    p = net.n_parameters
    rng = np.random.default_rng(2)
    v = rng.normal(size=p)
    eps = 1e-6
    loss = lambda t: net.loss(t[None], x, y)[0]  # noqa: E731
    grad = net.gradient(theta[None], x, y)[0]
    fd = np.array([(loss(theta + eps * e) - loss(theta - eps * e)) / (2 * eps) for e in np.eye(p)])
    assert np.allclose(grad, fd, rtol=1e-6, atol=1e-8)
    hv = net.hvp(theta[None], x, y, v[None])[0]
    fd_hv = (net.gradient((theta + eps * v)[None], x, y)[0] - net.gradient((theta - eps * v)[None], x, y)[0]) / (2 * eps)
    assert np.allclose(hv, fd_hv, rtol=1e-5, atol=1e-7)
    eta = 0.37
    grid = GridMap(net, x, y, [eta], [eta])
    map_fn, jacobian = grid.cell_map(0)
    fd_j = (map_fn(theta + eps * v) - map_fn(theta - eps * v)) / (2 * eps)
    assert np.allclose(jacobian(theta) @ v, fd_j, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not (data_root() / "external" / "uci-digits" / "digits.csv.gz").exists(),
                    reason="digits file not present")
def test_real_digits_file():
    pixels, labels = load_digits_array()
    assert pixels.shape == (1797, 64) and labels.shape == (1797,)
    assert pixels.min() == 0 and pixels.max() == 16
    assert np.array_equal(np.unique(labels), np.arange(10))
    assert len(DIGITS_SHA256) == 64
