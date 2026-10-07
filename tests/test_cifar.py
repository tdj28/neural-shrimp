import numpy as np
import pytest
import torch

from neural_shrimp import cifar


def tiny(seed=3, lam=1e-2):
    rng = np.random.default_rng(seed)
    model = cifar.MLP(d=5, width=3, k=4)
    x = rng.normal(size=(7, 5))
    y = cifar.one_hot(rng.integers(0, 4, size=7))[:, :4]
    problem = cifar.Problem(model, x, y, lam)
    theta = torch.as_tensor(rng.normal(size=model.n_parameters) * 0.7)
    return model, problem, theta, rng


def test_gradient_and_hvp_match_finite_differences():
    model, problem, theta, rng = tiny()
    v = torch.as_tensor(rng.normal(size=model.n_parameters))
    h = 1e-6
    fd_grad = (problem.loss(theta + h * v) - problem.loss(theta - h * v)) / (2 * h)
    assert abs(float(fd_grad) - float(problem.grad(theta) @ v)) < 1e-7
    fd_hvp = (problem.grad(theta + h * v) - problem.grad(theta - h * v)) / (2 * h)
    np.testing.assert_allclose(problem.hvp(theta, v).numpy(), fd_hvp.numpy(), atol=1e-6)


def test_two_step_jvp_matches_finite_differences():
    model, problem, theta, rng = tiny()
    eta = 0.3
    v = torch.as_tensor(rng.normal(size=model.n_parameters))
    h = 1e-6

    def f2(t):
        return problem.update(problem.update(t, eta), eta)

    fd = (f2(theta + h * v) - f2(theta - h * v)) / (2 * h)
    np.testing.assert_allclose(problem.jvp_two_step(theta, v, eta).numpy(), fd.numpy(), atol=1e-6)


def test_arnoldi_multipliers_match_dense_eigenvalues():
    model, problem, theta, rng = tiny(seed=5, lam=1e-3)
    eta = 0.9

    def f2(t):
        return problem.update(problem.update(t, eta), eta)

    dense = torch.func.jacfwd(f2)(theta).numpy()
    values = np.linalg.eigvals(dense)
    out = cifar.two_step_multipliers(problem, theta.numpy(), eta, k=4, tol=1e-12)
    lm = np.array([complex(*v) for v in out["modes"]["LM"]])
    np.testing.assert_allclose(np.sort(np.abs(lm))[::-1], np.sort(np.abs(values))[::-1][:4], rtol=1e-8)
    li = np.array([complex(*v) for v in out["modes"]["LI"]])
    assert np.isclose(li.imag.max(), values.imag.max(), rtol=1e-8, atol=1e-10)


def test_newton_period2_finds_a_fixed_point_solution():
    model, problem, theta, rng = tiny(seed=7, lam=5e-2)
    eta = 0.05
    for _ in range(4000):
        theta = problem.update(theta, eta)
    solved, history = cifar.newton_period2(problem, theta + 1e-4, eta, tol=1e-12)
    assert history[-1]["relative_residual"] < 1e-11
    assert float((problem.update(solved, eta) - solved).norm()) < 1e-9


def test_balanced_subsets_are_disjoint_balanced_and_reproducible():
    labels = np.repeat(np.arange(10), 300)
    a, b = cifar.balanced_subsets(labels, 28201, size=512)
    assert len(a) == len(b) == 512 and not set(a) & set(b)
    counts = np.bincount(labels[a], minlength=10)
    assert counts.min() == 51 and counts.max() == 52 and counts.sum() == 512
    np.testing.assert_array_equal(np.bincount(labels[b], minlength=10), counts)
    a2, b2 = cifar.balanced_subsets(labels, 28201, size=512)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    with pytest.raises(ValueError):
        cifar.balanced_subsets(labels, 29001)


def test_rotation_from_samples_recovers_a_planted_rotation():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(50, 2)))[0]
    k = np.arange(4000)
    angle = 2 * np.pi * 0.1234 * k
    curve = np.stack([np.cos(angle) + 0.1 * np.cos(2 * angle), np.sin(angle)], 1) @ basis.T
    rho, share = cifar.rotation_from_samples(curve + 3.0)
    assert abs(rho - 0.1234) < 1e-4 and share > 0.99
