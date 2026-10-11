import numpy as np
import torch

from neural_shrimp.cifar import MLP, Problem
from neural_shrimp.cifar_reduced import (dense_hessian, join_state, newton_period2, reduce_problem,
                                         split_state, two_step)


def small_problem(seed=0, n=8, d=20, width=3, k=2, lam=1e-2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x -= x.mean(axis=0)
    y = rng.normal(size=(n, k))
    model = MLP(d=d, width=width, k=k)
    theta = rng.normal(scale=0.5, size=model.n_parameters)
    return Problem(model, x, y, lam), model, theta


def test_reduced_step_reproduces_full_step_exactly():
    problem, model, theta = small_problem()
    reduced, q = reduce_problem(problem)
    eta = 0.3
    theta_r, complement = split_state(theta, model, q)
    full_next = problem.update(torch.as_tensor(theta), eta).numpy()
    reduced_next = reduced.update(torch.as_tensor(theta_r), eta).numpy()
    joined = join_state(reduced_next, (1 - eta * problem.lam) * complement, model, q)
    np.testing.assert_allclose(joined, full_next, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(join_state(theta_r, complement, model, q), theta, atol=1e-13)


def test_dense_hessian_and_two_step_jacobian_match_autodiff():
    problem, model, theta = small_problem(1)
    reduced, q = reduce_problem(problem)
    theta_r, _ = split_state(theta, model, q)
    t = torch.as_tensor(theta_r)
    exact = torch.func.hessian(reduced.loss)(t).numpy()
    np.testing.assert_allclose(dense_hessian(reduced, theta_r, chunk=7), exact, rtol=1e-10, atol=1e-12)
    eta = 0.2
    t2, _, m = two_step(reduced, theta_r, eta, chunk=5)
    f2 = lambda v: reduced.update(reduced.update(v, eta), eta)
    np.testing.assert_allclose(t2, f2(t).numpy(), atol=1e-14)
    np.testing.assert_allclose(m, torch.func.jacfwd(f2)(t).numpy(), rtol=1e-9, atol=1e-11)


def test_newton_recovers_a_fixed_point_as_a_period_two_orbit():
    problem, model, theta = small_problem(2)
    reduced, q = reduce_problem(problem)
    theta_r, _ = split_state(theta, model, q)
    eta = 0.05
    t = torch.as_tensor(theta_r)
    for _ in range(20000):
        t = reduced.update(t, eta)
    perturbed = t.numpy() + 1e-4 * np.random.default_rng(3).normal(size=t.numel())
    solution, _, history = newton_period2(reduced, perturbed, eta, tol=1e-13, max_iter=10)
    assert history[-1] < 1e-13
    assert np.linalg.norm(solution - t.numpy()) < 1e-8
