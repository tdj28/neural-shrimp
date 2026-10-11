"""Exact input-span reduction of the CIFAR tanh MLP and dense two-step analysis (exploratory).

Full-batch gradient descent minimizes
``0.5 * sum((f(x) - y)^2) / n + 0.5 * lam * |theta|^2`` for a d-h-k tanh MLP.
First-layer gradients lie in the row space of the inputs X (n x d). With an
orthonormal basis Q (d x r) of a space containing that row space, write
``W1 = Q B + N`` with ``Q^T N = 0``. Then exactly

    B <- B - eta * (Z^T Delta + lam * B),    N <- (1 - eta * lam) * N,

with ``Z = X Q``, and ``Delta`` depends on ``W1`` only through ``X W1 = Z B``.
So ``(B, b1, W2, b2)`` follow the gradient-descent map of the same MLP on
inputs ``Z``, while ``N`` decays on its own with multiplier ``1 - eta lam``.
This module builds the reduced problem, maps states between the two
coordinate systems, and computes dense Hessians, two-step Jacobians, Newton
corrections of period-2 orbits and spectra in reduced coordinates.
"""

import numpy as np
import torch

from .cifar import MLP, Problem


def reduce_problem(problem):
    """Reduced problem on Z = X Q and the basis Q (all thin-QR columns kept)."""
    x = problem.x.numpy()
    q, _ = np.linalg.qr(x.T, mode="reduced")
    z = x @ q
    model = MLP(d=q.shape[1], width=problem.model.width, k=problem.model.k)
    return Problem(model, z, problem.y.numpy(), problem.lam), q


def split_state(theta, full_model, q):
    """Full flat parameters -> (reduced flat parameters, complement N)."""
    theta = np.asarray(theta, dtype=np.float64)
    d, h = full_model.d, full_model.width
    w1 = theta[: d * h].reshape(d, h)
    b = q.T @ w1
    return np.concatenate([b.ravel(), theta[d * h:]]), w1 - q @ b


def join_state(theta_r, complement, full_model, q):
    """Reduced flat parameters and complement N -> full flat parameters."""
    theta_r = np.asarray(theta_r, dtype=np.float64)
    r, h = q.shape[1], full_model.width
    w1 = q @ theta_r[: r * h].reshape(r, h) + complement
    return np.concatenate([w1.ravel(), theta_r[r * h:]])


def dense_hessian(problem, theta, chunk=256):
    """Dense symmetric Hessian from batched exact Hessian-vector products."""
    t = torch.as_tensor(np.asarray(theta, dtype=np.float64))
    n = t.numel()
    batched = torch.func.vmap(lambda v: problem.hvp(t, v))
    out = np.empty((n, n))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        basis = torch.zeros((stop - start, n), dtype=torch.float64)
        basis[torch.arange(stop - start), torch.arange(start, stop)] = 1.0
        out[start:stop] = batched(basis).numpy()
    out += out.T
    out *= 0.5
    return out


def two_step(problem, theta, eta, chunk=256):
    """F^2(theta), the intermediate state F(theta), and the dense two-step Jacobian."""
    t0 = torch.as_tensor(np.asarray(theta, dtype=np.float64))
    t1 = problem.update(t0, eta)
    t2 = problem.update(t1, eta)
    a0 = dense_hessian(problem, t0.numpy(), chunk)
    a0 *= -eta
    a0[np.diag_indices_from(a0)] += 1.0
    a1 = dense_hessian(problem, t1.numpy(), chunk)
    a1 *= -eta
    a1[np.diag_indices_from(a1)] += 1.0
    m = a1 @ a0
    return t2.numpy(), t1.numpy(), m


def newton_period2(problem, theta, eta, *, tol=1e-12, max_iter=8, chunk=256, emit=None):
    """Dense Newton for F^2(theta) = theta in reduced coordinates.

    Returns (theta, monodromy at the last iterate, history of relative residuals).
    """
    from scipy.linalg import lu_factor, lu_solve

    theta = np.asarray(theta, dtype=np.float64).copy()
    history = []
    m = None
    for _ in range(max_iter):
        t2, _, m = two_step(problem, theta, eta, chunk)
        residual = t2 - theta
        rel = float(np.linalg.norm(residual) / max(np.linalg.norm(theta), 1e-300))
        history.append(rel)
        if emit is not None:
            emit({"relative_residual": rel})
        if rel < tol:
            break
        j = m.copy()
        j[np.diag_indices_from(j)] -= 1.0
        theta = theta + lu_solve(lu_factor(j, overwrite_a=True, check_finite=False), -residual)
        del j
    return theta, m, history
