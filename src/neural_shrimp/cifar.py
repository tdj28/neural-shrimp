"""Bounded CIFAR-10 networks for the exploratory real-data pilot (float64, full batch).

Data: the CIFAR-10 python archive (downloaded once through torchvision into
DATA/datasets/cifar10; its SHA-256 is recorded). Two disjoint, class-balanced
training subsets are drawn with one documented seed: A (development) and B
(held out). Only B's index list and its hash are written; nothing in this
module loads B's pixels.

Model: tanh MLP 3072-h-10 in standard parameterization with PyTorch-default
initialization (weights and biases uniform on +-1/sqrt(fan_in)), a flat float64
parameter vector theta = (W1, b1, W2, b2), and

    L(theta) = (1/n) sum_i 1/2 ||f(x_i) - y_i||^2 + (lam/2) ||theta||^2

with one-hot targets. Gradient descent is F(theta) = theta - eta grad L(theta).
Matrix-free tools: Hessian-vector products (forward over reverse), Jacobian-
vector products of F and F^2, two-step Floquet multipliers by Arnoldi on a
SciPy LinearOperator (largest-magnitude and largest-imaginary-part searches:
with weight decay the many flat directions form a real multiplier cluster at
(1 - eta lam)^2 just below 1, which the imaginary-part search ignores), and a
Newton-GMRES corrector for period-2 orbits.
"""

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

RESERVED_SEEDS = range(29000, 30000)
CLASSES = 10
INPUT_DIM = 3072


def data_root():
    return Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))


def check_seed(seed):
    if int(seed) in RESERVED_SEEDS:
        raise ValueError("seeds 29000-29999 are reserved for confirmatory tests")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cifar_train(root=None):
    """Return (pixels (50000, 3072) uint8, labels (50000,) int) from the extracted batches."""
    folder = Path(root or data_root() / "datasets" / "cifar10") / "cifar-10-batches-py"
    pixels, labels = [], []
    for k in range(1, 6):
        with open(folder / f"data_batch_{k}", "rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        pixels.append(np.asarray(batch[b"data"], dtype=np.uint8))
        labels.extend(batch[b"labels"])
    return np.concatenate(pixels), np.asarray(labels, dtype=int)


def balanced_subsets(labels, seed, size=512):
    """Two disjoint, class-balanced index sets (A, B) of ``size`` each from one seed.

    Every class contributes floor(size / 10) images to each subset; the
    remaining size mod 10 slots go to classes chosen by the same seed, the same
    classes for A and B.
    """
    check_seed(seed)
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    counts = np.full(CLASSES, size // CLASSES)
    counts[rng.choice(CLASSES, size=size % CLASSES, replace=False)] += 1
    a, b = [], []
    for c in range(CLASSES):
        members = rng.permutation(np.flatnonzero(labels == c))
        a.extend(members[:counts[c]])
        b.extend(members[counts[c]:2 * counts[c]])
    return np.sort(np.asarray(a)), np.sort(np.asarray(b))


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def seal_indices(indices, path, note):
    """Write a subset's index list and hash without touching its pixels."""
    record = {"note": note, "count": int(len(indices)), "sha256_int64": index_hash(indices),
              "indices": [int(i) for i in indices]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=1))
    return record["sha256_int64"]


def standardized(pixels):
    """Per-feature standardization on the given rows (constant features keep scale one)."""
    x = pixels.astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def one_hot(labels):
    out = np.zeros((len(labels), CLASSES))
    out[np.arange(len(labels)), labels] = 1.0
    return out


class MLP:
    """Flat-parameter tanh MLP d-h-k with standard parameterization."""

    def __init__(self, d=INPUT_DIM, width=32, k=CLASSES):
        self.d, self.width, self.k = int(d), int(width), int(k)
        self.sizes = (self.d * self.width, self.width, self.width * self.k, self.k)
        self.n_parameters = int(sum(self.sizes))

    def unflatten(self, theta):
        w1, b1, w2, b2 = torch.split(theta, self.sizes)
        return w1.view(self.d, self.width), b1, w2.view(self.width, self.k), b2

    def forward(self, theta, x):
        w1, b1, w2, b2 = self.unflatten(theta)
        return torch.tanh(x @ w1 + b1) @ w2 + b2

    def init(self, seed):
        """PyTorch nn.Linear default initialization (uniform on +-1/sqrt(fan_in))."""
        check_seed(seed)
        rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(2)[1])
        h, o = 1.0 / np.sqrt(self.d), 1.0 / np.sqrt(self.width)
        parts = [rng.uniform(-h, h, self.sizes[0]), rng.uniform(-h, h, self.sizes[1]),
                 rng.uniform(-o, o, self.sizes[2]), rng.uniform(-o, o, self.sizes[3])]
        return np.concatenate(parts)


class Problem:
    """Full-batch loss, gradient descent update and matrix-free derivatives."""

    def __init__(self, model, x, y, lam):
        self.model = model
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float64))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float64))
        self.lam = float(lam)
        self.n = self.x.shape[0]

    def loss(self, theta):
        out = self.model.forward(theta, self.x)
        return 0.5 * ((out - self.y) ** 2).sum() / self.n + 0.5 * self.lam * (theta * theta).sum()

    def grad(self, theta):
        return torch.func.grad(self.loss)(theta)

    def loss_and_grad(self, theta):
        return torch.func.grad_and_value(self.loss)(theta)[::-1]

    def update(self, theta, eta):
        return theta - eta * self.grad(theta)

    def hvp(self, theta, v):
        return torch.func.jvp(self.grad, (theta,), (v,))[1]

    def jvp_update(self, theta, v, eta):
        return v - eta * self.hvp(theta, v)

    def jvp_two_step(self, theta, v, eta, theta_next=None):
        t1 = self.update(theta, eta) if theta_next is None else theta_next
        return self.jvp_update(t1, self.jvp_update(theta, v, eta), eta)


def as_tensor(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def arnoldi_spectrum(operator, k=6, tol=1e-9, ncv=None, modes=("LM", "LI"), maxiter=300,
                     seed=28204, residual_tolerance=1e-7, return_vectors=False, max_matvec=2000, deadline=None):
    """Seeded partial spectrum with checked Ritz residuals, not spectral coverage.

    Small residuals are necessary numerical checks, not a proof that all
    leading modes were found or that eigenvalue errors equal residuals.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    check_seed(seed)
    n = operator.shape[0]
    if not 1 <= k < n - 1 or not modes or residual_tolerance <= 0:
        raise ValueError("invalid bounded Arnoldi request")
    found, diagnostics, vectors_by_mode = {}, {}, {}
    complex_candidates = []
    for index, mode in enumerate(modes):
        calls = 0

        def bounded_matvec(v):
            nonlocal calls
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Arnoldi wall-time budget exhausted')
            if calls >= max_matvec:
                raise RuntimeError('Arnoldi matvec budget exhausted')
            calls += 1
            return operator.matvec(v)

        bounded = LinearOperator(operator.shape, matvec=bounded_matvec, dtype=operator.dtype)
        v0 = np.random.default_rng(np.random.SeedSequence([seed, index])).normal(size=n)
        converged = True
        try:
            values, vectors = eigs(bounded, k=k, which=mode, tol=tol, ncv=ncv,
                                   maxiter=maxiter, v0=v0, return_eigenvectors=True)
        except ArpackNoConvergence as error:
            values, vectors = error.eigenvalues, error.eigenvectors
            converged = False
        values = np.asarray(values if values is not None else [], dtype=complex)
        vectors = np.asarray(vectors) if vectors is not None else np.empty((n, 0), dtype=complex)
        order = np.argsort(-np.abs(values))
        values, vectors = values[order], vectors[:, order]
        residuals = []
        for value, vector in zip(values, vectors.T):
            av = bounded.matvec(vector)
            denominator = max(float(np.linalg.norm(av) + abs(value) * np.linalg.norm(vector)), 1e-300)
            residuals.append(float(np.linalg.norm(av - value * vector) / denominator))
        qualified = bool(converged and len(values) == k and all(np.isfinite(r) and r <= residual_tolerance for r in residuals))
        found[mode] = [[float(v.real), float(v.imag)] for v in values]
        diagnostics[mode] = {"arpack_converged": converged, "requested_modes": k, "returned_modes": len(values),
                             "relative_ritz_residuals": residuals, "residual_tolerance": residual_tolerance,
                             "qualified": qualified, "matvec_calls": calls, "max_matvec": max_matvec, "start_vector_sha256": hashlib.sha256(v0.tobytes()).hexdigest()}
        if return_vectors:
            vectors_by_mode[mode] = vectors
        complex_candidates.extend(v for v in values if qualified and v.imag > 1e-9)
    qualified = all(d['qualified'] for d in diagnostics.values())
    dominant = max(complex_candidates, key=abs) if qualified and complex_candidates else None
    result = {"modes": found, "diagnostics": diagnostics, "qualified": qualified, "arnoldi_seed": seed,
              "spectrum_scope": "estimated partial spectrum; unchecked modes may change stability",
              "dominant_complex": None if dominant is None else [float(dominant.real), float(dominant.imag)],
              "dominant_complex_modulus": None if dominant is None else float(abs(dominant)),
              "dominant_complex_theta_over_2pi": None if dominant is None else float(abs(np.angle(dominant)) / (2 * np.pi))}
    if return_vectors:
        result['vectors'] = vectors_by_mode
    return result


def two_step_multipliers(problem, theta, eta, k=6, tol=1e-9, ncv=None, modes=("LM", "LI"), maxiter=300,
                         seed=28204, residual_tolerance=1e-7, return_vectors=False, max_matvec=2000, deadline=None):
    """Estimated two-step spectrum; caller must first validate orbit closure."""
    from scipy.sparse.linalg import LinearOperator

    theta = as_tensor(theta)
    t1 = problem.update(theta, eta)
    n = theta.numel()

    def matvec(v):
        out = problem.jvp_two_step(theta, as_tensor(np.real(v)), eta, theta_next=t1).numpy()
        if np.iscomplexobj(v):
            out = out + 1j * problem.jvp_two_step(theta, as_tensor(np.imag(v)), eta, theta_next=t1).numpy()
        return out

    operator = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    return arnoldi_spectrum(operator, k, tol, ncv, modes, maxiter, seed, residual_tolerance, return_vectors, max_matvec, deadline)


def newton_period2(problem, theta, eta, tol=1e-11, max_newton=12, gmres_tol=1e-10, gmres_restart=200,
                   gmres_maxiter=20, callback=None, deadline=None):
    """Correct F^2(theta)=theta and return diagnostics for the returned state.

    Residual is ||F^2(theta)-theta|| / max(1,||theta||). The final history row
    always describes the returned state, including exhaustion and failures.
    No failed or nonfinite GMRES correction is applied.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    if max_newton < 0 or tol <= 0:
        raise ValueError("invalid Newton settings")
    theta = as_tensor(theta).clone()
    n = theta.numel()
    history = []
    for iteration in range(max_newton + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Newton wall-time budget exhausted")
        t1 = problem.update(theta, eta)
        t2 = problem.update(t1, eta)
        residual = t2 - theta
        rel = float(residual.norm()) / max(1.0, float(theta.norm()))
        finite = bool(torch.isfinite(theta).all() and torch.isfinite(residual).all() and np.isfinite(rel))
        converged = bool(finite and rel < tol)
        status = ("nonfinite_residual" if not finite else "converged" if converged else
                  "max_iterations" if iteration == max_newton else "iterate")
        history.append({"iteration": iteration, "relative_residual": rel, "converged": converged, "status": status})
        if callback:
            callback(history[-1])
        if not finite or converged or iteration == max_newton:
            break

        def matvec(v, theta=theta, t1=t1):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("GMRES wall-time budget exhausted")
            return problem.jvp_two_step(theta, as_tensor(v), eta, theta_next=t1).numpy() - v

        operator = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        step, info = gmres(operator, -residual.numpy(), rtol=gmres_tol, restart=gmres_restart,
                           maxiter=gmres_maxiter)
        history[-1]["gmres_info"] = int(info)
        if info != 0 or not np.all(np.isfinite(step)):
            history[-1]['status'] = 'linear_solver_failed' if info != 0 else 'nonfinite_correction'
            break
        candidate = theta + as_tensor(step)
        if not torch.isfinite(candidate).all():
            history[-1]['status'] = 'nonfinite_correction'
            break
        theta = candidate
    return theta, history


def run_gd(problem, theta, eta, steps, record_from=None, record_every=1, losses=True):
    """Iterate gradient descent; return final theta, per-step losses and recorded states."""
    theta = as_tensor(theta).clone()
    loss_trace = np.empty(steps) if losses else None
    states = []
    record_from = steps if record_from is None else record_from
    for t in range(steps):
        value, gradient = problem.loss_and_grad(theta)
        if losses:
            loss_trace[t] = float(value)
        theta = theta - eta * gradient
        if not torch.isfinite(theta).all():
            return None, loss_trace[:t + 1] if losses else None, states
        if t >= record_from and (t - record_from) % record_every == 0:
            states.append(theta.numpy().copy())
    return theta, loss_trace, states


def rotation_from_samples(samples):
    """Rotation number (turns per sample) from samples of a closed curve.

    Samples (m, n) are projected on their top two principal directions; the
    unwrapped angle about the centroid over the sequence gives the rotation.
    """
    x = np.asarray(samples, dtype=np.float64)
    centred = x - x.mean(axis=0)
    gram = centred @ centred.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    coords = vectors[:, order[:2]] * np.sqrt(np.maximum(values[order[:2]], 0))
    angles = np.unwrap(np.arctan2(coords[:, 1], coords[:, 0]))
    share = float(values[order[:2]].sum() / max(values.sum(), 1e-300))
    return float(abs(angles[-1] - angles[0]) / (2 * np.pi * (len(angles) - 1))), share
