"""Small causal text transformer and matrix-free GD derivatives (exploratory).

All parameters, including the single tied token/output embedding, live in one
flat vector. Full-batch GD has no optimizer state beyond that vector. There is
no dropout, attention dropout, mixed precision, momentum or implicit cache.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


def check_seed(seed):
    if 29000 <= int(seed) <= 29999:
        raise ValueError('reserved confirmatory seed')


@dataclass(frozen=True)
class TinyTransformer:
    vocabulary: int
    context: int = 32
    width: int = 32
    expansion: int = 4
    norm_epsilon: float = 1e-5

    @property
    def shapes(self):
        d, h = self.width, self.expansion*self.width
        return {'embedding': (self.vocabulary, d), 'position': (self.context, d),
                'ln1_weight': (d,), 'ln1_bias': (d,), 'qkv': (3*d, d), 'projection': (d, d),
                'ln2_weight': (d,), 'ln2_bias': (d,), 'ff1': (h, d), 'ff1_bias': (h,),
                'ff2': (d, h), 'ff2_bias': (d,), 'lnf_weight': (d,), 'lnf_bias': (d,)}

    @property
    def n_parameters(self):
        return sum(math.prod(shape) for shape in self.shapes.values())

    def unflatten(self, theta):
        if theta.numel() != self.n_parameters:
            raise ValueError('parameter size mismatch')
        pieces = torch.split(theta, [math.prod(s) for s in self.shapes.values()])
        return {name: piece.reshape(shape) for (name, shape), piece in zip(self.shapes.items(), pieces)}

    def initialize(self, seed=28141, dtype=torch.float32, device='cpu'):
        check_seed(seed)
        rng = np.random.default_rng(seed)
        pieces = []
        for name, shape in self.shapes.items():
            value = rng.normal(0, .02, size=shape)
            if 'bias' in name:
                value = np.zeros(shape)
            elif name.startswith('ln') and name.endswith('weight'):
                value = np.ones(shape)
            pieces.append(value.ravel())
        return torch.tensor(np.concatenate(pieces), dtype=dtype, device=device)

    def forward(self, theta, tokens):
        p = self.unflatten(theta)
        t = tokens.shape[1]
        x = F.embedding(tokens, p['embedding']) + p['position'][:t]
        z = F.layer_norm(x, (self.width,), p['ln1_weight'], p['ln1_bias'], self.norm_epsilon)
        q, k, v = F.linear(z, p['qkv']).chunk(3, dim=-1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.width)
        mask = torch.arange(t, device=tokens.device)
        scores = scores.masked_fill(mask[None, :] > mask[:, None], float('-inf'))
        x = x + F.linear(torch.softmax(scores, dim=-1) @ v, p['projection'])
        z = F.layer_norm(x, (self.width,), p['ln2_weight'], p['ln2_bias'], self.norm_epsilon)
        x = x + F.linear(F.gelu(F.linear(z, p['ff1'], p['ff1_bias'])), p['ff2'], p['ff2_bias'])
        x = F.layer_norm(x, (self.width,), p['lnf_weight'], p['lnf_bias'], self.norm_epsilon)
        return F.linear(x, p['embedding'])  # Tied weight occurs only once in theta.


class LMProblem:
    def __init__(self, model, tokens, targets, ridge=.001):
        self.model, self.tokens, self.targets, self.ridge = model, tokens, targets, float(ridge)

    def cross_entropy(self, theta):
        logits = self.model.forward(theta, self.tokens)
        return F.cross_entropy(logits.reshape(-1, self.model.vocabulary), self.targets.reshape(-1))

    def loss(self, theta):
        return self.cross_entropy(theta) + .5*self.ridge*(theta@theta)

    def grad(self, theta):
        return torch.func.grad(self.loss)(theta)

    def update(self, theta, eta):
        return theta-eta*self.grad(theta)

    def hvp(self, theta, vector):
        return torch.func.jvp(self.grad, (theta,), (vector,))[1]

    def update_jvp(self, theta, vector, eta):
        return vector-eta*self.hvp(theta, vector)

    def two_step_jvp(self, theta, vector, eta):
        next_theta = self.update(theta, eta)
        return self.update_jvp(next_theta, self.update_jvp(theta, vector, eta), eta)


def lanczos_extremes(matvec, initial, steps=32):
    """Fully reorthogonalized symmetric Lanczos; return Ritz residuals, not guarantees.

    The residual norms are recomputed using the original operator. A small
    Ritz residual validates the returned eigenpair, not that no larger one exists.
    """
    q = initial / initial.norm()
    basis, diagonal, off_diagonal = [], [], []
    beta = torch.zeros((), dtype=q.dtype, device=q.device)
    previous = torch.zeros_like(q)
    for index in range(steps):
        basis.append(q)
        z = matvec(q)-beta*previous
        alpha = q@z
        diagonal.append(alpha)
        z = z-alpha*q
        Q = torch.stack(basis, dim=1)
        for _ in range(2):
            z = z-Q@(Q.T@z)
        new_beta = z.norm()
        if float(new_beta) < 100*torch.finfo(q.dtype).eps or index == steps-1:
            break
        off_diagonal.append(new_beta)
        previous, q, beta = q, z/new_beta, new_beta
    T = torch.diag(torch.stack(diagonal))
    if off_diagonal:
        off = torch.stack(off_diagonal)
        T += torch.diag(off, 1)+torch.diag(off, -1)
    values, vectors = torch.linalg.eigh(T)
    Q = torch.stack(basis, dim=1)
    result = {'steps': len(basis)}
    for name, index in (('minimum', 0), ('maximum', -1)):
        value, vector = values[index], Q@vectors[:, index]
        residual = (matvec(vector)-value*vector).norm()
        result[name] = {'value': float(value), 'absolute_residual': float(residual),
                        'relative_residual': float(residual/max(abs(float(value)), 1e-12))}
    return result


def ar_diagnostics(losses):
    """Exploratory detrended AR(1)/AR(2) one-step predictions; no multiplier claim."""
    y = np.asarray(losses, dtype=float)
    if len(y) < 32 or not np.isfinite(y).all():
        return {'outcome': 'insufficient_finite_samples'}
    time = np.arange(len(y), dtype=float)
    trend = np.polyfit(time, y, 1)
    residual = y-np.polyval(trend, time)
    scale = float(np.std(residual))
    out = {'samples': len(y), 'linear_slope': float(trend[0]), 'detrended_std': scale,
           'relative_trend_over_window': float(abs(trend[0])*len(y)/max(scale, 1e-15)),
           'interpretation': 'descriptive local system identification; not full-state or Floquet multipliers'}
    if scale <= 1e-12:
        return {**out, 'outcome': 'nearly_constant'}
    # Trend fitted on the training part only for the predictive baseline comparison.
    split = int(.75*len(y))
    fit_trend = np.polyfit(time[:split], y[:split], 1)
    z = (y-np.polyval(fit_trend, time))/max(float(np.std(y[:split])), 1e-12)
    for order in (1, 2):
        X = np.stack([z[order-k-1:len(z)-k-1] for k in range(order)], axis=1)
        target = z[order:]
        X = np.c_[np.ones(len(X)), X]
        beta = np.linalg.lstsq(X[:split-order], target[:split-order], rcond=None)[0]
        error = target[split-order:]-X[split-order:]@beta
        roots = np.roots(np.r_[1, -beta[1:]])
        out[f'ar{order}'] = {'coefficients': beta.tolist(), 'normalized_test_mse': float(np.mean(error**2)),
                              'roots': [[float(r.real), float(r.imag)] for r in roots]}
    return {**out, 'outcome': 'fit'}
