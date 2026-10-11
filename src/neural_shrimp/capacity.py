"""Driven tanh-network simulation and held-out information-processing capacities.

Each cell is ``dx/dt = -x + W tanh(x) + eps * u(t) * b`` with its own ``W``.
The input ``u`` is held constant for ``hold_steps`` RK4 steps of size ``dt``
per input value and is shared by all cells, so every cell receives exactly the
same stream, clock and channel. Features are the activations (or states)
sampled at ``virtual_nodes`` equally spaced times inside each hold interval.

Capacities follow Dambre et al. (2012): for each target the held-out capacity
is ``1 - NMSE`` of a ridge readout trained on an independent stream. Raw
values are returned; clipping or thresholds belong to the frozen analysis.
"""

import numpy as np


def hopfield_weights(alpha, gamma):
    """Weight matrices of the da Silva and Rech network, shape (cells, 3, 3)."""
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    gamma = np.atleast_1d(np.asarray(gamma, dtype=np.float64))
    if alpha.shape != gamma.shape or alpha.ndim != 1:
        raise ValueError("alpha and gamma must be one-dimensional and equal in shape")
    w = np.zeros((len(alpha), 3, 3))
    w[:, 0, 0], w[:, 0, 1], w[:, 0, 2] = 1.5, 2.9, alpha
    w[:, 1, 0], w[:, 1, 1] = -1.5, 1.18
    w[:, 2, 0], w[:, 2, 1], w[:, 2, 2] = gamma, -22.0, 0.47
    return w


def _rhs(x, weights, drive):
    return -x + np.einsum("cij,cj->ci", weights, np.tanh(x)) + drive


def _tangent_rhs(x, v, weights):
    s = 1.0 - np.tanh(x) ** 2
    return -v + np.einsum("cij,cj->ci", weights, s * v)


def simulate(weights, x0, inputs, eps, dt, hold_steps, virtual_nodes, b=(1.0, 0.0, 0.0),
             feature="activation", tangent=None):
    """Integrate every cell through the whole input stream.

    Returns ``(features, final_state, log_growth)``: features have shape
    (cells, len(inputs), 3 * virtual_nodes); ``log_growth`` accumulates the
    log norm growth of the tangent (renormalized after each input) when an
    initial ``tangent`` of shape (cells, 3) is given, otherwise ``None``.
    Sample times are the ends of ``virtual_nodes`` equal sub-intervals.
    """
    weights = np.asarray(weights, dtype=np.float64)
    x = np.array(x0, dtype=np.float64)
    if weights.ndim != 3 or weights.shape[1:] != (3, 3) or x.shape != (weights.shape[0], 3):
        raise ValueError("weights must be (cells, 3, 3) and x0 (cells, 3)")
    if hold_steps % virtual_nodes != 0 or hold_steps < 1 or virtual_nodes < 1:
        raise ValueError("hold_steps must be a positive multiple of virtual_nodes")
    if feature not in ("activation", "state"):
        raise ValueError("feature must be 'activation' or 'state'")
    inputs = np.asarray(inputs, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    eps = np.broadcast_to(np.asarray(eps, dtype=np.float64), (weights.shape[0],))
    cells = weights.shape[0]
    stride = hold_steps // virtual_nodes
    features = np.empty((cells, len(inputs), 3 * virtual_nodes))
    v = None if tangent is None else np.array(tangent, dtype=np.float64)
    if v is not None:
        if v.shape != (cells, 3):
            raise ValueError("tangent must have shape (cells, 3)")
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
    log_growth = None if v is None else np.zeros(cells)
    for k, u in enumerate(inputs):
        drive = (eps * u)[:, None] * b[None, :]
        for step in range(hold_steps):
            if v is None:
                k1 = _rhs(x, weights, drive)
                k2 = _rhs(x + 0.5 * dt * k1, weights, drive)
                k3 = _rhs(x + 0.5 * dt * k2, weights, drive)
                k4 = _rhs(x + dt * k3, weights, drive)
            else:
                k1, l1 = _rhs(x, weights, drive), _tangent_rhs(x, v, weights)
                x2, v2 = x + 0.5 * dt * k1, v + 0.5 * dt * l1
                k2, l2 = _rhs(x2, weights, drive), _tangent_rhs(x2, v2, weights)
                x3, v3 = x + 0.5 * dt * k2, v + 0.5 * dt * l2
                k3, l3 = _rhs(x3, weights, drive), _tangent_rhs(x3, v3, weights)
                x4, v4 = x + dt * k3, v + dt * l3
                k4, l4 = _rhs(x4, weights, drive), _tangent_rhs(x4, v4, weights)
                v = v + (dt / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if (step + 1) % stride == 0:
                node = (step + 1) // stride - 1
                sample = np.tanh(x) if feature == "activation" else x
                features[:, k, 3 * node:3 * node + 3] = sample
        if v is not None:
            norm = np.linalg.norm(v, axis=1)
            log_growth += np.log(norm)
            v = v / norm[:, None]
    return features, x, log_growth


def memory_targets(inputs, max_delay):
    """Targets ``u[k-d]`` for ``d = 1..max_delay``; rows before ``max_delay`` are invalid."""
    inputs = np.asarray(inputs, dtype=np.float64)
    targets = np.full((len(inputs), max_delay), np.nan)
    for d in range(1, max_delay + 1):
        targets[d:, d - 1] = inputs[:-d]
    return targets


def degree2_targets(inputs, max_square_delay, max_cross_delay):
    """Legendre ``P2(u[k-d])`` and products ``u[k-d1] u[k-d2]`` (d1 < d2).

    Returns (targets, labels). Inputs should be uniform on [-1, 1]; ``P2``
    and the products are then orthogonal and zero-mean.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    columns, labels = [], []
    n = len(inputs)

    def delayed(d):
        out = np.full(n, np.nan)
        out[d:] = inputs[:n - d] if d else inputs
        return out

    for d in range(1, max_square_delay + 1):
        u = delayed(d)
        columns.append(1.5 * u * u - 0.5)
        labels.append(("P2", d))
    for d1 in range(1, max_cross_delay + 1):
        for d2 in range(d1 + 1, max_cross_delay + 1):
            columns.append(delayed(d1) * delayed(d2))
            labels.append(("cross", d1, d2))
    return np.column_stack(columns), labels


def ridge_capacities(train_features, train_targets, test_features, test_targets, ridge):
    """Held-out ``1 - NMSE`` per cell and target with an unpenalized intercept.

    Features have shape (cells, samples, features); targets (samples, targets)
    are shared by all cells. Features are standardized with training
    statistics per cell; the penalty is ``ridge * samples`` on standardized
    weights. Rows with any nonfinite target are dropped consistently.
    Returns an array of shape (cells, targets); a constant feature column
    contributes nothing and is not an error.
    """
    tr_x = np.asarray(train_features, dtype=np.float64)
    te_x = np.asarray(test_features, dtype=np.float64)
    tr_y = np.asarray(train_targets, dtype=np.float64)
    te_y = np.asarray(test_targets, dtype=np.float64)
    keep_tr = np.all(np.isfinite(tr_y), axis=1)
    keep_te = np.all(np.isfinite(te_y), axis=1)
    tr_x, tr_y = tr_x[:, keep_tr], tr_y[keep_tr]
    te_x, te_y = te_x[:, keep_te], te_y[keep_te]
    mean = tr_x.mean(axis=1, keepdims=True)
    scale = tr_x.std(axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    a = (tr_x - mean) / scale
    b = (te_x - mean) / scale
    y_mean = tr_y.mean(axis=0)
    n, f = a.shape[1], a.shape[2]
    gram = np.einsum("csi,csj->cij", a, a) + ridge * n * np.eye(f)[None]
    cross = np.einsum("csi,st->cit", a, tr_y - y_mean)
    coef = np.linalg.solve(gram, cross)
    prediction = np.einsum("csi,cit->cst", b, coef) + y_mean
    error = np.mean((prediction - te_y[None]) ** 2, axis=1)
    variance = np.var(te_y, axis=0)
    return 1.0 - error / variance[None]


def consistency(features_a, features_b):
    """Mean Pearson correlation between two runs' responses, per cell.

    Each feature column's correlation across time is computed, then averaged
    over columns with nonzero variance in both runs (Uchida et al. 2004).
    """
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    valid = den > 0
    corr = np.where(valid, num / np.where(valid, den, 1.0), np.nan)
    return np.nanmean(corr, axis=1)
