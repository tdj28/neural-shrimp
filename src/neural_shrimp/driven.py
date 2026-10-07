"""Fast structure-of-arrays driven simulation of the three-neuron network.

Same model and semantics as :func:`neural_shrimp.capacity.simulate`
(``dx/dt = -x + W tanh(x) + eps * u(t) * b`` with held inputs, features
sampled at ``virtual_nodes`` equal sub-interval ends, optional tangent for
the conditional Lyapunov exponent), but arrays are component-first and
cell-last, roughly halving the cost of the 3 x 3 products. The capacity
module is kept unchanged because frozen plans bind it.
"""

import numpy as np


def _soa(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 3 or weights.shape[1:] != (3, 3):
        raise ValueError("weights must have shape (cells, 3, 3)")
    return np.ascontiguousarray(weights.transpose(1, 2, 0))


def _rhs(x, w, drive):
    t = np.tanh(x)
    return -x + w[:, 0] * t[0] + w[:, 1] * t[1] + w[:, 2] * t[2] + drive


def _tangent(x, w, v):
    sv = (1.0 - np.tanh(x) ** 2) * v
    return -v + w[:, 0] * sv[0] + w[:, 1] * sv[1] + w[:, 2] * sv[2]


def simulate_soa(weights, x0, inputs, eps, dt, hold_steps, virtual_nodes, b=(1.0, 0.0, 0.0),
                 feature="activation", tangent=None):
    """Drop-in faster equivalent of ``capacity.simulate`` (same return values)."""
    w = _soa(weights)
    cells = w.shape[2]
    x = np.ascontiguousarray(np.array(x0, dtype=np.float64).T)
    if x.shape != (3, cells):
        raise ValueError("x0 must have shape (cells, 3)")
    if hold_steps % virtual_nodes != 0 or hold_steps < 1 or virtual_nodes < 1:
        raise ValueError("hold_steps must be a positive multiple of virtual_nodes")
    if feature not in ("activation", "state"):
        raise ValueError("feature must be 'activation' or 'state'")
    inputs = np.asarray(inputs, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)[:, None]
    eps = np.broadcast_to(np.asarray(eps, dtype=np.float64), (cells,))
    stride = hold_steps // virtual_nodes
    features = np.empty((cells, len(inputs), 3 * virtual_nodes))
    v = None
    log_growth = None
    if tangent is not None:
        v = np.ascontiguousarray(np.array(tangent, dtype=np.float64).T)
        if v.shape != (3, cells):
            raise ValueError("tangent must have shape (cells, 3)")
        v = v / np.sqrt(np.sum(v * v, axis=0))
        log_growth = np.zeros(cells)
    for k, u in enumerate(inputs):
        drive = b * (eps * u)[None, :]
        for step in range(hold_steps):
            if v is None:
                k1 = _rhs(x, w, drive)
                k2 = _rhs(x + 0.5 * dt * k1, w, drive)
                k3 = _rhs(x + 0.5 * dt * k2, w, drive)
                k4 = _rhs(x + dt * k3, w, drive)
            else:
                k1, l1 = _rhs(x, w, drive), _tangent(x, w, v)
                x2, v2 = x + 0.5 * dt * k1, v + 0.5 * dt * l1
                k2, l2 = _rhs(x2, w, drive), _tangent(x2, w, v2)
                x3, v3 = x + 0.5 * dt * k2, v + 0.5 * dt * l2
                k3, l3 = _rhs(x3, w, drive), _tangent(x3, w, v3)
                x4, v4 = x + dt * k3, v + dt * l3
                k4, l4 = _rhs(x4, w, drive), _tangent(x4, w, v4)
                v = v + (dt / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if (step + 1) % stride == 0:
                node = (step + 1) // stride - 1
                sample = np.tanh(x) if feature == "activation" else x
                features[:, k, 3 * node:3 * node + 3] = sample.T
        if v is not None:
            norm = np.sqrt(np.sum(v * v, axis=0))
            log_growth += np.log(norm)
            v = v / norm
    return features, np.ascontiguousarray(x.T), log_growth
