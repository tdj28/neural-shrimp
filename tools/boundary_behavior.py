"""EXP-025 fixed-clock input assays; no phase alignment or baseline subtraction."""
from __future__ import annotations

import numpy as np

SIGNS = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])


def labels(n):
    return np.tile(np.column_stack((SIGNS, SIGNS.prod(axis=1))), (n, 1))


def features(states, window, quadratic=False):
    """Input (context, sign, time, neuron); fixed observation is tanh(state)."""
    z = np.tanh(states[:, :, window[0]:window[1] + 1]).reshape(len(states)*4, -1)
    if quadratic:
        i, j = np.triu_indices(z.shape[1])
        z = np.column_stack((z, z[:, i]*z[:, j]))
    return z


def fit_readout(x, y, ridge):
    mean = x.mean(axis=0)
    target_mean = y.mean(axis=0)
    centered = x - mean
    w = np.linalg.solve(centered.T @ centered / len(x) + ridge*np.eye(x.shape[1]),
                        centered.T @ (y-target_mean) / len(x))
    return w, target_mean - mean @ w


def score(train, test, window, ridge, quadratic=False):
    x = features(train, window, quadratic)
    z = features(test, window, quadratic)
    w, b = fit_readout(x, labels(len(train)), ridge)
    predictions = z @ w + b
    y = labels(len(test))
    # Balanced +/-1 targets have exactly unit variance. No clipping of bad R2.
    loss = ((predictions-y)**2).reshape(len(test), 4, 3).mean(axis=1)
    accuracy = ((predictions >= 0) == (y >= 0)).reshape(len(test), 4, 3).mean(axis=1)
    return dict(r2=1-loss.mean(axis=0), accuracy=accuracy.mean(axis=0),
                context_loss=loss, weights=w, intercept=b, predictions=predictions)


def bootstrap_mean(values, seed=2026092305, draws=4000):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), (draws, len(values)))], axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def crossing_pairs(points, classes):
    pairs = []
    for i in range(len(points)-1):
        if points[i]['alpha'] != points[i+1]['alpha']:
            continue
        if {classes[i], classes[i+1]} == {'periodic', 'irregular-supported'}:
            p, q = (i, i+1) if classes[i] == 'periodic' else (i+1, i)
            pairs.append((p, q))
    return pairs


def compare_boundaries(points, classes, context_losses, minimum_effect):
    pairs = crossing_pairs(points, classes)
    if not pairs:
        return dict(decision='inconclusive: no qualified adjacent regime crossings', pairs=[])
    # Common random context indices across all parameters; resample them together.
    # Each initialization contributes equally; both have the same context count.
    differences = np.array([context_losses[q]-context_losses[p] for p, q in pairs])
    per_pair = differences.mean(axis=1)
    context_difference = differences.mean(axis=0)
    mean = float(context_difference.mean())
    ci = bootstrap_mean(context_difference)
    two_transects = len({points[p]['alpha'] for p, q in pairs}) == 2
    if two_transects and mean > minimum_effect and ci[0] > 0 and np.mean(per_pair > 0) >= .75:
        decision = 'periodic advantage on frozen memory assay'
    elif two_transects and mean < -minimum_effect and ci[1] < 0 and np.mean(per_pair < 0) >= .75:
        decision = 'irregular advantage on frozen memory assay'
    else:
        decision = 'inconclusive or heterogeneous on frozen memory assay'
    return dict(decision=decision, pairs=pairs, mean_periodic_minus_irregular=mean,
                conditional_context_interval=ci, pair_differences=per_pair.tolist(),
                both_transects=two_transects)
