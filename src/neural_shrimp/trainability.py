"""Systems and frozen selection rules for period-resolved trainability maps.

Grid convention: rows index the hidden-layer rate, columns the output-layer
rate, and cell ``row * columns + col``. Rates are ``10 ** log10`` values.
All helpers are deterministic; none inspects anything beyond its arguments.
"""

import hashlib

import numpy as np

from .gdmaps import STATUS_CHAOTIC, STATUS_FIXED, STATUS_PERIODIC, TwoLayerNet, connected_components
from .models import TanhRegressor, make_teacher_regression


def array_sha256(array):
    """SHA-256 of a C-contiguous float64 array's bytes plus its shape."""
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def make_s1(seed, width=8):
    """Mean-field tanh network with dataset size equal to parameter count.

    Follows the Sohl-Dickstein (2024) notebook defaults: no biases, inputs of
    dimension ``width``, pre-activation ``sqrt(2) * x W0 / sqrt(width)``,
    output ``h w1 / width``, mean squared error, and standard-normal data,
    targets and weights.
    """
    net = TwoLayerNet(d=width, width=width, activation="tanh",
                      in_scale=np.sqrt(2.0 / width), out_scale=1.0 / width,
                      bias=False, loss_scale=1.0, ridge=0.0)
    streams = np.random.SeedSequence(int(seed)).spawn(3)
    n = net.n_parameters
    x = np.random.default_rng(streams[0]).normal(size=(n, width))
    y = np.random.default_rng(streams[1]).normal(size=n)
    theta0 = np.random.default_rng(streams[2]).normal(size=n)
    return net, x, y, theta0


def make_s2(seed):
    """The earlier 13-parameter tanh regressor with ridge 1e-3."""
    net = TwoLayerNet(d=2, width=3, activation="tanh", bias=True, loss_scale=0.5, ridge=1e-3)
    data = make_teacher_regression(seed=int(seed), d=2, teacher_width=3, n_train=24,
                                   n_validation=48, n_test=96)
    theta0 = TanhRegressor(2, 3, ridge=1e-3).initialize(seed=int(seed) + 100000, scale=0.2)
    return net, data.train_x, data.train_y, theta0


SYSTEMS = {"S1": make_s1, "S2": make_s2}


def edge_rates(net, theta0, x, y):
    """Initial-curvature edge rates ``2 / lambda`` for hidden and output blocks."""
    hidden, output = net.gauss_newton_layer_max(theta0, x, y)
    if not (np.isfinite(hidden) and np.isfinite(output) and hidden > 0 and output > 0):
        raise ValueError("Initial Gauss-Newton maxima must be finite and positive")
    return 2.0 / hidden, 2.0 / output


def log_axis(center, half_width, resolution):
    return np.linspace(center - half_width, center + half_width, int(resolution))


def zoom_window(status, period, window=32, stride=16):
    """Frozen zoom rule: window with most periodic (p >= 2) or chaotic cells.

    Returns ``None`` when every window count is zero, otherwise
    ``(row0, col0, count)`` with ties broken by smallest row, then column.
    """
    status = np.asarray(status)
    period = np.asarray(period)
    if status.shape != period.shape or status.ndim != 2:
        raise ValueError("status and period must be equal 2-D arrays")
    interest = ((status == STATUS_PERIODIC) & (period >= 2)) | (status == STATUS_CHAOTIC)
    rows, cols = interest.shape
    if rows < window or cols < window:
        raise ValueError("grid smaller than the zoom window")
    best = None
    for r0 in range(0, rows - window + 1, stride):
        for c0 in range(0, cols - window + 1, stride):
            count = int(interest[r0:r0 + window, c0:c0 + window].sum())
            if best is None or count > best[2]:
                best = (r0, c0, count)
    if best is None or best[2] == 0:
        return None
    return best


def zoom_axes(log_hidden, log_output, row0, col0, window=32, resolution=256):
    """Zoom axes spanning the centers of the window's first and last cells."""
    log_hidden = np.asarray(log_hidden, dtype=np.float64)
    log_output = np.asarray(log_output, dtype=np.float64)
    return (np.linspace(log_hidden[row0], log_hidden[row0 + window - 1], resolution),
            np.linspace(log_output[col0], log_output[col0 + window - 1], resolution))


def _representative(members, shape):
    rows, cols = np.unravel_index(members, shape)
    centroid = np.array([rows.mean(), cols.mean()])
    distance = (rows - centroid[0]) ** 2 + (cols - centroid[1]) ** 2
    order = np.lexsort((members, distance))
    return int(members[order[0]])


def periodic_components(status, period, min_size=4, max_components=16, fixed=False):
    """Frozen selection of equal-period components for certification.

    With ``fixed=False`` components contain periodic candidates of equal
    period >= 2; with ``fixed=True`` they contain fixed-point candidates.
    Components with at least ``min_size`` cells are ordered by size
    (descending), then smallest row-major member, and the first
    ``max_components`` are returned. Each record gives its period, size,
    representative cell (nearest the index centroid, ties to smallest index),
    and whether any member is 8-adjacent to a chaotic candidate.
    """
    status = np.asarray(status)
    period = np.asarray(period)
    if status.shape != period.shape or status.ndim != 2:
        raise ValueError("status and period must be equal 2-D arrays")
    if fixed:
        mask = status == STATUS_FIXED
    else:
        mask = (status == STATUS_PERIODIC) & (period >= 2)
    labels = connected_components(period, mask)
    chaotic = status == STATUS_CHAOTIC
    padded = np.pad(chaotic, 1)
    near_chaos = np.zeros_like(chaotic)
    rows, cols = chaotic.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            near_chaos |= padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    flat_labels = labels.ravel()
    records = []
    for component in range(int(labels.max()) + 1):
        members = np.flatnonzero(flat_labels == component)
        if len(members) < min_size:
            continue
        records.append({
            "component": component,
            "period": int(period.ravel()[members[0]]),
            "size": int(len(members)),
            "first_cell": int(members[0]),
            "representative": _representative(members, status.shape),
            "borders_chaos": bool(near_chaos.ravel()[members].any()),
        })
    records.sort(key=lambda item: (-item["size"], item["first_cell"]))
    return records[:max_components]


def chaos_sample(status, count, seed):
    """Up to ``count`` chaotic-candidate cells drawn without replacement."""
    cells = np.flatnonzero(np.asarray(status).ravel() == STATUS_CHAOTIC)
    if len(cells) == 0:
        return []
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(cells, size=min(int(count), len(cells)), replace=False)
    return [int(c) for c in chosen]
