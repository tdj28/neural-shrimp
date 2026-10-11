"""Systems and frozen selection rules for period-resolved trainability maps.

Grid convention: rows index the hidden-layer rate, columns the output-layer
rate, and cell ``row * columns + col``. Rates are ``10 ** log10`` values.
All helpers are deterministic; none inspects anything beyond its arguments.
"""

import hashlib

import numpy as np

from .gdmaps import (
    STATUS_CHAOTIC, STATUS_DIVERGENT, STATUS_FIXED, STATUS_PERIODIC, TwoLayerNet,
    connected_components,
)
from .models import TanhRegressor, make_teacher_regression


def array_sha256(array):
    """SHA-256 of a C-contiguous float64 array's bytes plus its shape."""
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(repr(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def make_s1(seed, width=8, ridge=0.0):
    """Mean-field tanh network with dataset size equal to parameter count.

    Configured after the published Sohl-Dickstein (2024) notebook settings:
    no biases, inputs of dimension ``width``, pre-activation
    ``sqrt(2) * x W0 / sqrt(width)``, output ``h w1 / width``, mean squared
    error, and standard-normal data, targets and weights. ``ridge`` adds
    ``ridge / 2 * ||theta||^2``; the published setting has none.
    """
    net = TwoLayerNet(d=width, width=width, activation="tanh",
                      in_scale=np.sqrt(2.0 / width), out_scale=1.0 / width,
                      bias=False, loss_scale=1.0, ridge=float(ridge))
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


def make_s1r(seed):
    """S1 with the same data and initial parameters plus ridge 1e-3."""
    return make_s1(seed, ridge=1e-3)


SYSTEMS = {"S1": make_s1, "S1R": make_s1r, "S2": make_s2}


def edge_rates(net, theta0, x, y):
    """Initial-curvature edge rates ``2 / lambda`` for hidden and output blocks."""
    hidden, output = net.gauss_newton_layer_max(theta0, x, y)
    if not (np.isfinite(hidden) and np.isfinite(output) and hidden > 0 and output > 0):
        raise ValueError("Initial Gauss-Newton maxima must be finite and positive")
    return 2.0 / hidden, 2.0 / output


def log_axis(center, half_width, resolution):
    return np.linspace(center - half_width, center + half_width, int(resolution))


def _neighbor_any(mask):
    """True where any of the 8 neighbors (not the cell itself) is True."""
    padded = np.pad(mask, 1)
    rows, cols = mask.shape
    out = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out |= padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    return out


def zoom_window(status, period, window=32, stride=16):
    """Frozen zoom rule targeting periodic windows embedded in chaos.

    Primary score: periodic candidates with period >= 3 having at least one
    chaotic 8-neighbor. Secondary score: chaotic candidates having at least
    one 8-neighbor that is neither chaotic nor divergent. The window with the
    lexicographically largest (primary, secondary) wins; ties go to the
    smallest row, then column. Returns ``None`` when both scores are zero in
    every window, otherwise ``(row0, col0, primary, secondary)``.
    """
    status = np.asarray(status)
    period = np.asarray(period)
    if status.shape != period.shape or status.ndim != 2:
        raise ValueError("status and period must be equal 2-D arrays")
    chaotic = status == STATUS_CHAOTIC
    primary = (status == STATUS_PERIODIC) & (period >= 3) & _neighbor_any(chaotic)
    bounded_other = ~chaotic & (status != STATUS_DIVERGENT)
    secondary = chaotic & _neighbor_any(bounded_other)
    rows, cols = status.shape
    if rows < window or cols < window:
        raise ValueError("grid smaller than the zoom window")
    best = None
    for r0 in range(0, rows - window + 1, stride):
        for c0 in range(0, cols - window + 1, stride):
            score = (int(primary[r0:r0 + window, c0:c0 + window].sum()),
                     int(secondary[r0:r0 + window, c0:c0 + window].sum()))
            if best is None or score > best[2:]:
                best = (r0, c0) + score
    if best is None or best[2:] == (0, 0):
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


def ring_statistics(status, period, members, p):
    """Exterior 8-neighbor ring of a component and its composition.

    Returns ring cells (row-major order), the ring cells whose label is a
    fixed point or a periodic candidate with a proper divisor of ``p`` as its
    period, and the ring cells labeled chaotic candidates.
    """
    shape = status.shape
    inside = np.zeros(shape, dtype=bool)
    inside.ravel()[members] = True
    ring = _neighbor_any(inside) & ~inside
    cells = np.flatnonzero(ring.ravel())
    flat_status, flat_period = status.ravel(), period.ravel()
    divisor = [int(c) for c in cells if flat_status[c] == STATUS_FIXED or (
        flat_status[c] == STATUS_PERIODIC and flat_period[c] < p and p % flat_period[c] == 0)]
    chaotic = [int(c) for c in cells if flat_status[c] == STATUS_CHAOTIC]
    return [int(c) for c in cells], divisor, chaotic


def select_components(status, period, min_size=4, max_embedded=16, max_other=8, max_fixed=4,
                      min_chaotic_fraction=0.5, min_embedded_period=3):
    """Frozen selection of components for certification.

    Components are 8-connected sets of equal-label cells: periodic candidates
    of equal period >= 2, or fixed-point candidates. Only components with at
    least ``min_size`` cells are eligible. A periodic component is an
    embedded candidate when its period is at least ``min_embedded_period``,
    no exterior-ring cell is a fixed point or has a proper-divisor period,
    and at least ``min_chaotic_fraction`` of its ring cells are chaotic
    candidates. Each list is ordered by size (descending), then smallest
    member, and truncated. Returns {"embedded": [...], "other": [...],
    "fixed": [...]}; records include ring cells for confirmation reruns.
    """
    status = np.asarray(status)
    period = np.asarray(period)
    if status.shape != period.shape or status.ndim != 2:
        raise ValueError("status and period must be equal 2-D arrays")
    lists = {"embedded": [], "other": [], "fixed": []}
    for kind, mask in (("periodic", (status == STATUS_PERIODIC) & (period >= 2)),
                       ("fixed", status == STATUS_FIXED)):
        labels = connected_components(period, mask).ravel()
        for component in range(int(labels.max()) + 1):
            members = np.flatnonzero(labels == component)
            if len(members) < min_size:
                continue
            p = int(period.ravel()[members[0]]) if kind == "periodic" else 1
            ring, divisor, chaotic = ring_statistics(status, period, members, p)
            fraction = len(chaotic) / len(ring) if ring else 0.0
            record = {"component": component, "period": p, "size": int(len(members)),
                      "first_cell": int(members[0]),
                      "representative": _representative(members, status.shape),
                      "ring_cells": ring, "ring_divisor_cells": divisor,
                      "ring_chaotic_cells": chaotic, "ring_chaotic_fraction": fraction}
            if kind == "fixed":
                lists["fixed"].append(record)
            elif (p >= min_embedded_period and not divisor and
                  fraction >= min_chaotic_fraction):
                lists["embedded"].append(record)
            else:
                lists["other"].append(record)
    limits = {"embedded": max_embedded, "other": max_other, "fixed": max_fixed}
    for key in lists:
        lists[key].sort(key=lambda item: (-item["size"], item["first_cell"]))
        lists[key] = lists[key][:limits[key]]
    return lists


def chaos_sample(status, count, seed):
    """Up to ``count`` chaotic-candidate cells drawn without replacement."""
    cells = np.flatnonzero(np.asarray(status).ravel() == STATUS_CHAOTIC)
    if len(cells) == 0:
        return []
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(cells, size=min(int(count), len(cells)), replace=False)
    return [int(c) for c in chosen]
