"""Model-free preparation for N100; no experiment or model is launched here.

Symmetric distance affinities A are normalized as W = D^-1 A. The real
spectrum is computed through D^-1/2 A D^-1/2, never by assuming W symmetric.
All reported stability predictions are local and conditional on the supplied
single-loop exponent. See docs/N100-plan.md for the pending pilot.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components


def distance_graph(positions, length_scale, neighbors=None):
    """Return (A, W) with no self edges; k-NN means the undirected union.

    Neighbor ties use node index. A positive edge lost to underflow is an
    error, not an implicit change to the requested graph.
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("positions must be a finite (n >= 2, 2) array")
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    n = len(p)
    if neighbors is not None and (not isinstance(neighbors, (int, np.integer)) or
                                  isinstance(neighbors, bool) or not 1 <= neighbors < n):
        raise ValueError("neighbors must be None or an integer from 1 to n-1")
    distance = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    mask = ~np.eye(n, dtype=bool)
    if neighbors is not None:
        ranked = distance.copy()
        np.fill_diagonal(ranked, np.inf)
        order = np.argsort(ranked, axis=1, kind="stable")[:, :neighbors]
        mask = np.zeros((n, n), dtype=bool)
        mask[np.arange(n)[:, None], order] = True
        mask |= mask.T
    a = np.zeros((n, n), dtype=np.float64)
    with np.errstate(under="ignore"):
        a[mask] = np.exp(-distance[mask] / length_scale)
    if (a[mask] == 0).any():
        raise ValueError("distance weights underflowed; requested graph is not represented")
    return a, normalize_affinity(a)


def normalize_affinity(affinity):
    """Validate symmetric nonnegative affinities and return row-stochastic W."""
    a = np.asarray(affinity, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 2:
        raise ValueError("affinity must be square with at least two nodes")
    if not np.isfinite(a).all() or (a < 0).any() or not np.array_equal(a, a.T):
        raise ValueError("affinity must be finite, nonnegative and exactly symmetric")
    if np.any(np.diag(a) != 0):
        raise ValueError("self affinities must be zero; self share is controlled by s")
    d = a.sum(axis=1)
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("every node must have a finite positive weighted degree")
    return a / d[:, None]


def laplacian_spectrum(affinity):
    """Return all eigenvalues of I-W and reject unresolved spectral gaps.

    Disconnected graphs retain every zero mode; none is silently discarded.
    An unresolved positive gap must not be rounded into a disconnected graph.
    """
    a = np.asarray(affinity, dtype=np.float64)
    normalize_affinity(a)
    components = int(connected_components(a > 0, directed=False, return_labels=False))
    invroot = 1 / np.sqrt(a.sum(axis=1))
    symmetric = (a * invroot[:, None]) * invroot[None, :]
    symmetric = (symmetric + symmetric.T) / 2
    mu = np.linalg.eigvalsh(np.eye(len(a)) - symmetric)
    floor = 64 * len(a) * np.finfo(np.float64).eps
    if np.max(np.abs(mu[:components])) > floor or mu[-1] > 2 + floor:
        raise ValueError("normalized Laplacian spectrum failed its numerical bounds")
    if mu[components] <= floor:
        raise ValueError("positive spectral gap is unresolved in float64")
    mu[:components] = 0
    return mu, components


def stability_prediction(affinity, leading_exponent, strengths):
    """Conditional transverse exponents and strict stability interval in 0<=s<=1.

    Valid for a chaotic synchronized orbit (finite lambda > 0), identical
    K=1 maps, simultaneous mixing of their full outputs, and this reversible W.
    This does not predict capture probabilities, cluster formation or the
    exponent of an orbit reached after coupling changes the attractor.
    """
    lam = float(leading_exponent)
    s = np.asarray(strengths, dtype=np.float64)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("leading_exponent must be finite and positive")
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("strengths must be a nonempty finite vector in [0, 1]")
    mu, components = laplacian_spectrum(affinity)
    decay = np.exp(-lam)
    with np.errstate(divide="ignore"):
        transverse = lam + np.log(np.abs(1 - s[:, None] * mu[None, 1:]))
    low = float(-np.expm1(-lam) / mu[1]) if components == 1 else None
    high = float((1 + decay) / mu[-1])
    exists = low is not None and low < min(high, 1.0)
    return {
        "laplacian_eigenvalues": mu,
        "components": components,
        "eigenratio": float(mu[-1] / mu[1]) if components == 1 else None,
        "eigenratio_limit": float((1 + decay) / -np.expm1(-lam)),
        "unrestricted_open_interval": [low, high] if components == 1 else None,
        "has_stable_strength_in_0_1": bool(exists),
        "strengths": s,
        "transverse_exponents": transverse,
        "stable": np.max(transverse, axis=1) < 0,
    }


def mix_outputs(outputs, weights, strength):
    """Simultaneous output mixing for a NumPy (nodes, state_dimensions) array."""
    x = np.asarray(outputs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("outputs must be a finite (nodes >= 2, dimensions) array")
    if (w.shape != (len(x), len(x)) or not np.isfinite(w).all() or (w < 0).any() or
            not np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-14)):
        raise ValueError("weights must be a finite nonnegative row-stochastic matrix")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    return (1 - strength) * x + strength * (w @ x)


def pairwise_relative_distances(states):
    """Direct full-state differences; no Gram-matrix cancellation near 1e-8.

    Divide by min(||x_i||, ||x_j||), requiring the article's directed criterion
    in both orientations. Memory is O(nodes * dimensions), not O(nodes^2*d).
    """
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("states must be a finite (nodes >= 2, dimensions) array")
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("relative distances require finite positive state norms")
    out = np.zeros((len(x), len(x)))
    for i in range(len(x) - 1):
        distance = np.linalg.norm(x[i + 1:] - x[i], axis=1) / np.minimum(norms[i], norms[i + 1:])
        out[i, i + 1:] = distance
        out[i + 1:, i] = distance
    if not np.isfinite(out).all():
        raise ValueError("pairwise distance overflow")
    return out


def persistent_pair_summary(tail_max_distances, tolerance=1e-8):
    """Describe pairs sustained below tolerance; components are NOT clusters.

    Input is the elementwise maximum distance over the entire stated tail.
    A connected chain of close pairs need not have a small diameter. Retain
    and flag that fact instead of labeling its component synchronized.
    """
    d = np.asarray(tail_max_distances, dtype=np.float64)
    if (d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 2 or not np.isfinite(d).all()
            or (d < 0).any() or not np.array_equal(d, d.T) or np.any(np.diag(d) != 0)):
        raise ValueError("tail distances must be finite, nonnegative, symmetric with zero diagonal")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    adjacent = d < tolerance
    count, labels = connected_components(adjacent, directed=False)
    components = []
    for label in range(count):
        members = np.flatnonzero(labels == label)
        diameter = float(d[np.ix_(members, members)].max())
        components.append({"nodes": members.tolist(), "max_relative_distance": diameter,
                           "all_pairs_sustained": bool(diameter < tolerance)})
    return {"global_sync": bool(d.max() < tolerance),
            "sustained_pair_count": int(np.triu(adjacent, 1).sum()),
            "pair_count": len(d) * (len(d) - 1) // 2,
            "components": components}
