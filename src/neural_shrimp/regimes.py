"""Regime features of long gradient-descent trajectories (exploratory A3b).

Beyond the edge of stability, a full-batch gradient-descent trajectory can sit
on an ordinary period-2 orbit, on an invariant circle of the two-step map
(quasi-periodic training), on a phase-locked periodic orbit of that circle,
or on a chaotic attractor. The helpers here turn saved trajectory pieces into
regime labels and features, and compare predictive models of training
outcomes with and without rotation-number information.

``rotation_number`` and ``cycle_rotation`` measure turns of two-step points
around their centroid in the plane of their first two principal axes;
``spectrum_features`` summarizes a loss series; ``classify`` applies fixed
regime rules; ``blocked_cv_r2``, ``partial_correlation`` and
``block_bootstrap`` support nested model comparisons on cells ordered along a
rate slice, where neighbouring cells are strongly dependent.
"""

from math import gcd

import numpy as np

REGIMES = ("divergent", "fixed", "period2", "locked", "other_periodic", "quasiperiodic",
           "chaotic", "irregular")


def _plane_angles(points):
    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ vt[:2].T
    share = float((singular[:2] ** 2).sum() / max(float((singular ** 2).sum()), 1e-300))
    return np.arctan2(planar[:, 1], planar[:, 0]), share


def rotation_number(lobe):
    """Rotation number (turns per point) of ordered points on a closed curve.

    ``lobe`` holds consecutive two-step states, shape (points, parameters).
    Returns (rho, rho_first_half, rho_second_half, planar_share), with rho in
    [0, 1/2] because a projection cannot tell rho from 1 - rho.
    """
    lobe = np.asarray(lobe, dtype=np.float64)
    if lobe.ndim != 2 or len(lobe) < 8:
        raise ValueError("lobe must have shape (points >= 8, parameters)")
    angles, share = _plane_angles(lobe)
    steps = np.diff(angles)
    steps = (steps + np.pi) % (2 * np.pi) - np.pi
    half = len(steps) // 2

    def rate(part):
        return float(abs(part.sum()) / (2 * np.pi * len(part)))

    return rate(steps), rate(steps[:half]), rate(steps[half:]), share


def cycle_rotation(cycle):
    """Rotation m/q of a periodic cycle of the map, from its two-step points.

    ``cycle`` holds one period of consecutive states, shape (p, parameters).
    For even p the two-step points are every other state (q = p / 2); for odd
    p they visit all states (q = p). Returns (m, q, rho) with m the absolute
    net number of turns around the centroid, reduced to lowest terms in rho.
    """
    cycle = np.asarray(cycle, dtype=np.float64)
    p = len(cycle)
    if p < 2:
        raise ValueError("cycle must hold at least two states")
    order = np.arange(0, p, 2) if p % 2 == 0 else (2 * np.arange(p)) % p
    points = cycle[order]
    q = len(points)
    if q < 3:
        return 0, q, 0.0
    angles, _ = _plane_angles(points)
    steps = np.diff(np.r_[angles, angles[0]])
    steps = (steps + np.pi) % (2 * np.pi) - np.pi
    m = int(round(abs(steps.sum()) / (2 * np.pi)))
    if m == 0:
        return 0, q, 0.0
    g = gcd(m, q)
    return m // g, q // g, m / q


def spectrum_features(series, peaks=8):
    """Line fraction and normalized spectral entropy of a scalar series.

    The series is demeaned and Hann-windowed. ``line_fraction`` is the share of
    power in the ``peaks`` strongest frequency bins plus one neighbour on each
    side (1 for a few pure lines, small for broadband noise); ``entropy`` is
    the Shannon entropy of the normalized power spectrum divided by its
    maximum. A constant series returns (1.0, 0.0).
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1 or len(x) < 16 or not np.all(np.isfinite(x)):
        raise ValueError("series must be a finite one-dimensional array of length >= 16")
    x = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    power[0] = 0.0
    total = power.sum()
    if total <= 1e-300:
        return 1.0, 0.0
    p = power / total
    keep = np.zeros(len(p), dtype=bool)
    for i in np.argsort(p)[::-1][:peaks]:
        keep[max(i - 1, 0):i + 2] = True
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(len(p) - 1))
    return float(p[keep].sum()), entropy


def peak_frequencies(series, count=3, low=0.005):
    """Strongest spectral peaks (cycles per sample) above ``low``, by power."""
    x = np.asarray(series, dtype=np.float64)
    x = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x))
    found = []
    for i in np.argsort(power)[::-1]:
        if freq[i] < low or i == 0 or i == len(power) - 1:
            continue
        if power[i] < power[i - 1] or power[i] < power[i + 1]:
            continue
        a, b, c = np.log(power[i - 1] + 1e-300), np.log(power[i] + 1e-300), np.log(power[i + 1] + 1e-300)
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        found.append(float(freq[i] + shift * (freq[1] - freq[0])))
        if len(found) == count:
            break
    return found


def dominant_ar_root(series, order=6):
    """Complex root of largest modulus of a least-squares AR(order) fit.

    For a noise-driven weakly damped oscillation (for example the training
    loss sampled every two steps near a Neimark-Sacker point) this estimates
    the two-step multiplier: returns (modulus, frequency = angle / 2 pi in
    [0, 1/2]). Returns (nan, nan) when the fit has no complex root.
    """
    z = np.asarray(series, dtype=np.float64)
    if z.ndim != 1 or len(z) < 10 * order or not np.all(np.isfinite(z)):
        raise ValueError("series must be finite, one-dimensional and at least 10 * order long")
    z = z - z.mean()
    lagged = np.column_stack([z[order - k - 1:len(z) - k - 1] for k in range(order)])
    coef, *_ = np.linalg.lstsq(lagged, z[order:], rcond=None)
    roots = np.roots(np.r_[1.0, -coef])
    roots = roots[np.abs(roots.imag) > 1e-9]
    if len(roots) == 0:
        return float("nan"), float("nan")
    top = roots[np.argmax(np.abs(roots))]
    return float(abs(top)), float(abs(np.angle(top)) / (2 * np.pi))


def classify(status, period, ftle, rho_halves, rho, separation, cycle_m,
             qp_ftle=1e-3, qp_rho_tol=5e-4, chaos_ftle=5e-3, separation_floor=1e-6):
    """Regime label from screening, Lyapunov and rotation features.

    ``ftle`` is a pair of largest-exponent estimates from independent tangent
    vectors; ``rho_halves`` the rotation numbers of the two halves of the
    kept two-step points; ``separation`` the median final distance between
    the base tail and copies perturbed by 1e-10 at the start of the tail (on
    the attractor); ``cycle_m`` the net turns of a screened cycle. Quasi-
    periodic cells need both exponents below ``qp_ftle``, a converged nonzero
    rotation number and no separation growth beyond ``separation_floor``;
    chaotic cells need both exponents above ``chaos_ftle`` and separation
    beyond the floor. Remaining nonperiodic cells are "irregular".
    """
    if status == 0:
        return "divergent"
    if status == 1:
        return "fixed"
    if status == 2:
        if period == 2:
            return "period2"
        if period >= 3 and period % 2 == 0 and cycle_m >= 1:
            return "locked"
        return "other_periodic"
    a, b = ftle
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(separation)):
        return "irregular"
    if (max(a, b) < qp_ftle and rho is not None and rho > 0.005
            and abs(rho_halves[0] - rho_halves[1]) < qp_rho_tol and separation < separation_floor):
        return "quasiperiodic"
    if min(a, b) > chaos_ftle and separation > separation_floor:
        return "chaotic"
    return "irregular"


def _design(columns, n):
    return np.column_stack([np.ones(n)] + [np.asarray(c, dtype=np.float64) for c in columns])


def blocked_cv_r2(features, target, blocks, ridge=1e-8):
    """Out-of-sample R^2 of least squares with leave-one-block-out folds.

    ``features`` is a list of 1-D arrays (an intercept is added), ``blocks``
    an integer fold label per row. Predictions for each held-out block use a
    fit on all other blocks; R^2 is computed over all held-out predictions.
    """
    y = np.asarray(target, dtype=np.float64)
    x = _design(features, len(y))
    blocks = np.asarray(blocks)
    prediction = np.empty_like(y)
    for b in np.unique(blocks):
        test = blocks == b
        train = ~test
        xt = x[train]
        coef = np.linalg.solve(xt.T @ xt + ridge * np.eye(x.shape[1]), xt.T @ y[train])
        prediction[test] = x[test] @ coef
    resid = y - prediction
    return float(1.0 - resid @ resid / max(float(((y - y.mean()) ** 2).sum()), 1e-300))


def partial_correlation(feature, target, controls):
    """Correlation of ``feature`` and ``target`` after regressing out ``controls``."""
    x = _design(controls, len(target))
    pinv = np.linalg.pinv(x)
    rf = np.asarray(feature, dtype=np.float64) - x @ (pinv @ np.asarray(feature, dtype=np.float64))
    rt = np.asarray(target, dtype=np.float64) - x @ (pinv @ np.asarray(target, dtype=np.float64))
    denom = np.sqrt((rf @ rf) * (rt @ rt))
    return float(rf @ rt / denom) if denom > 0 else 0.0


def block_bootstrap(statistic, n_rows, block, draws=500, seed=0):
    """Percentile interval of ``statistic(index)`` under a moving-block bootstrap.

    Rows are ordered (for example along a rate slice); each draw concatenates
    random contiguous blocks of length ``block`` until ``n_rows`` indices are
    collected. Returns (low, high) at 2.5% and 97.5%.
    """
    rng = np.random.default_rng(seed)
    starts_max = max(n_rows - block, 0)
    values = []
    for _ in range(draws):
        parts = []
        while sum(len(p) for p in parts) < n_rows:
            s = int(rng.integers(0, starts_max + 1))
            parts.append(np.arange(s, min(s + block, n_rows)))
        index = np.concatenate(parts)[:n_rows]
        values.append(statistic(index))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
