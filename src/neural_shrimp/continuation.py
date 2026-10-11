"""Ray continuation of periodic orbits and typed stability boundaries.

Given a two-parameter family of smooth maps and a certified attracting
period-``p`` orbit at a center point, follow the orbit along a ray in
parameter space with predictor-corrector steps (secant predictor, exact
Newton corrector from :func:`neural_shrimp.orbits.certify_orbit`), locate
where it stops being attracting by bisection, and type the boundary from the
dominant multiplier at the last attracting point:

* ``fold``: real multiplier approaching +1 (saddle-node of the orbit);
* ``flip``: real multiplier approaching -1 (period doubling);
* ``neimark_sacker``: complex pair approaching the unit circle;
* ``unresolved``: stability lost or orbit lost without a multiplier within
  ``unit_band`` of the unit circle;
* ``none_within_range``: still attracting at ``s_max``.

Typed points along many rays describe boundary organization; they are not a
continuation of the bifurcation curves themselves.
"""

import numpy as np

from .orbits import certify_orbit


def _dominant(certificate):
    m = certificate["multipliers"][0]
    return complex(m["real"], m["imag"])


def classify_multiplier(mu, unit_band=0.02, real_tolerance=1e-6):
    """Boundary type implied by the dominant multiplier near the unit circle."""
    if abs(abs(mu) - 1.0) > unit_band:
        return "unresolved"
    if abs(mu.imag) <= real_tolerance * max(1.0, abs(mu)):
        return "fold" if mu.real > 0 else "flip"
    return "neimark_sacker"


def continue_ray(family, x0, period, center, direction, s_max, h0, h_min=None, h_max=None,
                 bisection_tolerance=None, unit_band=0.02, cert_options=None, max_steps=2000):
    """Follow an attracting orbit from ``center`` along ``direction``.

    ``family(params)`` returns ``(map_fn, jacobian)`` for a parameter vector.
    Parameters along the ray are ``center + s * direction`` for ``s`` in
    ``[0, s_max]``. Returns a dict with the accepted path (s, dominant
    multiplier modulus), the boundary location ``s_boundary`` (midpoint of the
    final bracket), its bracket, ``type`` and the dominant multiplier at the
    last attracting point.
    """
    center = np.asarray(center, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    h_min = h0 / 256.0 if h_min is None else h_min
    h_max = 4.0 * h0 if h_max is None else h_max
    tol = h0 / 64.0 if bisection_tolerance is None else bisection_tolerance
    options = {"candidate_tolerance": 0.05, **(cert_options or {})}

    def certify(s, seed):
        map_fn, jacobian = family(center + s * direction)
        return certify_orbit(map_fn, jacobian, seed, period, **options)

    start = certify(0.0, np.asarray(x0, dtype=np.float64))
    if not (start["verified"] and start["attracting"]):
        return {"type": "start_not_attracting", "start": start, "path": [], "s_boundary": None,
                "bracket": None, "dominant": None}
    path = [(0.0, np.asarray(start["orbit"][0]), _dominant(start))]
    s, h = 0.0, h0
    previous_state = None
    failure_s = None
    for _ in range(max_steps):
        if s >= s_max:
            break
        trial_s = min(s + h, s_max)
        state = path[-1][1]
        seed = state if previous_state is None else state + (state - previous_state) * (
            (trial_s - s) / max(path[-1][0] - path[-2][0], 1e-300))
        cert = certify(trial_s, seed)
        if cert["verified"] and cert["attracting"]:
            previous_state = state
            path.append((trial_s, np.asarray(cert["orbit"][0]), _dominant(cert)))
            s, h = trial_s, min(h * 1.5, h_max)
            continue
        if h > h_min:
            h *= 0.5
            failure_s = trial_s
            continue
        failure_s = trial_s
        break
    last_s, last_state, last_mu = path[-1]
    if failure_s is None and last_s >= s_max:
        return {"type": "none_within_range", "path": [(p[0], abs(p[2])) for p in path],
                "s_boundary": None, "bracket": None, "dominant": [last_mu.real, last_mu.imag]}
    low, high = last_s, failure_s if failure_s is not None else min(last_s + h_min, s_max)
    low_state, low_mu = last_state, last_mu
    while high - low > tol:
        mid = 0.5 * (low + high)
        cert = certify(mid, low_state)
        if cert["verified"] and cert["attracting"]:
            low, low_state, low_mu = mid, np.asarray(cert["orbit"][0]), _dominant(cert)
        else:
            high = mid
    kind = classify_multiplier(low_mu, unit_band=unit_band)
    return {"type": kind, "path": [(p[0], abs(p[2])) for p in path],
            "s_boundary": 0.5 * (low + high), "bracket": [low, high],
            "dominant": [low_mu.real, low_mu.imag], "last_attracting_state": low_state.tolist()}


def ray_directions(count):
    """Unit directions at equal angles in normalized parameter coordinates."""
    angles = 2.0 * np.pi * np.arange(count) / count
    return np.column_stack((np.cos(angles), np.sin(angles)))
