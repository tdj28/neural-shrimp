"""Resonance-anchored ancestry of phase-locked windows (exploratory A3).

A period-``P`` window of a two-parameter family of maps is an Arnold tongue
of a Neimark-Sacker bifurcation of a parent orbit when the region where its
orbit exists (bounded by two fold curves, multiplier +1) narrows onto the
point of the parent Neimark-Sacker curve whose birth rotation equals the
window's rotation, while the orbit collapses onto the parent orbit.

This module provides the numerical pieces, all built on the hardened Newton
certification of :func:`neural_shrimp.orbits.certify_orbit`:

* :func:`existence_ray` follows a verified orbit of any stability along a ray
  and locates where it ceases to exist, typing the end by the multiplier
  closest to +1 (``fold``) or not (``lost``);
* :func:`continue_segment` carries an orbit along a straight parameter segment
  with adaptive steps, refusing steps whose orbit jumps (branch switching);
* :func:`same_cycle` compares two periodic orbits up to cyclic shift;
* :func:`cycle_amplitude` measures how far an orbit lies from a parent orbit;
* :func:`classify_ancestry` applies frozen acceptance rules to per-level
  cross-section records.

Floating-point evidence, not a computer-assisted proof.
"""

import numpy as np

from .orbits import certify_orbit

OUTCOMES = ("pass", "alternative_ancestry", "exhausted_budget", "no_qualifying_window", "invalid")


def closest_to_one(certificate):
    """The multiplier closest to +1 and its distance."""
    values = [complex(m["real"], m["imag"]) for m in certificate.get("multipliers", [])]
    if not values:
        return None, np.inf
    mu = min(values, key=lambda z: abs(z - 1.0))
    return mu, float(abs(mu - 1.0))


def _certify(family, params, seed, period, options):
    map_fn, jacobian = family(np.asarray(params, dtype=np.float64))
    return certify_orbit(map_fn, jacobian, np.asarray(seed, dtype=np.float64), period, **options)


def same_cycle(a, b, tolerance):
    """True when orbits ``a`` and ``b`` (period x dim) coincide up to cyclic shift."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    scale = max(1.0, float(np.max(np.abs(a))))
    return any(float(np.max(np.abs(np.roll(b, -k, axis=0) - a))) <= tolerance * scale
               for k in range(len(a)))


def orbit_jump(a, b):
    """Smallest max-norm distance between orbits ``a`` and ``b`` over cyclic shifts."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return min(float(np.max(np.abs(np.roll(b, -k, axis=0) - a))) for k in range(len(a)))


def cycle_amplitude(orbit, parent):
    """Max over orbit points of the distance to the nearest parent-orbit point."""
    orbit, parent = np.asarray(orbit, dtype=np.float64), np.asarray(parent, dtype=np.float64)
    return float(max(np.min(np.linalg.norm(parent - p, axis=1)) for p in orbit))


def existence_ray(family, x0, period, center, direction, s_max, h0, h_min=None,
                  bisection_tolerance=1e-10, cert_options=None, max_steps=4000, fold_band=0.05,
                  jump_factor=50.0):
    """Follow a verified period-``period`` orbit of any stability along a ray.

    Parameters are ``center + s * direction`` for ``s`` in ``[0, s_max]``. Steps
    whose orbit moves more than ``jump_factor`` times the previous accepted
    per-step motion (scaled by the step ratio) are refused as possible branch
    switches. Returns a dict with ``type`` (``fold``: the multiplier closest to
    +1 at the last existing point lies within ``fold_band`` of +1; ``lost``:
    existence ended otherwise; ``none_within_range``; ``start_failed``),
    ``s_boundary`` (midpoint of the final bracket), ``bracket``, the last
    orbit and its multipliers, and ``steps``.
    """
    center = np.asarray(center, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    h_min = h0 / 1024.0 if h_min is None else h_min
    options = {"candidate_tolerance": 0.05, **(cert_options or {})}
    start = _certify(family, center, x0, period, options)
    if not start["verified"]:
        return {"type": "start_failed", "start_reason": start["reason"], "s_boundary": None,
                "bracket": None, "steps": 0}
    path = [(0.0, np.asarray(start["orbit"]), start)]
    s, h, motion = 0.0, h0, None
    failure_s, steps = None, 0
    while steps < max_steps and s < s_max:
        steps += 1
        trial_s = min(s + h, s_max)
        orbit = path[-1][1]
        if len(path) >= 2:
            ds = path[-1][0] - path[-2][0]
            seed = orbit[0] + (orbit[0] - path[-2][1][0]) * ((trial_s - s) / max(ds, 1e-300))
        else:
            seed = orbit[0]
        cert = _certify(family, center + trial_s * direction, seed, period, options)
        ok = cert["verified"]
        if ok:
            new = np.asarray(cert["orbit"])
            jump = orbit_jump(orbit, new)
            if motion is not None and jump > jump_factor * motion * (trial_s - s) + 1e-12:
                ok = False
        if ok:
            motion = jump / max(trial_s - s, 1e-300)
            path.append((trial_s, new, cert))
            s, h = trial_s, min(h * 1.5, 16.0 * h0)
            continue
        failure_s = trial_s
        if h > h_min:
            h *= 0.5
            continue
        break
    last_s, last_orbit, last_cert = path[-1]
    if failure_s is None:
        mu, dist = closest_to_one(last_cert)
        return {"type": "none_within_range", "s_boundary": None, "bracket": None, "steps": steps,
                "last_orbit": last_orbit.tolist(), "closest_to_one": [mu.real, mu.imag], "distance_to_one": dist}
    # Near a fold the orbit moves like the square root of the distance to it,
    # so the bisection guards continuity with a tight seed-distance tolerance
    # instead of the per-step motion test (node and saddle share the fold).
    low, high, low_orbit, low_cert = last_s, failure_s, last_orbit, last_cert
    tight = {**options, "candidate_tolerance": min(options["candidate_tolerance"], 1e-2)}
    while high - low > bisection_tolerance:
        mid = 0.5 * (low + high)
        cert = _certify(family, center + mid * direction, low_orbit[0], period, tight)
        if cert["verified"]:
            low, low_orbit, low_cert = mid, np.asarray(cert["orbit"]), cert
        else:
            high = mid
    mu, dist = closest_to_one(low_cert)
    return {"type": "fold" if dist <= fold_band else "lost", "s_boundary": 0.5 * (low + high),
            "bracket": [low, high], "steps": steps, "last_orbit": low_orbit.tolist(),
            "closest_to_one": [mu.real, mu.imag], "distance_to_one": dist,
            "spectral_radius": low_cert["spectral_radius"], "reason_at_failure": "existence_ended"}


def continue_segment(family, x0, period, start, end, h0=0.02, h_min=1e-6, cert_options=None,
                     jump_factor=50.0, max_steps=4000):
    """Carry an orbit from parameters ``start`` to ``end`` (fraction t in [0, 1]).

    Accepts verified orbits of any stability; refuses steps whose orbit jumps
    by more than ``jump_factor`` times the running per-unit-t motion. Returns
    a dict with ``ok``, the final certificate and orbit, ``t_reached`` and
    ``steps``.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    options = {"candidate_tolerance": 0.05, **(cert_options or {})}
    cert = _certify(family, start, x0, period, options)
    if not cert["verified"]:
        return {"ok": False, "t_reached": 0.0, "steps": 0, "reason": f"start:{cert['reason']}"}
    t, h, motion, steps = 0.0, h0, None, 0
    orbit, previous = np.asarray(cert["orbit"]), None
    while t < 1.0 and steps < max_steps:
        steps += 1
        trial = min(1.0, t + h)
        seed = orbit[0] if previous is None else orbit[0] + (orbit[0] - previous[1][0]) * (
            (trial - t) / max(t - previous[0], 1e-300))
        new_cert = _certify(family, start + trial * (end - start), seed, period, options)
        ok = new_cert["verified"]
        if ok:
            new = np.asarray(new_cert["orbit"])
            jump = orbit_jump(orbit, new)
            if motion is not None and jump > jump_factor * motion * (trial - t) + 1e-12:
                ok = False
        if ok:
            motion = jump / max(trial - t, 1e-300)
            previous = (t, orbit)
            orbit, cert, t = new, new_cert, trial
            h = min(2.0 * h, 0.25)
            continue
        h *= 0.5
        if h < h_min:
            return {"ok": False, "t_reached": t, "steps": steps, "reason": "step_underflow",
                    "orbit": orbit.tolist(), "certificate": cert}
    return {"ok": t >= 1.0, "t_reached": t, "steps": steps, "reason": "reached" if t >= 1.0 else "max_steps",
            "orbit": orbit.tolist(), "certificate": cert}


def classify_ancestry(levels, root_tolerance, amplitude_factor=10.0, controls_passed=True):
    """Apply frozen acceptance rules to cross-section records ordered by decreasing distance.

    Each level is a dict with ``delta`` (distance beyond the parent curve),
    ``left`` and ``right`` (fold-ray results with ``type`` and ``u``, the
    tangential coordinate of the fold relative to the predicted root), the
    interior orbit ``amplitude``, ``primitive`` (bool) and ``continuity``
    (bool: reached from the previous level without branch switching). Returns
    (outcome, details).
    """
    if not controls_passed:
        return "invalid", {"reason": "controls failed"}
    if not levels:
        return "exhausted_budget", {"reason": "no level completed"}
    complete = [lv for lv in levels if lv.get("left", {}).get("type") == "fold"
                and lv.get("right", {}).get("type") == "fold"]
    details = {"levels_attempted": len(levels), "levels_fold_bounded": len(complete)}
    if len(complete) < len(levels) or not all(lv.get("continuity") and lv.get("primitive") for lv in levels):
        details["reason"] = "a level lacked two folds, primitivity or continuity"
        return "exhausted_budget", details
    centers = np.array([0.5 * (lv["left"]["u"] + lv["right"]["u"]) for lv in complete])
    widths = np.array([lv["right"]["u"] - lv["left"]["u"] for lv in complete])
    deltas = np.array([lv["delta"] for lv in complete])
    amps = np.array([lv["amplitude"] for lv in complete])
    # centre of the locked interval extrapolated to delta -> 0 (linear in delta)
    slope, intercept = np.polyfit(deltas, centers, 1)
    details.update({"centers": centers.tolist(), "widths": widths.tolist(), "amplitudes": amps.tolist(),
                    "extrapolated_center": float(intercept), "center_slope": float(slope),
                    "root_tolerance": float(root_tolerance),
                    "amplitude_ratio": float(amps[0] / max(amps[-1], 1e-300))})
    widths_shrink = bool(np.all(np.diff(widths) < 0) and widths[-1] > 0)
    amp_ok = amps[0] / max(amps[-1], 1e-300) >= amplitude_factor
    nearest_ok = abs(centers[-1]) <= max(root_tolerance, abs(widths[-1]))
    details.update({"widths_shrink": widths_shrink, "amplitude_ok": bool(amp_ok),
                    "center_mismatch": float(abs(intercept)), "last_center_within": bool(nearest_ok)})
    if widths_shrink and amp_ok and abs(intercept) <= root_tolerance and nearest_ok:
        return "pass", details
    details["reason"] = "fold-bounded branch resolved but does not converge onto the predicted root"
    return "alternative_ancestry", details
