"""Batched autonomous maps of the three-neuron tanh network.

For many weight matrices at once: settle, record local maxima of ``x3`` as
full-state events (cubic Hermite localization), screen the event sequence for
the smallest full-state recurrence period, estimate the largest finite-time
Lyapunov exponent, detect equilibria, and, for periodic candidates, integrate
the variational equations over one recurrence period to obtain Floquet
multipliers. Periods count ``x3`` maxima per cycle (the EXP-004 convention).
Everything is a screening label; verified orbits need shooting certificates.

Internally arrays use a structure-of-arrays layout (component first, cells
last), which is about twice as fast as batched ``einsum`` for 3 x 3 systems.
The public functions take and return cell-first arrays.
"""

import numpy as np


def _soa_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 3 or weights.shape[1:] != (3, 3):
        raise ValueError("weights must have shape (cells, 3, 3)")
    return np.ascontiguousarray(weights.transpose(1, 2, 0))


def _rhs(x, w):
    t = np.tanh(x)
    return -x + w[:, 0] * t[0] + w[:, 1] * t[1] + w[:, 2] * t[2]


def _jvp(x, w, v):
    """Jacobian of the flow applied to vectors (3, cells) or matrices (3, k, cells)."""
    s = 1.0 - np.tanh(x) ** 2
    if v.ndim == 2:
        sv = s * v
        return -v + w[:, 0] * sv[0] + w[:, 1] * sv[1] + w[:, 2] * sv[2]
    sv = s[:, None, :] * v
    return -v + (w[:, 0, None, :] * sv[0][None] + w[:, 1, None, :] * sv[1][None]
                 + w[:, 2, None, :] * sv[2][None])


def _rk4(x, w, dt):
    k1 = _rhs(x, w)
    k2 = _rhs(x + 0.5 * dt * k1, w)
    k3 = _rhs(x + 0.5 * dt * k2, w)
    k4 = _rhs(x + dt * k3, w)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _rk4_tangent(x, v, w, dt):
    k1, l1 = _rhs(x, w), _jvp(x, w, v)
    x2, v2 = x + 0.5 * dt * k1, v + 0.5 * dt * l1
    k2, l2 = _rhs(x2, w), _jvp(x2, w, v2)
    x3, v3 = x + 0.5 * dt * k2, v + 0.5 * dt * l2
    k3, l3 = _rhs(x3, w), _jvp(x3, w, v3)
    x4, v4 = x + dt * k3, v + dt * l3
    k4, l4 = _rhs(x4, w), _jvp(x4, w, v4)
    return (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4),
            v + (dt / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4))


def _hermite_maximum(x0, x1, f0, f1, dt):
    """Locate the maximum of x3 between two steps (cell-first arrays).

    Newton iterations on the derivative of the Hermite cubic of ``x3`` in the
    step fraction ``t``; the full state is interpolated at the result.
    """
    d0, d1 = f0[:, 2], f1[:, 2]
    denom = d0 - d1
    frac = np.clip(np.where(denom != 0, d0 / np.where(denom != 0, denom, 1.0), 0.5), 0.0, 1.0)
    p0, p1, m0, m1 = x0[:, 2], x1[:, 2], dt * d0, dt * d1
    for _ in range(4):
        t = frac
        dh = ((6 * t * t - 6 * t) * p0 + (3 * t * t - 4 * t + 1) * m0
              + (-6 * t * t + 6 * t) * p1 + (3 * t * t - 2 * t) * m1)
        d2h = (12 * t - 6) * p0 + (6 * t - 4) * m0 + (-12 * t + 6) * p1 + (6 * t - 2) * m1
        step = np.where(d2h != 0, dh / np.where(d2h != 0, d2h, 1.0), 0.0)
        frac = np.clip(frac - step, 0.0, 1.0)
    t = frac[:, None]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return frac, h00 * x0 + h10 * dt * f0 + h01 * x1 + h11 * dt * f1


def screen_events(events, times, max_period=32, repeats=4, atol=2e-6, rtol=2e-5):
    """Smallest event period with full-state recurrence over the last events.

    ``events`` has shape (cells, count, 3), chronological, with NaN padding
    after the last valid event; ``times`` has shape (cells, count). Period
    ``p`` requires ``repeats * p + 1`` valid events and agreement of every
    event with the event ``p`` later across the final ``repeats * p + 1``
    events. Returns (period, cycle_time): period 0 means none found and
    cycle_time is the mean duration of one ``p``-event cycle.
    """
    cells = events.shape[0]
    period = np.zeros(cells, dtype=np.int64)
    cycle = np.full(cells, np.nan)
    valid = np.sum(np.all(np.isfinite(events), axis=2), axis=1)
    for p in range(1, max_period + 1):
        need = repeats * p + 1
        idx = np.flatnonzero((period == 0) & (valid >= need))
        if len(idx) == 0:
            continue
        positions = valid[idx, None] - need + np.arange(need)[None, :]
        seq = events[idx[:, None], positions]
        a, b = seq[:, p:], seq[:, :-p]
        ok = np.all(np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b)), axis=(1, 2))
        hits = idx[ok]
        period[hits] = p
        last = valid[hits] - 1
        cycle[hits] = (times[hits, last] - times[hits, last - repeats * p]) / repeats
    return period, cycle


def autonomous_map(weights, x0, dt, settle, measure, max_events=160, max_period=32, repeats=4,
                   atol=2e-6, rtol=2e-5, ftle_align=100.0, tangent_seed=0,
                   divergence_threshold=1e6, equilibrium_tolerance=1e-8):
    """Settle, record x3-maximum events, screen periods and estimate the FTLE.

    Returns a dict with ``period`` (-1 diverged, 0 no recurrence),
    ``cycle_time``, ``ftle`` (per unit time over ``measure - ftle_align``),
    ``equilibrium`` (no events in the measurement window and final vector-field
    norm below ``equilibrium_tolerance``), ``event_states`` (cells,
    max_events, 3), ``event_times``, ``event_counts``, ``final_state`` and
    ``diverged``.
    """
    w = _soa_weights(weights)
    x = np.ascontiguousarray(np.array(x0, dtype=np.float64).T)
    cells = x.shape[1]
    settle_steps, measure_steps = int(round(settle / dt)), int(round(measure / dt))
    align_steps = int(round(ftle_align / dt))
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(settle_steps):
            x = _rk4(x, w, dt)
    diverged = ~np.all(np.isfinite(x), axis=0) | (np.max(np.abs(x), axis=0) > divergence_threshold)
    rng = np.random.default_rng(tangent_seed)
    v = rng.normal(size=(cells, 3)).T.copy()
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    log_growth = np.zeros(cells)
    events = np.full((cells, max_events, 3), np.nan)
    times = np.full((cells, max_events), np.nan)
    counts = np.zeros(cells, dtype=np.int64)
    f_prev = _rhs(x, w)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(measure_steps):
            x_new, v = _rk4_tangent(x, v, w, dt)
            norm = np.sqrt(np.sum(v * v, axis=0))
            good = norm > 0
            if step >= align_steps:
                log_growth += np.log(np.where(good, norm, 1.0))
            v = v / np.where(good, norm, 1.0)
            f_new = _rhs(x_new, w)
            crossing = (f_prev[2] > 0) & (f_new[2] <= 0)
            if np.any(crossing):
                idx = np.flatnonzero(crossing)
                frac, state = _hermite_maximum(x[:, idx].T, x_new[:, idx].T,
                                               f_prev[:, idx].T, f_new[:, idx].T, dt)
                slot = counts[idx] % max_events
                events[idx, slot] = state
                times[idx, slot] = (step + frac) * dt
                counts[idx] += 1
            x, f_prev = x_new, f_new
    wrapped = np.flatnonzero(counts > max_events)
    if len(wrapped):
        order = (np.arange(max_events)[None, :] + (counts[wrapped] % max_events)[:, None]) % max_events
        events[wrapped] = np.take_along_axis(events[wrapped], order[:, :, None], axis=1)
        times[wrapped] = np.take_along_axis(times[wrapped], order, axis=1)
    diverged |= ~np.all(np.isfinite(x), axis=0) | (np.max(np.abs(x), axis=0) > divergence_threshold)
    period, cycle = screen_events(events, times, max_period, repeats, atol, rtol)
    period[diverged] = -1
    field = np.sqrt(np.sum(_rhs(x, w) ** 2, axis=0))
    equilibrium = ~diverged & (counts == 0) & (field < equilibrium_tolerance)
    ftle = log_growth / ((measure_steps - align_steps) * dt)
    return {"period": period, "cycle_time": cycle, "ftle": np.where(diverged, np.nan, ftle),
            "equilibrium": equilibrium, "final_field_norm": field,
            "event_states": events, "event_times": times, "event_counts": counts,
            "final_state": np.ascontiguousarray(x.T), "diverged": diverged}


def floquet_multipliers(weights, start_states, periods_time, dt):
    """Monodromy eigenvalues over one cycle time from each start state.

    Integrates state and the 3 x 3 variational matrix with RK4 for
    ``floor(T / dt)`` steps plus one final partial step for the remainder.
    Returns (multipliers sorted by decreasing modulus, end_state), cell-first.
    One multiplier should be near 1 (the flow direction) for a periodic orbit.
    """
    w_all = _soa_weights(weights)
    x_all = np.ascontiguousarray(np.array(start_states, dtype=np.float64).T)
    cells = x_all.shape[1]
    m_all = np.repeat(np.eye(3)[:, :, None], cells, axis=2)
    total = np.asarray(periods_time, dtype=np.float64)
    full_steps = np.floor(total / dt).astype(np.int64)
    remainder = total - full_steps * dt
    for step in range(int(full_steps.max()) if cells else 0):
        active = np.flatnonzero(full_steps > step)
        if len(active) == 0:
            break
        xa, ma = _rk4_tangent(x_all[:, active], m_all[:, :, active], w_all[:, :, active], dt)
        x_all[:, active], m_all[:, :, active] = xa, ma
    rem = np.flatnonzero(remainder > 0)
    if len(rem):
        h = remainder[rem]
        xa, ma, wa = x_all[:, rem], m_all[:, :, rem], w_all[:, :, rem]
        k1, l1 = _rhs(xa, wa), _jvp(xa, wa, ma)
        x2, m2 = xa + 0.5 * h * k1, ma + 0.5 * h * l1
        k2, l2 = _rhs(x2, wa), _jvp(x2, wa, m2)
        x3, m3 = xa + 0.5 * h * k2, ma + 0.5 * h * l2
        k3, l3 = _rhs(x3, wa), _jvp(x3, wa, m3)
        x4, m4 = xa + h * k3, ma + h * l3
        k4, l4 = _rhs(x4, wa), _jvp(x4, wa, m4)
        x_all[:, rem] = xa + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        m_all[:, :, rem] = ma + (h / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4)
    eigenvalues = np.linalg.eigvals(m_all.transpose(2, 0, 1))
    order = np.argsort(-np.abs(eigenvalues), axis=1)
    return np.take_along_axis(eigenvalues, order, axis=1), np.ascontiguousarray(x_all.T)
