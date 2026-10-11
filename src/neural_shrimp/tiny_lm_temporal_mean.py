"""Retrospective temporal-mean curvature sufficiency on fixed development states.

No forecast: every mean action uses the full observed theta[0:16] path.
Callbacks retain acquired native returns before this module validates them.
"""
import math
import numpy as np

HORIZONS = (1, 4, 16)
TARGET_COUNTS = dict(hvp=1088, gradient=162, ce=58)
PROFILE_COUNTS = dict(hvp=2, gradient=2, ce=2)
# Upper bound on separately retained state, tangent and CE-gradient vectors.
# 204 reference/perturbed states + 136 mean/frozen/ordered vectors + 12 CE gradients.
AUXILIARY_VECTORS = 352


def need(ok, message):
    if not ok:
        raise ValueError(message)


def vector(value, shape=None):
    a = np.asarray(value, dtype=np.float64)
    need(a.ndim == 1 and np.isfinite(a).all(), 'finite flat vector')
    need(shape is None or a.shape == shape, 'same vector shape')
    return a


def relative(a, b):
    return float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-12)


def forecast(times, return_bytes, dimension=15776):
    """Full measured costs plus explicit retained-vector upper bound, no compression."""
    need(set(times) == set(return_bytes) == set(TARGET_COUNTS), 'complete measured categories')
    seconds = 0.0
    size = 0
    for name, count in TARGET_COUNTS.items():
        t, b = times[name], return_bytes[name]
        need(len(t) == len(b) == 2, 'two full-size returns per category')
        need(all(type(v) is float and math.isfinite(v) and v > 0 for v in t), 'positive measured time')
        need(all(type(v) is int and v > 0 for v in b), 'positive retained return bytes')
        seconds += count * max(t)
        size += count * max(b)
    # NPY header is 128 bytes for the prescribed flat float64 vector.
    size += AUXILIARY_VECTORS * (dimension * 8 + 128)
    # Twice all native returns and unique input/summary vectors, plus metadata reserve.
    return dict(seconds=2 * seconds + 30, bytes=2 * size + 64 * 1024**2)


def mean_power(reference, direction, eta, hvp, save, check):
    reference = np.asarray(reference, dtype=np.float64)
    u = vector(direction)
    need(reference.shape == (17, len(u)) and np.isfinite(reference).all(), '17 fixed reference states')
    need(math.isfinite(eta) and eta > 0, 'positive physical rate')
    powers = [u.copy()]
    try:
        for h in range(16):
            # All 16 actions use the SAME current vector; never update it within the sum.
            current = powers[-1]
            total = np.zeros_like(current)
            for k in range(16):
                check()
                action = vector(hvp(reference[k], current), current.shape)
                total += action
                vector(total, current.shape)
            powers.append(vector(current - (eta / 16) * total, current.shape))
    finally:
        save('mean-powers', np.asarray(powers))
    return np.asarray(powers)


def acquire_control(theta, direction, eta, ridge, gradient, hvp, ce, save, check):
    """Exactly 81 gradients, 32 HVPs and 29 CE calls for one eta/4 control.

    CE slopes reuse the already acquired objective gradients minus ridge*theta.
    The original observed direction is used literally, without renormalization.
    """
    theta, u = vector(theta), vector(direction)
    need(theta.shape == u.shape and abs(float(np.linalg.norm(u)) - 1) < 1e-12,
         'literal original unit direction')
    need(math.isfinite(eta) and eta > 0 and math.isfinite(ridge) and ridge >= 0,
         'finite control map')
    reference, gradients, ce_values = [theta.copy()], [], []
    try:
        for h in range(17):
            check()
            g = vector(gradient(reference[-1]), theta.shape)
            gradients.append(g)
            c = float(ce(reference[-1]))
            ce_values.append(c)
            need(math.isfinite(c), 'finite reference CE')
            if h < 16:
                reference.append(vector(reference[-1] - eta * g, theta.shape))
    finally:
        save('reference', np.asarray(reference))
        save('reference-gradients', np.asarray(gradients))
        save('reference-ce', np.asarray(ce_values))
    frozen, ordered = [u.copy()], [u.copy()]
    try:
        for h in range(16):
            check()
            frozen.append(vector(frozen[-1] - eta * vector(hvp(reference[0], frozen[-1]), theta.shape)))
            ordered.append(vector(ordered[-1] - eta * vector(hvp(reference[h], ordered[-1]), theta.shape)))
    finally:
        save('frozen', np.asarray(frozen))
        save('ordered', np.asarray(ordered))
    epsilons = [e * max(1.0, float(np.linalg.norm(theta))) for e in (1e-4, 5e-5)]
    branches = []
    for k, eps in enumerate(epsilons):
        row = {}
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            states, readout = [vector(theta + sign * eps * u)], {}
            try:
                for h in range(1, 17):
                    check()
                    states.append(vector(states[-1] - eta * vector(gradient(states[-1]), theta.shape)))
                    if h in HORIZONS:
                        readout[h] = float(ce(states[-1]))
                        need(math.isfinite(readout[h]), 'finite perturbed CE')
            finally:
                save(f'epsilon-{k}-{label}', np.asarray(states))
                save(f'epsilon-{k}-{label}-ce', np.asarray(sorted(readout.items())).reshape(-1, 2))
            row[label], row['ce_' + label] = np.asarray(states), readout
        branches.append(row)
    reference = np.asarray(reference)
    readout_gradients = np.asarray([gradients[h] - ridge * reference[h] for h in HORIZONS])
    save('readout-gradients', readout_gradients)
    objective = np.asarray(ce_values) + .5 * ridge * np.sum(reference * reference, axis=1)
    increments = np.diff(objective)
    need(np.isfinite(increments).all(), 'finite objective increments')
    save('objective-increments', increments)
    return dict(reference=reference, reference_gradients=np.asarray(gradients), direction=u, frozen=np.asarray(frozen), ordered=np.asarray(ordered),
                branches=branches, epsilons=epsilons, readout_gradients=readout_gradients,
                objective_increments=increments.tolist(), small_step_monotone=bool(np.all(increments <= 0)))


def measurements(case, mean, additional=None):
    """All original 1% FD/transport/CE gates stay unchanged; mean is not a qualifier."""
    u = vector(case['direction'])
    additional = {} if additional is None else additional
    predictors = dict(mean=mean, frozen=case['frozen'], ordered=case['ordered'],
                      persistence=np.repeat(u[None, :], 17, axis=0), **additional)
    for p in predictors.values():
        need(np.asarray(p).shape == (17, len(u)) and np.isfinite(p).all(), 'complete comparator')
    rows = []
    for i, h in enumerate(HORIZONS):
        responses = [(r['plus'][h] - r['minus'][h]) / (2 * e)
                     for r, e in zip(case['branches'], case['epsilons'])]
        need(len(responses) == 2, 'exact two amplitude responses')
        small = vector(responses[1], u.shape)
        agreement = relative(responses[0], small)
        ordered_error = relative(predictors['ordered'][h], small)
        slopes = [(r['ce_plus'][h] - r['ce_minus'][h]) / (2 * e)
                  for r, e in zip(case['branches'], case['epsilons'])]
        slope_scale = max(abs(slopes[1]), 1e-8)
        ce_gradient = vector(case['readout_gradients'][i], u.shape)
        gain = float(np.linalg.norm(small))
        comparisons = {}
        for name, values in predictors.items():
            value = vector(values[h], u.shape)
            pnorm = float(np.linalg.norm(value))
            dot = float(np.dot(value, small))
            comparisons[name] = dict(vector_error=relative(value, small), gain=pnorm,
                signed_alignment=None if pnorm == 0 or gain == 0 else dot / (pnorm * gain),
                training_ce_slope=float(np.dot(ce_gradient, value)))
        pred = comparisons['ordered']['training_ce_slope']
        rows.append(dict(horizon=h, response_gain=gain, finite_difference_relative_disagreement=agreement,
            transported_vector_error=ordered_error, resolved=bool(agreement <= .01 and ordered_error <= .01),
            training_ce_slopes=slopes, training_ce_slope_disagreement=abs(slopes[0]-slopes[1])/slope_scale,
            training_ce_slope_prediction_error=abs(pred-slopes[1])/slope_scale, comparisons=comparisons))
    for row in rows:
        row['mechanism_qualified'] = bool(row['resolved'] and rows[0]['resolved'])
        row['training_ce_slope_resolved'] = bool(row['training_ce_slope_disagreement'] <= .01 and
                                                 row['training_ce_slope_prediction_error'] <= .01)
    return rows


def descriptive_gate(cases):
    selected = [c for c in cases if c['rate_role'] == 'original']
    need(len(selected) == 2 and {c['seed'] for c in selected} == {28161, 28162}, 'two fixed original seeds')
    rows = [[r for r in c['rows'] if r['horizon'] == 16] for c in selected]
    need(all(len(r) == 1 for r in rows), 'one h16 per original')
    return bool(all(r[0]['mechanism_qualified'] and r[0]['comparisons']['mean']['vector_error'] < .2 and
                    10 * r[0]['comparisons']['mean']['vector_error'] < r[0]['comparisons']['frozen']['vector_error']
                    for r in rows))
