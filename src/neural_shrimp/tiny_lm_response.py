"""Fixed-horizon perturbation response: frozen versus transported derivatives.

Transported response is retrospective chain-rule validation, never a forecast.
The two-amplitude finite-difference gate precedes interpretation of frozen error.
"""
import math
import numpy as np
HORIZONS = (1, 4, 16)

def finite(v):
    a = np.asarray(v, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('finite flat state required')
    return a

def relative(a, b):
    return float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(b)), 1e-12)

def profile_forecast(rows):
    """Price all four fixed cases from two measured operation costs per arm."""
    expected = dict(gradient=145, hvp=64, ce=24)
    if len(rows) != 2 or sorted(row['arm'] for row in rows) != [8, 256]:
        raise ValueError('both fixed training arms required')
    total = 0.0
    for row in rows:
        if set(row['times']) != set(expected):
            raise ValueError('complete operation profile required')
        for name, count in expected.items():
            values = row['times'][name]
            if len(values) != 2 or not all(type(v) is float and math.isfinite(v) and v > 0 for v in values):
                raise ValueError('two finite positive measured costs required')
            total += count * max(values)
    return 4 * total + 20


def measurements(reference, tangents, branches, readout_gradients, epsilons):
    out = []
    for direction in range(2):
        for h in HORIZONS:
            rows = [branches[direction, k] for k in range(2)]
            responses = [(row['plus'][h] - row['minus'][h]) / (2 * epsilons[k]) for k, row in enumerate(rows)]
            small = finite(responses[1])
            agreement = relative(responses[0], small)
            transported = tangents[direction]['transported'][h]
            frozen = tangents[direction]['frozen'][h]
            u = tangents[direction]['transported'][0]
            terr = relative(transported, small)
            ferr = relative(frozen, small)
            gain = float(np.linalg.norm(small))
            scalar = float(np.linalg.norm(tangents[direction]['frozen'][1])) ** h
            slopes = [(row['ce_plus'][h] - row['ce_minus'][h]) / (2 * epsilons[k]) for k, row in enumerate(rows)]
            pred = float(np.dot(readout_gradients[h], transported))
            scale = max(abs(slopes[1]), 1e-08)
            row = dict(direction='gradient' if direction == 0 else 'random', horizon=h, response_gain=gain, finite_difference_relative_disagreement=agreement, transported_vector_error=terr, frozen_vector_error=ferr, persistence_vector_error=relative(u, small), scalar_gain=scalar, scalar_gain_relative_error=abs(scalar - gain) / max(gain, 1e-12), resolved=bool(agreement <= 0.01 and terr <= 0.01), training_ce_slopes=slopes, transported_training_ce_slope=pred, training_ce_slope_disagreement=abs(slopes[0] - slopes[1]) / scale, training_ce_slope_prediction_error=abs(pred - slopes[1]) / scale)
            if not all((math.isfinite(v) for v in row.values() if type(v) is float)):
                raise ValueError('nonfinite response metric')
            row['training_ce_slope_resolved'] = bool(row['training_ce_slope_disagreement'] <= 0.01 and row['training_ce_slope_prediction_error'] <= 0.01)
            out.append(row)
    first = {r['direction']: r['resolved'] for r in out if r['horizon'] == 1}
    for row in out:
        row['mechanism_qualified'] = bool(row['resolved'] and first[row['direction']])
    return out

def run_case(theta, eta, gradient, hvp, ce, ridge, random_direction, save, check):
    theta = finite(theta)
    random_direction = finite(random_direction)
    if theta.shape != random_direction.shape or not math.isfinite(eta) or eta <= 0 or (not math.isfinite(ridge)) or (ridge < 0):
        raise ValueError('valid map and direction')
    reference = [theta.copy()]
    gradients = []
    try:
        for h in range(17):
            check()
            g = np.asarray(gradient(reference[-1]), dtype=np.float64)
            gradients.append(g)
            finite(g)
            if h < 16:
                reference.append(reference[-1] - eta * g)
                finite(reference[-1])
    finally:
        save('reference', np.asarray(reference))
        save('reference-gradients', np.asarray(gradients))
    directions = []
    for v in (gradients[0], random_direction):
        size = float(np.linalg.norm(v))
        if not math.isfinite(size) or size <= 0:
            raise ValueError('nonzero prescribed direction')
        directions.append(v / size)
    save('directions', np.asarray(directions))
    epsilons = [e * max(1.0, float(np.linalg.norm(theta))) for e in (0.0001, 5e-05)]
    tangents = []
    for i, u in enumerate(directions):
        f = [u.copy()]
        t = [u.copy()]
        try:
            for h in range(16):
                check()
                f.append(finite(f[-1] - eta * finite(hvp(reference[0], f[-1]))))
                t.append(finite(t[-1] - eta * finite(hvp(reference[h], t[-1]))))
        finally:
            save(f'direction-{i}-frozen', np.asarray(f))
            save(f'direction-{i}-transported', np.asarray(t))
        tangents.append(dict(frozen=f, transported=t))
    branches = {}
    for i, u in enumerate(directions):
        for k, eps in enumerate(epsilons):
            row = {}
            for sign, label in [(1, 'plus'), (-1, 'minus')]:
                states = [finite(theta + sign * eps * u)]
                values = {}
                try:
                    for h in range(1, 17):
                        check()
                        states.append(states[-1] - eta * finite(gradient(states[-1])))
                        finite(states[-1])
                        if h in HORIZONS:
                            values[h] = float(ce(states[-1]))
                            if not math.isfinite(values[h]):
                                raise ValueError('nonfinite training CE')
                finally:
                    save(f'direction-{i}-epsilon-{k}-{label}', np.asarray(states))
                    save(f'direction-{i}-epsilon-{k}-{label}-ce', np.asarray(sorted(values.items()), dtype=np.float64).reshape(-1, 2))
                row[label] = states
                row['ce_' + label] = values
            branches[i, k] = row
    readout_gradients = {h: gradients[h] - ridge * reference[h] for h in HORIZONS}
    save('readout-gradients', np.asarray([readout_gradients[h] for h in HORIZONS]))
    rows = measurements(reference, tangents, branches, readout_gradients, epsilons)
    return dict(rows=rows, epsilon_values=epsilons, branch_training_ce={f'{i}:{k}': {key: value for key, value in row.items() if key.startswith('ce_')} for (i, k), row in branches.items()}, scope='Frozen-curvature approximation test on a float64 counterfactual trajectory; transported derivative is retrospective; no utility or chaos claim.')
