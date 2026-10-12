"""Independent controlled passages with fixed saved entrances.

The caller supplies a model, source-bound saved vectors and optional consumed
reference trajectories. No pretrained asset is loaded here. All output labels
describe floating-point cross-checks, not certificates or fresh holdouts.
"""
from __future__ import annotations

import math
import numpy as np
import torch


REFERENCE_EPSILONS = (5e-5, 5e-4, 3e-6, 1e-5, 3e-5, 1e-4)
ENTRANCES = ("full", "midpoint", "full+transverse")


def _array(value, n=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (n is not None and result.size != n) or not np.isfinite(result).all():
        raise ValueError("finite full-state one-dimensional vector required")
    return result.copy()


def quadratic_prediction(u_in, a, b, epsilon, u_out, max_iter=100000):
    """First strict exit for the positive-oriented quadratic G-map and integral."""
    if not all(math.isfinite(v) for v in [u_in, a, b, epsilon, u_out]):
        raise ValueError("finite quadratic parameters required")
    if min(a, b, epsilon) <= 0 or u_in > u_out or type(max_iter) is not int or max_iter < 1:
        raise ValueError("positive drift, legal entrance and positive iteration cap required")
    value, count = float(u_in), None
    for iteration in range(1, max_iter + 1):
        value = value + a * epsilon + b * value * value
        if value > u_out:
            count = iteration
            break
    scale = math.sqrt(b / (a * epsilon))
    integral = (math.atan(u_out * scale) - math.atan(u_in * scale)) / math.sqrt(a * b * epsilon)
    return dict(discrete_G=count, arctan_G=integral)


def fixed_entrances(x_c, right, left, U, a, b):
    """Preserve supplied vectors, orient signs, and reproduce transverse RNG0."""
    x_c = _array(x_c)
    right, left = _array(right, x_c.size), _array(left, x_c.size)
    if not all(math.isfinite(v) for v in [U, a, b]) or U <= 0 or a * b <= 0:
        raise ValueError("fixed positive channel width and same-sign nonzero a,b required")
    if abs(np.linalg.norm(right) - 1) > 1e-8 or abs(left @ right - 1) > 1e-8:
        raise ValueError("saved modes must already satisfy norm(r)=1 and l.r=1")
    sign = 1. if a > 0 else -1.
    right, left = sign * right, sign * left
    random = np.random.default_rng(0).normal(size=x_c.size)
    random -= (left @ random) / (left @ right) * right
    norm = np.linalg.norm(random)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("transverse direction is unresolved")
    random /= norm
    starts = np.stack((x_c - U * right, x_c, x_c - U * right + U * random))
    return dict(x_c=x_c, right=right, left=left, transverse=random, states=starts,
                orientation=sign, a=abs(float(a)), b=abs(float(b)), U=float(U))


def compare_reference(u, rows, reference_u, U, reference_rows=None):
    """Compare only common pre-crossing coordinates; never chaotic late tails."""
    current, reference = np.asarray(u), np.asarray(reference_u, dtype=np.float64)
    if current.ndim != 2 or reference.ndim != 2 or reference.shape[1] != len(rows) or current.shape[1] != len(rows):
        raise ValueError("reference/current trajectories must have the same ordered columns")
    if not np.isfinite(reference).all():
        raise ValueError("reference coordinate matrix must be finite")
    if reference_rows is not None and len(reference_rows) != len(rows):
        raise ValueError("reference metadata must include every ordered entrance")
    comparisons = []
    for column, row in enumerate(rows):
        crossings = np.flatnonzero(reference[:, column] > U)
        old_count = int(crossings[0] + 1) if len(crossings) else None
        if reference_rows is not None:
            old = reference_rows[column]
            if old["eps"] != row["eps"] or old["entrance"] != row["entrance"] or old["measured_G"] != old_count:
                raise ValueError("reference metadata/first-strict-exit identity mismatch")
        new_count = row["measured_G"]
        old_pre = len(reference) if old_count is None else old_count - 1
        new_pre = row["recorded_G"] if new_count is None else new_count - 1
        shared = min(old_pre, new_pre)
        delta = current[:shared, column] - reference[:shared, column]
        if not np.isfinite(delta).all():
            raise ValueError("nonfinite coordinate before the first exit")
        maximum = float(np.max(np.abs(delta))) if shared else None
        rms = float(np.sqrt(np.mean(delta * delta))) if shared else None
        comparisons.append(dict(column=column, eps=row["eps"], entrance=row["entrance"],
                                reference_G=old_count, measured_G=new_count,
                                count_difference=None if old_count is None or new_count is None else new_count - old_count,
                                shared_pre_exit_G=shared, maximum_absolute_u_error=maximum,
                                rms_u_error=rms, maximum_error_over_U=None if maximum is None else maximum / U))
    return comparisons


def run_controlled_passages(model, x_c, right, left, U, alpha_c, log_temperature, a, b,
                           *, epsilons=REFERENCE_EPSILONS, max_g=512,
                           reference_u=None, reference_rows=None, tick=None, on_group=None):
    """Run six alpha groups of three fixed entrances, with no transient removal.

    ``model`` follows ManualQwenSoftMap's prefix_cache/period_map interface.
    One G evaluation is exactly two model steps. Every group retains each
    column through its first STRICT u>U crossing, including that crossing;
    later coordinates are NaN. The supplied callback can save each completed
    group immediately. Missing exits remain censored, never default to max_g.
    ``tick`` runs before each group and G evaluation for caller-owned budgets.
    """
    if type(max_g) is not int or max_g < 1 or not all(math.isfinite(v) for v in [alpha_c, log_temperature]):
        raise ValueError("finite map parameters and positive integer max_g required")
    epsilons = tuple(float(eps) for eps in epsilons)
    if not epsilons or not all(math.isfinite(eps) and eps > 0 for eps in epsilons):
        raise ValueError("all epsilon groups must be positive and finite")
    fixed = fixed_entrances(x_c, right, left, U, a, b)
    center = torch.tensor(fixed["x_c"], dtype=torch.float64, device="cpu")
    lt = torch.tensor(fixed["left"], dtype=torch.float64, device="cpu")
    rt = torch.tensor(fixed["right"], dtype=torch.float64, device="cpu")
    tick = tick or (lambda: None)
    groups, rows = [], []
    with torch.no_grad():
        for group_index, epsilon in enumerate(epsilons):
            tick()
            alpha = float(alpha_c + epsilon)
            cache = model.prefix_cache(alpha)
            states = torch.tensor(fixed["states"], dtype=torch.float64, device="cpu")
            counts, us, transverse = [None] * 3, [], []
            for g in range(1, max_g + 1):
                tick()
                states = model.period_map(states, alpha, log_temperature, 2, cache=cache)
                if states.shape != fixed["states"].shape or states.dtype != torch.float64 or states.device.type != "cpu" or not torch.isfinite(states).all():
                    raise ValueError("map returned invalid full-state passage batch")
                difference = states - center
                coordinates = difference @ lt
                distance = torch.linalg.vector_norm(difference - coordinates[:, None] * rt, dim=1)
                coordinate_row, transverse_row = np.full(3, np.nan), np.full(3, np.nan)
                for column in range(3):
                    if counts[column] is None:
                        coordinate_row[column] = float(coordinates[column])
                        transverse_row[column] = float(distance[column])
                        if coordinate_row[column] > U:
                            counts[column] = g
                us.append(coordinate_row)
                transverse.append(transverse_row)
                if all(count is not None for count in counts):
                    break
            group_rows = []
            for column, entrance in enumerate(ENTRANCES):
                u_in = 0. if entrance == "midpoint" else -float(U)
                prediction = quadratic_prediction(u_in, fixed["a"], fixed["b"], epsilon, float(U))
                count, predicted = counts[column], prediction["discrete_G"]
                group_rows.append(dict(column=3 * group_index + column, eps=epsilon, entrance=entrance,
                                       u_in=u_in, u_out=float(U), measured_G=count,
                                       recorded_G=len(us) if count is None else count,
                                       censored=count is None, predicted_discrete_G=predicted,
                                       predicted_arctan_G=prediction["arctan_G"],
                                       relative_count_error=None if count is None or predicted is None else (count - predicted) / predicted))
            group = dict(index=group_index, eps=epsilon, alpha=alpha, rows=group_rows,
                         G_evaluations=len(us), batch_model_steps=2 * len(us), batch_size=3,
                         u=np.asarray(us), transverse=np.asarray(transverse),
                         states_at_group_stop=states.detach().cpu().numpy().copy())
            groups.append(group)
            rows.extend(group_rows)
            if on_group is not None:
                on_group(group_index, group)
    height = max(group["G_evaluations"] for group in groups)
    combined_u, combined_transverse = np.full((height, len(rows)), np.nan), np.full((height, len(rows)), np.nan)
    for j, group in enumerate(groups):
        n = group["G_evaluations"]
        combined_u[:n, 3*j:3*j+3] = group["u"]
        combined_transverse[:n, 3*j:3*j+3] = group["transverse"]
    compared = None if reference_u is None else compare_reference(combined_u, rows, reference_u, U, reference_rows)
    return dict(scope="Fixed saved entrances, consumed six-alpha development cross-check; no new holdout or global claim.",
                rows=rows, groups=groups, u=combined_u, transverse=combined_transverse,
                reference_comparison=compared, fixed_inputs=fixed,
                all_exits_observed=all(not row["censored"] for row in rows),
                batch_model_steps=sum(group["batch_model_steps"] for group in groups),
                single_state_equivalent_steps=3 * sum(group["batch_model_steps"] for group in groups),
                prefix_evaluations=len(groups), period=2, max_g=max_g)
