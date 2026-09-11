"""Prospective finite-probe action selection; this module never integrates ODEs.

Every requested direction is evaluated, including duplicates. The fixed score
is the existing full-activation RMSE with one global phase offset. Candidate
failures invalidate the affected selector instead of silently shrinking its
action set. Cached probes and costs remain available to the experiment driver.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from numbers import Real
import time

import numpy as np

from hopfield_drift_metrics import waveform_errors

PROBE_DURATION = 75.0
SCORE_NAME = "metrics.aligned_rms.aggregate"
COORDINATE_INDICES = (0, 4, 8, 12)
POLICY_INDICES = {"local_signed": (16, 17), "tangent_signed": (18, 19),
                  "coordinate_signed": COORDINATE_INDICES,
                  "search_all_21": tuple(range(21))}


def _unit_direction(value):
    direction = np.asarray(value, dtype=float)
    if (direction.shape != (2,) or not np.isfinite(direction).all() or
            not np.isclose(np.linalg.norm(direction), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Each candidate must be a finite native-unit direction of length two")
    return direction.copy()


def direction_candidates(policy, *, local_direction=None, tangent_direction=None):
    """Return candidates in fixed order without discarding duplicate directions.

    Uniform direction k has angle 2*pi*k/16. Signed sets always put the supplied
    positive direction first. Coordinate order is +alpha,+gamma,-alpha,-gamma,
    matching uniform indices 0,4,8,12.
    """
    if policy == "uniform_16":
        angles = np.arange(16) * (2 * np.pi / 16)
        vectors = np.column_stack((np.cos(angles), np.sin(angles)))
        vectors[::4] = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]
        labels = [f"uniform-{index:02d}" for index in range(16)]
    elif policy == "local_signed":
        value = _unit_direction(local_direction)
        vectors, labels = [value, -value], ["local_plus", "local_minus"]
    elif policy == "tangent_signed":
        value = _unit_direction(tangent_direction)
        vectors, labels = [value, -value], ["tangent_plus", "tangent_minus"]
    elif policy == "coordinate_signed":
        vectors = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]
        labels = ["alpha_plus", "gamma_plus", "alpha_minus", "gamma_minus"]
    else:
        raise ValueError(f"Unknown fixed candidate policy: {policy}")
    return [{"index": index, "label": label, "direction": _unit_direction(vector).tolist()}
            for index, (label, vector) in enumerate(zip(labels, vectors))]


def experiment_candidates(local_direction, tangent_direction, atlas_direction):
    """EXP-007 order: uniform16, local+/-, tangent+/-, atlas (21 total)."""
    result = direction_candidates("uniform_16")
    result += direction_candidates("local_signed", local_direction=local_direction)
    result += direction_candidates("tangent_signed", tangent_direction=tangent_direction)
    result += [{"index": 20, "label": "atlas_margin", "direction": _unit_direction(atlas_direction).tolist()}]
    return [{**candidate, "index": index} for index, candidate in enumerate(result)]


def score_probe(reference_activations, probe_times, probe_activations, period,
                *, phase_epoch=0.0, duration=PROBE_DURATION, phase_grid=128):
    """Score the entire prescribed probe, never a selected sub-window.

    Times are elapsed probe time starting at zero and ending at 75. The epoch
    is the original baseline orbit phase in physical time. No frequency is
    fitted. Primary held-out outcomes after time 75 belong to the driver and
    are never available to this scoring helper.
    """
    times = np.asarray(probe_times, dtype=float)
    if duration != PROBE_DURATION:
        raise ValueError("The prospectively fixed probe duration is 75")
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            abs(times[0]) > 1e-12 or abs(times[-1] - duration) > 1e-12 or
            np.any(times < 0) or np.any(times > duration) or
            not np.isfinite(phase_epoch)):
        raise ValueError("Probe must contain the full elapsed interval from zero through 75")
    return waveform_errors(reference_activations, times + float(phase_epoch),
                           probe_activations, period, phase_grid=phase_grid)


def _score(metrics):
    try:
        value = metrics["aligned_rms"]["aggregate"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing fixed score {SCORE_NAME}") from error
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or
            not np.isfinite(value) or not 0 <= value <= 2):
        raise ValueError("Aligned full-activation RMSE must be finite and lie in [0,2]")
    return float(value)


def _candidate_list(candidates):
    if len(candidates) == 0:
        raise ValueError("At least one declared candidate is required")
    result = []
    for index, item in enumerate(candidates):
        if isinstance(item, Mapping):
            supplied_index = item.get("index", index)
            if isinstance(supplied_index, bool) or supplied_index != index:
                raise ValueError("Candidate indices must match their declared zero-based order")
            label, vector = str(item.get("label", f"candidate-{index:02d}")), item["direction"]
        else:
            label, vector = f"candidate-{index:02d}", item
        result.append({"index": index, "label": label, "direction": _unit_direction(vector).tolist()})
    return result


def select_from_records(records, indices):
    """Select within a prospectively fixed subset of an already evaluated pool.

    No new evaluations occur here. A failure elsewhere in a larger pool does
    not affect a policy whose declared subset does not use that candidate.
    Subsets must never be chosen after seeing failures or held-out outcomes.
    """
    indices = tuple(indices)
    if (not indices or len(set(indices)) != len(indices) or
            any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or index < 0 for index in indices)):
        raise ValueError("A nonempty, unique set of declared candidate indices is required")
    indexed = {}
    for record in records:
        index = record["index"]
        if index in indexed:
            raise ValueError("Duplicate candidate record index")
        indexed[index] = record
    missing = [index for index in indices if index not in indexed]
    invalid = []
    eligible = []
    for index in indices:
        if index not in indexed:
            continue
        record = indexed[index]
        try:
            if record.get("status") != "complete":
                raise ValueError("Candidate evaluation did not complete")
            value = _score(record["metrics"])
            if value != record.get("score"):
                raise ValueError("Recorded candidate score differs from its metric")
            _unit_direction(record["direction"])
        except (ValueError, KeyError, TypeError):
            invalid.append(index)
        else:
            eligible.append((value, index))
    valid = not missing and not invalid
    selected = min(eligible, key=lambda item: (item[0], item[1])) if valid else None
    return {"valid": valid, "candidate_indices": list(indices),
            "selected_index": None if selected is None else selected[1],
            "selected_score": None if selected is None else selected[0],
            "selected_direction": None if selected is None else list(indexed[selected[1]]["direction"]),
            "missing_indices": missing, "invalid_indices": invalid,
            "required_probe_calls": len(indices), "new_probe_calls": 0,
            "score_definition": SCORE_NAME,
            "tie_rule": "Minimum score; exact score ties choose the lowest candidate index",
            "reason": "passed" if valid else "A required candidate is missing or failed; no alternative is selected"}


def select_with_probe(evaluator, candidates, radius, *, duration=PROBE_DURATION,
                      on_candidate=None):
    """Evaluate every candidate and select the fixed-score minimum.

    `evaluator(direction, *, radius, duration, index)` must return a mapping with
    `metrics`, optional `cost`, and optional `cache`. The caller owns integration
    and persistence. `cache` can contain exact NumPy probe arrays and is returned
    separately from compact decision records in `cached_probes`.

    Exceptions are retained as failed candidate records; an exception's `payload`
    attribute is preserved as its cached probe. Remaining candidates are still
    attempted. `on_candidate(record, cache)` can persist each result immediately.
    Recorded wall time includes the evaluator's integration and scoring. The
    function makes no equal-total-compute or control-proof claim.
    """
    candidates = _candidate_list(candidates)
    if isinstance(radius, bool) or not isinstance(radius, Real) or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Probe radius must be positive and finite")
    if duration != PROBE_DURATION:
        raise ValueError("The prospectively fixed probe duration is 75")
    records, cached = [], {}
    for candidate in candidates:
        index = candidate["index"]
        record = {**candidate, "radius": float(radius), "duration": float(duration),
                  "weight_delta": (radius * np.asarray(candidate["direction"])).tolist(),
                  "status": "failed", "score": None}
        started = time.perf_counter()
        try:
            evaluated = evaluator(np.asarray(candidate["direction"]).copy(), radius=float(radius),
                                  duration=float(duration), index=index)
            if not isinstance(evaluated, Mapping):
                raise ValueError("Probe evaluator must return metrics, optional cost and optional cache")
            # Retain returned evidence before validating or interpreting its score.
            record["metrics"] = copy.deepcopy(evaluated.get("metrics"))
            record["cost"] = copy.deepcopy(evaluated.get("cost", {}))
            cached[index] = copy.deepcopy(evaluated.get("cache"))
            if not isinstance(record["cost"], Mapping):
                raise ValueError("Evaluator cost must be a mapping")
            record["score"] = _score(record["metrics"])
            record["status"] = "complete"
        except Exception as error:
            record["error"] = repr(error)
            if hasattr(error, "payload"):
                cached[index] = copy.deepcopy(error.payload)
            elif index not in cached:
                cached[index] = None
        record["evaluation_wall_seconds"] = time.perf_counter() - started
        records.append(record)
        if on_candidate:
            on_candidate(record, cached[index])
    decision = select_from_records(records, range(len(candidates)))
    decision["new_probe_calls"] = len(candidates)
    return {**decision, "radius": float(radius), "duration": float(duration),
            "candidate_records": records, "cached_probes": cached,
            "evaluation_wall_seconds": float(sum(row["evaluation_wall_seconds"] for row in records)),
            "candidate_count": len(candidates), "failed_candidate_count": sum(row["status"] != "complete" for row in records),
            "scope": "Finite-probe selector ablation. Probe cost is retained; no equal-total-cost superiority or control proof is inferred."}
