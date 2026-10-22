"""Bounded searches for resolved minima on one declared finite section.

The caller supplies already-qualified objective evaluations and owns their raw
artifacts. Only strict *interior* mesh brackets are searched. Refinement uses
golden-section steps, with no polynomial fit, derivative or trace-zero test.
Success means a narrow numerical bracket and resolved finite-offset comparisons;
it is neither a proof of a mathematical local minimum nor evidence of ridge
topology. Unsampled extrema and flat-bottomed minima can be missed by the mesh.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import math
from numbers import Integral


SCOPE = (
    "Numerical section minima supported by declared mesh brackets and finite "
    "offsets only; no derivative, exact-minimum, trace-zero or ridge-topology claim."
)
_GOLDEN_STEP = (3.0 - math.sqrt(5.0)) / 2.0


class _BudgetExhausted(Exception):
    pass


class _EvaluationFailed(Exception):
    pass


def _finite_float(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def search_section_minima(
    evaluate: Callable[[float], Mapping],
    mesh: Sequence[float],
    *,
    coordinate_tolerance: float,
    max_evaluations: int,
    radius_floor: float,
    signal_multiple: float,
    validation_offsets: Sequence[float],
    max_refinement_steps: int = 128,
    record: Callable[[dict], None] | None = None,
):
    """Search every strict local-minimum bracket on ``mesh``.

    ``evaluate(s)`` must return a mapping containing a finite nonnegative
    ``radius`` and a non-null integer or string ``index`` identifying the actual
    evaluation. Optional ``qualified`` or ``passed`` fields must be true. Its
    other fields are left to the caller's raw evidence. Coordinates are cached
    exactly, without rounding; repeat requests reuse the original evaluation.

    The mesh must be finite and strictly increasing, with at least three
    points. A strict bracket requires the interior value to be smaller than
    both neighbors. Every such bracket is retained, including low-signal ones.
    The signal gate is an *absolute contrast* greater than
    ``radius_floor * signal_multiple``. A weak mesh bracket is reported without
    further evaluations. For other brackets, fit-free refinement stays inside
    the original bracket. The final bracket width must meet the coordinate
    tolerance, then the actual objective is compared at symmetric offsets at
    exactly two distinct declared scales. Both scales must be larger than the
    coordinate tolerance, and their points must stay in the full declared
    section support. They may extend beyond the original refinement bracket.
    All four contrasts must exceed the signal gate to qualify the candidate.

    ``max_evaluations`` bounds actual callback attempts, including failed ones;
    cached requests are free. ``max_refinement_steps`` bounds each refinement.
    ``record`` receives progress events, including every attempted coordinate,
    before the next model call. Recording failures propagate rather than being
    disguised as objective failures. Ordinary objective exceptions instead
    produce an ``evaluation_failed`` result with all partial evidence retained.

    Top-level ``status`` distinguishes ``no_bracket``, ``converged``,
    ``low_signal``, ``nonconverged``, ``budget_exhausted`` and
    ``evaluation_failed``. Always inspect every candidate: mixed outcomes are
    retained individually. ``minima`` contains only qualified candidates;
    ``candidates`` and ``mesh_brackets`` also retain unresolved attempts.
    """
    if not callable(evaluate) or (record is not None and not callable(record)):
        raise ValueError("evaluate and optional record must be callable")
    coordinates = [_finite_float(s, "mesh coordinate") for s in mesh]
    offsets = [_finite_float(h, "validation offset") for h in validation_offsets]
    tolerance = _finite_float(coordinate_tolerance, "coordinate tolerance")
    floor = _finite_float(radius_floor, "radius floor")
    multiple = _finite_float(signal_multiple, "signal multiple")
    if len(coordinates) < 3 or any(a >= b for a, b in zip(coordinates, coordinates[1:])):
        raise ValueError("at least three strictly increasing finite mesh coordinates required")
    if not math.isfinite(coordinates[-1]-coordinates[0]):
        raise ValueError("mesh span must be finite in the working precision")
    if tolerance <= 0 or floor < 0 or multiple < 1:
        raise ValueError("positive tolerance, nonnegative radius floor and signal multiple >= 1 required")
    if len(offsets) != 2 or len(set(offsets)) != 2 or min(offsets) <= tolerance:
        raise ValueError("exactly two distinct positive offsets larger than coordinate tolerance required")
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral)
            or max_evaluations < 1 or isinstance(max_refinement_steps, bool)
            or not isinstance(max_refinement_steps, Integral) or max_refinement_steps < 1):
        raise ValueError("positive integer evaluation and refinement budgets required")
    threshold = floor * multiple
    if not math.isfinite(threshold):
        raise ValueError("signal threshold must be finite")
    offsets.sort(reverse=True)
    result = {
        "status": "running", "complete": False, "scope": SCOPE,
        "settings": {
            "mesh": coordinates, "support": [coordinates[0], coordinates[-1]],
            "coordinate_tolerance": tolerance, "max_evaluations": int(max_evaluations),
            "max_refinement_steps": int(max_refinement_steps), "radius_floor": floor,
            "signal_multiple": multiple, "signal_threshold": threshold,
            "validation_offsets": offsets,
        },
        "evaluations": [], "cache_hits": 0, "mesh_samples": [],
        "mesh_complete": False, "mesh_brackets": [], "candidates": [], "minima": [],
    }
    cache = {}

    def emit(event):
        if record is not None:
            record(event)

    def sample(s):
        s = float(s)
        if s in cache:
            result["cache_hits"] += 1
            return cache[s]
        if len(result["evaluations"]) >= max_evaluations:
            raise _BudgetExhausted
        attempt = {"s": s, "status": "started", "radius": None, "index": None}
        result["evaluations"].append(attempt)
        emit({"event": "evaluation_started", "attempt": len(result["evaluations"]), **attempt})
        try:
            raw = evaluate(s)
            if not isinstance(raw, Mapping) or "radius" not in raw or "index" not in raw:
                raise ValueError("objective must return a mapping containing radius and index")
            for key in ("qualified", "passed"):
                if key in raw and not bool(raw[key]):
                    raise ValueError(f"objective {key} gate failed")
            radius = _finite_float(raw["radius"], "objective radius")
            if radius < 0:
                raise ValueError("objective radius must be nonnegative")
            index = raw["index"]
            if isinstance(index, bool) or not isinstance(index, (Integral, str)):
                raise ValueError("objective index must be an integer or string")
            if isinstance(index, Integral):
                index = int(index)
        except Exception as error:
            attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
            emit({"event": "evaluation_failed", **attempt})
            raise _EvaluationFailed from error
        attempt.update(status="complete", radius=radius, index=index)
        emit({"event": "evaluation_complete", **attempt})
        cache[s] = {"s": s, "radius": radius, "index": index}
        return cache[s]

    interruption = None
    try:
        for s in coordinates:
            result["mesh_samples"].append(sample(s))
        result["mesh_complete"] = True
    except (_BudgetExhausted, _EvaluationFailed) as error:
        interruption = "budget_exhausted" if isinstance(error, _BudgetExhausted) else "evaluation_failed"

    # Retain every observed strict bracket even when later mesh evaluation fails.
    samples = result["mesh_samples"]
    for i in range(1, len(samples) - 1):
        left, center, right = samples[i-1:i+2]
        if center["radius"] < left["radius"] and center["radius"] < right["radius"]:
            bracket = {
                "mesh_center_position": i, "left": left, "center": center, "right": right,
                "minimum_mesh_contrast": min(left["radius"], right["radius"]) - center["radius"],
            }
            result["mesh_brackets"].append(bracket)
            result["candidates"].append({
                "bracket": bracket, "status": "pending", "qualified": False,
                "best_sample": center, "refinement": {"status": "not_started", "steps": 0},
                "validation": [],
            })
    emit({"event": "mesh_finished", "complete": result["mesh_complete"],
          "strict_brackets": len(result["mesh_brackets"]), "interruption": interruption})

    for candidate in result["candidates"]:
        if interruption:
            candidate.update(status=interruption, reason="remaining search not evaluated")
            continue
        bracket = candidate["bracket"]
        if bracket["minimum_mesh_contrast"] <= threshold:
            candidate.update(status="low_signal", reason="mesh contrast does not exceed declared signal threshold")
            candidate["refinement"]["status"] = "not_started_low_signal_mesh"
            emit({"event": "candidate_finished", "candidate": candidate})
            continue
        a, x, b = bracket["left"], bracket["center"], bracket["right"]
        refinement = candidate["refinement"]

        def retain_bracket():
            refinement.update(left=a, center=x, right=b, width=b["s"]-a["s"])
            candidate["best_sample"] = x

        retain_bracket()
        try:
            for step in range(max_refinement_steps):
                if refinement["width"] <= tolerance:
                    break
                # Choose the larger side. Only actual sampled objective values
                # decide how to shrink the bracket; no fit smooths away a cusp.
                boundary = b if b["s"]-x["s"] >= x["s"]-a["s"] else a
                s = x["s"] + _GOLDEN_STEP * (boundary["s"]-x["s"])
                if not a["s"] < s < b["s"] or s == x["s"]:
                    refinement.update(status="nonconverged", reason="floating-point coordinate stagnation")
                    break
                trial = sample(s)
                if s < x["s"]:
                    if trial["radius"] < x["radius"]:
                        b, x = x, trial
                    else:
                        a = trial
                elif trial["radius"] < x["radius"]:
                    a, x = x, trial
                else:
                    b = trial
                refinement["steps"] = step + 1
                retain_bracket()
                emit({"event": "refinement_step", "bracket_position": bracket["mesh_center_position"],
                      "step": step + 1, "width": refinement["width"], "best_sample": x})
            if refinement["width"] > tolerance:
                refinement.update(status="nonconverged")
                candidate.update(status="nonconverged", reason=refinement.get("reason", "refinement step budget exhausted"))
            else:
                refinement["status"] = "converged"
                if any(x["s"]-h < coordinates[0] or x["s"]+h > coordinates[-1] for h in offsets):
                    candidate.update(status="nonconverged", reason="declared validation offsets leave section support")
                else:
                    for h in offsets:
                        row = {"offset": h, "status": "started", "samples": [], "contrasts": []}
                        candidate["validation"].append(row)
                        for sign in (-1, 1):
                            point = sample(x["s"] + sign*h)
                            row["samples"].append(point)
                            row["contrasts"].append(point["radius"]-x["radius"])
                        row.update(status="complete", resolved=all(d > threshold for d in row["contrasts"]),
                                   lower_neighbor_resolved=any(d < -threshold for d in row["contrasts"]))
                    if all(row["resolved"] for row in candidate["validation"]):
                        candidate.update(status="converged", qualified=True)
                        result["minima"].append(candidate)
                    elif any(row["lower_neighbor_resolved"] for row in candidate["validation"]):
                        candidate.update(status="nonconverged", reason="finite-offset check resolves a lower neighboring objective")
                    else:
                        candidate.update(status="low_signal", reason="finite-offset contrasts do not exceed declared signal threshold")
        except (_BudgetExhausted, _EvaluationFailed) as error:
            interruption = "budget_exhausted" if isinstance(error, _BudgetExhausted) else "evaluation_failed"
            candidate.update(status=interruption, reason="search interrupted; partial evidence retained")
            if refinement["status"] == "not_started":
                refinement["status"] = interruption
        emit({"event": "candidate_finished", "candidate": candidate})

    statuses = Counter(c["status"] for c in result["candidates"])
    result["candidate_status_counts"] = dict(statuses)
    result["evaluation_count"] = len(result["evaluations"])
    if interruption:
        result["status"] = interruption
    elif not result["candidates"]:
        result["status"] = "no_bracket"
    elif statuses["nonconverged"]:
        result["status"] = "nonconverged"
    elif statuses["low_signal"]:
        result["status"] = "low_signal"
    else:
        result["status"] = "converged"
    result["complete"] = result["mesh_complete"] and result["status"] in {"converged", "low_signal", "no_bracket"}
    emit({"event": "section_finished", "status": result["status"], "complete": result["complete"],
          "evaluation_count": result["evaluation_count"], "qualified_minima": len(result["minima"])})
    return result
