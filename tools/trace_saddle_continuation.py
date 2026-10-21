"""Continue a numerically qualified trace saddle in a third parameter.

The callback owns the full orbit/spectral qualification and durable raw evidence.
This module only handles two-dimensional finite-difference stationary points.
Every attempted evaluation and iteration is recorded, including failures. A
height root is always evaluated at a corrected saddle, never at a fixed slice.
Coordinates should be dimensionless, with scales declared by the caller.
"""
from __future__ import annotations

import math
import json
from typing import Callable

import numpy as np


class ContinuationError(RuntimeError):
    """A failed numerical gate; ``record`` contains the recorded failure."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


class TraceSaddleContinuation:
    """Bounded predictor/corrector with an optional durable event callback.

    ``evaluate(center, parameter)`` returns a finite scalar or a dictionary with
    a finite ``tau``. The callback must fail if any orbit or spectral gate fails.
    If it supplies ``qualified`` or ``passed``, these must be true. ``record``
    receives each JSON-friendly event before the next operation. ``events`` is
    also retained on this object if a callback or numerical gate raises.

    Convergence means the Newton correction is below ``correction_tolerance``
    sequentially at every declared stencil scale. A finer scale may move the
    center after a coarser scale passes; only the final scale is guaranteed at
    the returned center. Saved centers permit a separate cross-scale audit.
    It is a numerical
    criterion, not an interval certificate of a stationary point.
    """

    def __init__(self, evaluate: Callable, *, stencils=(0.1, 0.03),
                 correction_tolerance=0.003, max_iterations=6,
                 max_evaluations=1000, trust_radius=2.0,
                 condition_limit=1e8, coordinate_bounds=None,
                 max_continuation_points=100, record=None):
        self.evaluate = evaluate
        self.stencils = tuple(float(h) for h in stencils)
        self.correction_tolerance = float(correction_tolerance)
        self.max_iterations = int(max_iterations)
        self.max_evaluations = int(max_evaluations)
        self.trust_radius = float(trust_radius)
        self.condition_limit = float(condition_limit)
        self.max_continuation_points = int(max_continuation_points)
        self.record = record
        self.events = []
        self.evaluations = []
        self.results = []
        self._cache = {}
        self.bounds = None if coordinate_bounds is None else np.asarray(coordinate_bounds, float)
        if (not self.stencils or not all(math.isfinite(h) and h > 0 for h in self.stencils)
                or any(a <= b for a, b in zip(self.stencils, self.stencils[1:]))):
            raise ValueError("positive strictly decreasing stencil scales required")
        if (not math.isfinite(self.correction_tolerance) or self.correction_tolerance <= 0
                or not math.isfinite(self.trust_radius) or self.trust_radius <= 0
                or not math.isfinite(self.condition_limit) or self.condition_limit <= 1
                or self.max_iterations < 1 or self.max_evaluations < 1
                or self.max_continuation_points < 1):
            raise ValueError("finite positive tolerances and bounded positive iteration counts required")
        if self.bounds is not None and (self.bounds.shape != (2, 2)
                or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError("coordinate_bounds must give increasing [low, high] for each coordinate")

    def _emit(self, stage, **fields):
        event = _plain({"stage": stage, **fields})
        self.events.append(event)
        if self.record is not None:
            self.record(event)
        return event

    def _fail(self, message, **fields):
        event = self._emit("failed", error=message, **fields)
        raise ContinuationError(message, event)

    def _center(self, center):
        x = np.asarray(center, dtype=float)
        if x.shape != (2,) or not np.isfinite(x).all():
            self._fail("finite two-coordinate center required")
        if self.bounds is not None and (np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1])):
            self._fail("evaluation left declared coordinate bounds", center=x)
        return x.copy()

    def _point(self, center, parameter):
        x = self._center(center)
        parameter = float(parameter)
        if not math.isfinite(parameter):
            self._fail("finite third parameter required")
        key = (float(x[0]), float(x[1]), parameter)
        if key in self._cache:
            return self._cache[key]
        if len(self.evaluations) >= self.max_evaluations:
            self._fail("maximum callback evaluation count reached", parameter=parameter, center=x)
        row = {"index": len(self.evaluations), "center": x.tolist(),
               "parameter": parameter, "status": "started"}
        self.evaluations.append(row)
        self._emit("evaluation_started", **row)
        try:
            value = self.evaluate(x.copy(), parameter)
            if isinstance(value, dict):
                if any(key in value and not value[key] for key in ("qualified", "passed")):
                    raise ValueError("callback qualification failed")
                tau = float(value["tau"])
                metadata = _plain(value)
                json.dumps(metadata, allow_nan=False)
                row["callback"] = metadata
            else:
                tau = float(value)
            if not math.isfinite(tau):
                raise ValueError("callback returned a nonfinite trace")
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            self._emit("evaluation_failed", **row)
            raise ContinuationError(row["error"], _plain(row)) from error
        row.update(status="complete", tau=tau)
        self._cache[key] = row
        self._emit("evaluation_complete", **row)
        return row

    def _stencil(self, center, parameter, h):
        rows = [[self._point(center + h * np.array([dx, dy]), parameter)
                 for dx in (-1, 0, 1)] for dy in (-1, 0, 1)]
        f = np.array([[r["tau"] for r in row] for row in rows])
        gradient = np.array([(f[1, 2] - f[1, 0]) / (2 * h),
                             (f[2, 1] - f[0, 1]) / (2 * h)])
        cross = (f[2, 2] - f[0, 2] - f[2, 0] + f[0, 0]) / (4 * h * h)
        hessian = np.array([[(f[1, 2] - 2 * f[1, 1] + f[1, 0]) / h**2, cross],
                            [cross, (f[2, 1] - 2 * f[1, 1] + f[0, 1]) / h**2]])
        eigenvalues = np.linalg.eigvalsh(hessian)
        condition = float(np.linalg.cond(hessian))
        evidence = {"center": center, "parameter": parameter, "stencil_step": h,
                    "gradient": gradient, "hessian": hessian,
                    "hessian_eigenvalues": eigenvalues,
                    "point_indices": [[r["index"] for r in row] for row in rows]}
        if not math.isfinite(condition) or condition >= self.condition_limit:
            self._fail("ill-conditioned trace Hessian", **evidence)
        if not eigenvalues[0] < 0 < eigenvalues[1]:
            self._fail("stationary-point candidate is not a nondegenerate saddle", **evidence)
        correction = -np.linalg.solve(hessian, gradient)
        return rows[1][1], evidence, correction, condition

    def correct(self, parameter, initial_center):
        """Correct a predictor, retaining every iteration and rejecting bad gates."""
        parameter = float(parameter)
        if not math.isfinite(parameter):
            self._fail("finite third parameter required")
        center = self._center(initial_center)
        history = []
        for h in self.stencils:
            for iteration in range(self.max_iterations):
                point, evidence, correction, condition = self._stencil(center, parameter, h)
                move = float(np.linalg.norm(correction))
                row = _plain({**evidence, "iteration": iteration, "tau": point["tau"],
                              "hessian_condition": condition, "correction": correction,
                              "correction_norm": move, "converged": move <= self.correction_tolerance})
                history.append(row)
                self._emit("saddle_iteration", **row)
                if not math.isfinite(move) or move > self.trust_radius:
                    self._fail("saddle correction exceeds declared trust radius", parameter=parameter,
                               center=center, correction=correction, history=history)
                if move <= self.correction_tolerance:
                    break
                center = self._center(center + correction)
            else:
                self._fail("saddle correction did not converge at declared stencil scale",
                           parameter=parameter, center=center, history=history)
        result = _plain({"parameter": parameter, "center": center, "tau": point["tau"],
                         "converged": True, "history": history, "point_index": point["index"],
                         "hessian": evidence["hessian"],
                         "hessian_eigenvalues": evidence["hessian_eigenvalues"],
                         "correction_norm": move, "stencils": self.stencils})
        self.results.append(result)
        self._emit("saddle_complete", **result)
        return result

    def continue_values(self, values, initial_center):
        """Correct the baseline and then use a secant predictor from accepted centers."""
        values = [float(v) for v in values]
        if (not values or len(values) > self.max_continuation_points
                or not all(math.isfinite(v) for v in values)
                or any(a == b for a, b in zip(values, values[1:]))):
            self._fail("finite distinct consecutive parameter values within the point bound required")
        accepted = []
        initial = self._center(initial_center)
        for parameter in values:
            predictor = initial.copy() if not accepted else np.array(accepted[-1]["center"])
            if len(accepted) >= 2:
                a, b = accepted[-2:]
                predictor += ((parameter - b["parameter"]) / (b["parameter"] - a["parameter"])) * (
                    np.array(b["center"]) - np.array(a["center"]))
                if np.linalg.norm(predictor - np.array(b["center"])) > self.trust_radius:
                    self._fail("saddle predictor exceeds declared trust radius", parameter=parameter,
                               predictor=predictor)
            self._emit("saddle_predictor", parameter=parameter, center=predictor,
                       based_on=[r["parameter"] for r in accepted[-2:]])
            accepted.append(self.correct(parameter, predictor))
        return {"complete": True, "points": accepted, "evaluations": len(self.evaluations)}

    def solve_height_zero(self, left, right, *, height_tolerance=1e-8,
                          parameter_tolerance=1e-8, max_iterations=30):
        """Safeguarded secant/bisection of the height of corrected saddles.

        Success requires the height residual gate. The parameter tolerance stops
        an unresolved search; it never substitutes for that gate. The surviving
        bracket and its width are retained and are not a rigorous enclosure.
        """
        if (not math.isfinite(height_tolerance) or height_tolerance <= 0
                or not math.isfinite(parameter_tolerance) or parameter_tolerance <= 0
                or not 1 <= max_iterations <= self.max_continuation_points):
            raise ValueError("positive root tolerances and bounded iteration count required")
        for endpoint in (left, right):
            if (not endpoint.get("converged") or not math.isfinite(float(endpoint["tau"]))
                    or not math.isfinite(float(endpoint["parameter"]))):
                self._fail("height bracket endpoints must be converged finite saddles")
        lo, hi = sorted((left, right), key=lambda row: row["parameter"])
        if lo["parameter"] == hi["parameter"] or lo["tau"] * hi["tau"] >= 0:
            self._fail("no strict sign-changing saddle-height bracket",
                       parameters=[lo["parameter"], hi["parameter"]], heights=[lo["tau"], hi["tau"]])
        history = []
        for iteration in range(int(max_iterations)):
            width = hi["parameter"] - lo["parameter"]
            if width <= parameter_tolerance:
                self._fail("height bracket reached parameter tolerance without height convergence",
                           bracket=[lo, hi], history=history)
            candidate = (lo["parameter"] * hi["tau"] - hi["parameter"] * lo["tau"]) / (hi["tau"] - lo["tau"])
            if not lo["parameter"] + 0.1 * width <= candidate <= hi["parameter"] - 0.1 * width:
                candidate = (lo["parameter"] + hi["parameter"]) / 2
            weight = (candidate - lo["parameter"]) / width
            predictor = (1 - weight) * np.array(lo["center"]) + weight * np.array(hi["center"])
            saddle = self.correct(candidate, predictor)
            row = {"iteration": iteration, "parameter": candidate, "tau": saddle["tau"],
                   "bracket": [lo["parameter"], hi["parameter"]],
                   "bracket_heights": [lo["tau"], hi["tau"]], "center": saddle["center"]}
            history.append(row)
            self._emit("height_root_iteration", **row)
            if abs(saddle["tau"]) <= height_tolerance:
                result = {"complete": True, "saddle": saddle, "history": history,
                          "height_tolerance": height_tolerance,
                          "parameter_tolerance": parameter_tolerance,
                          "bracket": [lo["parameter"], hi["parameter"]],
                          "bracket_width": width, "evaluations": len(self.evaluations)}
                self._emit("height_root_complete", **result)
                return result
            if lo["tau"] * saddle["tau"] < 0:
                hi = saddle
            else:
                lo = saddle
        self._fail("saddle-height root did not converge", bracket=[lo, hi], history=history)
