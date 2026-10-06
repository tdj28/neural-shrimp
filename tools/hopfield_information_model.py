"""Prospective EXP008 development-only prediction and exhaustive-action analysis.

This module contains no trajectory integrator or target-data access. The driver
constructs the frozen features, supplies the complete development row census,
seals the fitted JSON packet, and seals all test decisions before test targets.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from numbers import Real

import numpy as np


LAMBDA_GRID = (1e-6, 1e-4, 1e-2, 1., 100.)
LEARNERS = ("baseline", "geometry", "shuffled_geometry")
POLICIES = (*LEARNERS, "min_preview")
CANDIDATES = tuple(range(18))
PHASES = (675, 1350)
RADII = (.00025, .001, .004, .016)
DTS = (.01, .005)
PRIMARY_DT = .005
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260913


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def packet_digest(packet):
    """Hash the JSON contents excluding the digest field itself."""
    contents = {k: v for k, v in packet.items() if k != "packet_sha256"}
    return hashlib.sha256(json.dumps(contents, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _finish(packet):
    result = _safe(packet)
    result["packet_sha256"] = packet_digest(result)
    return result


def _matrix(value, name, rows=None):
    result = np.asarray(value, dtype=float)
    if (result.ndim != 2 or result.shape[1] == 0 or not np.isfinite(result).all()
            or (rows is not None and len(result) != rows)):
        raise ValueError(f"{name} must be a complete finite two-dimensional feature matrix")
    return result.copy()


def _names(value, width, name):
    result = list(value)
    if (len(result) != width or len(set(result)) != width
            or any(not isinstance(s, str) or not s for s in result)):
        raise ValueError(f"{name} must give one unique nonempty name per column")
    return result


def _ids(value, n, name):
    result = list(value)
    if len(result) != n or any(not isinstance(s, str) or not s for s in result):
        raise ValueError(f"{name} must contain one nonempty string per row")
    return result


def _weights(groups):
    counts = Counter(groups)
    return np.array([1. / (len(counts) * counts[g]) for g in groups])


def _standardizer(x, groups):
    weights = _weights(groups)
    mean = weights @ x
    centered = x - mean
    scale = np.sqrt(weights @ (centered * centered))
    constant = np.ptp(x, axis=0) == 0
    scale[constant | (scale == 0)] = 1.
    return mean, scale, weights


def _ridge_grid(x, y, groups):
    mean, scale, weights = _standardizer(x, groups)
    z = (x - mean) / scale
    intercept = float(weights @ y)
    gram = z.T @ (weights[:, None] * z)
    gram = .5 * (gram + gram.T)
    rhs = z.T @ (weights * (y - intercept))
    coefficients = [np.linalg.solve(gram + lam * np.eye(x.shape[1]), rhs)
                    for lam in LAMBDA_GRID]
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite ridge coefficients")
    return mean, scale, intercept, coefficients


def _fit_one(x, y, groups, names):
    unique = sorted(set(groups))
    groups_array = np.asarray(groups)
    folds = []
    for held_out in unique:
        train = groups_array != held_out
        test = ~train
        mean, scale, intercept, coefficients = _ridge_grid(
            x[train], y[train], groups_array[train].tolist())
        predictions = np.array([intercept + ((x[test] - mean) / scale) @ coef
                                for coef in coefficients])
        mse = np.mean((predictions - y[test]) ** 2, axis=1)
        if not np.isfinite(mse).all():
            raise ValueError("Nonfinite leave-one-anchor-out score")
        folds.append(dict(held_out_group=held_out, train_groups=unique[:unique.index(held_out)]
                          + unique[unique.index(held_out)+1:], train_rows=int(train.sum()),
                          validation_rows=int(test.sum()), feature_mean=mean,
                          feature_scale=scale, mse_by_lambda=mse))
    cv_mse = np.mean([f["mse_by_lambda"] for f in folds], axis=0)
    selected = int(np.argmin(cv_mse))  # Exact ties: first (smallest) lambda.
    mean, scale, intercept, coefficients = _ridge_grid(x, y, groups)
    coefficient = coefficients[selected]
    fitted = intercept + ((x - mean) / scale) @ coefficient
    return dict(feature_names=names, feature_mean=mean, feature_scale=scale,
                intercept=intercept, coefficients=coefficient,
                chosen_lambda=LAMBDA_GRID[selected], chosen_lambda_index=selected,
                cv=dict(lambda_grid=LAMBDA_GRID, equal_group_mse=cv_mse, folds=folds),
                training_equal_group_mse=float(_weights(groups) @ ((fitted-y) ** 2)),
                cost=dict(cv_folds=len(unique), ridge_solves=(len(unique)+1)*len(LAMBDA_GRID)))


def fit_information_models(base_features, geometry_features, target_rmse, group_ids, *,
                           shuffled_geometry_features, base_feature_names,
                           geometry_feature_names, row_ids, expected_row_ids=None,
                           expected_group_ids=None, shuffle_metadata=None):
    """Fit three learners using development rows only; never filter failed rows.

    The driver must pass its independently constructed complete expected row
    and group census. Explicit feature columns include any frozen nonlinear
    terms; this function adds no polynomial features. A failed packet retains
    completed model diagnostics but cannot be used for prediction.
    """
    packet = dict(schema_version=1, method="EXP008_group_weighted_linear_ridge_v1",
                  status="training_failed", lambda_grid=LAMBDA_GRID, models={},
                  label="Fine-step activation RMSE over 75<t<=300",
                  prediction_clipping=False, shuffle_metadata=shuffle_metadata,
                  failure_rule="Any required development failure invalidates all learners")
    try:
        base = _matrix(base_features, "base_features")
        geometry = _matrix(geometry_features, "geometry_features", len(base))
        shuffled = _matrix(shuffled_geometry_features, "shuffled_geometry_features", len(base))
        if geometry.shape != shuffled.shape:
            raise ValueError("Shuffled geometry must preserve the geometry column schema")
        names = _names(base_feature_names, base.shape[1], "base_feature_names")
        geometry_names = _names(geometry_feature_names, geometry.shape[1], "geometry_feature_names")
        if set(names) & set(geometry_names):
            raise ValueError("Base and geometry feature names must be disjoint")
        ids = _ids(row_ids, len(base), "row_ids")
        groups = _ids(group_ids, len(base), "group_ids")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate development row ID")
        if expected_row_ids is None or expected_group_ids is None:
            raise ValueError("The independent expected development row and group census is required")
        expected_ids = list(expected_row_ids)
        expected_groups = list(expected_group_ids)
        if len(set(expected_ids)) != len(expected_ids) or set(ids) != set(expected_ids):
            raise ValueError("Incomplete or unplanned development row census")
        if (len(expected_groups) != 12 or len(set(expected_groups)) != 12
                or set(groups) != set(expected_groups)):
            raise ValueError("Exactly twelve complete declared development groups are required")
        if any(isinstance(v, (bool, np.bool_)) for v in target_rmse):
            raise ValueError("Boolean development labels are not activation RMSE")
        y = np.asarray(target_rmse, dtype=float)
        if y.shape != (len(base),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 2)):
            raise ValueError("All development labels must be finite activation RMSE in [0,2]")
        packet.update(base_feature_names=names, geometry_feature_names=geometry_names,
                      development_row_ids=ids, development_groups=sorted(set(groups)),
                      development_group_counts=dict(Counter(groups)), development_rows=len(ids))
        data_hash = hashlib.sha256()
        for array in (base, geometry, shuffled, y):
            data_hash.update(str(array.shape).encode())
            data_hash.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        data_hash.update(json.dumps([ids, groups], separators=(",", ":")).encode())
        packet["development_data_sha256"] = data_hash.hexdigest()
        for policy, features in zip(LEARNERS, (base, np.column_stack((base, geometry)),
                                                np.column_stack((base, shuffled)))):
            packet["models"][policy] = _fit_one(features, y, groups,
                names if policy == "baseline" else names + geometry_names)
        packet["status"] = "complete"
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError, FloatingPointError) as error:
        packet["error"] = repr(error)
    return _finish(packet)


def predict_information_models(packet, base_features, geometry_features, *,
                               shuffled_geometry_features, base_feature_names=None,
                               geometry_feature_names=None):
    """Apply an immutable development model without fitting or test outcomes."""
    result = dict(status="prediction_failed", scores={p: None for p in LEARNERS})
    try:
        if packet.get("packet_sha256") != packet_digest(packet) or packet.get("status") != "complete":
            raise ValueError("A valid complete sealed development packet is required")
        base = _matrix(base_features, "base_features")
        geometry = _matrix(geometry_features, "geometry_features", len(base))
        shuffled = _matrix(shuffled_geometry_features, "shuffled_geometry_features", len(base))
        if base.shape[1] != len(packet["base_feature_names"]) or geometry.shape[1] != len(packet["geometry_feature_names"]):
            raise ValueError("Test feature width differs from the locked development schema")
        if shuffled.shape != geometry.shape:
            raise ValueError("Shuffled test geometry schema differs")
        for supplied, key in ((base_feature_names, "base_feature_names"),
                              (geometry_feature_names, "geometry_feature_names")):
            if supplied is not None and list(supplied) != packet[key]:
                raise ValueError("Test feature order differs from the locked development schema")
        for policy, features in zip(LEARNERS, (base, np.column_stack((base, geometry)),
                                                np.column_stack((base, shuffled)))):
            model = packet["models"][policy]
            prediction = model["intercept"] + ((features - np.asarray(model["feature_mean"]))
                         / np.asarray(model["feature_scale"])) @ np.asarray(model["coefficients"])
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite test prediction")
            result["scores"][policy] = prediction
        result.update(status="complete", model_packet_sha256=packet["packet_sha256"])
    except (ValueError, TypeError, KeyError, FloatingPointError) as error:
        result.update(error=repr(error), scores={p: None for p in LEARNERS})
    return _safe(result)


def _finite_score(value, bounded=False):
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and (not bounded or 0 <= value <= 2))


def select_candidate_actions(prediction_result, preview_scores, *, candidate_ids=CANDIDATES):
    """Use all eighteen entries, retaining duplicates; exact ties use lowest ID."""
    ids = list(candidate_ids)
    if len(ids) != 18 or set(ids) != set(CANDIDATES) or any(type(i) is not int for i in ids):
        raise ValueError("All eighteen unique declared candidate IDs are required")
    preview = list(preview_scores)
    policies = {}
    for policy in POLICIES:
        values = preview if policy == "min_preview" else prediction_result.get("scores", {}).get(policy)
        valid = (policy == "min_preview" or prediction_result.get("status") == "complete")
        valid = valid and values is not None and len(values) == 18
        valid = valid and all(_finite_score(x, bounded=policy == "min_preview") for x in values)
        if valid:
            position = min(range(18), key=lambda i: (values[i], ids[i]))
            policies[policy] = dict(status="complete", candidate=ids[position],
                                    selection_score=float(values[position]))
        else:
            policies[policy] = dict(status="selector_failed", candidate=None)
    return _safe(dict(status="complete" if all(p["status"] == "complete" for p in policies.values())
                      else "selector_failed", candidate_ids=ids, policies=policies,
                      predictions=prediction_result.get("scores"), preview_scores=preview,
                      model_packet_sha256=prediction_result.get("model_packet_sha256"),
                      required_preview_calls={p: 18 for p in POLICIES}))


def _condition_prediction_metrics(decision, policy, rmses, outcomes_complete):
    """Score every declared prediction, without clipping or survivor averaging."""
    result = dict(status="undefined_incomplete", prediction_census_complete=False,
                  mse=None, minimum=None, maximum=None, below_zero_fraction=None,
                  above_two_fraction=None, out_of_range_fraction=None)
    predictions = decision.get("predictions")
    if not isinstance(predictions, dict):
        return result
    values = predictions.get(policy)
    ids = decision.get("candidate_ids", CANDIDATES)
    if (not isinstance(values, (list, tuple, np.ndarray)) or np.ndim(values) != 1
            or len(values) != 18 or not isinstance(ids, (list, tuple, np.ndarray))
            or np.ndim(ids) != 1 or len(ids) != 18
            or any(type(i) is not int for i in ids) or set(ids) != set(CANDIDATES)
            or not all(_finite_score(v) for v in values)):
        return result
    indexed = dict(zip(ids, values))
    prediction = np.array([indexed[c] for c in CANDIDATES], dtype=float)
    result.update(prediction_census_complete=True,
                  minimum=float(prediction.min()), maximum=float(prediction.max()),
                  below_zero_fraction=float(np.mean(prediction < 0)),
                  above_two_fraction=float(np.mean(prediction > 2)),
                  out_of_range_fraction=float(np.mean((prediction < 0) | (prediction > 2))))
    if outcomes_complete:
        with np.errstate(over="ignore", invalid="ignore"):
            mse = float(np.mean((prediction - np.asarray(rmses)) ** 2))
        if np.isfinite(mse):
            result.update(status="complete", mse=mse)
        else:
            result["status"] = "undefined_nonfinite_metric"
    return result


def summarize_information_outcomes(candidate_records, decision_records, *, block_ids,
                                   phases=PHASES, radii=RADII, dts=DTS):
    """Analyze TEST-only exhaustive candidate outcomes without dropping failures.

    Candidate rows: block, phase, radius, candidate, dt, status, rmse.
    Decision rows: block, phase, radius, policies[policy]={status,candidate},
    predictions[learner]=eighteen scores, optional explicit candidate_ids.
    Zero-dose records are a separate driver assay and must not be passed here.
    """
    blocks = list(block_ids)
    if (len(blocks) != 12 or len(set(blocks)) != 12
            or any(not isinstance(b, str) or not b for b in blocks)
            or tuple(phases) != PHASES or tuple(radii) != RADII or tuple(dts) != DTS):
        raise ValueError("The frozen twelve-block, two-phase, four-dose, two-step design is required")
    contexts = [(b, ph, r) for b in blocks for ph in phases for r in radii]
    expected = {(b, ph, r, c, dt) for b, ph, r in contexts for c in CANDIDATES for dt in dts}
    lookup = {}
    for row in candidate_records:
        key = tuple(row[k] for k in ("block", "phase", "radius", "candidate", "dt"))
        if type(row["candidate"]) is not int or key not in expected or key in lookup:
            raise ValueError("Unplanned or duplicate candidate outcome")
        lookup[key] = row
    decisions = {}
    for row in decision_records:
        key = tuple(row[k] for k in ("block", "phase", "radius"))
        if key not in contexts or key in decisions:
            raise ValueError("Unplanned or duplicate decision context")
        decisions[key] = row
    census = Counter(r.get("status", "missing_status") for r in candidate_records)
    census["missing"] += len(expected) - len(lookup)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(12, size=(BOOTSTRAP_SAMPLES, 12))
    result = dict(planned_candidates=len(expected), received_candidates=len(lookup),
                  candidate_status_counts=dict(census), planned_decisions=len(contexts),
                  received_decisions=len(decisions), primary_dt=PRIMARY_DT,
                  blocks=blocks, steps={}, bootstrap=dict(samples=BOOTSTRAP_SAMPLES,
                  seed=BOOTSTRAP_SEED, unit="Whole paired test block; descriptive percentile interval"),
                  primary_identity="baseline_regret - geometry_regret = baseline_RMSE - geometry_RMSE")
    for dt in dts:
        condition_rows = {}
        for context in contexts:
            rows = [lookup.get((*context, c, dt)) for c in CANDIDATES]
            complete = all(row is not None and row.get("status") == "complete"
                           and _finite_score(row.get("rmse"), bounded=True) for row in rows)
            rmses = [float(row["rmse"]) if row is not None and row.get("status") == "complete"
                     and _finite_score(row.get("rmse"), bounded=True) else None for row in rows]
            oracle = min(rmses) if complete else None
            policies = {}
            for policy in POLICIES:
                choice = decisions.get(context, {}).get("policies", {}).get(policy, {})
                c = choice.get("candidate")
                valid_choice = choice.get("status") == "complete" and type(c) is int and c in CANDIDATES
                selected = rmses[c] if valid_choice else None
                policies[policy] = dict(status="complete" if complete and valid_choice else "undefined_incomplete",
                                        candidate=c if valid_choice else None, selected_rmse=selected,
                                        regret=selected-oracle if complete and valid_choice else None)
            condition_rows[context] = dict(block=context[0], phase=context[1], radius=context[2],
                candidate_complete=complete, oracle_rmse=oracle,
                oracle_candidates=[c for c, value in enumerate(rmses) if value == oracle]
                                  if complete else [], candidate_rmse=rmses, policies=policies,
                prediction={policy: _condition_prediction_metrics(decisions.get(context, {}),
                             policy, rmses, complete) for policy in LEARNERS})
        panels = {}
        for dose in (*radii, "pooled"):
            selected_radii = radii if dose == "pooled" else (dose,)
            per_block = {policy: dict(selected_rmse=[], regret=[]) for policy in POLICIES}
            prediction_per_block = {policy: {metric: [] for metric in
                ("mse", "minimum", "maximum", "below_zero_fraction", "above_two_fraction", "out_of_range_fraction")}
                for policy in LEARNERS}
            for block in blocks:
                conditions = [condition_rows[(block, ph, r)] for r in selected_radii for ph in phases]
                for policy in POLICIES:
                    valid = all(c["policies"][policy]["status"] == "complete" for c in conditions)
                    for metric in ("selected_rmse", "regret"):
                        per_block[policy][metric].append(float(np.mean([c["policies"][policy][metric]
                                                        for c in conditions])) if valid else None)
                for policy in LEARNERS:
                    metrics = [c["prediction"][policy] for c in conditions]
                    for metric in prediction_per_block[policy]:
                        values = [row[metric] for row in metrics]
                        if any(v is None for v in values):
                            value = None
                        elif metric == "minimum":
                            value = min(values)
                        elif metric == "maximum":
                            value = max(values)
                        else:
                            value = float(np.mean(values))
                        prediction_per_block[policy][metric].append(value)
            means = {policy: {metric: float(np.mean(values)) if all(v is not None for v in values)
                              else None for metric, values in metrics.items()}
                     for policy, metrics in per_block.items()}
            contrasts = {}
            for name, left, right in (("baseline_minus_geometry", "baseline", "geometry"),
                                      ("baseline_minus_shuffled", "baseline", "shuffled_geometry"),
                                      ("shuffled_minus_geometry", "shuffled_geometry", "geometry"),
                                      ("min_preview_minus_geometry", "min_preview", "geometry")):
                x = per_block[left]["selected_rmse"]
                y = per_block[right]["selected_rmse"]
                if any(v is None for v in (*x, *y)):
                    contrasts[name] = dict(status="undefined_incomplete")
                    continue
                # Oracle cancellation is exact algebraically. Compute paired
                # benefit from task errors directly, avoiding subtraction of
                # two separately rounded regret means.
                benefit = np.asarray(x) - np.asarray(y)
                ci = np.percentile(benefit[draws].mean(axis=1), [2.5, 97.5])
                contrasts[name] = dict(status="complete", mean_benefit=float(benefit.mean()),
                    ci95=ci, positive_blocks=int((benefit > 0).sum()), per_block=benefit)
            prediction = {}
            for policy, metrics in prediction_per_block.items():
                aggregate = {}
                for metric, values in metrics.items():
                    if any(v is None for v in values):
                        aggregate[metric] = None
                    elif metric == "minimum":
                        aggregate[metric] = min(values)
                    elif metric == "maximum":
                        aggregate[metric] = max(values)
                    else:
                        aggregate[metric] = float(np.mean(values))
                prediction[policy] = dict(status="complete" if aggregate["mse"] is not None
                    else "undefined_incomplete", range_status="complete" if aggregate["minimum"] is not None
                    else "undefined_incomplete", **aggregate)
            panels[str(dose)] = dict(per_block=per_block, means=means, contrasts=contrasts,
                                    prediction=prediction, prediction_per_block=prediction_per_block)
        result["steps"][str(dt)] = dict(conditions=list(condition_rows.values()), panels=panels,
            complete_candidate_conditions=sum(c["candidate_complete"] for c in condition_rows.values()))
    complete_census = all(result["steps"][str(dt)]["complete_candidate_conditions"] == len(contexts)
                          for dt in dts)
    result["primary_required_candidate_census_complete"] = complete_census
    result["primary"] = (result["steps"][str(PRIMARY_DT)]["panels"]["pooled"]["contrasts"]["baseline_minus_geometry"]
                         if complete_census else dict(status="undefined_incomplete",
                            reason="Required test candidate census includes both numerical steps"))
    return _safe(result)
