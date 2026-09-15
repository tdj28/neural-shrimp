"""Frozen EXP-011 ridge recipe. Pure numerical functions; no label or file loader."""
from __future__ import annotations

import hashlib
import json
import time
import numpy as np
from scipy.linalg import qr, solve_triangular

LAMBDA_GRID = (1e-6, 1e-4, .01, 1., 100.)
CHECK_ATOL = 1e-9
CHECK_RTOL = 1e-7


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def finite(array, name):
    a = np.asarray(array, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite {name}; no fallback")
    return a


def equal_origin_weights(groups):
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    if not len(unique):
        raise ValueError("Empty training origins")
    count = dict(zip(unique, counts))
    return np.asarray([1. / (len(unique) * count[g]) for g in groups])


def ridge_grid(x, y, groups, penalties=LAMBDA_GRID):
    x, y = finite(x, "features"), finite(y, "labels")
    if x.ndim != 2 or y.shape != (len(x),) or len(groups) != len(x) or not len(x):
        raise ValueError("Incompatible training arrays")
    if tuple(penalties) != LAMBDA_GRID:
        raise ValueError("Frozen penalty grid differs")
    w = equal_origin_weights(groups)
    mean = w @ x
    centered = x - mean
    scale = np.sqrt(w @ (centered * centered))
    constant = (np.ptp(x, axis=0) == 0) | (scale == 0)
    scale[constant] = 1.
    z = centered / scale
    intercept = float(w @ y)
    a, b = np.sqrt(w)[:, None] * z, np.sqrt(w) * (y - intercept)
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    projected = u.T @ b
    fits = []
    for penalty in penalties:
        svd = vt.T @ ((singular / (singular**2 + penalty)) * projected)
        # Direct augmented-design QR is independent of the SVD path. Neither
        # solver forms the ill-conditioned normal-equation Gram matrix.
        augmented = np.vstack((a, np.sqrt(penalty) * np.eye(x.shape[1])))
        target = np.concatenate((b, np.zeros(x.shape[1])))
        q, r = qr(augmented, mode="economic", check_finite=True)
        alternative = solve_triangular(r, q.T @ target, check_finite=True)
        fits.append(dict(penalty=penalty, svd=finite(svd, "SVD coefficients").tolist(),
                         qr=finite(alternative, "QR coefficients").tolist()))
    return dict(mean=mean.tolist(), scale=scale.tolist(), intercept=intercept,
                exact_constant_columns=np.flatnonzero(constant).tolist(),
                training_groups=sorted({str(g) for g in groups}), training_rows=len(y),
                singular_values=singular.tolist(), fits=fits)


def predict(grid, x, index, solver="svd"):
    if solver not in ("svd", "qr"):
        raise ValueError("Unknown frozen solver")
    x = finite(x, "prediction features")
    coefficient = np.asarray(grid["fits"][index][solver])
    return finite(grid["intercept"] + ((x - grid["mean"]) / grid["scale"]) @ coefficient,
                  "predicted scores")


def score_agreement(a, b):
    a, b = finite(a, "SVD scores"), finite(b, "QR scores")
    if a.shape != b.shape:
        raise ValueError("Numerical score shapes differ")
    delta = np.abs(a - b)
    return dict(passed=bool(np.all(delta <= CHECK_ATOL + CHECK_RTOL * np.maximum(np.abs(a), np.abs(b)))),
                maximum_absolute_difference=float(delta.max()) if delta.size else 0.)


def context_key(row):
    return (row["group"], row["phase"], row["radius"])


def shortlist_rows(rows, scores):
    scores = finite(scores, "shortlist scores")
    if len(rows) != len(scores):
        raise ValueError("Prediction census differs")
    contexts = {}
    for row, value in zip(rows, scores):
        contexts.setdefault(context_key(row), []).append((row["candidate"], row["id"], float(value)))
    answer = []
    for context, entries in sorted(contexts.items()):
        if len(entries) != 18 or {r[0] for r in entries} != set(range(18)):
            raise ValueError("Every context requires all 18 separate candidate entries")
        ranked = sorted(entries, key=lambda r: (r[2], r[0]))
        answer.append(dict(context=list(context), candidate_order=[r[0] for r in ranked],
                           top4=[r[0] for r in ranked[:4]],
                           top4_ids=[r[1] for r in ranked[:4]]))
    return answer


def fit_nested(x, y, groups, names, held_x, held_rows, *, stage_timings=None):
    """Tune on inner physical-origin folds; caller provides outer-training labels only."""
    x, y, held_x = finite(x, "features"), finite(y, "labels"), finite(held_x, "held features")
    groups = np.asarray(groups)
    unique = sorted({str(g) for g in groups})
    held_groups = {r["group"] for r in held_rows}
    if (len(unique) < 2 or set(unique) & held_groups or len(held_groups) != 1
            or x.shape[1] != len(names) or held_x.shape[1] != len(names)):
        raise ValueError("Physical-origin fold isolation/schema differs")
    timing = dict(grid_fit_seconds=0., inner_validation_scoring_seconds=0., held_prediction_and_shortlist_seconds=0.)
    inner, cv_svd, cv_qr, checks = [], [], [], []
    for held_group in unique:
        train = groups != held_group
        val = ~train
        started = time.monotonic()
        grid = ridge_grid(x[train], y[train], groups[train])
        timing['grid_fit_seconds'] += time.monotonic() - started
        started = time.monotonic()
        svd = np.stack([predict(grid, x[val], i) for i in range(len(LAMBDA_GRID))])
        alternative = np.stack([predict(grid, x[val], i, "qr") for i in range(len(LAMBDA_GRID))])
        mse = np.mean((svd - y[val])**2, axis=1)
        qr_mse = np.mean((alternative - y[val])**2, axis=1)
        check = score_agreement(svd, alternative)
        timing['inner_validation_scoring_seconds'] += time.monotonic() - started
        checks.append(check)
        cv_svd.append(mse)
        cv_qr.append(qr_mse)
        inner.append(dict(held_group=held_group, validation_rows=int(val.sum()),
                          training_label_sha256=digest(y[train].tolist()), grid=grid,
                          svd_mse=mse.tolist(), qr_mse=qr_mse.tolist(), score_agreement=check))
    mean_svd, mean_qr = np.mean(cv_svd, axis=0), np.mean(cv_qr, axis=0)
    selected, selected_qr = int(np.argmin(mean_svd)), int(np.argmin(mean_qr))
    started = time.monotonic()
    final_grid = ridge_grid(x, y, groups)
    timing['grid_fit_seconds'] += time.monotonic() - started
    started = time.monotonic()
    primary = predict(final_grid, held_x, selected)
    # QR chooses its own CV penalty, exposing selection and action consequences.
    alternative = predict(final_grid, held_x, selected_qr, "qr")
    primary_list = shortlist_rows(held_rows, primary)
    alternative_list = shortlist_rows(held_rows, alternative)
    final_check = score_agreement(primary, alternative)
    shortlist_same = all(a["top4"] == b["top4"] for a, b in zip(primary_list, alternative_list))
    rank_same = all(a["candidate_order"] == b["candidate_order"] for a, b in zip(primary_list, alternative_list))
    checks_pass = (all(c["passed"] for c in checks) and final_check["passed"]
                   and selected == selected_qr and shortlist_same)
    timing['held_prediction_and_shortlist_seconds'] = time.monotonic() - started
    if stage_timings is not None:
        stage_timings.update(timing)
    packet = dict(schema_version="exp011-ridge-svd-qr-v1", feature_names=list(names),
        training_groups=unique, held_group=next(iter(held_groups)), training_rows=len(y),
        training_label_sha256=digest(y.tolist()), penalty_grid=list(LAMBDA_GRID), inner_folds=inner,
        mean_inner_mse_svd=mean_svd.tolist(), mean_inner_mse_qr=mean_qr.tolist(),
        selected_index_svd=selected, selected_index_qr=selected_qr,
        selected_penalty_svd=LAMBDA_GRID[selected], selected_penalty_qr=LAMBDA_GRID[selected_qr],
        final_grid=final_grid, held_row_ids=[r["id"] for r in held_rows],
        held_predictions_svd=primary.tolist(), held_predictions_qr=alternative.tolist(),
        shortlists_svd=primary_list, shortlists_qr=alternative_list,
        numerical_audit=dict(passed=checks_pass, all_inner_scores_agree=all(c["passed"] for c in checks),
             final_score_agreement=final_check, selected_penalty_agrees=selected == selected_qr,
             shortlist_agrees=shortlist_same, full_ranking_agrees=rank_same,
             coefficient_max_absolute_difference=max(float(np.max(np.abs(np.asarray(f["svd"]) - f["qr"])))
                 for g in [final_grid, *[i["grid"] for i in inner]] for f in g["fits"])))
    packet["packet_sha256"] = digest(packet)
    return packet
