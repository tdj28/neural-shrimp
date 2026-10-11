#!/usr/bin/env python3
"""Post-run arithmetic diagnosis on sealed EXP008 data; never a release gate.

No source tolerances, stored features, coefficients, or outcomes are replaced.
The two experiments isolate directional Gram cancellation and ridge solvers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy import linalg

import audit_information_platform as bindings
import hopfield_information_model as model
from hopfield_drift_probe import phase_projected_gram

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / 'docs/EXP-008-arithmetic-method.md'
SOLVERS = ('normal', 'svd', 'augmented_qr')


def projected_columns(sensitivity, velocity, dtype=np.float64):
    """Residualize parameter columns without forming a difference of Grams."""
    s = np.asarray(sensitivity, dtype=dtype)
    v = np.asarray(velocity, dtype=dtype)
    if (s.ndim != 3 or s.shape[1:] != (3, 2) or len(s) == 0
            or v.shape != s.shape[:2] or not np.isfinite(s).all()
            or not np.isfinite(v).all()):
        raise ValueError('Expected finite sensitivity (N,3,2), velocity (N,3)')
    columns, phase = s.reshape(-1, 2), v.ravel()
    norm = np.sum(phase * phase, dtype=dtype)
    coefficients = (np.sum(phase[:, None] * columns, axis=0, dtype=dtype) / norm
                    if norm > 0 else np.zeros(2, dtype=dtype))
    return columns - phase[:, None] * coefficients, coefficients


def gram_context(sensitivity, velocity):
    """Preserve the exact frozen projection while adding separate references."""
    frozen = phase_projected_gram(sensitivity, velocity)
    columns = np.asarray(sensitivity, float).reshape(-1, 2)
    phase = np.asarray(velocity, float).ravel()
    residual = columns - phase[:, None] * frozen['phase_coefficients']
    extended, coefficients = projected_columns(sensitivity, velocity, np.longdouble)
    return dict(frozen=frozen, residual=residual, extended=extended,
                extended_coefficients=coefficients)


def directional_diagnostic(context, delta):
    """Separate errors from the quadratic form, projection, and Gram storage."""
    delta = np.asarray(delta, float)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError('Expected a finite two-component delta')
    gram = context['frozen']['gram']
    q = float(delta @ gram @ delta)
    residual = context['residual']
    response = residual @ delta
    q_direct = float(np.mean(response * response))
    ld = np.longdouble
    delta_ld = delta.astype(ld)
    # These references preserve the exact binary64 inputs. They do not recover
    # integration accuracy lost before the tensors were saved.
    response_reference = np.sum(context['extended'] * delta_ld, axis=1, dtype=ld)
    q_reference = np.mean(response_reference * response_reference, dtype=ld)
    q_stored_gram = np.sum(delta_ld[:, None] * gram.astype(ld) * delta_ld[None, :], dtype=ld)
    frozen_response_ld = np.sum(residual.astype(ld) * delta_ld, axis=1, dtype=ld)
    q_frozen_residual_ld = np.mean(frozen_response_ld * frozen_response_ld, dtype=ld)
    absolute_terms = np.sum(np.abs(delta_ld[:, None] * gram.astype(ld) * delta_ld[None, :]), dtype=ld)
    rms_reference = np.sqrt(q_reference)
    result = dict(frozen_quadratic=q, frozen_clamped_rmse=float(np.sqrt(max(0., q))),
        direct_residual_quadratic=q_direct, direct_residual_rmse=float(np.sqrt(q_direct)),
        extended_reference_quadratic=float(q_reference), extended_reference_rmse=float(rms_reference),
        stored_gram_extended_quadratic=float(q_stored_gram),
        frozen_residual_extended_quadratic=float(q_frozen_residual_ld),
        directional_cancellation_ratio=(float(absolute_terms / q_reference) if q_reference > 0 else None),
        frozen_quadratic_negative=bool(q < 0),
        frozen_rmse_reference_error=float(abs(np.sqrt(max(0., q))-rms_reference)),
        direct_rmse_reference_error=float(abs(np.sqrt(q_direct)-rms_reference)),
        quadratic_evaluation_error=float(abs(ld(q)-q_stored_gram)),
        gram_formation_error=float(abs(q_stored_gram-q_frozen_residual_ld)),
        projection_error=float(abs(q_frozen_residual_ld-q_reference)))
    return result


def solver_grid(x, y, groups, lambdas=model.LAMBDA_GRID):
    """Compare three algorithms on ONE frozen-standardized ridge objective."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or not x.shape[0] or not x.shape[1] or y.shape != (len(x),)
            or len(groups) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas)):
        raise ValueError('Complete finite rows and positive penalties required')
    mean, scale, weights = model._standardizer(x, groups)
    z, intercept = (x-mean)/scale, float(weights @ y)
    a, b = np.sqrt(weights)[:, None] * z, np.sqrt(weights) * (y-intercept)
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    projection = u.T @ b
    # Match the original arithmetic, including its nonsquare-root weighting.
    gram = z.T @ (weights[:, None] * z)
    gram = .5 * (gram + gram.T)
    rhs = z.T @ (weights * (y-intercept))
    eye = np.eye(x.shape[1])
    a_ld, b_ld = a.astype(np.longdouble), b.astype(np.longdouble)
    largest = float(singular[0] ** 2)
    smallest = float(singular[-1] ** 2) if len(singular) == x.shape[1] else 0.
    results = []
    for lam in lambdas:
        beta = dict(normal=np.linalg.solve(gram + lam*eye, rhs),
            svd=vh.T @ ((singular/(singular**2+lam)) * projection),
            augmented_qr=linalg.lstsq(np.vstack((a, np.sqrt(lam)*eye)),
                np.r_[b, np.zeros(x.shape[1])], lapack_driver='gelsy')[0])
        metrics = {}
        for name, coefficient in beta.items():
            c = coefficient.astype(np.longdouble)
            residual = np.sum(a_ld*c, axis=1, dtype=np.longdouble)-b_ld
            gradient = np.sum(a_ld*residual[:, None], axis=0, dtype=np.longdouble) + lam*c
            stationarity = np.sqrt(np.sum(gradient*gradient, dtype=np.longdouble))
            denominator = ((largest+lam)*np.linalg.norm(coefficient)
                           + np.sqrt(largest)*np.linalg.norm(b))
            metrics[name] = dict(coefficients=coefficient.tolist(),
                objective_extended=float(np.sum(residual*residual, dtype=np.longdouble)
                                         + lam*np.sum(c*c, dtype=np.longdouble)),
                stationarity_norm_extended=float(stationarity),
                relative_stationarity=float(stationarity/denominator) if denominator else 0.,
                coefficient_error_upper_bound_from_stationarity=float(stationarity/lam))
        differences = {}
        for left, right in [('normal', 'svd'), ('normal', 'augmented_qr'), ('svd', 'augmented_qr')]:
            differences[left+'_vs_'+right] = dict(
                max_coefficient_difference=float(np.max(np.abs(beta[left]-beta[right]))),
                max_training_prediction_difference=float(np.max(np.abs(z @ (beta[left]-beta[right])))))
        condition = (largest+lam)/(smallest+lam)
        results.append(dict(penalty=float(lam), regularized_normal_condition_number=condition,
            augmented_design_condition_number=float(np.sqrt(condition)),
            solvers=metrics, differences=differences))
    return dict(mean=mean, scale=scale, intercept=intercept, standardized=z,
                singular_values=singular, results=results)


def ridge_diagnostic(x, y, groups, saved):
    group_array = np.asarray(groups)
    folds = []
    for held_out in sorted(set(groups)):
        train = group_array != held_out
        fit = solver_grid(x[train], y[train], group_array[train].tolist())
        validation = (x[~train]-fit['mean'])/fit['scale']
        losses = {name: [float(np.mean((fit['intercept']+validation @ np.asarray(
            item['solvers'][name]['coefficients'])-y[~train])**2)) for item in fit['results']]
                  for name in SOLVERS}
        folds.append(dict(held_out=held_out, mean=fit['mean'].tolist(), scale=fit['scale'].tolist(),
            intercept=fit['intercept'], mse_by_solver=losses, grid=fit['results']))
    cv = {name: np.mean([fold['mse_by_solver'][name] for fold in folds], axis=0) for name in SOLVERS}
    selected = {name: int(np.argmin(cv[name])) for name in SOLVERS}
    final = solver_grid(x, y, groups)
    old_index = saved['chosen_lambda_index']
    old_coefficients = np.asarray(saved['coefficients'])
    comparison = {}
    for name in SOLVERS:
        beta = np.asarray(final['results'][old_index]['solvers'][name]['coefficients'])
        comparison[name] = dict(at_original_selected_lambda=True,
            max_coefficient_difference=float(np.max(np.abs(beta-old_coefficients))),
            frozen_coefficient_comparison=bindings.compare_all(old_coefficients.tolist(), beta.tolist()),
            chosen_lambda_index=selected[name], chosen_lambda=float(model.LAMBDA_GRID[selected[name]]),
            choice_matches_saved=selected[name] == old_index)
    return dict(folds=folds, cv_mse={k:v.tolist() for k,v in cv.items()},
        saved_chosen_lambda_index=old_index, comparison_to_saved=comparison,
        final_mean=final['mean'].tolist(), final_scale=final['scale'].tolist(),
        final_intercept=final['intercept'], final_singular_values=final['singular_values'].tolist(),
        final_grid=final['results'],
        scope='All solvers use identical original standardization; no test-label fitting or amended features')


def diagnose(attempt):
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        raise RuntimeError('This method requires longdouble wider than binary64')
    p, receipt, features, records, packet, _ = bindings.bound_inputs(attempt)
    contexts, directions = [], []
    for role in bindings.driver.ROLES:
        rows = features[role]
        for anchor in bindings.driver.anchors_for(p, role):
            for phase in p['phases']:
                path = attempt/'sensitivity'/f"{anchor['id']}-p{phase}.npz"
                with np.load(path, allow_pickle=False) as raw:
                    tail = raw['times'] > 75
                    context = gram_context(raw['activation_sensitivity'][tail], raw['activation_velocity'][tail])
                    direction = raw['tail_direction'].tolist()
                frozen = context['frozen']
                contexts.append(dict(role=role, anchor=anchor['id'], phase=phase,
                    sensitivity_sha256=bindings.sha(path), gram=frozen['gram'].tolist(),
                    eigenvalues=frozen['eigenvalues'].tolist(), saved_tail_direction=direction,
                    direction_comparison=bindings.compare_all(direction, frozen['minimum_eigenvector'].tolist(),
                        rtol=1e-12, atol=1e-14),
                    phase_coefficients=frozen['phase_coefficients'].tolist(),
                    extended_phase_coefficients=[float(v) for v in context['extended_coefficients']]))
                for row in rows:
                    if row['anchor'] != anchor['id'] or row['phase'] != phase:
                        continue
                    delta = row['radius'] * np.asarray(row['direction'])
                    item = directional_diagnostic(context, delta)
                    saved = row['base'][34]
                    item.update(id=row['id'], role=role, delta=delta.tolist(), saved_projected_rmse=saved,
                        saved_reference_error=abs(saved-item['extended_reference_rmse']),
                        frozen_comparison=bindings.compare_all(saved, item['frozen_clamped_rmse'],
                            rtol=1e-12, atol=1e-14, path=row['id']+'.local_projected_rmse'))
                    directions.append(item)
    development = features['development']
    targets = {r['id']: r for r in records['development']}
    y = np.asarray([targets[row['id']+'-dt0.005']['rmse'] for row in development])
    groups = [row['block'] for row in development]
    base = np.asarray([row['base'] for row in development])
    geometry = np.asarray([row['geometry'] for row in development])
    shuffled = np.asarray(bindings.driver.shuffled_geometry(development, p, 'development'))
    matrices = dict(baseline=base, geometry=np.column_stack((base, geometry)),
                    shuffled_geometry=np.column_stack((base, shuffled)))
    input_hash = hashlib.sha256()
    for array in (base, geometry, shuffled, y):
        input_hash.update(str(array.shape).encode())
        input_hash.update(np.ascontiguousarray(array, dtype='<f8').tobytes())
    input_hash.update(json.dumps([[r['id'] for r in development], groups], separators=(',', ':')).encode())
    if input_hash.hexdigest() != packet['development_data_sha256']:
        raise ValueError('Reconstructed saved development data hash changed')
    if len(contexts) != 48 or len(directions) != 3456:
        raise ValueError('Incomplete sensitivity-context or directional-row census')
    ridge = {name: ridge_diagnostic(x, y, groups, packet['models'][name]) for name,x in matrices.items()}
    return dict(raw_bindings_passed=True, artifact_count=len(receipt['artifacts']),
        original_source=bindings.SOURCE, manifest_sha256=bindings.MANIFEST, receipt_sha256=bindings.RECEIPT,
        development_data_sha256=input_hash.hexdigest(), contexts=contexts, directions=directions,
        direction_count=len(directions), context_count=len(contexts),
        frozen_projected_feature_failure_count=sum(not d['frozen_comparison']['passed'] for d in directions),
        ridge=ridge, integrations=0, new_targets=0, diagnostic_status='completed',
        interpretation='Post-run numerical development evidence, not a repaired frozen gate, new utility result, '
            'proof of exact arithmetic, or validation on another platform.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-source', required=True)
    args = parser.parse_args()
    attempt, out = args.attempt.resolve(), args.out.resolve()
    if out.is_relative_to(attempt) or out.is_relative_to(ROOT):
        parser.error('Output must be outside the immutable attempt and repository')
    if out.exists():
        parser.error('Output already exists; preserve attempts with a new path')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if source != args.expected_source:
        parser.error('Current source does not match --expected-source')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        parser.error('Commit the diagnostic plan and source before execution; working tree must be clean')
    remote_refs = subprocess.check_output(['git', 'for-each-ref', '--contains', source,
        '--format=%(refname)', 'refs/remotes/origin/'], cwd=ROOT, text=True).splitlines()
    if not remote_refs:
        parser.error('Expected source must be contained in a local origin remote-tracking ref after push')
    report = dict(diagnostic_status='started', diagnostic_source=source, method_sha256=bindings.sha(METHOD),
        containing_origin_remote_refs=remote_refs,
        script_sha256=bindings.sha(Path(__file__)), started_at=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), python=sys.version, numpy_version=np.__version__, scipy_version=scipy.__version__,
        numerical_libraries=np.show_config(mode='dicts'),
        thread_environment={k:os.environ[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'] if k in os.environ},
        float64_mantissa_bits=np.finfo(np.float64).nmant+1,
        longdouble_mantissa_bits=np.finfo(np.longdouble).nmant+1)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new report and retain launch provenance even if interrupted.
    with out.open('x') as handle:
        json.dump(bindings.safe(report), handle, indent=2, allow_nan=False)
        handle.write('\n')
    started = time.perf_counter()
    try:
        report.update(diagnose(attempt))
        exit_code = 0
    except Exception as error:
        report.update(diagnostic_status='failed', error=repr(error))
        exit_code = 2
    report['wall_seconds'] = time.perf_counter()-started
    with out.open('w') as handle:
        json.dump(bindings.safe(report), handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k:report[k] for k in ('diagnostic_status', 'diagnostic_source', 'wall_seconds')}))
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
