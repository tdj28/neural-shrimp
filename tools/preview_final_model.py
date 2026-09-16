"""DRAFT EXP-013 final old-data recipes and label-free fresh predictions.

Pure array/metadata API: no file loader, dynamics, label capability, or outcome
execution on import. The caller must bind the exact historical origin registry,
qualified 75-unit labels and feature inputs in its prospectively frozen plan.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import numpy as np

import preview_shortlist_features as features
import preview_shortlist_model as model

RESOLUTIONS = ('rk4-0.00125', 'rk4-0.0025')
SCHEMAS = ('shared', 'geometry')
BUDGETS = (2, 4, 8)
PHASES = (675, 1350)
RADII = (.00025, .001, .004, .016)
PACKET_SCHEMA = 'EXP-013-draft-final-old-ridge-v1'
PREDICTION_SCHEMA = 'EXP-013-draft-prepreview-predictions-v1'
ROW_REQUIRED = {'id', 'group', 'phase', 'radius', 'candidate', 'direction'}
ROW_ALLOWED = ROW_REQUIRED | {'role', 'anchor'}
UNIFORM_DIRECTIONS = np.asarray([r['direction'] for r in features.direction_candidates('uniform_16')])
SOURCE_PATHS = list(dict.fromkeys(['tools/preview_final_model.py',
    'tools/preview_shortlist_model.py', *features.SOURCE_PATHS]))


class FinalFitFailure(RuntimeError):
    """A solver exception retains completed fold evidence without a fallback."""
    def __init__(self, message, packet):
        super().__init__(message)
        self.packet = packet


def feature_names(schema):
    if schema not in SCHEMAS:
        raise ValueError('Unknown frozen feature schema')
    return list(features.SHARED_FEATURE_NAMES) + (
        list(features.GEOMETRY_FEATURE_NAMES) if schema == 'geometry' else [])


def _seal(packet):
    packet['packet_sha256'] = model.digest(packet)
    return packet


def _array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()


def _registry(groups):
    groups = list(groups)
    if len(groups) != 24 or len(set(groups)) != 24:
        raise ValueError('Exactly 24 distinct historical physical origins required')
    for group in groups:
        try:
            prefix, alpha, gamma = group.split(':')
            canonical = features.physical_group(float.fromhex(alpha), float.fromhex(gamma))
        except (AttributeError, TypeError, ValueError):
            raise ValueError('Historical registry must contain canonical physical origins') from None
        if group != canonical or prefix != 'hopfield-da-silva-rech-2022':
            raise ValueError('Historical registry must contain canonical physical origins')
    return sorted(groups)


def _matrix(x, names, schema):
    x = np.asarray(x)
    if (list(names) != feature_names(schema) or x.dtype != np.float64 or x.ndim != 2
            or x.shape[1] != len(names) or not 0 < len(x) <= 3456 or not np.isfinite(x).all()):
        raise ValueError('Exact finite float64 pre-preview feature schema required')
    return x


def _rows(rows, x, *, training_groups=None):
    """Reconstruct coordinate identities independently of supplied aliases."""
    if len(rows) != len(x):
        raise ValueError('Row/feature census differs')
    ids, groups, contexts, aliases, directions = set(), [], defaultdict(list), {}, {}
    normalized = []
    for row, values in zip(rows, x, strict=True):
        if (not isinstance(row, dict) or not ROW_REQUIRED <= set(row) or set(row)-ROW_ALLOWED
                or not isinstance(row['id'], str) or not row['id'] or row['id'] in ids):
            raise ValueError('Unique IDs and pre-preview-only row fields required')
        group = features.physical_group(values[0], values[1])
        if row['group'] != group:
            raise ValueError('Physical-origin alias differs from feature coordinates')
        phase, radius, candidate = row['phase'], row['radius'], row['candidate']
        if (type(phase) is not int or phase not in PHASES or type(candidate) is not int
                or candidate not in range(18) or isinstance(radius, (bool, np.bool_))
                or radius not in RADII):
            raise ValueError('Unplanned phase/radius/candidate identity')
        if (not np.array_equal(values[7:9], [np.sin(2*np.pi*phase/2700), np.cos(2*np.pi*phase/2700)])
                or not np.array_equal(values[12:15], [radius, np.log(radius), radius*radius])):
            raise ValueError('Row phase/radius differs from pre-preview features')
        direction = np.asarray(row['direction'])
        if (direction.shape != (2,) or direction.dtype.kind not in 'fi'
                or not np.isfinite(direction).all() or not np.array_equal(direction, values[15:17])
                or not np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-12)):
            raise ValueError('Row direction differs from unit pre-preview features')
        if candidate < 16 and not np.array_equal(direction, UNIFORM_DIRECTIONS[candidate]):
            raise ValueError('Uniform candidate identity differs from canonical direction')
        identity = (group, phase, candidate)
        if identity in directions and not np.array_equal(direction, directions[identity]):
            raise ValueError('Candidate direction changed across radii at one origin/phase')
        directions[identity] = direction.copy()
        if 'anchor' in row or 'role' in row:
            if any(not isinstance(row.get(k, ''), str) for k in ('role', 'anchor')):
                raise ValueError('Origin alias metadata must be text')
            alias = (row.get('role', ''), row.get('anchor', ''))
            if alias in aliases and aliases[alias] != group:
                raise ValueError('Origin alias maps to more than one physical setting')
            aliases[alias] = group
        ids.add(row['id'])
        groups.append(group)
        contexts[(group, phase, radius)].append(candidate)
        normalized.append(dict(id=row['id'], group=group, phase=phase, radius=float(radius),
                               candidate=candidate, direction=direction.tolist()))
    if any(len(c) != 18 or set(c) != set(range(18)) for c in contexts.values()):
        raise ValueError('Every planned context requires 18 distinct candidate entries')
    if any(not np.array_equal(directions[(g,p,17)], -directions[(g,p,16)]) for g,p,_ in contexts):
        raise ValueError('Local candidate16/17 directions must be a signed pair')
    if training_groups is not None:
        expected = {(g, p, r) for g in training_groups for p in PHASES for r in RADII}
        if (len(rows) != 3456 or set(groups) != set(training_groups) or set(contexts) != expected
                or any(n != 144 for n in Counter(groups).values())):
            raise ValueError('Complete 24-origin/192-context/3456-row historical census required')
    return np.asarray(groups), normalized


def _labels(y, n):
    y = np.asarray(y)
    if (y.dtype != np.float64 or y.shape != (n,) or not np.isfinite(y).all()
            or np.any(y < 0) or np.any(y > 2)):
        raise ValueError('Finite float64 historical 75-unit tanh RMSE labels in [0,2] required')
    return y


def _fit_arrays(x, y, groups, packet):
    """Numerical core, separately exercised with tiny synthetic arrays."""
    unique = sorted(set(groups.tolist()))
    packet.update(inner_folds=[], final_grid=None, final_grid_score_checks=[])
    cv_svd, cv_qr = [], []
    try:
        for held in unique:
            train, val = groups != held, groups == held
            grid = model.ridge_grid(x[train], y[train], groups[train])
            fold = dict(held_group=held, training_groups=sorted(set(groups[train])),
                training_rows=int(train.sum()), validation_rows=int(val.sum()),
                validation_indices=np.flatnonzero(val).tolist(),
                training_label_sha256=model.digest(y[train].tolist()),
                validation_label_sha256=model.digest(y[val].tolist()), grid=grid,
                status='grid_complete', predictions_svd=[], predictions_qr=[], all_penalty_score_checks=[])
            packet['inner_folds'].append(fold)
            for index in range(len(model.LAMBDA_GRID)):
                svd = model.predict(grid, x[val], index, 'svd')
                fold['predictions_svd'].append(svd.tolist())
                qr = model.predict(grid, x[val], index, 'qr')
                fold['predictions_qr'].append(qr.tolist())
                fold['all_penalty_score_checks'].append(model.score_agreement(svd, qr))
            mse = model.finite(np.mean((np.asarray(fold['predictions_svd'])-y[val])**2, axis=1), 'CV SVD MSE')
            qr_mse = model.finite(np.mean((np.asarray(fold['predictions_qr'])-y[val])**2, axis=1), 'CV QR MSE')
            fold.update(status='complete', svd_mse=mse.tolist(), qr_mse=qr_mse.tolist())
            cv_svd.append(mse)
            cv_qr.append(qr_mse)
        mean_svd = model.finite(np.mean(cv_svd, axis=0), 'mean-origin SVD MSE')
        mean_qr = model.finite(np.mean(cv_qr, axis=0), 'mean-origin QR MSE')
        selected, selected_qr = int(np.argmin(mean_svd)), int(np.argmin(mean_qr))
        packet.update(mean_origin_mse_svd=mean_svd.tolist(), mean_origin_mse_qr=mean_qr.tolist(),
            selected_index_svd=selected, selected_index_qr=selected_qr,
            selected_penalty_svd=model.LAMBDA_GRID[selected], selected_penalty_qr=model.LAMBDA_GRID[selected_qr])
        final = model.ridge_grid(x, y, groups)
        packet['final_grid'] = final
        for index, penalty in enumerate(model.LAMBDA_GRID):
            check = dict(penalty=penalty, status='grid_complete')
            packet['final_grid_score_checks'].append(check)
            svd = model.predict(final, x, index, 'svd')
            check['predictions_svd'] = svd.tolist()
            qr = model.predict(final, x, index, 'qr')
            check.update(status='complete', predictions_qr=qr.tolist(), **model.score_agreement(svd, qr))
        cv_pass = all(c['passed'] for f in packet['inner_folds'] for c in f['all_penalty_score_checks'])
        final_pass = all(c['passed'] for c in packet['final_grid_score_checks'])
        packet.update(status='complete', numerical_audit=dict(passed=cv_pass and final_pass and selected == selected_qr,
            all_cv_penalty_scores_agree=cv_pass, all_final_penalty_scores_agree=final_pass,
            selected_penalty_agrees=selected == selected_qr,
            coefficient_max_absolute_difference=max(float(np.max(np.abs(np.asarray(f['svd'])-f['qr'])))
                for grid in [final, *[f['grid'] for f in packet['inner_folds']]] for f in grid['fits'])))
    except Exception as error:
        packet.update(status='incomplete', numerical_audit=dict(passed=False), error=repr(error))
        raise FinalFitFailure(str(error), _seal(packet)) from error
    return _seal(packet)


def fit_final_recipe(old_x, old_y, old_rows, feature_names, *, schema, resolution, old_origin_groups):
    """Tune only on all 24 old origins, then fit once on those same old origins.

    No fresh feature, preview, target or label argument is accepted. Registry
    and qualified-label source authorization remain the frozen caller's job.
    A numerical disagreement is retained in a complete packet with failed
    gates; a solver exception raises FinalFitFailure with completed evidence.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError('Unplanned historical label resolution')
    registry = _registry(old_origin_groups)
    x = _matrix(old_x, feature_names, schema)
    groups, rows = _rows(old_rows, x, training_groups=registry)
    y = _labels(old_y, len(x))
    packet = dict(schema_version=PACKET_SCHEMA, recipe=f'{resolution}--{schema}',
        feature_schema=schema, label_resolution=resolution, feature_names=list(feature_names),
        training_groups=registry, training_rows=len(x), training_row_ids=[r['id'] for r in rows],
        training_identity_sha256=model.digest(rows), training_features_sha256=_array_hash(x),
        training_label_sha256=model.digest(y.tolist()), penalty_grid=list(model.LAMBDA_GRID),
        score_atol=model.CHECK_ATOL, score_rtol=model.CHECK_RTOL,
        scope='All training/tuning data are the 24 old physical origins; raw75-unit RMSE, no label/prediction clipping.')
    return _fit_arrays(x, y, groups, packet)


def fit_four_recipes(old_bundle, old_labels_by_resolution, *, old_origin_groups):
    """Exactly four old-data recipes; no fresh-data access or fit-time I/O."""
    allowed = {'schema_version', 'shared_names', 'geometry_names', 'rows', 'shared', 'geometry',
               'source_bindings', 'timings'}
    required = allowed-{'source_bindings', 'timings'}
    if (not required <= set(old_bundle) or set(old_bundle)-allowed
            or old_bundle['schema_version'] != features.SCHEMA_VERSION
            or old_bundle['shared_names'] != list(features.SHARED_FEATURE_NAMES)
            or old_bundle['geometry_names'] != list(features.GEOMETRY_FEATURE_NAMES)
            or set(old_labels_by_resolution) != set(RESOLUTIONS)):
        raise ValueError('Exact historical pre-preview bundle and two old label resolutions required')
    registry = _registry(old_origin_groups)
    shared = _matrix(old_bundle['shared'], old_bundle['shared_names'], 'shared')
    _rows(old_bundle['rows'], shared, training_groups=registry)
    geometry = np.asarray(old_bundle['geometry'])
    if geometry.dtype != np.float64 or geometry.shape != (len(shared), 6) or not np.isfinite(geometry).all():
        raise ValueError('Exact finite six-field geometry matrix required')
    labels = {r: _labels(old_labels_by_resolution[r], len(shared)) for r in RESOLUTIONS}
    packets = {}
    for resolution in RESOLUTIONS:
        for schema in SCHEMAS:
            x = shared if schema == 'shared' else np.column_stack((shared, geometry))
            try:
                packet = fit_final_recipe(x, labels[resolution], old_bundle['rows'], feature_names(schema),
                    schema=schema, resolution=resolution, old_origin_groups=registry)
            except FinalFitFailure as error:
                packet = error.packet
            packets[packet['recipe']] = packet
    return _seal(dict(schema_version='EXP-013-draft-four-final-models-v1', models=packets,
        required_recipes=4, required_cv_folds=96, required_final_grids=4,
        complete=all(p['status'] == 'complete' for p in packets.values()),
        all_model_checks_pass=all(p['numerical_audit']['passed'] for p in packets.values())))


def rank_prepreview(rows, scores):
    """Complete 18-entry context orders and charged 2/4/8 prefixes, no labels."""
    scores = model.finite(scores, 'pre-preview predicted scores')
    if scores.shape != (len(rows),) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Unique complete prediction census required')
    grouped = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[model.context_key(row)].append((float(score), row['candidate'], row['id']))
    result = []
    for context, entries in sorted(grouped.items()):
        if len(entries) != 18 or {r[1] for r in entries} != set(range(18)):
            raise ValueError('Every context requires 18 distinct candidate entries')
        ranked = sorted(entries, key=lambda r: (r[0], r[1]))
        result.append(dict(context=list(context), candidate_order=[r[1] for r in ranked],
            row_id_order=[r[2] for r in ranked], predicted_scores=[r[0] for r in ranked],
            shortlists={str(b): dict(candidates=[r[1] for r in ranked[:b]], ids=[r[2] for r in ranked[:b]],
                cutoff_gap=ranked[b][0]-ranked[b-1][0]) for b in BUDGETS}))
    return result


def predict_prepreview(packet, x, rows, feature_names):
    """Fresh planned context subsets only. No fresh-label argument or loader."""
    unsealed = {k:v for k,v in packet.items() if k != 'packet_sha256'}
    if (packet.get('packet_sha256') != model.digest(unsealed) or packet.get('schema_version') != PACKET_SCHEMA
            or packet.get('status') != 'complete' or not packet.get('numerical_audit', {}).get('passed')):
        raise ValueError('Complete numerically qualified sealed final model required')
    if list(feature_names) != packet['feature_names']:
        raise ValueError('Prediction feature ordering differs from sealed model')
    registry = _registry(packet['training_groups'])
    x = _matrix(x, feature_names, packet['feature_schema'])
    groups, normalized = _rows(rows, x)
    if set(groups) & set(registry):
        raise ValueError('Fresh physical origin aliases historical training data')
    svd = model.predict(packet['final_grid'], x, packet['selected_index_svd'], 'svd')
    qr = model.predict(packet['final_grid'], x, packet['selected_index_qr'], 'qr')
    primary, alternative = rank_prepreview(normalized, svd), rank_prepreview(normalized, qr)
    score_check = model.score_agreement(svd, qr)
    checks = {str(b): all(a['shortlists'][str(b)]['candidates'] == c['shortlists'][str(b)]['candidates']
        and a['shortlists'][str(b)]['ids'] == c['shortlists'][str(b)]['ids']
        for a,c in zip(primary, alternative, strict=True)) for b in BUDGETS}
    return _seal(dict(schema_version=PREDICTION_SCHEMA, model_packet_sha256=packet['packet_sha256'],
        recipe=packet['recipe'], feature_names=list(feature_names), rows=normalized,
        feature_sha256=_array_hash(x), predictions_svd=svd.tolist(), predictions_qr=qr.tolist(),
        rankings_svd=primary, rankings_qr=alternative,
        numerical_audit=dict(passed=score_check['passed'] and all(checks.values()),
            score_agreement=score_check, ordered_shortlists_agree=checks,
            full_ranking_agrees=all(a['candidate_order'] == b['candidate_order']
                for a,b in zip(primary, alternative, strict=True))),
        scope='Pre-preview predicted scores and shortlists only; no selected or full18 fresh labels accessed.'))
