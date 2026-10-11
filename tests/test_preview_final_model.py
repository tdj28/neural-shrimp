"""DRAFT EXP-013 synthetic checks; never read old/fresh research outcomes.

Production-sized census tests replace ridge fitting with a cheap deterministic
stub. Actual SVD/QR arithmetic tests use only 15 synthetic rows/three columns.
"""
from __future__ import annotations

import copy
import builtins
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_final_model as final


def synthetic_bundle(groups=24, *, start=0, full=True):
    rows, shared = [], []
    phases, radii = (final.PHASES, final.RADII) if full else ([675], [.001])
    registry = []
    directions = final.features.direction_candidates('uniform_16') + final.features.direction_candidates(
        'local_signed', local_direction=[1., 0.])
    for origin in range(start, start+groups):
        alpha, gamma = 2.+origin*.01, 30.+origin*.02
        group = final.features.physical_group(alpha, gamma)
        registry.append(group)
        for phase in phases:
            for radius in radii:
                for candidate in range(18):
                    values = np.full(37, candidate/1000.+phase/10000., dtype=np.float64)
                    values[:2] = [alpha, gamma]
                    values[7:9] = [np.sin(2*np.pi*phase/2700), np.cos(2*np.pi*phase/2700)]
                    values[12:15] = [radius, np.log(radius), radius*radius]
                    direction = directions[candidate]['direction']
                    values[15:17] = direction
                    shared.append(values)
                    rows.append(dict(id=f'dummy-g{origin}-p{phase}-r{radius}-c{candidate}',
                        group=group, role='synthetic', anchor=f'dummy-alias-{origin}',
                        phase=phase, radius=radius, candidate=candidate, direction=list(direction)))
    return dict(schema_version=final.features.SCHEMA_VERSION,
        shared_names=list(final.features.SHARED_FEATURE_NAMES),
        geometry_names=list(final.features.GEOMETRY_FEATURE_NAMES), rows=rows,
        shared=np.asarray(shared), geometry=np.zeros((len(rows), 6), dtype=np.float64)), registry


def cheap_grid(x, y, groups, penalties=final.model.LAMBDA_GRID):
    """No decomposition; production-sized tests exercise orchestration only."""
    weights = final.model.equal_origin_weights(groups)
    return dict(mean=(weights@x).tolist(), scale=np.ones(x.shape[1]).tolist(),
        intercept=float(weights@y), exact_constant_columns=[],
        training_groups=sorted(set(groups.tolist())), training_rows=len(y), singular_values=[],
        fits=[dict(penalty=p, svd=np.zeros(x.shape[1]).tolist(), qr=np.zeros(x.shape[1]).tolist())
              for p in penalties])


@pytest.fixture(scope='module')
def complete_packet():
    bundle, registry = synthetic_bundle()
    y = np.asarray([.25+r['candidate']/1000. for r in bundle['rows']], dtype=np.float64)
    labels = {r: y.copy() for r in final.RESOLUTIONS}
    observed = []
    def recorded(*args, **kwargs):
        observed.append((len(args[0]), len(set(args[2]))))
        return cheap_grid(*args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(final.model, 'ridge_grid', recorded)
        packet = final.fit_four_recipes(bundle, labels, old_origin_groups=registry)
    return bundle, registry, labels, packet, observed


def test_four_recipe_complete_fold_final_grid_and_all_penalty_census(complete_packet):
    bundle, registry, labels, packet, observed = complete_packet
    assert packet['complete'] and packet['all_model_checks_pass']
    assert packet['required_cv_folds'] == 96 and packet['required_final_grids'] == 4
    assert list(packet['models']) == [f'{r}--{s}' for r in final.RESOLUTIONS for s in final.SCHEMAS]
    assert observed == [(3312, 23)]*24+[(3456, 24)] + [(3312, 23)]*24+[(3456, 24)] + [(3312, 23)]*24+[(3456, 24)] + [(3312, 23)]*24+[(3456, 24)]
    for item in packet['models'].values():
        assert len(item['inner_folds']) == 24 and len(item['final_grid']['fits']) == 5
        assert len(item['final_grid_score_checks']) == 5
        assert item['selected_index_svd'] == item['selected_index_qr'] == 0
        assert item['selected_penalty_svd'] == item['selected_penalty_qr'] == 1e-6
        assert item['training_groups'] == sorted(registry)
        for fold in item['inner_folds']:
            assert fold['held_group'] not in fold['training_groups']
            assert fold['training_rows'] == 3312 and fold['validation_rows'] == 144
            assert len(fold['grid']['fits']) == len(fold['all_penalty_score_checks']) == 5
            assert np.shape(fold['predictions_svd']) == (5, 144)
            assert np.shape(fold['predictions_qr']) == (5, 144)
        assert item['packet_sha256'] == final.model.digest({k:v for k,v in item.items() if k!='packet_sha256'})


@pytest.mark.parametrize('mutation', ['phase_tags', 'radius_tags', 'candidate_tags', 'missing_direction',
    'unsigned_local', 'local_across_radii'])
def test_complete_but_mislabeled_contexts_or_candidate_directions_are_rejected(monkeypatch, complete_packet, mutation):
    bundle, registry, labels, _, _ = complete_packet
    bundle = copy.deepcopy(bundle)
    if mutation == 'phase_tags':
        for row in bundle['rows']:
            row['phase'] = 1350 if row['phase'] == 675 else 675
    elif mutation == 'radius_tags':
        for row in bundle['rows']:
            row['radius'] = final.RADII[(final.RADII.index(row['radius'])+1)%4]
    elif mutation == 'candidate_tags':
        for row in bundle['rows']:
            if row['candidate'] in (0, 1):
                row['candidate'] = 1-row['candidate']
    elif mutation == 'missing_direction':
        del bundle['rows'][0]['direction']
    elif mutation == 'unsigned_local':
        for index, row in enumerate(bundle['rows']):
            if row['candidate'] == 17:
                row['direction'] = [1., 0.]
                bundle['shared'][index, 15:17] = [1., 0.]
    else:
        for index, row in enumerate(bundle['rows'][:36]):
            if row['radius'] == .001 and row['candidate'] >= 16:
                direction = [0., 1.] if row['candidate'] == 16 else [0., -1.]
                row['direction'] = direction
                bundle['shared'][index, 15:17] = direction
    monkeypatch.setattr(final.model, 'ridge_grid', lambda *_: pytest.fail('Forged identity reached fitting'))
    with pytest.raises(ValueError):
        final.fit_final_recipe(bundle['shared'], labels[final.RESOLUTIONS[0]], bundle['rows'],
            bundle['shared_names'], schema='shared', resolution=final.RESOLUTIONS[0], old_origin_groups=registry)


def tiny_arrays():
    rng = np.random.default_rng(13013)
    x = rng.normal(size=(15, 3)); x[:, 2] = 1.
    y = .7+.03*x[:, 0]-.02*x[:, 1]
    groups = np.repeat(['a', 'b', 'c'], [3, 5, 7])
    return x, y, groups


def test_actual_tiny_cv_uses_train_only_equal_origin_scalers_and_equal_fold_mean():
    x, y, groups = tiny_arrays()
    packet = final._fit_arrays(x, y, groups, {})
    assert packet['numerical_audit']['passed']
    for fold in packet['inner_folds']:
        train = groups != fold['held_group']
        weights = final.model.equal_origin_weights(groups[train])
        assert np.array_equal(fold['grid']['mean'], weights@x[train])
        assert fold['grid']['scale'][2] == 1.
        for group in set(groups[train]):
            assert np.isclose(weights[groups[train] == group].sum(), .5)
    assert np.array_equal(packet['mean_origin_mse_svd'],
                          np.mean([f['svd_mse'] for f in packet['inner_folds']], axis=0))
    changed = y.copy(); changed[groups == 'a'] += .5
    other = final._fit_arrays(x, changed, groups, {})
    assert packet['inner_folds'][0]['grid'] == other['inner_folds'][0]['grid']
    assert packet['inner_folds'][0]['svd_mse'] != other['inner_folds'][0]['svd_mse']


def test_final_all_five_penalty_score_checks_catch_nonselected_solver_disagreement(monkeypatch):
    x, y, groups = tiny_arrays()
    original = final.model.predict
    def changed(grid, values, index, solver='svd'):
        scores = original(grid, values, index, solver)
        if grid['training_rows'] == 15 and index == 4 and solver == 'qr':
            scores = scores+.01
        return scores
    monkeypatch.setattr(final.model, 'predict', changed)
    packet = final._fit_arrays(x, y, groups, {})
    assert packet['status'] == 'complete'
    assert packet['selected_index_svd'] != 4
    assert packet['numerical_audit']['all_cv_penalty_scores_agree']
    assert not packet['numerical_audit']['all_final_penalty_scores_agree']
    assert not packet['numerical_audit']['passed']
    assert packet['final_grid_score_checks'][4]['maximum_absolute_difference'] > .009


def test_selected_penalty_disagreement_is_retained_independently_of_score_tolerance(monkeypatch):
    x, y, groups = tiny_arrays()
    def custom(grid, values, index, solver='svd'):
        # A sub-tolerance difference changes the exact MSE argmin across solvers.
        preferred = 0 if solver == 'svd' else 1
        return np.full(len(values), .7+(0. if index == preferred else 1e-12))
    monkeypatch.setattr(final.model, 'predict', custom)
    packet = final._fit_arrays(x, np.full(len(y), .7), groups, {})
    assert packet['numerical_audit']['all_cv_penalty_scores_agree']
    assert packet['numerical_audit']['all_final_penalty_scores_agree']
    assert packet['selected_index_svd'] == 0 and packet['selected_index_qr'] == 1
    assert not packet['numerical_audit']['passed']


def test_solver_exception_preserves_completed_folds_without_replacement(monkeypatch):
    x, y, groups = tiny_arrays()
    original, calls = final.model.ridge_grid, []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError('synthetic solver failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(final.model, 'ridge_grid', fail_second)
    with pytest.raises(final.FinalFitFailure, match='synthetic solver failure') as raised:
        final._fit_arrays(x, y, groups, {})
    packet = raised.value.packet
    assert packet['status'] == 'incomplete' and not packet['numerical_audit']['passed']
    assert len(packet['inner_folds']) == 1 and packet['final_grid'] is None
    assert packet['packet_sha256'] == final.model.digest({k:v for k,v in packet.items() if k!='packet_sha256'})


def test_prediction_failure_keeps_current_completed_grid_and_partial_predictions(monkeypatch):
    x, y, groups = tiny_arrays()
    original = final.model.predict
    def fail_qr(grid, values, index, solver='svd'):
        if solver == 'qr' and index == 1:
            raise ValueError('synthetic prediction failure')
        return original(grid, values, index, solver)
    monkeypatch.setattr(final.model, 'predict', fail_qr)
    with pytest.raises(final.FinalFitFailure, match='synthetic prediction failure') as raised:
        final._fit_arrays(x, y, groups, {})
    packet = raised.value.packet
    assert len(packet['inner_folds']) == 1
    current = packet['inner_folds'][0]
    assert current['status'] == 'grid_complete' and len(current['grid']['fits']) == 5
    assert len(current['predictions_svd']) == 2 and len(current['predictions_qr']) == 1
    assert len(current['all_penalty_score_checks']) == 1
    assert not packet['numerical_audit']['passed']


@pytest.mark.parametrize('mutation', ['alias', 'duplicate_registry', 'missing_origin', 'missing_candidate',
    'repeated_candidate', 'extra_label_field', 'wrong_names', 'float32', 'nan', 'negative_label',
    'oversize_label', 'wrong_resolution', 'changed_direction'])
def test_public_fit_rejects_leak_or_incomplete_schema_before_any_solver(monkeypatch, complete_packet, mutation):
    original, registry, labels, _, _ = complete_packet
    bundle = copy.deepcopy(original)
    registry = list(registry)
    y = labels[final.RESOLUTIONS[0]].copy()
    names, resolution = bundle['shared_names'], final.RESOLUTIONS[0]
    if mutation == 'alias':
        bundle['rows'][0]['group'] = 'fake-role-alias'
    elif mutation == 'duplicate_registry':
        registry[-1] = registry[0]
    elif mutation == 'missing_origin':
        registry.pop()
    elif mutation == 'missing_candidate':
        bundle['rows'].pop(); bundle['shared'] = bundle['shared'][:-1]; y = y[:-1]
    elif mutation == 'repeated_candidate':
        bundle['rows'][1]['candidate'] = 0
    elif mutation == 'extra_label_field':
        bundle['rows'][0]['preview_rmse'] = .1
    elif mutation == 'wrong_names':
        names = list(reversed(names))
    elif mutation == 'float32':
        bundle['shared'] = bundle['shared'].astype(np.float32)
    elif mutation == 'nan':
        bundle['shared'][0, 5] = np.nan
    elif mutation == 'negative_label':
        y[0] = -.1
    elif mutation == 'oversize_label':
        y[0] = np.nextafter(2., np.inf)
    elif mutation == 'wrong_resolution':
        resolution = 'fresh-labels'
    elif mutation == 'changed_direction':
        bundle['rows'][0]['direction'] = [0., 1.]
    monkeypatch.setattr(final.model, 'ridge_grid', lambda *_: pytest.fail('Invalid data reached fitting'))
    with pytest.raises(ValueError):
        final.fit_final_recipe(bundle['shared'], y, bundle['rows'], names,
            schema='shared', resolution=resolution, old_origin_groups=registry)


def test_rank_ties_preserve_separately_charged_ids_and_exact_budget_prefixes():
    bundle, _ = synthetic_bundle(1, start=100, full=False)
    result = final.rank_prepreview(bundle['rows'], np.ones(18))[0]
    assert result['candidate_order'] == list(range(18))
    for budget in (2, 4, 8):
        entry = result['shortlists'][str(budget)]
        assert entry['candidates'] == list(range(budget)) and len(set(entry['ids'])) == budget
        assert entry['cutoff_gap'] == 0.


def test_fresh_prediction_is_label_free_complete_and_rejects_historical_coordinate_aliases(complete_packet):
    bundle, registry, labels, packet, _ = complete_packet
    trained = packet['models'][f'{final.RESOLUTIONS[0]}--shared']
    fresh, _ = synthetic_bundle(1, start=100, full=False)
    prediction = final.predict_prepreview(trained, fresh['shared'], fresh['rows'], fresh['shared_names'])
    assert prediction['numerical_audit']['passed']
    assert set(prediction['numerical_audit']['ordered_shortlists_agree']) == {'2', '4', '8'}
    assert not any('label' in name for name in inspect.signature(final.predict_prepreview).parameters)
    with pytest.raises(TypeError):
        final.predict_prepreview(trained, fresh['shared'], fresh['rows'], fresh['shared_names'], fresh_labels=[0.])
    alias_rows = copy.deepcopy(bundle['rows'][:18])
    for row in alias_rows:
        row['id'] = 'fresh-looking-'+row['id']; row['role'] = 'fresh'; row['anchor'] = 'new alias'
    with pytest.raises(ValueError, match='aliases historical'):
        final.predict_prepreview(trained, bundle['shared'][:18], alias_rows, bundle['shared_names'])


@pytest.mark.parametrize('position,expected', [(0, {'2': False, '4': False, '8': False}),
    (2, {'2': True, '4': False, '8': False}), (6, {'2': True, '4': True, '8': False}),
    (10, {'2': True, '4': True, '8': True})])
def test_each_ordered_budget_gate_is_independent_of_full_rank_diagnostics(monkeypatch, complete_packet, position, expected):
    packet = complete_packet[3]['models'][f'{final.RESOLUTIONS[0]}--shared']
    fresh, _ = synthetic_bundle(1, start=100, full=False)
    def scores(grid, x, index, solver='svd'):
        values = np.arange(18)*1e-12
        if solver == 'qr':
            values[[position, position+1]] = values[[position+1, position]]
        return values
    monkeypatch.setattr(final.model, 'predict', scores)
    prediction = final.predict_prepreview(packet, fresh['shared'], fresh['rows'], fresh['shared_names'])
    assert prediction['numerical_audit']['score_agreement']['passed']
    assert prediction['numerical_audit']['ordered_shortlists_agree'] == expected
    assert not prediction['numerical_audit']['full_ranking_agrees']
    assert prediction['numerical_audit']['passed'] == all(expected.values())


def test_tampered_or_failed_model_cannot_predict_and_unknown_bundle_fields_rejected(complete_packet):
    bundle, registry, labels, all_models, _ = complete_packet
    packet = copy.deepcopy(all_models['models'][f'{final.RESOLUTIONS[0]}--shared'])
    fresh, _ = synthetic_bundle(1, start=100, full=False)
    packet['selected_index_svd'] = 4
    with pytest.raises(ValueError, match='sealed final model'):
        final.predict_prepreview(packet, fresh['shared'], fresh['rows'], fresh['shared_names'])
    extra = dict(bundle, fresh_preview_scores=[.1])
    with pytest.raises(ValueError, match='pre-preview bundle'):
        final.fit_four_recipes(extra, labels, old_origin_groups=registry)


def test_four_recipe_orchestrator_retains_one_failed_recipe_and_attempts_all_four(monkeypatch, complete_packet):
    bundle, registry, labels, _, _ = complete_packet
    calls = []
    def fit(*args, schema, resolution, **kwargs):
        recipe = f'{resolution}--{schema}'
        calls.append(recipe)
        packet = dict(recipe=recipe, status='complete', numerical_audit=dict(passed=True), inner_folds=[])
        if len(calls) == 1:
            packet.update(status='incomplete', numerical_audit=dict(passed=False), error='synthetic partial fit')
            packet['inner_folds'] = [dict(status='grid_complete', grid={'synthetic': True})]
            raise final.FinalFitFailure('synthetic partial fit', final._seal(packet))
        return final._seal(packet)
    monkeypatch.setattr(final, 'fit_final_recipe', fit)
    packet = final.fit_four_recipes(bundle, labels, old_origin_groups=registry)
    assert list(packet['models']) == calls and len(calls) == 4
    assert not packet['complete'] and not packet['all_model_checks_pass']
    assert packet['models'][calls[0]]['inner_folds'][0]['grid'] == {'synthetic': True}
    assert sum(m['status'] == 'complete' for m in packet['models'].values()) == 3


def test_prediction_reads_no_files_and_preserves_unclipped_negative_scores(monkeypatch, complete_packet):
    packet = complete_packet[3]['models'][f'{final.RESOLUTIONS[0]}--shared']
    fresh, _ = synthetic_bundle(1, start=100, full=False)
    monkeypatch.setattr(final.model, 'predict', lambda grid, x, index, solver='svd': np.linspace(-.2, .2, len(x)))
    def forbidden(*args, **kwargs):
        pytest.fail('Pre-preview prediction attempted file access')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    prediction = final.predict_prepreview(packet, fresh['shared'], fresh['rows'], fresh['shared_names'])
    assert prediction['predictions_svd'][0] == -.2 and prediction['predictions_qr'][0] == -.2
    assert prediction['numerical_audit']['passed']
