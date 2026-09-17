"""EXP-013 policy/statistics tests use fabricated rows/predictions/labels only.

No fitting, integrations, research data, network or runtime source guards.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import fresh_preview_evaluation as evaluation


def reseal(packet):
    packet.pop('packet_sha256', None)
    packet['packet_sha256'] = evaluation.model.digest(packet)
    return packet


@pytest.fixture(scope='module')
def fixture():
    rows, registry = [], []
    for origin in range(24):
        group = evaluation.final.features.physical_group(2.+origin*.01, 30.+origin*.02)
        registry.append(group)
        for phase in evaluation.PHASES:
            for radius in evaluation.RADII:
                for candidate in range(18):
                    direction = evaluation.final.UNIFORM_DIRECTIONS[candidate].tolist() if candidate < 16 else [1. if candidate == 16 else -1., 0.]
                    rows.append(dict(id=f'dummy-{origin}-{phase}-{radius}-{candidate}', group=group,
                        phase=phase, radius=radius, candidate=candidate, direction=direction))
    packets, labels = {}, {}
    for resolution in evaluation.RESOLUTIONS:
        labels[resolution] = {}
        for row in rows:
            origin = registry.index(row['group'])
            # Distinct phase/radius/origin contributions expose weighting errors.
            value = (.01 + origin*.001 + (row['phase'] == 1350)*.02
                     + evaluation.RADII.index(row['radius'])*.003
                     + ((row['candidate']+origin) % 18)*.0007)
            if resolution == evaluation.RESOLUTIONS[1]:
                value += row['candidate']*.0000001
            labels[resolution][row['id']] = dict(status='complete', rmse=value, qualified=True,
                scalar_pass=True, waveform_pass=True, source_id=row['id'])
        for schema in evaluation.final.SCHEMAS:
            recipe = f'{resolution}--{schema}'
            offset = 2 if schema == 'shared' else 9
            values = np.asarray([((row['candidate']+offset) % 18)*.001 for row in rows])
            rankings = evaluation.final.rank_prepreview(rows, values)
            packets[recipe] = reseal(dict(schema_version=evaluation.final.PREDICTION_SCHEMA,
                model_packet_sha256=hashlib.sha256(recipe.encode()).hexdigest(), recipe=recipe,
                feature_names=evaluation.final.feature_names(schema), rows=copy.deepcopy(rows),
                feature_sha256='a'*64, predictions_svd=values.tolist(), predictions_qr=values.tolist(),
                rankings_svd=rankings, rankings_qr=copy.deepcopy(rankings),
                numerical_audit=dict(passed=True, score_agreement=dict(passed=True, maximum_absolute_difference=0.),
                    ordered_shortlists_agree={'2': True, '4': True, '8': True}, full_ranking_agrees=True),
                scope='Synthetic pre-preview predictions only.'))
    policies = evaluation.seal_policies(rows, packets, origin_groups=registry)
    qualification = {k: True for k in evaluation.QUALIFICATION_KEYS}
    result = evaluation.evaluate(policies, labels, qualification=qualification)
    return dict(rows=rows, registry=registry, packets=packets, policies=policies,
                labels=labels, qualification=qualification, result=result)


def test_exact_fixed_policy_catalog_and_endpoint_deduplication():
    catalog = evaluation.policy_catalog()
    assert len(catalog) == 16
    assert [sum(p['budget'] == b for p in catalog) for b in (2, 4, 8, 18)] == [6, 5, 4, 1]
    fixed = {p['id']: p['fixed_candidates'] for p in catalog if p['kind'] == 'fixed'}
    assert fixed == {
        'alpha_pair-b2': [0, 8], 'gamma_pair-b2': [4, 12], 'diagonal_pair-b2': [2, 10],
        'local_pair-b2': [16, 17], 'axes-b4': [0, 4, 8, 12], 'diagonals-b4': [2, 6, 10, 14],
        'local_coverage-b4': [0, 8, 16, 17], 'octagon-b8': [0, 2, 4, 6, 8, 10, 12, 14],
        'local_plus_six-b8': [0, 2, 4, 8, 10, 12, 16, 17], 'full18-b18': list(range(18))}
    assert len({p['id'] for p in catalog}) == 16


def test_complete_6144_census_and_30_fixed_intervals(fixture):
    result = fixture['result']
    assert result['qualified'] and result['complete']
    assert result['retained_readouts'] == result['observed_readouts'] == 6144
    assert len(result['contexts']) == 192
    stats = result['cohort_statistics']
    assert sum(len(s['four_preview_contrasts']) for r in stats['readouts'].values() for s in r.values()) == 30
    assert stats['bootstrap']['seed'] == 20260915
    assert stats['bootstrap']['resamples'] == 20000
    assert stats['bootstrap']['origin_order'] == sorted(fixture['registry'])
    assert result['label_failures'] == result['prediction_failures'] == []


def test_equal_origin_means_and_independent_paired_resampling(fixture):
    rng = np.random.default_rng(20260915)
    draw = rng.integers(0, 24, (20000, 24))
    stats = fixture['result']['cohort_statistics']
    assert stats['bootstrap']['shared_indices_sha256'] == hashlib.sha256(np.asarray(draw, dtype='<i8').tobytes()).hexdigest()
    # Independent direct scalar reconstruction bypasses evaluator context scores.
    definitions = {p['id']: p for p in evaluation.policy_catalog()}
    for resolution in evaluation.RESOLUTIONS:
        for stratum, radius in [('pooled', None), *[(str(r), r) for r in evaluation.RADII]]:
            original = stats['readouts'][resolution][stratum]
            policy_means = {p: [] for p in definitions}
            for group in sorted(fixture['registry']):
                for policy in definitions:
                    losses, excess, included = [], [], []
                    for phase in evaluation.PHASES:
                        for r in evaluation.RADII if radius is None else [radius]:
                            rows = [v for v in fixture['rows'] if (v['group'], v['phase'], v['radius']) == (group, phase, r)]
                            values = {v['candidate']: fixture['labels'][resolution][v['id']]['rmse'] for v in rows}
                            definition = definitions[policy]
                            if definition['kind'] == 'fixed':
                                selected = definition['fixed_candidates']
                            else:
                                offset = 2 if definition['feature_schema'] == 'shared' else 9
                                selected = sorted(range(18), key=lambda c: (((c+offset) % 18)*.001, c))[:definition['budget']]
                            chosen = min(selected, key=lambda c: (values[c], c))
                            oracle = min(range(18), key=lambda c: (values[c], c))
                            losses.append(values[chosen]); excess.append(values[chosen]-values[oracle]); included.append(oracle in selected)
                    policy_means[policy].append(float(np.mean(losses)))
                    observed_origin = next(o for o in original['per_origin'] if o['group'] == group)
                    assert observed_origin['policies'][policy]['rmse'] == np.mean(losses)
                    assert observed_origin['policies'][policy]['excess_rmse'] == np.mean(excess)
                    assert observed_origin['policies'][policy]['winner_inclusion'] == np.mean(included)
            for policy, values in policy_means.items():
                assert original['policies'][policy]['rmse'] == np.mean(values)
            for comparison in ('shared-b4', 'axes-b4', 'local_coverage-b4'):
                effects = np.asarray(policy_means['geometry-b4']) - policy_means[comparison]
                target = original['four_preview_contrasts'][f'geometry-b4_minus_{comparison}']
                assert target['mean'] == np.mean(effects)
                assert target['descriptive_central_95_interval'] == np.percentile(np.mean(effects[draw], axis=1), [2.5, 97.5]).tolist()
                assert [x['effect'] for x in target['per_origin']] == effects.tolist()
                assert sum(target['signs'].values()) == 24


def test_selected_only_ties_choose_candidate_id_and_reject_oracle_access():
    rows = [dict(id='eight', candidate=8), dict(id='zero', candidate=0)]
    result = evaluation.observe_selected(rows, {'eight': .1, 'zero': .1})
    assert result['chosen_candidate'] == 0 and result['chosen_id'] == 'zero'
    with pytest.raises(ValueError, match='selected-only'):
        evaluation.observe_selected(rows, {'eight': .1, 'zero': .1, 'not-selected': .01})
    with pytest.raises(ValueError):
        evaluation.observe_selected([dict(id='a', candidate=0)]*2, {'a': .1})


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1, 2.1, True, '0.1'])
def test_selected_only_refuses_invalid_rmse(bad):
    with pytest.raises(ValueError):
        evaluation.observe_selected([dict(id='a', candidate=0), dict(id='b', candidate=8)], {'a': bad, 'b': .1})


def test_exact_winner_vs_any_minimizer_inclusion(fixture):
    context = fixture['policies']['contexts'][0]
    labels = {row_id: dict(status='complete', qualified=True, rmse=.1) for row_id in context['candidate_ids']}
    policies = context['policies'][evaluation.RESOLUTIONS[0]]
    scored, _, order = evaluation._observe_full(context, labels, policies)
    assert order == list(range(18))
    diagonal = scored['diagonal_pair-b2']['observed']
    assert diagonal['chosen_candidate'] == 2
    assert not diagonal['winner_inclusion'] and diagonal['any_minimizer_inclusion']
    assert diagonal['excess_rmse'] == 0.
    assert diagonal['required_preview_calls'] == 2


@pytest.mark.parametrize('mutation', ['missing-row', 'duplicate-row', 'alias-group', 'unplanned-phase', 'label-field'])
def test_planned_census_and_physical_alias_guard(fixture, mutation):
    rows = copy.deepcopy(fixture['rows'])
    registry = fixture['registry'].copy()
    if mutation == 'missing-row':
        rows.pop()
    elif mutation == 'duplicate-row':
        rows[-1] = copy.deepcopy(rows[0]); rows[-1]['id'] = 'different-id'
    elif mutation == 'alias-group':
        registry[-1] = registry[0]
    elif mutation == 'unplanned-phase':
        rows[0]['phase'] = 1
    else:
        rows[0]['preview_rmse'] = .1
    with pytest.raises(ValueError):
        evaluation.seal_policies(rows, {}, origin_groups=registry)


def test_missing_labels_preserved_no_partial_cohort_mean(fixture):
    labels = copy.deepcopy(fixture['labels'])
    removed = fixture['rows'][0]['id']
    del labels[evaluation.RESOLUTIONS[0]][removed]
    result = evaluation.evaluate(fixture['policies'], labels, qualification=fixture['qualification'])
    assert not result['complete'] and not result['qualified']
    assert result['retained_readouts'] == 6144 and result['cohort_statistics'] is None
    assert result['label_failures'][0]['id'] == removed
    assert result['label_failures'][0]['status'] == 'missing'
    assert 0 < result['observed_readouts'] < 6144
    assert len(result['contexts']) == 192


@pytest.mark.parametrize('gate', evaluation.QUALIFICATION_KEYS)
def test_every_required_external_failure_blocks_all_aggregates(fixture, gate):
    qualification = fixture['qualification'].copy()
    qualification[gate] = False
    result = evaluation.evaluate(fixture['policies'], fixture['labels'], qualification=qualification)
    assert not result['qualified'] and result['cohort_statistics'] is None
    assert result['observed_readouts'] == result['retained_readouts'] == 6144
    assert result['qualification'][gate] is False


def test_qualification_missing_or_truthy_nonboolean_refused(fixture):
    for qualification in ({}, dict(fixture['qualification'], native_shams=1)):
        with pytest.raises(ValueError):
            evaluation.evaluate(fixture['policies'], fixture['labels'], qualification=qualification)


def test_failed_numerical_label_preserves_flags_even_when_scalar_available(fixture):
    labels = copy.deepcopy(fixture['labels'])
    row_id = fixture['rows'][0]['id']
    labels[evaluation.RESOLUTIONS[0]][row_id].update(qualified=False, waveform_pass=False,
        scalar_nonincrease=False, failure_reason='synthetic numerical disagreement')
    result = evaluation.evaluate(fixture['policies'], labels, qualification=fixture['qualification'])
    assert result['complete'] and not result['qualified']
    assert result['cohort_statistics'] is None and result['observed_readouts'] == 6144
    evidence = result['label_failures'][0]['evidence']
    assert evidence['waveform_pass'] is False and evidence['scalar_nonincrease'] is False
    assert evidence['failure_reason'] == 'synthetic numerical disagreement'


def test_nonfinite_label_is_retained_as_failure_not_filtered(fixture):
    labels = copy.deepcopy(fixture['labels'])
    row_id = fixture['rows'][0]['id']
    labels[evaluation.RESOLUTIONS[0]][row_id]['rmse'] = float('nan')
    result = evaluation.evaluate(fixture['policies'], labels, qualification=fixture['qualification'])
    assert not result['qualified'] and result['cohort_statistics'] is None
    assert result['label_failures'][0]['evidence']['rmse'] == 'nan'
    evaluation._verify_hash(result)


def test_missing_predictions_keep_fixed_and_declared_learned_rows(fixture):
    policies = evaluation.seal_policies(fixture['rows'], {}, origin_groups=fixture['registry'])
    result = evaluation.evaluate(policies, fixture['labels'], qualification=fixture['qualification'])
    assert not result['qualified'] and not result['complete']
    assert len(result['prediction_failures']) == 4
    assert result['observed_readouts'] == 10*192*2
    assert result['retained_readouts'] == 6144 and result['cohort_statistics'] is None
    first = result['contexts'][0]['resolutions'][evaluation.RESOLUTIONS[0]]['policies']
    assert first['shared-b4']['observed'] is None
    assert first['axes-b4']['observed'] is not None


def test_prediction_numerical_disagreement_retained_without_fallback(fixture):
    packets = copy.deepcopy(fixture['packets'])
    packet = packets[f'{evaluation.RESOLUTIONS[0]}--shared']
    packet['predictions_qr'][0] += 1.
    packet['rankings_qr'] = evaluation.final.rank_prepreview(packet['rows'], packet['predictions_qr'])
    a, b = packet['rankings_svd'], packet['rankings_qr']
    check = evaluation.model.score_agreement(packet['predictions_svd'], packet['predictions_qr'])
    prefixes = {str(budget): all(x['shortlists'][str(budget)]['ids'] == y['shortlists'][str(budget)]['ids'] for x, y in zip(a, b))
                for budget in (2, 4, 8)}
    packet['numerical_audit'] = dict(passed=False, score_agreement=check,
        ordered_shortlists_agree=prefixes, full_ranking_agrees=False)
    reseal(packet)
    policies = evaluation.seal_policies(fixture['rows'], packets, origin_groups=fixture['registry'])
    assert policies['complete'] and not policies['numerical_passed']
    result = evaluation.evaluate(policies, fixture['labels'], qualification=fixture['qualification'])
    assert result['complete'] and not result['qualified']
    assert result['observed_readouts'] == 6144 and result['cohort_statistics'] is None
    assert result['prediction_failures'][0]['numerical_passed'] is False


@pytest.mark.parametrize('mutation', ['hash', 'forged-ranking', 'forged-audit', 'label-descendant', 'row-identity', 'duplicate'])
def test_prediction_seal_reconstructs_not_trusts_reported_checks(fixture, mutation):
    packets = copy.deepcopy(fixture['packets'])
    key = f'{evaluation.RESOLUTIONS[0]}--shared'
    packet = packets[key]
    if mutation == 'hash':
        packet['predictions_svd'][0] += 1.
    else:
        if mutation == 'forged-ranking':
            packet['rankings_svd'][0]['candidate_order'] = list(reversed(range(18)))
        elif mutation == 'forged-audit':
            packet['numerical_audit']['passed'] = False
        elif mutation == 'label-descendant':
            packet['fresh_labels'] = [1.]*3456
        elif mutation == 'row-identity':
            packet['rows'][0]['radius'] = .016
        elif mutation == 'duplicate':
            packets[key] = [packet, packet]
        reseal(packet)
    with pytest.raises(ValueError):
        evaluation.seal_policies(fixture['rows'], packets, origin_groups=fixture['registry'])


def make_winner_seal(fixture):
    winners = []
    for context in fixture['policies']['contexts']:
        for resolution in evaluation.RESOLUTIONS:
            for policy, selection in context['policies'][resolution].items():
                if selection['budget'] == 18:
                    continue
                rows = [dict(id=context['candidate_ids'][c], candidate=c) for c in selection['selected']]
                values = {r['id']: fixture['labels'][resolution][r['id']]['rmse'] for r in rows}
                winners.append(dict(context=context['context'], resolution=resolution, policy=policy,
                    winner=evaluation.observe_selected(rows, values)))
    return reseal(dict(policy_seal_sha256=fixture['policies']['packet_sha256'], winners=winners))


def test_complete_selected_winner_seal_replay_and_bad_winner_failure(fixture):
    winner_seal = make_winner_seal(fixture)
    assert len(winner_seal['winners']) == 5760
    result = evaluation.evaluate(fixture['policies'], fixture['labels'], qualification=fixture['qualification'], winner_seal=winner_seal)
    assert result['qualified']
    assert result['selected_winner_replay']['complete']
    assert result['selected_winner_replay']['checked'] == 5760
    winner_seal['winners'][0]['winner']['rmse'] += .0001
    reseal(winner_seal)
    result = evaluation.evaluate(fixture['policies'], fixture['labels'], qualification=fixture['qualification'], winner_seal=winner_seal)
    assert not result['qualified'] and result['cohort_statistics'] is None
    assert len(result['selected_winner_replay']['failures']) == 1


def test_missing_and_duplicate_winner_rows_cannot_qualify(fixture):
    seal = make_winner_seal(fixture)
    seal['winners'].pop()
    reseal(seal)
    result = evaluation.evaluate(fixture['policies'], fixture['labels'], qualification=fixture['qualification'], winner_seal=seal)
    assert not result['qualified'] and result['cohort_statistics'] is None
    assert len(result['selected_winner_replay']['missing']) == 1
    seal['winners'].append(seal['winners'][0])
    reseal(seal)
    with pytest.raises(ValueError, match='duplicated'):
        evaluation.evaluate(fixture['policies'], fixture['labels'], qualification=fixture['qualification'], winner_seal=seal)


def test_resolution_diagnostics_match_labels_and_keep_signs(fixture):
    result = fixture['result']
    for context in result['contexts']:
        primary, comparison = [context['resolutions'][r] for r in evaluation.RESOLUTIONS]
        delta = np.asarray(primary['candidate_rmse']) - comparison['candidate_rmse']
        diagnostic = context['numerical_resolution']
        assert diagnostic['signed_label_changes'] == delta.tolist()
        assert diagnostic['empirical_max_absolute_score_change'] == np.max(np.abs(delta))
        assert diagnostic['envelope_is_certified_error_bound'] is False
        assert diagnostic['full18_minimum_signed_change'] == min(primary['candidate_rmse'])-min(comparison['candidate_rmse'])
        for policy, change in diagnostic['policies'].items():
            a, b = primary['policies'][policy]['observed'], comparison['policies'][policy]['observed']
            assert change['selected_error_signed_change'] == a['rmse']-b['rmse']
            assert change['selected_action_changed'] == (a['chosen_candidate'] != b['chosen_candidate'])


def test_cost_dependencies_and_non_latency_scope():
    cost = evaluation.cost_contract()
    definitions = {p['id']: p for p in evaluation.policy_catalog()}
    for policy in cost['policies']:
        definition = definitions[policy['id']]
        assert policy['preview_calls'] == definition['budget']
        assert ('300-unit unperturbed sensitivity' in policy['prerequisites']) == definition['needs_sensitivity']
        assert ('historical EXP-004 atlas' in policy['prerequisites']) == definition['needs_atlas']
    assert definitions['shared-b2']['needs_sensitivity'] and definitions['full18-b18']['needs_sensitivity']
    assert not definitions['axes-b4']['needs_sensitivity']
    assert cost['amortization_horizons'] == [1, 8, 192]
    assert cost['acquisition']['total_short_traces'] == 10608
    assert cost['duplicated_physical_entries_stay_charged']
    assert not cost['selected_full_batch_cost_is_measured_policy_latency'] and not cost['net_speedup_claim']


def test_module_exposes_no_file_or_fit_capability(monkeypatch, fixture):
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure evaluator attempted I/O or fitting')
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(evaluation.model, 'ridge_grid', forbidden)
    policies = evaluation.seal_policies(fixture['rows'], {}, origin_groups=fixture['registry'])
    result = evaluation.evaluate(policies, {}, qualification=fixture['qualification'])
    assert not result['qualified'] and result['cohort_statistics'] is None
    assert 'labels' not in inspect.signature(evaluation.seal_policies).parameters
    assert 'selected_labels' in inspect.signature(evaluation.observe_selected).parameters


def test_failed_nested_nonfinite_diagnostics_preserved_in_strict_json(fixture):
    labels = copy.deepcopy(fixture['labels'])
    row_id = fixture['rows'][0]['id']
    labels[evaluation.RESOLUTIONS[0]][row_id].update(status='failed', qualified=False,
        diagnostics={'activation': np.array([float('inf'), float('nan')]), 'passed': False})
    result = evaluation.evaluate(fixture['policies'], labels, qualification=fixture['qualification'])
    assert not result['qualified'] and result['cohort_statistics'] is None
    retained = result['label_failures'][0]['evidence']['diagnostics']
    assert retained == {'activation': [{'nonfinite_float': 'inf'}, {'nonfinite_float': 'nan'}], 'passed': False}
    evaluation._verify_hash(result)


def test_predicted_exact_ties_are_candidate_id_order(fixture):
    packets = copy.deepcopy(fixture['packets'])
    for packet in packets.values():
        packet['predictions_svd'] = packet['predictions_qr'] = [0.]*3456
        packet['rankings_svd'] = evaluation.final.rank_prepreview(packet['rows'], packet['predictions_svd'])
        packet['rankings_qr'] = copy.deepcopy(packet['rankings_svd'])
        reseal(packet)
    policies = evaluation.seal_policies(fixture['rows'], packets, origin_groups=fixture['registry'])
    for context in policies['contexts']:
        for resolution in evaluation.RESOLUTIONS:
            for schema in ('shared', 'geometry'):
                for budget in (2, 4, 8):
                    assert context['policies'][resolution][f'{schema}-b{budget}']['selected'] == list(range(budget))
    assert policies['numerical_passed']


def test_solver_rank_change_beyond_top8_is_diagnostic_not_failed_action_gate(fixture):
    packets = copy.deepcopy(fixture['packets'])
    packet = packets[f'{evaluation.RESOLUTIONS[0]}--shared']
    a = np.asarray(packet['predictions_svd'])*1e-9
    b = a.copy()
    b[6], b[7] = b[7], b[6]
    packet['predictions_svd'], packet['predictions_qr'] = a.tolist(), b.tolist()
    packet['rankings_svd'] = evaluation.final.rank_prepreview(packet['rows'], a)
    packet['rankings_qr'] = evaluation.final.rank_prepreview(packet['rows'], b)
    packet['numerical_audit'].update(full_ranking_agrees=False, score_agreement=evaluation.model.score_agreement(a, b))
    reseal(packet)
    policies = evaluation.seal_policies(fixture['rows'], packets, origin_groups=fixture['registry'])
    assert policies['numerical_passed'] and not policies['failures']
    evidence = policies['prediction_evidence'][packet['recipe']]['packets'][0]['numerical_audit']
    assert not evidence['full_ranking_agrees']
    assert all(evidence['ordered_shortlists_agree'].values()) and evidence['passed']


def test_qualified_resolution_argmin_change_retained_without_policy_switch(fixture):
    labels = copy.deepcopy(fixture['labels'])
    context = fixture['policies']['contexts'][0]
    ids = context['candidate_ids']
    for resolution in evaluation.RESOLUTIONS:
        for row_id in ids:
            labels[resolution][row_id]['rmse'] = .1
    labels[evaluation.RESOLUTIONS[0]][ids[1]]['rmse'] -= 1e-7
    result = evaluation.evaluate(fixture['policies'], labels, qualification=fixture['qualification'])
    assert result['qualified']
    changed = next(c for c in result['contexts'] if c['context'] == context['context'])['numerical_resolution']
    assert changed['full18_winner_changed'] and changed['full_ranking_changed']
    assert changed['full18_minimum_signed_change'] < 0
    assert result['cohort_statistics']['bootstrap']['interval_count'] == 30


@pytest.mark.parametrize('mutation', ['wrong-direction', 'bool-candidate', 'bool-phase', 'bool-direction'])
def test_prediction_realized_direction_and_strict_scalar_identity(fixture, mutation):
    packets = copy.deepcopy(fixture['packets'])
    packet = packets[f'{evaluation.RESOLUTIONS[0]}--shared']
    if mutation == 'wrong-direction': packet['rows'][0]['direction'] = [999., 999.]
    elif mutation == 'bool-candidate': packet['rows'][1]['candidate'] = True
    elif mutation == 'bool-phase': packet['rows'][0]['phase'] = True
    else: packet['rows'][0]['direction'] = [True, False]
    reseal(packet)
    with pytest.raises(ValueError, match='Prediction identity'):
        evaluation.seal_policies(fixture['rows'], packets, origin_groups=fixture['registry'])


def test_prediction_requires_realized_directions_but_unavailable_ledger_does_not(fixture):
    rows = [{k: v for k, v in row.items() if k != 'direction'} for row in fixture['rows']]
    empty = evaluation.seal_policies(rows, {}, origin_groups=fixture['registry'])
    assert not empty['complete']
    with pytest.raises(ValueError, match='Prediction identity'):
        evaluation.seal_policies(rows, fixture['packets'], origin_groups=fixture['registry'])
    assert all(row['direction'] == fixture['rows'][next(i for i, source in enumerate(fixture['rows']) if source['id'] == row['id'])]['direction']
               for row in fixture['policies']['rows'])


@pytest.mark.parametrize('mutation', ['partial-directions', 'wrong-uniform', 'wrong-local-pair', 'radius-direction-change'])
def test_planned_realized_direction_census_is_bound(fixture, mutation):
    rows = copy.deepcopy(fixture['rows'])
    if mutation == 'partial-directions': del rows[0]['direction']
    elif mutation == 'wrong-uniform': rows[0]['direction'] = [0., 1.]
    elif mutation == 'wrong-local-pair':
        for row in rows:
            if row['candidate'] == 17: row['direction'] = [0., 1.]
    else: rows[16]['direction'] = [.6, .8]
    with pytest.raises(ValueError):
        evaluation.seal_policies(rows, {}, origin_groups=fixture['registry'])


@pytest.fixture(scope='module')
def costs(fixture):
    values = dict(preview_task_seconds={resolution: {r['id']: 2. for r in fixture['rows']} for resolution in evaluation.RESOLUTIONS},
        reference_task_seconds={g: 8. for g in fixture['registry']},
        sensitivity_task_seconds={(g, p): 4. for g in fixture['registry'] for p in evaluation.PHASES},
        final_recipe_seconds={recipe: 192. for recipe in fixture['packets']}, joint_setup_seconds=96.)
    return values, evaluation.amortized_costs(fixture['policies'], **values)


def test_fixed_block_amortization_exact_scopes_and_formulas(costs):
    _, costs = costs
    for resolution in evaluation.RESOLUTIONS:
        result = costs['accounting'][resolution]
        for horizon, blocks in [(1, 192), (8, 24), (192, 1)]:
            h = str(horizon)
            assert result['axes-b4'][h]['blocks'] == blocks
            assert result['axes-b4'][h]['full_total_mean_per_decision_seconds'] == (16. if horizon == 1 else 9.)
            assert result['local_coverage-b4'][h]['full_total_mean_per_decision_seconds'] == (20. if horizon == 1 else 10.)
            assert result['full18-b18'][h]['full_total_mean_per_decision_seconds'] == (48. if horizon == 1 else 38.)
            learned = result['shared-b4'][h]
            assert learned['known_subtotal_mean_per_decision_seconds'] == (20. if horizon == 1 else 10.) + 288./horizon
            assert learned['full_total_mean_per_decision_seconds'] is None
            assert not learned['all_required_costs_known']
            block = learned['accounting_blocks'][0]
            assert block['components']['reference_per_origin']['required_items'] == (24 if horizon == 192 else 1)
            assert block['components']['sensitivity_per_origin_phase']['required_items'] == {1: 1, 8: 2, 192: 48}[horizon]
            assert block['components']['final_model_once']['required_items'] == 1
            assert block['components']['joint_setup_once']['total_seconds'] == 96.
            assert block['required_preview_calls'] == 4*horizon
            assert [x['component'] for x in block['unknown_components']] == ['historical_training_label_creation']
            geometry = result['geometry-b4'][h]['accounting_blocks'][0]
            assert [x['component'] for x in geometry['unknown_components']] == ['historical_training_label_creation', 'historical_atlas_creation']
    assert costs['measured_policy_latency'] is False and costs['historical_reuse_is_zero_cost'] is False


def test_missing_prerequisite_cost_stays_unknown_not_implicitly_zero(fixture, costs):
    values, _ = costs
    values = copy.deepcopy(values)
    first = sorted(fixture['registry'])[0]
    del values['reference_task_seconds'][first]
    values['final_recipe_seconds'] = None
    values['joint_setup_seconds'] = None
    result = evaluation.amortized_costs(fixture['policies'], **values)['accounting'][evaluation.RESOLUTIONS[0]]
    axes = result['axes-b4']['192']
    assert axes['full_total_mean_per_decision_seconds'] is None
    assert axes['known_subtotal_mean_per_decision_seconds'] == (1536.+23*8)/192
    assert axes['accounting_blocks'][0]['unknown_components'] == [dict(component='reference_per_origin', key=first)]
    learned = result['shared-b4']['1']['accounting_blocks'][0]
    assert {'final_model_once', 'joint_setup_once', 'historical_training_label_creation'} <= {x['component'] for x in learned['unknown_components']}


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf'), True, '1'])
def test_cost_accounting_refuses_nonfinite_or_invalid_time(fixture, costs, bad):
    values = dict(costs[0], joint_setup_seconds=bad)
    with pytest.raises(ValueError, match='nonnegative finite'):
        evaluation.amortized_costs(fixture['policies'], **values)


def test_cost_accounting_rejects_unplanned_task_identity(fixture, costs):
    values = copy.deepcopy(costs[0])
    values['preview_task_seconds'][evaluation.RESOLUTIONS[0]]['unplanned'] = 1.
    with pytest.raises(ValueError, match='Unplanned cost'):
        evaluation.amortized_costs(fixture['policies'], **values)
