"""Synthetic regeneration/dataflow failures; never execute against EXP-008."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import audit_information_regeneration as audit


def wave_fixture(duration=75):
    period = 12.
    phase = np.arange(2700) / 2700
    reference = np.column_stack((.4*np.sin(2*np.pi*phase), .2*np.cos(2*np.pi*phase),
        .3*np.sin(4*np.pi*phase)))
    times = np.arange(round(duration/.1)+1)*.1
    times[-1] = duration
    epoch = 3.
    values = audit.periodic_interpolate(reference, (times+epoch)/period)
    initial = np.arctanh(values[0])
    local = np.array([1., 0.])
    context = dict(reference=reference, T=period, epoch=epoch, initial=initial,
        certificate={}, gram={'marker':7}, directions=audit.driver.candidates(local))
    anchor = dict(id='synthetic', alpha=1.06, gamma=10.3)
    radius = .001
    raw = dict(times=times, states=np.arctanh(values), initial=initial,
        delta=np.array([radius, 0.]), alpha=anchor['alpha']+radius, gamma=anchor['gamma'], dt=.005)
    return context, anchor, raw, radius


def test_comparison_enumerates_all_failures_and_uses_original_combined_bound():
    log = audit.Recorder()
    log.compare(dict(x=[.2, .3], identity=2, missing=7),
        dict(x=[.21, .31, .4], identity=3), 'model', 'root', **audit.MODEL_ARITHMETIC)
    assert len(log.failures) == 5  # Keys, length, both coefficients, identity.
    numeric = [f for f in log.failures if f['kind'] == 'arithmetic']
    assert len(numeric) == 2
    assert numeric[0]['allowed_difference'] == 1e-9 + 1e-7*.21
    assert log.counts['model']['numeric'] == 2


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_identical_nonfinite_numeric_values_never_pass_as_serialized_strings(value):
    log = audit.Recorder()
    log.compare(dict(values=np.asarray([value])), dict(values=np.asarray([value])), 'model', 'root')
    assert len(log.failures) == 1
    assert log.failures[0]['kind'] == 'nonfinite_or_type'
    assert log.failures[0]['path'] == 'root.values[0]'
    assert log.counts['model']['numeric'] == 1


def test_census_preserves_missing_duplicate_unplanned_and_order_failures():
    log = audit.Recorder()
    rows = [{'id':'a'}, {'id':'a'}, {'id':'unexpected'}]
    found = audit.census(rows, ['a', 'missing'], log, 'test', ordered=True)
    assert found == {}
    assert counter_kinds(log) == {'unexpected_id':1, 'missing_or_duplicate':2, 'identity':1}


def counter_kinds(log):
    from collections import Counter
    return dict(Counter(f['kind'] for f in log.failures))


def test_fresh_phase_is_used_while_equivalent_saved_phase_retains_frozen_check():
    context, _, raw, _ = wave_fixture()
    times, values = raw['times']+context['epoch'], np.tanh(raw['states'])
    original = audit.driver.waveform_errors(context['reference'], times, values, context['T'])
    saved = copy.deepcopy(original)
    saved['phase_offset_cycles'] += 1e-9
    errors = values-audit.periodic_interpolate(context['reference'],
        times/context['T']+saved['phase_offset_cycles'])
    saved['aligned_rms'] = audit._rms(errors)
    for window in saved['windows']:
        mask = (times >= window['first_sample_time']) & (times <= window['last_sample_time'])
        window['aligned_rms'] = audit._rms(errors[mask])
    saved['phase_search']['refinement_evaluations'] += 1
    log = audit.Recorder()
    fresh, detail = audit.regenerate_metrics(saved, context['reference'], times, values, context['T'], log, 'fixture')
    assert not log.failures
    assert fresh['phase_offset_cycles'] == original['phase_offset_cycles']
    assert fresh['phase_offset_cycles'] != saved['phase_offset_cycles']
    assert detail['optimizer_evaluations']['saved'] != detail['optimizer_evaluations']['fresh']
    assert detail['saved_phase_recomputed_rmse'] == saved['aligned_rms']['aggregate']


def test_metric_comparison_reports_multiple_corrupt_fields_without_discarding_fresh_output():
    context, _, raw, _ = wave_fixture()
    times, values = raw['times']+context['epoch'], np.tanh(raw['states'])
    saved = audit.driver.waveform_errors(context['reference'], times, values, context['T'])
    saved['synchronous_rms']['aggregate'] = .8
    saved['aligned_rms']['per_component'][1] = .9
    log = audit.Recorder()
    fresh, _ = audit.regenerate_metrics(saved, context['reference'], times, values, context['T'], log, 'fixture')
    assert fresh['aligned_rms']['aggregate'] < 1e-10
    assert {f['path'] for f in log.failures} == {
        'fixture.synchronous_rms.aggregate', 'fixture.aligned_rms.per_component[1]'}


def test_preview_features_receive_new_metrics_and_gram_not_saved_feature_values(monkeypatch):
    context, anchor, raw, radius = wave_fixture()
    original = audit.driver.waveform_errors(context['reference'], raw['times']+context['epoch'],
        np.tanh(raw['states']), context['T'])
    saved = copy.deepcopy(original)
    saved['aligned_rms']['aggregate'] = .75
    row = dict(id='synthetic-c00', role='development', anchor=anchor['id'], block='block',
        phase=675, radius=radius, candidate=0, status='complete', raw='unused',
        direction=[1., 0.], delta=[radius, 0.], alpha=raw['alpha'], gamma=raw['gamma'],
        metrics=saved, base=[999.], geometry=[1.], geometry_diagnostics={}, preview_score=.75)
    monkeypatch.setattr(audit.driver, 'geometry_features', lambda *a:dict(values=[1.], diagnostics={}))
    seen = {}
    def build_features(anchor, phase, direction, radius, cert, gram, metrics):
        seen.update(gram=gram, metrics=metrics)
        return [metrics['aligned_rms']['aggregate']]
    monkeypatch.setattr(audit.driver, 'base_features', build_features)
    log = audit.Recorder()
    result, _ = audit.regenerate_preview(row, raw, anchor, context, None, log)
    assert seen['gram'] is context['gram']
    assert seen['metrics'] is result['metrics']
    assert result['base'] == [result['preview_score']]
    assert result['preview_score'] < 1e-10
    assert row['base'] == [999.] and row['preview_score'] == .75
    assert len(log.failures) >= 2


def test_target_regenerates_tail_label_and_rejects_changed_raw_intervention():
    context, anchor, raw, radius = wave_fixture(300)
    mask = raw['times'] > 75
    metrics = audit.driver.waveform_errors(context['reference'], raw['times'][mask]+context['epoch'],
        np.tanh(raw['states'][mask]), context['T'])
    row = dict(id='synthetic-c00-dt0.005', role='development', anchor=anchor['id'], block='block',
        phase=675, radius=radius, candidate=0, dt=.005, status='complete',
        direction=[1., 0.], delta=[radius, 0.], alpha=raw['alpha'], gamma=raw['gamma'],
        phase_epoch=context['epoch'], rmse=.7, tail_metrics=metrics)
    log = audit.Recorder()
    result, _ = audit.regenerate_target(row, raw, anchor, context, log)
    assert result['rmse'] < 1e-10 and row['rmse'] == .7
    assert any(f['path'].endswith('saved_score_identity') for f in log.failures)
    changed = copy.deepcopy(raw)
    changed['delta'][0] += 1e-8
    changed['alpha'] += 1e-8
    log = audit.Recorder()
    with pytest.raises(ValueError, match='declared realized intervention'):
        audit.regenerate_target(row, changed, anchor, context, log)
    assert {f['path'].rsplit('.', 1)[-1] for f in log.failures} == {'delta', 'parameters'}


def test_sensitivity_readouts_are_derived_from_saved_states_not_saved_activations():
    context, anchor, _, _ = wave_fixture(300)
    orbit = np.vstack((np.arctanh(context['reference']), np.arctanh(context['reference'][:1])))
    cert = dict(orbit_states=orbit, orbit_activations=np.tanh(orbit), period=12., transverse_spectral_radius=.2)
    times = np.linspace(0, 300, 3001)
    states = np.tile(orbit[675], (3001, 1))
    state_sens = np.zeros((3001, 3, 2))
    state_sens[:, 0, 0] = times
    state_sens[:, 1, 1] = 2*times
    deriv = 1-np.tanh(states)**2
    activation_sens = deriv[:, :, None]*state_sens
    velocity = deriv*np.array([audit.hopfield_rhs(t, x, anchor['alpha'], anchor['gamma']) for t, x in zip(times, states)])
    gram = audit.driver.phase_projected_gram(activation_sens[times>75], velocity[times>75])
    raw = dict(times=times, states=states, state_sensitivity=state_sens,
        activation_sensitivity=activation_sens.copy(), activation_velocity=velocity.copy(),
        tail_gram=gram['gram'], tail_direction=gram['minimum_eigenvector'])
    raw['activation_sensitivity'][10, 0, 0] += .01
    log = audit.Recorder()
    rebuilt, detail = audit.build_context(raw, dict(tail_gram=gram), anchor, cert, 675, log, 'fixture')
    np.testing.assert_array_equal(rebuilt['gram']['gram'], gram['gram'])
    assert detail['directions_bitwise_equal']
    assert len(log.failures) == 1
    assert log.failures[0]['path'] == 'fixture.activation_sensitivity[10][0][0]'
    changed = copy.deepcopy(raw)
    changed['tail_direction'] = -changed['tail_direction']
    log = audit.Recorder()
    rebuilt, detail = audit.build_context(changed, dict(tail_gram=gram), anchor, cert, 675, log, 'fixture')
    assert not detail['directions_bitwise_equal']
    np.testing.assert_array_equal(rebuilt['directions'][16], changed['tail_direction'])
    np.testing.assert_array_equal(detail['regenerated_local_direction'], gram['minimum_eigenvector'])
    assert any(f['path'].startswith('fixture.tail_direction') for f in log.failures)


def decision_fixture(packet='old', score=.1):
    scores = (score+np.arange(18)*.01).tolist()
    return [dict(anchor='a', block='b', phase=675, radius=.001,
        model_packet_sha256=packet, status='complete', candidate_ids=list(range(18)),
        required_preview_calls={p:18 for p in audit.POLICIES}, preview_scores=scores,
        predictions={p:scores.copy() for p in audit.LEARNERS},
        policies={p:dict(status='complete', candidate=0, selection_score=score) for p in audit.POLICIES})]


def test_action_changes_and_complete_score_vectors_are_retained_without_new_gate():
    sealed = decision_fixture()
    fresh = copy.deepcopy(sealed)
    fresh[0]['predictions']['baseline'][17] = .01
    fresh[0]['policies']['baseline'].update(candidate=17, selection_score=.01)
    log = audit.Recorder()
    report = audit.decision_effects(sealed, fresh, log, 'synthetic')
    assert report['action_changes'] == [dict(condition=0, policy='baseline', sealed=0, regenerated=17)]
    assert len(report['conditions'][0]['policies']['baseline']['regenerated_scores']) == 18
    assert report['score_difference_pass_threshold'] is None
    assert not log.failures  # Changed valid choices are descriptive, not an invented score-tolerance failure.


def test_fitting_uses_regenerated_development_labels_and_never_falls_back(monkeypatch):
    monkeypatch.setattr(audit.driver, 'expected_features', lambda p, role:[role])
    monkeypatch.setattr(audit.driver, 'specs', lambda p, role:iter([dict(role=role)]))
    packet = dict(status='complete', packet_sha256='old', development_data_sha256='old_data',
        models={k:dict(chosen_lambda=.01, chosen_lambda_index=2, coefficients=[.1]) for k in audit.LEARNERS})
    old = dict(packet=packet, decisions=decision_fixture(),
        features={role:{role:dict(id=role, role=role, base=[-99.])} for role in audit.driver.ROLES},
        records={role:{role:dict(role=role, rmse=.9)} for role in audit.driver.ROLES})
    new_features = {role:[dict(id=role, role=role, base=[.2])] for role in audit.driver.ROLES}
    new_records = {role:[dict(role=role, rmse=.3)] for role in audit.driver.ROLES}
    calls = []
    def fit(rows, records, p):
        calls.append((rows, records))
        assert rows is new_features['development'] and records is new_records['development']
        return {**packet, 'packet_sha256':'new', 'development_data_sha256':'new_data'}
    monkeypatch.setattr(audit.driver, 'fit_development', fit)
    monkeypatch.setattr(audit.driver, 'decide_test', lambda rows, packet, p:decision_fixture(packet['packet_sha256']))
    monkeypatch.setattr(audit.driver, 'summarize', lambda records, decisions, p:dict(labels=[r['rmse'] for r in records]))
    report = audit.fit_and_assess(dict(planned_targets=2), old, new_features, new_records, audit.Recorder())
    assert len(calls) == 1
    assert report['data_hashes'] == dict(saved='old_data', regenerated='new_data')
    assert report['outcomes']['sealed_actions_saved_labels']['labels'] == [.9, .9]
    assert report['outcomes']['regenerated_actions_regenerated_labels']['labels'] == [.3, .3]
    new_records['test'].clear()
    report = audit.fit_and_assess(dict(planned_targets=2), old, new_features, new_records, audit.Recorder())
    assert report['outcomes'] == dict(status='undefined_incomplete', received=1, required=2, primary=None)
    assert len(calls) == 2
    new_records['development'].clear()
    report = audit.fit_and_assess(dict(planned_targets=2), old, new_features, new_records, audit.Recorder())
    assert report['status'] == 'unavailable_incomplete_development_labels'
    assert len(calls) == 2


@pytest.mark.parametrize('failure', ['head', 'dirty', 'unpublished'])
def test_source_gate_prevents_execution_on_unfrozen_sources(monkeypatch, failure):
    commit = 'a'*40
    def git(args, **kwargs):
        if args[1:] == ['rev-parse', 'HEAD']:
            return ('b'*40 if failure == 'head' else commit)+'\n'
        if args[1:] == ['status', '--porcelain']:
            return ' M tools/file.py\n' if failure == 'dirty' else ''
        if args[1] == 'for-each-ref':
            return '' if failure == 'unpublished' else 'refs/remotes/origin/test\n'
        return 'tree\n'
    monkeypatch.setattr(audit.subprocess, 'check_output', git)
    with pytest.raises(ValueError):
        audit.execution_provenance(commit)


def test_binding_failure_reports_failure_and_never_regenerates_or_fits(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'execution_provenance', lambda head:dict(audit_source_commit=head))
    def fail(attempt, log):
        log.fail('bindings', 'first', 'changed_hash')
        log.fail('bindings', 'second', 'missing_file')
        return None
    monkeypatch.setattr(audit, 'bindings', fail)
    monkeypatch.setattr(audit, 'load_inputs', lambda *a:pytest.fail('No reads after failed bindings'))
    monkeypatch.setattr(audit.driver, 'fit_development', lambda *a:pytest.fail('No fitting after failed bindings'))
    result = audit.audit(tmp_path, expected_head='a'*40)
    assert result['status'] == 'binding_failure_no_regeneration'
    assert result['exit_code'] == 2 and result['failure_count'] == 2
    assert result['new_integrations'] == 0 and result['frozen_gate_not_replaced']


def test_interruption_preserves_completed_records_and_bypasses_per_row_handlers(monkeypatch, tmp_path):
    from contextlib import nullcontext
    monkeypatch.setattr(audit, 'execution_provenance', lambda head:{})
    plan = dict(source_files=[], phases=[675], selection=dict(core_index=[0, 0]))
    monkeypatch.setattr(audit, 'bindings', lambda *a:(plan, {}, {}))
    old = dict(features={}, records={}, baselines={})
    anchors = {role:dict(id=role) for role in audit.driver.ROLES}
    for role in audit.driver.ROLES:
        old['features'][role] = {role:dict(id=role, anchor=role, phase=675, raw='preview.npz')}
        old['records'][role] = {role:dict(id=role, anchor=role, phase=675, raw='target.npz')}
        old['baselines'][role] = {role:dict(eligible=True,
            certificate=dict(verified=True, attracting=True), steps=[dict(period=9), dict(period=9)])}
    monkeypatch.setattr(audit, 'load_inputs', lambda *a:old)
    monkeypatch.setattr(audit.driver.base, 'load_grid', lambda *a:{})
    monkeypatch.setattr(audit.driver, 'build_atlas_geometry', lambda *a:{})
    monkeypatch.setattr(audit.driver, 'anchors_for', lambda p, role:[anchors[role]])
    monkeypatch.setattr(audit.driver, 'expected_features', lambda p, role:[role])
    monkeypatch.setattr(audit.driver, 'specs', lambda p, role:iter([dict(id=role, anchor=role)]))
    monkeypatch.setattr(audit.driver, 'trial_id', lambda spec:spec['id'])
    monkeypatch.setattr(audit.np, 'load', lambda *a, **kw:nullcontext({}))
    monkeypatch.setattr(audit, 'read', lambda *a:{})
    monkeypatch.setattr(audit, 'build_context', lambda *a:({}, {'id':'context'}))
    monkeypatch.setattr(audit, 'regenerate_preview', lambda row, *a:({**row, 'base':[.1]}, {'id':row['id']}))
    def interrupt(*args):
        raise audit.AuditInterrupted('synthetic timeout')
    monkeypatch.setattr(audit, 'regenerate_target', interrupt)
    monkeypatch.setattr(audit, 'fit_and_assess', lambda *a:pytest.fail('No fit after interruption'))
    result = audit.audit(tmp_path, expected_head='a'*40)
    assert result['status'] == 'incomplete_exception' and result['exit_code'] == 2
    assert result['received'] == dict(contexts=1, previews=1, targets=0)
    assert result['regenerated_features']['development'][0]['base'] == [.1]
    assert result['regenerated_features']['test'] == []
    assert result['failures'][-1]['error'] == 'AuditInterrupted: synthetic timeout'


def test_output_rejects_entire_repo_symlink_escape_and_existing_attempt(monkeypatch, tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    attempt = repo/'evidence'
    attempt.mkdir()
    monkeypatch.setattr(audit, 'ROOT', repo)
    with pytest.raises(ValueError, match='entire repository'):
        audit.output_path(repo/'report.json', attempt)
    link = tmp_path/'link'
    link.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match='entire repository'):
        audit.output_path(link/'report.json', attempt)
    report = tmp_path/'report.json'
    report.write_text('preserve')
    with pytest.raises(ValueError, match='already exists'):
        audit.output_path(report, attempt)
    assert report.read_text() == 'preserve'
    with pytest.raises(ValueError, match='Unsafe'):
        audit.bounded_path(tmp_path, 'link/../../outside')
    (repo/'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        audit.bounded_path(repo, 'escape/report.json')
