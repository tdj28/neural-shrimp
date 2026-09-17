"""EXP-013 runner contracts: fabricated metadata/arrays, no fits or dynamics.

Git/remote checks are fully mocked. Temporary files are synthetic only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_fresh_preview as run


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def lines(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def forbidden(*args, **kwargs):
    raise AssertionError('Synthetic runner test attempted a forbidden fit/dynamics/network path')


@pytest.fixture(scope='module')
def plan():
    anchors = []
    for index in range(24):
        alpha, gamma = 1.06+index*.0001, 10.2+index*.001
        anchors.append(dict(id=f'fresh-{index:02d}', alpha=alpha, gamma=gamma,
            group=run.old_features.physical_group(alpha, gamma), row=25+index, column=30+index))
    selection = dict(anchors=anchors, synthetic=True)
    return dict(**copy.deepcopy(run.CONTRACT), selection=selection, census=run.fresh.build_census(anchors),
        fixed_policies=run.evaluation.policy_catalog(), source_files=[dict(path=p, sha256='0'*64, bytes=0) for p in run.SOURCE_PATHS],
        old_plan=dict(path='synthetic-old-plan.json', sha256='0'*64, bytes=0), origin_source_bindings=[], claim_boundary='Synthetic fixture only.')


@pytest.fixture(scope='module')
def policy(plan):
    rows = run.planned_rows(plan)
    catalog = run.evaluation.policy_catalog()
    contexts = []
    for c in plan['census']['decision_contexts']:
        policies = {}
        for resolution in run.final.RESOLUTIONS:
            policies[resolution] = {}
            for definition in catalog:
                selected = definition['fixed_candidates'] if definition['kind'] == 'fixed' else list(range(definition['budget']))
                policies[resolution][definition['id']] = dict(status='complete', budget=definition['budget'],
                    selected=selected, selected_ids=[c['candidate_ids'][i] for i in selected])
        contexts.append(dict(context=[c['group'], c['phase'], c['radius']],
            candidate_ids=c['candidate_ids'], policies=policies))
    return run.seal(dict(rows=rows, catalog=catalog, contexts=contexts, synthetic=True))


def test_metadata_plan_and_exact_scheduled_workload(plan):
    assert run.validate(plan) == dict(origins=24, candidates=3456, shams=48, short_traces=10608, readouts=6144)
    specs = run.stage_specs(plan)
    assert {k: len(v) for k, v in specs.items()} == dict(models=1, baselines=24, sensitivities=48, previews=10608)
    assert len({x['id'] for x in specs['previews']}) == 10608
    native = [x for x in specs['previews'] if x['method']['kind'] == 'RK4']
    adaptive = [x for x in specs['previews'] if x['method']['kind'] == 'DOP853']
    assert len(native) == 10512 and len(adaptive) == 96
    assert {x['case']['id'] for x in adaptive} == set(plan['census']['sentinel_ids'])
    assert not ({x['id'] for x in plan['census']['shams']} & {x['case']['id'] for x in adaptive})
    rows = run.planned_rows(plan)
    assert len(rows) == 3456
    assert all(set(r) == {'id', 'role', 'anchor', 'group', 'phase', 'radius', 'candidate'} for r in rows)


@pytest.mark.parametrize('mutation', ['backend', 'bootstrap_seed', 'workers', 'candidate', 'sham', 'sentinel', 'policy', 'source'])
def test_plan_rejects_contract_census_policy_source_drift(plan, mutation):
    changed = copy.deepcopy(plan)
    if mutation in ('backend', 'bootstrap_seed', 'workers'): changed[mutation] = 'changed'
    elif mutation == 'candidate': changed['census']['cases'][0]['candidate'] = 17
    elif mutation == 'sham': changed['census']['shams'].pop()
    elif mutation == 'sentinel': changed['census']['sentinel_ids'][0] = changed['census']['sentinel_ids'][1]
    elif mutation == 'policy': changed['fixed_policies'][2]['fixed_candidates'] = [1, 9]
    else: changed['source_files'].pop()
    with pytest.raises(ValueError): run.validate(changed)


def test_plan_binding_and_reconstruction_checks_are_not_optional_when_requested(plan, monkeypatch):
    bound = {item['path']: item for item in plan['source_files']+[plan['old_plan']]}
    monkeypatch.setattr(run.legacy, 'binding', lambda path: copy.deepcopy(bound[str(Path(path).relative_to(run.ROOT))]))
    monkeypatch.setattr(run, 'read', lambda path: {'synthetic': True})
    validations = []
    monkeypatch.setattr(run.old, 'validate', lambda p, **kw: validations.append(kw))
    monkeypatch.setattr(run, 'build_plan', lambda: copy.deepcopy(plan))
    assert run.validate(plan, bindings=True, reconstruct=True)['origins'] == 24
    assert validations == [dict(check_bindings=True, reconstruct=False)]
    changed = copy.deepcopy(plan); changed['source_files'][0]['sha256'] = '1'*64
    with pytest.raises(ValueError, match='binding mismatch'): run.validate(changed, bindings=True)
    changed = copy.deepcopy(plan); changed['claim_boundary'] = 'mutated'
    with pytest.raises(ValueError, match='reconstruction'): run.validate(changed, reconstruct=True)


def test_build_plan_has_metadata_only_capabilities(plan, monkeypatch):
    marker = object(); reads = []
    monkeypatch.setattr(run, 'input_metadata', lambda: (marker, marker, []))
    def selected(grid, prior):
        assert grid is marker and prior is marker
        return copy.deepcopy(plan['selection'])
    monkeypatch.setattr(run.fresh, 'select_origins', selected)
    def read_old(path):
        assert Path(path) == run.OLD_PLAN
        reads.append(path); return {'metadata_only': True}
    monkeypatch.setattr(run, 'read', read_old)
    monkeypatch.setattr(run.old, 'validate', lambda p, **kw: p == {'metadata_only': True} or forbidden())
    monkeypatch.setattr(run.legacy, 'binding', lambda p: dict(path=str(Path(p).relative_to(run.ROOT)), bytes=0, sha256='0'*64))
    monkeypatch.setattr(run.final, 'fit_four_recipes', forbidden)
    monkeypatch.setattr(run.final, 'predict_prepreview', forbidden)
    monkeypatch.setattr(run.old_features, 'load_features', forbidden)
    monkeypatch.setattr(run.work, 'execute', forbidden)
    built = run.build_plan()
    assert built['census'] == plan['census'] and built['fixed_policies'] == plan['fixed_policies']
    assert reads == [run.OLD_PLAN]


def metadata_tree(tmp_path, monkeypatch):
    root = tmp_path/'metadata'; root.mkdir()
    monkeypatch.setattr(run, 'ROOT', root)
    def binding(path):
        path = Path(path)
        return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=run.legacy.sha(path))
    monkeypatch.setattr(run.legacy, 'binding', binding)
    artifacts = []
    for i in range(181):
        p = root/f'evidence/EXP-004/attempt-001/focus/row-{i:03d}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, alpha=np.linspace(1.056, 1.08, 181), gamma=np.full(181, 10.14+i*.001),
            period=np.full(181, 9), forbidden_preview_labels=np.array([object()], dtype=object))
        artifacts.append(dict(path=f'focus/row-{i:03d}.npz', bytes=p.stat().st_size, sha256=run.legacy.sha(p)))
    hashes = {}
    for rel in run.HISTORICAL_HASHES:
        value = {'artifacts': artifacts} if rel.endswith('EXP-004/attempt-001/receipt.json') else {'metadata_only': True}
        write(root/rel, value); hashes[rel] = run.legacy.sha(root/rel)
    monkeypatch.setattr(run, 'HISTORICAL_HASHES', hashes)
    return root, hashes


def test_input_metadata_decodes_only_six_jsons_and_three_fields_from_181_bound_rows(tmp_path, monkeypatch):
    root, hashes = metadata_tree(tmp_path, monkeypatch)
    read = run.read; load = np.load; decoded, arrays = [], []
    def limited_read(path):
        rel = str(Path(path).relative_to(root)); assert rel in hashes
        decoded.append(rel); return read(path)
    def limited_load(path, **kw):
        assert kw == {'allow_pickle': False}
        assert Path(path).parent == root/'evidence/EXP-004/attempt-001/focus'
        arrays.append(Path(path).name); return load(path, **kw)
    monkeypatch.setattr(run, 'read', limited_read); monkeypatch.setattr(np, 'load', limited_load)
    grid, prior, bindings = run.input_metadata()
    assert set(decoded) == set(hashes) and len(arrays) == 181
    assert set(grid) == {'alpha', 'gamma', 'period'} and grid['period'].shape == (181, 181)
    assert set(prior) == {'EXP-006', 'EXP-007', 'EXP-008'} and len(bindings) == 187


@pytest.mark.parametrize('mutation', ['historical-json', 'raw-hash', 'raw-bytes', 'missing-row'])
def test_input_metadata_rejects_rebound_or_missing_evidence(tmp_path, monkeypatch, mutation):
    root, hashes = metadata_tree(tmp_path, monkeypatch)
    if mutation == 'historical-json':
        write(root/'evidence/EXP-006/attempt-001/manifest.json', {'changed': True})
    elif mutation in ('raw-hash', 'raw-bytes'):
        path = root/'evidence/EXP-004/attempt-001/receipt.json'; receipt = run.read(path)
        receipt['artifacts'][0]['sha256' if mutation == 'raw-hash' else 'bytes'] = '0'*64 if mutation == 'raw-hash' else 1
        write(path, receipt); hashes[str(path.relative_to(root))] = run.legacy.sha(path)
    else:
        (root/'evidence/EXP-004/attempt-001/focus/row-013.npz').unlink()
    with pytest.raises(ValueError): run.input_metadata()


def capability(tmp_path, monkeypatch, policy):
    records = [dict(id=f'{r["id"]}--{resolution}', case_id=r['id'], method=resolution,
        status='complete', rmse=.01+r['candidate']*.001) for r in policy['rows'] for resolution in run.final.RESOLUTIONS]
    indexed = {r['id']: copy.deepcopy(r) for r in records}
    reads, events = [], []
    actual_read = run.read
    def saved(path):
        path = Path(path)
        if path.parent == tmp_path/'records':
            ident = path.stem; reads.append(ident); return copy.deepcopy(indexed[ident])
        return actual_read(path)
    monkeypatch.setattr(run, 'read', saved)
    monkeypatch.setattr(run, 'append_event', lambda path, stage, **kw: events.append(dict(stage=stage, **kw)))
    store = run.FreshLabels(tmp_path, records, policy, tmp_path/'access.jsonl')
    return store, records, indexed, reads, events


def reveal_and_seal(store, policy, tmp_path):
    winners = []
    for context in policy['contexts']:
        for resolution in run.final.RESOLUTIONS:
            for definition in policy['catalog']:
                if definition['budget'] == 18: continue
                rows, values = store.selected(context['context'], resolution, definition['id'])
                winners.append(dict(context=context['context'], resolution=resolution, policy=definition['id'],
                    winner=run.evaluation.observe_selected(rows, values)))
    packet = run.seal(dict(policy_seal_sha256=policy['packet_sha256'], winners=winners))
    path = tmp_path/'winner.json'; write(path, packet)
    return path, packet


def test_selected_capability_only_opens_sealed_ids_and_blocks_early_full(tmp_path, monkeypatch, policy):
    store, _, _, reads, _ = capability(tmp_path, monkeypatch, policy)
    context = policy['contexts'][0]; res = run.final.RESOLUTIONS[0]
    with pytest.raises(ValueError, match='before winner'): store.full(res)
    with pytest.raises(ValueError, match='selected-label'): store.selected(context['context'], res, 'full18-b18')
    assert reads == []
    rows, values = store.selected(context['context'], res, 'axes-b4')
    ids = context['policies'][res]['axes-b4']['selected_ids']
    assert set(values) == set(ids) and [r['id'] for r in rows] == ids
    assert reads == [f'{i}--{res}' for i in ids]
    with pytest.raises(ValueError, match='duplicate'): store.selected(context['context'], res, 'axes-b4')


def test_all_selected_winners_sealed_before_exact_two_full_reveals(tmp_path, monkeypatch, policy):
    store, _, _, reads, events = capability(tmp_path, monkeypatch, policy)
    path, packet = reveal_and_seal(store, policy, tmp_path)
    assert len(packet['winners']) == 5760
    assert all(e['stage'] == 'selected_reveal' for e in events)
    selected_reads = len(reads)
    store.attach_winners(path)
    assert events[-1]['stage'] == 'winner_seal' and events[-1]['count'] == 5760
    for resolution in run.final.RESOLUTIONS:
        values = store.full(resolution)
        assert len(values) == 3456
        with pytest.raises(ValueError, match='duplicate'): store.full(resolution)
    assert len(reads)-selected_reads == 6912
    assert [e['stage'] for e in events[-3:]] == ['winner_seal', 'full18_reveal', 'full18_reveal']
    with pytest.raises(ValueError, match='selected-label'):
        store.selected(policy['contexts'][0]['context'], run.final.RESOLUTIONS[0], 'axes-b4')


@pytest.mark.parametrize('mutation', ['unrevealed', 'duplicate', 'missing', 'wrong-winner', 'wrong-policy-seal', 'bad-hash'])
def test_winner_capability_rejects_incomplete_or_forged_seals_before_full(tmp_path, monkeypatch, policy, mutation):
    store, _, _, _, _ = capability(tmp_path, monkeypatch, policy)
    path, packet = reveal_and_seal(store, policy, tmp_path)
    if mutation == 'unrevealed': store.observed.pop(next(iter(store.observed)))
    elif mutation == 'duplicate': packet['winners'][-1] = copy.deepcopy(packet['winners'][0])
    elif mutation == 'missing': packet['winners'].pop()
    elif mutation == 'wrong-winner': packet['winners'][0]['winner']['rmse'] += .001
    elif mutation == 'wrong-policy-seal': packet['policy_seal_sha256'] = '0'*64
    else: packet['packet_sha256'] = '0'*64
    if mutation != 'bad-hash': packet.pop('packet_sha256'); run.seal(packet)
    write(path, packet)
    with pytest.raises(ValueError): store.attach_winners(path)
    assert store.winners is None
    with pytest.raises(ValueError): store.full(run.final.RESOLUTIONS[0])


def test_changed_saved_selected_label_refused(tmp_path, monkeypatch, policy):
    store, _, saved, _, _ = capability(tmp_path, monkeypatch, policy)
    context = policy['contexts'][0]; resolution = run.final.RESOLUTIONS[0]
    ident = context['policies'][resolution]['axes-b4']['selected_ids'][0]
    saved[f'{ident}--{resolution}']['rmse'] += .1
    with pytest.raises(ValueError, match='identity/score'):
        store.selected(context['context'], resolution, 'axes-b4')
    assert not store.observed


def chronology_fixture(tmp_path, policy):
    model = run.seal(dict(synthetic='old models'))
    model_rel = 'models/payloads/final-old-models/models.json'
    write(tmp_path/model_rel, model); write(tmp_path/'policy-seal.json', policy)
    write(tmp_path/'fresh-features.json', dict(synthetic='metadata'))
    write(tmp_path/'predictions.json', dict(synthetic='predictions'))
    np.savez(tmp_path/'fresh-features.npz', shared=np.zeros((1,1)))
    events = [dict(stage='models_sealed', monotonic_ns=10, path=model_rel,
            sha256=run.legacy.sha(tmp_path/model_rel), packet_sha256=model['packet_sha256']),
        dict(stage='fresh_dynamics_started', monotonic_ns=20),
        dict(stage='policies_sealed', monotonic_ns=30, sha256=run.legacy.sha(tmp_path/'policy-seal.json'),
            packet_sha256=policy['packet_sha256'], features_sha256=run.legacy.sha(tmp_path/'fresh-features.npz'),
            feature_metadata_sha256=run.legacy.sha(tmp_path/'fresh-features.json'), predictions_sha256=run.legacy.sha(tmp_path/'predictions.json')),
        dict(stage='previews_started', monotonic_ns=40)]
    labels, winners = [], []
    by_id = {r['id']: r for r in policy['rows']}
    for context in policy['contexts']:
        for resolution in run.final.RESOLUTIONS:
            for definition in policy['catalog']:
                if definition['budget'] == 18: continue
                ident = definition['id']; ids = context['policies'][resolution][ident]['selected_ids']
                labels.append(dict(stage='selected_reveal', monotonic_ns=100+len(labels), context=context['context'],
                    resolution=resolution, policy=ident, ids=ids, policy_seal_sha256=policy['packet_sha256']))
                winner = run.evaluation.observe_selected([by_id[i] for i in ids], {i: .1 for i in ids})
                winners.append(dict(context=context['context'], resolution=resolution, policy=ident, winner=winner))
    packet = run.seal(dict(policy_seal_sha256=policy['packet_sha256'], winners=winners))
    write(tmp_path/'winner-seal.json', packet)
    labels.append(dict(stage='winner_seal', monotonic_ns=6000, sha256=run.legacy.sha(tmp_path/'winner-seal.json'), packet_sha256=packet['packet_sha256'], count=5760))
    for i, resolution in enumerate(run.final.RESOLUTIONS):
        labels.append(dict(stage='full18_reveal', monotonic_ns=6001+i, resolution=resolution,
            ids=[r['id'] for r in policy['rows']], winner_seal_sha256=packet['packet_sha256']))
    lines(tmp_path/'events.jsonl', events); lines(tmp_path/'label-access.jsonl', labels)
    lines(tmp_path/'models/payloads/final-old-models/access.jsonl', [dict(monotonic_ns=1), dict(monotonic_ns=2)])
    return events, labels


def test_complete_artifact_bound_chronology(tmp_path, policy):
    chronology_fixture(tmp_path, policy)
    audit = run.chronology(tmp_path, policy)
    assert audit['passed'] and audit['selected_reveals'] == 5760 and audit['full_reveals'] == 2
    assert audit['models_before_fresh_dynamics'] and audit['policies_before_previews'] and audit['winners_before_full18']


@pytest.mark.parametrize('mutation', ['stage-order', 'duplicate-stage', 'selected-order', 'selected-id', 'selected-seal',
    'duplicate-selected', 'missing-selected', 'full-order', 'full-ids', 'full-seal', 'feature-hash', 'late-training', 'artifact'])
def test_chronology_rejects_order_census_and_bound_artifact_changes(tmp_path, policy, mutation):
    events, labels = chronology_fixture(tmp_path, policy)
    if mutation == 'stage-order': events[0]['monotonic_ns'] = 25
    elif mutation == 'duplicate-stage': events.append(events[0])
    elif mutation == 'selected-order': labels[0]['monotonic_ns'] = 20
    elif mutation == 'selected-id': labels[0]['ids'] = ['not-selected']
    elif mutation == 'selected-seal': labels[0]['policy_seal_sha256'] = '0'*64
    elif mutation == 'duplicate-selected': labels[1] = copy.deepcopy(labels[0])
    elif mutation == 'missing-selected': labels.pop(0)
    elif mutation == 'full-order': labels[-1]['monotonic_ns'] = 5999
    elif mutation == 'full-ids': labels[-1]['ids'].pop()
    elif mutation == 'full-seal': labels[-1]['winner_seal_sha256'] = '0'*64
    elif mutation == 'feature-hash': events[2]['feature_metadata_sha256'] = '0'*64
    elif mutation == 'late-training': lines(tmp_path/'models/payloads/final-old-models/access.jsonl', [dict(monotonic_ns=11)])
    else: write(tmp_path/'predictions.json', {'changed': True})
    lines(tmp_path/'events.jsonl', events); lines(tmp_path/'label-access.jsonl', labels)
    with pytest.raises(ValueError): run.chronology(tmp_path, policy)


def tiny_qualification(tmp_path, monkeypatch, *, missing=()):
    cases = [dict(id='candidate')]; shams = [dict(id='sham')]
    plan = dict(selection=dict(anchors=[]), census=dict(cases=cases, shams=shams, sentinel_ids=['candidate'], sensitivity_contexts=[]))
    specs = run.stage_specs(plan)['previews']
    contexts = {case['id']: dict(initial=np.full(3, .1), alpha=1.06, gamma=10.3, delta=np.array([.00025, 0.]),
        reference=np.zeros((2700,3)), epoch=0., period=20.) for case in cases+shams}
    values = {'rk4-0.005': .1, 'rk4-0.0025': .099, 'rk4-0.00125': .09895,
              'dop853-1e-11': .0989499, 'dop853-1e-12': .0989501}
    records = []
    for task in specs:
        row = dict(id=task['id'], kind='preview', case_id=task['case']['id'], method=task['method']['id'], status='failed', rmse=None)
        if task['id'] not in missing:
            name = row['method']; context = contexts[row['case_id']]
            states = np.full((751,3), values[name]); states[0] = context['initial']
            path = tmp_path/'payloads'/row['id']/'trace.npz'; path.parent.mkdir(parents=True)
            np.savez(path, times=run.work.TIMES, states=states, **{k: context[k] for k in ('initial','alpha','gamma','delta')})
            solver = dict(task['method'])
            solver.update(steps=round(75/solver['dt'])) if solver['kind'] == 'RK4' else solver.update(success=True, nfev=13)
            row.update(status='complete', rmse=values[name], metrics=dict(phase_offset_cycles=0.), solver=solver,
                raw=str(path.relative_to(tmp_path)), raw_sha256=run.legacy.sha(path), raw_bytes=path.stat().st_size)
        records.append(row)
    monkeypatch.setattr(run.work, 'score', lambda context, times, states: dict(rmse=float(states[-1,0]), metrics=dict(phase_offset_cycles=0.)))
    return plan, records, contexts


def test_native_and_all_three_independent_pairs_qualify_with_saved_score_replay(tmp_path, monkeypatch):
    plan, records, contexts = tiny_qualification(tmp_path, monkeypatch)
    result = run.qualify(plan, records, tmp_path, contexts, replay=True)
    assert result['native_candidates'] and result['native_shams'] and result['independent_panel']
    assert result['score_packets_replayed'] == 8
    case = result['details'][0]
    assert len(case['independent_comparisons']) == 3 and all(x['passed'] for x in case['independent_comparisons'])
    assert all(x['common_native_finest_phase'] == 0. and len(x['common_phase_rmse']) == 2 for x in case['independent_comparisons'])


def test_missing_independent_retains_native_diagnostics_and_all_panel_rows(tmp_path, monkeypatch):
    plan, records, contexts = tiny_qualification(tmp_path, monkeypatch, missing=['candidate--dop853-1e-12'])
    result = run.qualify(plan, records, tmp_path, contexts, replay=True)
    case = result['details'][0]
    assert case['all_native_complete'] and case['native_scalar_pass'] and case['native_waveform_pass']
    assert case['d1'] > case['d2'] and not case['qualified']
    assert not case['all_required_complete'] and not case['independent_pass']
    assert len(case['independent_comparisons']) == 3
    assert sum(x['complete'] for x in case['independent_comparisons']) == 1
    assert case['statuses']['dop853-1e-12'] == 'failed'
    labels = run.numerical_labels(plan, records, result)
    assert all(labels[r]['candidate']['qualified'] is False and labels[r]['candidate']['numerical'] == case for r in run.final.RESOLUTIONS)
    assert result['native_candidates'] and not result['independent_panel']


def test_missing_native_keeps_every_status_without_complete_case_qualification(tmp_path, monkeypatch):
    plan, records, contexts = tiny_qualification(tmp_path, monkeypatch, missing=['candidate--rk4-0.00125'])
    result = run.qualify(plan, records, tmp_path, contexts, replay=True)
    case = result['details'][0]
    assert not case['all_native_complete'] and not case['qualified'] and not result['native_candidates']
    assert len(case['statuses']) == 5 and len(result['details']) == 2
    labels = run.numerical_labels(plan, records, result)
    fine = labels['rk4-0.00125']['candidate']
    assert fine['status'] == 'failed' and fine['rmse'] is None and not fine['qualified']


def test_native_nonincrease_gate_is_enforced_even_near_roundoff(tmp_path, monkeypatch):
    plan, records, contexts = tiny_qualification(tmp_path, monkeypatch)
    for row in records:
        if row['case_id'] == 'candidate' and row['method'].startswith('rk4'):
            row['rmse'] = .1 + (1e-12 if row['method'] == 'rk4-0.00125' else 0.)
    case = run.qualify(plan, records, tmp_path, contexts, replay=False)['details'][0]
    assert case['d1'] == 0. and 0. < case['d2'] < 1e-4
    assert not case['native_scalar_pass'] and not case['qualified']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'case', 'method', 'raw-path', 'solver', 'steps', 'hash'])
def test_qualification_rejects_census_or_realized_identity_drift(tmp_path, monkeypatch, mutation):
    plan, records, contexts = tiny_qualification(tmp_path, monkeypatch)
    if mutation == 'missing': records.pop()
    elif mutation == 'duplicate': records.append(copy.deepcopy(records[0]))
    elif mutation == 'case': records[0]['case_id'] = 'sham'
    elif mutation == 'method': records[0]['method'] = 'rk4-0.00125'
    elif mutation == 'raw-path': records[0]['raw'] = records[1]['raw']
    elif mutation == 'solver': records[0]['solver']['dt'] = .001
    elif mutation == 'steps': records[0]['solver']['steps'] += 1
    else: records[0]['raw_sha256'] = '0'*64
    with pytest.raises(ValueError): run.qualify(plan, records, tmp_path, contexts, replay=False)


def test_fill_terminal_never_discards_pending_or_failed_tasks(plan):
    specs = run.stage_specs(plan)['previews']
    first = dict(id=specs[0]['id'], case_id=specs[0]['case']['id'], method=specs[0]['method']['id'],
        kind='preview', status='failed', error='synthetic prefix retained', rmse=None)
    records = run.fill_terminal(specs, [first], 'not_started_test')
    assert len(records) == 10608 and records[0] == first
    assert sum(r['status'] == 'not_started_test' for r in records) == 10607
    assert [r['id'] for r in records] == [s['id'] for s in specs]
    with pytest.raises(ValueError): run.fill_terminal(specs, [first, first], 'invalid')


@pytest.mark.parametrize('attempt', ['', '.', '..', '../escape', 'nested/path', '/absolute'])
def test_external_attempt_rejects_invalid_names_and_repository_root(tmp_path, monkeypatch, attempt):
    repo = tmp_path/'repo'; repo.mkdir(); monkeypatch.setattr(run, 'ROOT', repo)
    with pytest.raises(ValueError): run.external_attempt(tmp_path/'data', attempt)
    with pytest.raises(ValueError): run.external_attempt(repo/'raw', 'attempt-001')


def test_external_attempt_is_new_exclusive_and_rejects_symlink_escapes(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir(); monkeypatch.setattr(run, 'ROOT', repo)
    dest = run.external_attempt(tmp_path/'data', 'attempt-001')
    assert dest.is_dir() and not dest.resolve().is_relative_to(repo)
    with pytest.raises(ValueError): run.external_attempt(tmp_path/'data', 'attempt-001')
    escaped = tmp_path/'escaped'; escaped.mkdir(); (escaped/'EXP-013').symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError): run.external_attempt(escaped, 'attempt-002')
    (tmp_path/'data'/'EXP-013'/'attempt-link').symlink_to(tmp_path/'absent', target_is_directory=True)
    with pytest.raises(ValueError): run.external_attempt(tmp_path/'data', 'attempt-link')
    link = tmp_path/'rootlink'; link.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError): run.external_attempt(link, 'attempt-001')


def source_fixture(tmp_path, monkeypatch, *, dirty=False, remote=True):
    repo = tmp_path/'repo'; repo.mkdir(); monkeypatch.setattr(run, 'ROOT', repo)
    paths = ['tools/a.py', 'tests/b.py']; monkeypatch.setattr(run, 'SOURCE_PATHS', paths)
    plan_path = repo/'experiments/plan.json'; sources = {p: b'synthetic '+p.encode() for p in paths}
    sources['experiments/plan.json'] = b'{}'
    for p, data in sources.items():
        path = repo/p; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    calls, shows = [], []
    def git(*args):
        calls.append(args)
        if args[0] == 'status': return '?? changed' if dirty else ''
        if args == ('rev-parse', 'HEAD'): return '1'*40
        if args == ('branch', '--show-current'): return 'codex/synthetic'
        if args[0] == 'ls-remote': return ('1' if remote else '2')*40+'\trefs/heads/codex/synthetic'
        return forbidden()
    def show(command):
        assert command[:4] == ['git', '-C', str(repo), 'show']
        commit, path = command[4].split(':',1); assert commit == '1'*40
        shows.append(path); return sources[path]
    monkeypatch.setattr(run.legacy, 'git', git); monkeypatch.setattr(run.subprocess, 'check_output', show)
    return repo, plan_path, calls, shows


def test_source_guard_binds_every_source_and_plan_to_exact_remote_head(tmp_path, monkeypatch):
    repo, plan_path, calls, shows = source_fixture(tmp_path, monkeypatch)
    result = run.source_guard({}, plan_path)
    assert result['source_commit'] == '1'*40 and result['pushed_branch'] == 'codex/synthetic'
    assert shows == ['tools/a.py', 'tests/b.py', 'experiments/plan.json']
    assert calls[-1] == ('ls-remote', '--exit-code', 'origin', 'refs/heads/codex/synthetic')


@pytest.mark.parametrize('mutation', ['dirty', 'remote', 'source', 'plan-outside'])
def test_source_guard_fails_closed_without_real_git_or_network(tmp_path, monkeypatch, mutation):
    repo, plan_path, _, shows = source_fixture(tmp_path, monkeypatch, dirty=mutation=='dirty', remote=mutation!='remote')
    if mutation == 'source': (repo/'tools/a.py').write_text('changed locally')
    if mutation == 'plan-outside': plan_path = tmp_path/'outside.json'; plan_path.write_text('{}')
    with pytest.raises(ValueError): run.source_guard({}, plan_path)
    if mutation == 'dirty': assert shows == []


def test_guarded_run_headroom_failure_seals_complete_unavailable_denominator(tmp_path, monkeypatch, plan):
    """No compiler/process/model/ODE launch: exercise real failure finalization."""
    repo = tmp_path/'repo'; repo.mkdir(); monkeypatch.setattr(run, 'ROOT', repo)
    path = repo/'experiments/EXP-013-plan.json'; write(path, plan)
    review_path = repo/'docs/EXP-013-local-review.json'
    write(review_path, dict(status='ready_for_exploratory_execution', plan_sha256=run.legacy.sha(path)))
    validations = []
    monkeypatch.setattr(run, 'validate', lambda value, **kw: validations.append(kw))
    monkeypatch.setattr(run, 'source_guard', lambda *a: dict(source_commit='1'*40, pushed_branch='codex/synthetic', remote_head_verified_at='synthetic'))
    monkeypatch.setattr(run.subprocess, 'check_output', lambda command: review_path.read_bytes()
        if command[-1] == '1'*40+':docs/EXP-013-local-review.json' else forbidden())
    monkeypatch.setattr(run.legacy, 'binding', lambda p: dict(path=str(Path(p).relative_to(repo)), sha256=run.legacy.sha(p), bytes=Path(p).stat().st_size))
    monkeypatch.setattr(run.importlib.metadata, 'version', lambda name: run.CONTRACT['required_packages'][name])
    monkeypatch.setattr(run, 'resources', lambda *a, **kw: dict(available_ram_bytes=1, disk_free_bytes=1, synthetic=True))
    monkeypatch.setattr(run, 'require_headroom', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('synthetic headroom failure')))
    monkeypatch.setattr(run.work, 'build_libraries', forbidden)
    monkeypatch.setattr(run.work, 'execute', forbidden)
    monkeypatch.setattr(run.final, 'fit_four_recipes', forbidden)
    monkeypatch.setattr(run.work, 'identity', lambda pid: dict(pid=pid, start_ticks='synthetic', boot_id='synthetic'))
    monkeypatch.setattr(run.platform, 'platform', lambda: 'synthetic-platform')
    timers = []
    monkeypatch.setattr(run.signal, 'setitimer', lambda *args: timers.append(args))
    result = run.run(path, tmp_path/'data', 'attempt-001')
    directory = tmp_path/'data/EXP-013/attempt-001'
    terminal = run.read(directory/'all-terminal-records.json')
    assert {stage: len(rows) for stage, rows in terminal.items()} == dict(models=1, baselines=24, sensitivities=48, previews=10608)
    assert sum(len(rows) for rows in terminal.values()) == 10681
    assert all(row['status'] == 'not_started_prerequisite_or_producer_failure' for rows in terminal.values() for row in rows)
    assert not result['complete'] and not result['qualified'] and 'headroom failure' in result['error']
    assert result['cohort_statistics'] is None and not any(result['qualification'].values())
    unavailable = run.read(directory/'evaluation-unavailable.json')
    assert unavailable['retained_readouts'] == 6144 and unavailable['observed_readouts'] == 0
    assert unavailable['cohort_statistics'] is None
    assert (directory/'amortization.json').is_file()
    receipt = run.read(directory/'receipt.json')
    assert receipt['status'] == 'incomplete' and receipt['source_commit'] == '1'*40
    assert receipt['artifacts'] == run.work.inventory(directory)
    assert validations == [dict(bindings=True, reconstruct=True), dict(bindings=True, reconstruct=False)]
    assert timers[-1] == (run.signal.ITIMER_REAL, 0)
