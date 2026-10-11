"""Synthetic-only EXP-011 model, capability, seal and failure-contract tests."""
from pathlib import Path
import copy
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_preview_shortlist as run
import preview_shortlist_model as model
import numpy as np
import pytest


def toy(tmp_path, groups=4, contexts=2):
    rng = np.random.default_rng(711)
    rows, shared, geometry, labels = [], [], [], {}
    for group in range(groups):
        for phase in range(contexts):
            for candidate in range(18):
                ident = f'g{group}-p{phase}-c{candidate:02d}'
                row = dict(id=ident, group=f'physical-{group}', role='old-role', anchor=f'alias-{group}',
                           phase=phase, radius=.001, candidate=candidate, direction=[1., 0.])
                x = rng.normal(size=3)
                g = rng.normal(size=1)
                shared.append(x)
                geometry.append(g)
                rows.append(row)
                labels[ident] = {}
                for resolution, offset in zip(run.RESOLUTIONS, [0., 1e-8]):
                    path = tmp_path / 'labels' / f'{ident}--{resolution}.json'
                    value = float(.5 + .02 * x[0] + .005 * x[1] + .003 * g[0] + offset)
                    run.write(path, dict(id=f'{ident}--{resolution}', case_id=ident, method=resolution,
                        status='complete', rmse=value, metrics=dict(aligned_rms=dict(aggregate=value))))
                    labels[ident][resolution] = run.external_binding(path)
    bundle = dict(schema_version='toy', shared_names=['a', 'b', 'c'], geometry_names=['g'], rows=rows,
                  shared=np.asarray(shared), geometry=np.asarray(geometry), source_bindings=[], timings={})
    plan = dict(rows=rows, label_records=labels, independent_case_ids=[], feature_identity=run.feature_identity(bundle))
    return bundle, plan


def fit_one(tmp_path, bundle, plan, name='attempt'):
    directory = tmp_path / name
    (directory / 'records').mkdir(parents=True)
    task = run.tasks(plan)[0]
    run.fold_worker(task, dict(plan=plan, bundle=bundle), directory, None, -1)
    record = run.read(directory / 'records' / f'{task["id"]}.json')
    assert record['status'] == 'complete', record
    return directory, record


def test_equal_origin_weights_and_train_only_scalers():
    rng = np.random.default_rng(221)
    x = rng.normal(size=(72, 4))
    x[:, 3] = 1.
    y = .4 + .05 * x[:, 0] - .03 * x[:, 1]
    groups = np.repeat(['a', 'b', 'c', 'd'], 18)
    held_x = rng.normal(size=(18, 4)); held_x[:, 3] = 1.
    rows = [dict(id=f'h-{i}', group='held', phase=0, radius=.1, candidate=i) for i in range(18)]
    packet = model.fit_nested(x, y, groups, ['a', 'b', 'c', 'constant'], held_x, rows)
    assert packet['numerical_audit']['passed']
    for fold in packet['inner_folds']:
        train = groups != fold['held_group']
        assert fold['held_group'] not in fold['grid']['training_groups']
        assert np.array_equal(fold['grid']['mean'], model.equal_origin_weights(groups[train]) @ x[train])
        assert fold['grid']['scale'][3] == 1.
    duplicate = groups == 'a'
    other = model.fit_nested(np.vstack((x, x[duplicate])), np.r_[y, y[duplicate]],
        np.r_[groups, groups[duplicate]], ['a', 'b', 'c', 'constant'], held_x, rows)
    assert other['selected_penalty_svd'] == packet['selected_penalty_svd']
    assert np.allclose(other['held_predictions_svd'], packet['held_predictions_svd'], atol=1e-12, rtol=1e-12)


def test_physical_aliases_cannot_cross_outer_boundary():
    x = np.ones((4, 2))
    rows = [dict(id=f'h-{i}', group='same-physical-origin', phase=0, radius=.1, candidate=i) for i in range(18)]
    with pytest.raises(ValueError, match='Physical-origin'):
        model.fit_nested(x, np.ones(4), ['a', 'a', 'same-physical-origin', 'same-physical-origin'], ['x', 'y'], np.ones((18, 2)), rows)


def test_exact_four_ties_and_duplicate_directions():
    rows = [dict(id=str(i), group='g', phase=0, radius=.1, candidate=i, direction=[1, 0]) for i in range(18)]
    ranked = model.shortlist_rows(rows, np.ones(18))[0]
    assert ranked['top4'] == [0, 1, 2, 3]
    assert len(set(ranked['top4_ids'])) == 4
    assert run.observe_shortlist(rows, np.ones(18), ranked['top4'])['chosen_candidate'] == 0
    with pytest.raises(ValueError, match='18 separate'):
        model.shortlist_rows(rows[:-1], np.ones(17))


def test_nonfinite_has_no_fallback():
    with pytest.raises(ValueError, match='Nonfinite'):
        model.ridge_grid([[0.], [float('nan')]], [1., 2.], ['a', 'b'])


def test_training_capability_rejects_held_alias_before_read(tmp_path, monkeypatch):
    bundle, plan = toy(tmp_path, contexts=1)
    store = run.LabelStore(bundle['rows'], plan['label_records'], allowed_training_groups=['physical-1'], held_group='physical-0')
    monkeypatch.setattr(store, '_values', lambda *_: pytest.fail('Forbidden label read'))
    with pytest.raises(ValueError, match='outside allowed'):
        store.training([bundle['rows'][0]['id']], run.RESOLUTIONS[0])
    with pytest.raises(ValueError, match='Held physical'):
        run.LabelStore(bundle['rows'], plan['label_records'], allowed_training_groups=['physical-0'], held_group='physical-0')


def test_no_held_label_read_and_mutation_does_not_change_models(tmp_path, monkeypatch):
    bundle, plan = toy(tmp_path, contexts=1)
    held = {r['id'] for r in bundle['rows'] if r['group'] == 'physical-0'}
    original_read = run.read
    def spy(path):
        if Path(path).parent.name == 'labels' and any(Path(path).name.startswith(i + '--') for i in held):
            pytest.fail('Fitting touched held-origin labels')
        return original_read(path)
    monkeypatch.setattr(run, 'read', spy)
    directory, record = fit_one(tmp_path, bundle, plan)
    for ident in held:
        for resolution in run.RESOLUTIONS:
            path = Path(plan['label_records'][ident][resolution]['path'])
            row = original_read(path)
            row['rmse'] = row['metrics']['aligned_rms']['aggregate'] = .99
            run.write(path, row)
            plan['label_records'][ident][resolution] = run.external_binding(path)
    directory2, _ = fit_one(tmp_path, bundle, plan, 'changed-hidden-labels')
    for resolution in run.RESOLUTIONS:
        for schema in run.SCHEMAS:
            filename = f'{run.recipe_name(resolution, schema)}.json'
            assert original_read(directory / 'folds' / record['id'] / filename) == original_read(directory2 / 'folds' / record['id'] / filename)


def test_winner_sealed_before_oracle_and_full_replay(tmp_path, monkeypatch):
    bundle, plan = toy(tmp_path)
    directory, record = fit_one(tmp_path, bundle, plan)
    original = run.LabelStore.reveal
    def checked(self, ids, resolution, seal_path, *, purpose):
        if purpose == 'full18_evaluation':
            assert (Path(seal_path).parent / 'winner-seal.json').exists()
        return original(self, ids, resolution, seal_path, purpose=purpose)
    monkeypatch.setattr(run.LabelStore, 'reveal', checked)
    evaluated = run.evaluate(plan, bundle, [record], directory, persist=True)
    assert len(evaluated['contexts']) == 2
    assert run.evaluate(plan, bundle, [record], directory, persist=False) == evaluated
    assert run.replay_models(plan, bundle, [record], directory)['replayed_recipes'] == 4
    for context in evaluated['contexts']:
        for policies in context['policies'].values():
            assert all(p['required_preview_calls'] == 4 for p in policies.values())


def test_fixed_control_context_bijection_and_model_tamper(tmp_path):
    bundle, plan = toy(tmp_path)
    directory, record = fit_one(tmp_path, bundle, plan)
    path = directory / record['seal']
    saved = run.read(path)
    corrupt = copy.deepcopy(saved)
    corrupt['shortlist_ids']['axes'][1] = corrupt['shortlist_ids']['axes'][0]
    run.write(path, corrupt)
    with pytest.raises(ValueError, match='bijection'):
        run.validate_seal(path, bundle['rows'], expected_group=record['case_id'])
    run.write(path, saved)
    packet_path = path.parent / saved['artifacts'][0]['path']
    packet = run.read(packet_path); packet['held_predictions_svd'][0] += 1
    run.write(packet_path, packet)
    with pytest.raises(ValueError, match='bytes changed'):
        run.validate_seal(path, bundle['rows'], expected_group=record['case_id'])


def test_oracle_without_winner_is_rejected(tmp_path):
    bundle, plan = toy(tmp_path, contexts=1)
    directory, record = fit_one(tmp_path, bundle, plan)
    seal_path = directory / record['seal']
    store = run.LabelStore(bundle['rows'], plan['label_records'], held_group=record['case_id'])
    with pytest.raises(FileNotFoundError):
        store.reveal(run.read(seal_path)['held_ids'], run.RESOLUTIONS[0], seal_path, purpose='full18_evaluation')


def test_numerical_failure_retained_and_primary_undefined(tmp_path, monkeypatch):
    bundle, plan = toy(tmp_path, contexts=1)
    original = model.fit_nested
    def discrepant(*args, **kwargs):
        packet = original(*args, **kwargs)
        packet.pop('packet_sha256')
        packet['numerical_audit']['passed'] = False
        packet['packet_sha256'] = model.digest(packet)
        return packet
    monkeypatch.setattr(model, 'fit_nested', discrepant)
    directory = tmp_path / 'disagreement'; (directory / 'records').mkdir(parents=True)
    task = run.tasks(plan)[0]
    run.fold_worker(task, dict(plan=plan, bundle=bundle), directory, None, -1)
    record = run.read(directory / 'records' / f'{task["id"]}.json')
    assert record['status'] == 'numerical_disagreement'
    assert len(record['completed_recipes']) == 4
    evaluated = run.evaluate(plan, bundle, [record], directory, persist=True)
    summary = run.summarize(plan, [record], evaluated)
    assert summary['primary_equal_origin_readout'] is None
    assert summary['fixed_denominators']['origins'] == 24


def test_label_failure_keeps_failed_stage(tmp_path):
    bundle, plan = toy(tmp_path, contexts=1)
    train_id = next(r['id'] for r in bundle['rows'] if r['group'] == 'physical-1')
    path = Path(plan['label_records'][train_id][run.RESOLUTIONS[1]]['path'])
    row = run.read(path); row['status'] = 'failed'; run.write(path, row)
    plan['label_records'][train_id][run.RESOLUTIONS[1]] = run.external_binding(path)
    directory = tmp_path / 'failure'; (directory / 'records').mkdir(parents=True)
    task = run.tasks(plan)[0]
    run.fold_worker(task, dict(plan=plan, bundle=bundle), directory, None, -1)
    record = run.read(directory / 'records' / f'{task["id"]}.json')
    assert record['status'] == 'failed'
    assert len(record['completed_recipes']) == 2
    assert (directory / 'folds' / task['id'] / 'failure.json').exists()
    assert not (directory / 'folds' / task['id'] / 'pre-reveal-seal.json').exists()


def test_qualification_rejects_scalar_or_independent_waveform_failure():
    certificate = run.read(run.ROOT / 'docs/EXP-010-qualification.json')
    assert run.qualification_certificate(certificate) == certificate
    for key in ['native_scalar_qualified', 'independent_waveform_qualified', 'all_scalar_and_waveform_checks_pass']:
        changed = copy.deepcopy(certificate); changed[key] = 0
        with pytest.raises(ValueError, match='qualification certificate'):
            run.qualification_certificate(changed)


def test_prepare_does_not_decode_label_bearing_sources(monkeypatch):
    original = run.read
    def spy(path):
        path = Path(path)
        if path.resolve().is_relative_to(run.LABEL_ROOT):
            assert path.name == 'receipt.json', f'Label-bearing EXP-010 decode: {path}'
        return original(path)
    monkeypatch.setattr(run, 'read', spy)
    # Read-only input construction/hashing; no label store access or fitting.
    plan = run.build_plan()
    assert len(plan['rows']) == 3456
    assert run.validate(plan)['model_recipes'] == 96


def test_exact_feature_schema_not_only_width():
    bundle = dict(schema_version=run.features.SCHEMA_VERSION, rows=[], shared_names=['x'] * 37,
                  geometry_names=['g'] * 6, shared=np.zeros((3456, 37)), geometry=np.zeros((3456, 6)))
    with pytest.raises(ValueError, match='schema'):
        run.validate_feature_census(bundle)


def test_output_path_rechecks_final_symlink(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'; checkout.mkdir()
    data = tmp_path / 'data'; data.mkdir()
    (data / 'EXP-011').symlink_to(checkout, target_is_directory=True)
    monkeypatch.setattr(run, 'ROOT', checkout)
    with pytest.raises(ValueError, match='enters checkout'):
        run.external_attempt(data, 'attempt-001')
    with pytest.raises(ValueError):
        run.external_attempt(tmp_path, '../bad')


def test_complete_attempt_requires_summary_and_replay(tmp_path):
    records = [dict(id=f'outer-{i:02d}', status='complete') for i in range(24)]
    with pytest.raises(ValueError, match='Required completed artifacts'):
        run.require_artifacts(records, tmp_path, {'manifest.json', 'started.json', 'terminal-records.json', 'timings.json'})


def test_rank_flip_can_have_tiny_or_material_minimum_change():
    rows = [dict(id=str(i), group='g', phase=0, radius=.1, candidate=i) for i in range(18)]
    a = np.arange(18, dtype=float)
    b = a.copy(); b[0] = 1. + 1e-12; b[1] = 1.
    small = run.numerical_context(rows, b, np.r_[1. - 1e-12, a[1:]])
    assert small['oracle_argmin_changed']
    assert abs(small['oracle_minimum_signed_change']) < 1e-10
    large = run.numerical_context(rows, a, b)
    assert large['oracle_argmin_changed'] and abs(large['oracle_minimum_signed_change']) == 1.
    assert large['envelope_is_certified_error_bound'] is False


def test_descriptive_interval_needs_complete_origin_denominator():
    with pytest.raises(ValueError, match='24 equal'):
        run.interval(np.ones(23))
    summary = run.interval(np.arange(24, dtype=float))
    assert summary == run.interval(np.arange(24, dtype=float))
    assert summary['mean'] == 11.5


def test_dirty_source_blocks_before_fitting(monkeypatch):
    monkeypatch.setattr(run.runtime, 'git', lambda *args: '?? new-source.py')
    with pytest.raises(ValueError, match='clean before fitting'):
        run.source_guard(run.PLAN)


def test_unpushed_source_blocks_before_fitting(tmp_path, monkeypatch):
    plan_path = tmp_path / 'plan.json'; plan_path.write_text('{}')
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'SOURCE_PATHS', [])
    def git(*args):
        return {'status': '', 'rev-parse': 'expected', 'symbolic-ref': 'codex/toy', 'ls-remote': 'different refs/heads/codex/toy'}[args[0]]
    monkeypatch.setattr(run.runtime, 'git', git)
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *_args, **_kwargs: b'{}')
    with pytest.raises(ValueError, match='pushed before fitting'):
        run.source_guard(plan_path)


def test_preflight_failure_has_all_terminal_rows_and_exclusive_attempt(tmp_path, monkeypatch):
    plan = dict(run.CONTRACT, rows=[dict(id=f'{i}', group=f'{i}') for i in range(24)])
    plan_path = tmp_path / 'plan.json'; run.write(plan_path, plan)
    monkeypatch.setattr(run, 'validate', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, 'source_guard', lambda *_: dict(source_commit='synthetic', pushed_branch='codex/synthetic'))
    monkeypatch.setattr(run.features, 'load_features', lambda **_: (_ for _ in ()).throw(ValueError('synthetic stage failure')))
    directory = run.run(plan_path, tmp_path / 'external', 'attempt-001')
    receipt = run.read(directory / 'receipt.json')
    assert receipt['status'] == 'incomplete'
    assert receipt['observed_records'] == receipt['expected_records'] == 24
    assert len(run.read(directory / 'terminal-records.json')) == 24
    assert (directory / 'failure.json').exists()
    with pytest.raises(FileExistsError):
        run.run(plan_path, tmp_path / 'external', 'attempt-001')


def sleeping_worker(*args):
    time.sleep(30)


def test_reused_scheduler_deadline_leaves_no_owned_child(tmp_path):
    directory = tmp_path / 'deadline'; (directory / 'records').mkdir(parents=True)
    task = dict(id='outer-00', case=dict(id='group'), method=dict(id='nested'))
    records = run.runtime.execute_tasks([task], {'group': {}}, directory, None, workers=1,
        deadline=time.monotonic() + .08, per_task=.05, worker_fn=sleeping_worker)
    assert records[0]['status'] in ('task_timeout', 'whole_attempt_timeout')
    assert not Path(f'/proc/{records[0]["pid"]}').exists()
