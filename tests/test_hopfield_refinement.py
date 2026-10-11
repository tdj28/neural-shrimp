"""Meaningful synthetic guards for known-case refinement; no target integration."""
import copy
import json
import os
from pathlib import Path
import sys
import signal
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_hopfield_refinement as driver


def old_fixture():
    rows = []
    for role, anchor in [('development', 'dev'), ('test', 'test')]:
        for candidate in [-1, 0, 4, 8, 12, 16, 17]:
            radius = 0. if candidate == -1 else .001
            for dt in [.01, .005]:
                value = .1 + (.01 if candidate == 0 and dt == .01 else 0.)
                rows.append(dict(role=role, anchor=anchor, phase=675, radius=radius,
                    candidate=candidate, dt=dt, status='complete', rmse=value,
                    raw=f'{role}-{candidate}-{dt}.npz'))
    decisions = [dict(anchor='test', phase=675, radius=.001, block='block', status='complete',
        policies={p: dict(candidate=4 if p != 'shuffled_geometry' else 8, status='complete')
                  for p in driver.POLICIES})]
    return rows, decisions


def test_selection_preserves_shared_actions_and_stable_control_ties():
    rows, decisions = old_fixture()
    selected = driver.select_cases(rows, decisions)
    assert selected['counts'] == dict(prior_pairs=14, sensitive=2, selected_unique=2,
        stable_controls=2, zero_dose=2, cohort=7, independent=7, assignments=4)
    test_four = next(c for c in selected['census'] if c['key'] == ['test', 'test', 675, .001, 4])
    assert test_four['reasons'] == ['stable_control', 'selected-baseline', 'selected-geometry', 'selected-min_preview']
    assert len(selected['assignments']) == 4
    assert driver.select_cases(list(reversed(rows)), decisions) == selected


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'nan', 'boolean', 'failed'])
def test_old_census_rejects_bad_pairs_without_substitution(kind):
    rows, decisions = old_fixture()
    if kind == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif kind == 'missing':
        rows.pop()
    elif kind == 'failed':
        rows[0]['status'] = 'failed'
    else:
        rows[0]['rmse'] = float('nan') if kind == 'nan' else True
    with pytest.raises(ValueError):
        driver.select_cases(rows, decisions)


def test_no_stable_control_is_recorded_as_unmatched():
    rows, decisions = old_fixture()
    for r in rows:
        if r['role'] == 'development' and r['candidate'] >= 0 and r['dt'] == .01:
            r['rmse'] = .5
    result = driver.select_cases(rows, decisions)
    assert result['unmatched_controls'] == [['development', 'dev', 675, .001]]
    assert result['counts']['stable_controls'] == 1


def test_real_old_evidence_census_has_expected_cohort_and_sentinels():
    # Reads prior artifacts only. This cannot call any dynamics implementation.
    rows, decisions = driver.read_prior()
    result = driver.select_cases(rows, decisions)
    assert result['cohort_sha256'] == driver.COHORT_SHA
    assert result['counts'] == dict(prior_pairs=3504, sensitive=1012, selected_unique=220,
        stable_controls=108, zero_dose=48, cohort=1369, independent=19, assignments=384)
    assert sum(bool(r['independent_labels']) for r in result['census']) == 19
    assert sum(len(r['independent_labels']) for r in result['census']) == 20
    assert not any('sensitive' in c['reasons'] and any(s.startswith('selected-') for s in c['reasons'])
                   for c in result['census'])


def test_output_rejects_inside_repo_and_symlink_escape(tmp_path):
    with pytest.raises(ValueError):
        driver.external_attempt(driver.ROOT / 'evidence', 'attempt-001')
    with pytest.raises(ValueError):
        driver.external_attempt('relative', 'attempt-001')
    with pytest.raises(ValueError):
        driver.external_attempt(tmp_path, '../attempt-001')
    link = tmp_path / 'link'
    link.symlink_to(driver.ROOT, target_is_directory=True)
    with pytest.raises(ValueError):
        driver.external_attempt(link, 'attempt-001')
    assert driver.external_attempt(tmp_path, 'attempt-001') == tmp_path / 'EXP-009/attempt-001'


def test_native_call_shapes_match_new_steps_without_integration():
    from hopfield_drift_probe import integrate_trace
    for dt, expected_steps, expected_stride in [(.0025, 120000, 40), (.00125, 240000, 80)]:
        captured = {}

        def fake(alpha, gamma, initial, step, steps, stride, capacity, states, final, progress):
            captured.update(steps=steps, stride=stride, capacity=capacity)
            a = np.ctypeslib.as_array(states, shape=(capacity * 3,)).reshape(capacity, 3)
            a[:] = [1., 2., 3.]
            np.ctypeslib.as_array(final, shape=(3,))[:] = [1., 2., 3.]
            np.ctypeslib.as_array(progress, shape=(2,))[:] = [steps, capacity]
            return capacity

        times, states = integrate_trace(fake, 1., 10., [1., 2., 3.], dt)
        assert captured == dict(steps=expected_steps, stride=expected_stride, capacity=3001)
        np.testing.assert_array_equal(times, np.arange(3001) * .1)
        assert states.shape == (3001, 3)


def sleeping_worker(task, context, directory, library, parent_pid):
    time.sleep(10)


def complete_worker(task, context, directory, library, parent_pid):
    driver.write(directory / 'records' / f'{task["id"]}.json', dict(
        id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
        status='complete', rmse=.1, pid=os.getpid()))


def scheduler_fixture(tmp_path, n=4):
    (tmp_path / 'records').mkdir()
    specs = [dict(id=f'job-{i}', case=dict(id=f'case-{i}'), method=dict(id='test')) for i in range(n)]
    return specs, {f'case-{i}': {} for i in range(n)}


def test_per_task_timeout_joins_workers_and_keeps_complete_census(tmp_path):
    specs, contexts = scheduler_fixture(tmp_path)
    records = driver.execute_tasks(specs, contexts, tmp_path, '', workers=2,
        deadline=time.monotonic() + 3, per_task=.05, worker_fn=sleeping_worker)
    assert len(records) == 4 and {r['status'] for r in records} == {'task_timeout'}
    for row in records:
        with pytest.raises(ProcessLookupError):
            os.kill(row['pid'], 0)
    assert len((tmp_path / 'journal.jsonl').read_text().splitlines()) == 4


def test_whole_deadline_retains_unstarted_jobs_and_no_children(tmp_path):
    specs, contexts = scheduler_fixture(tmp_path, 6)
    records = driver.execute_tasks(specs, contexts, tmp_path, '', workers=2,
        deadline=time.monotonic() + .08, per_task=10, worker_fn=sleeping_worker)
    assert len(records) == 6
    assert sum(r['status'] == 'whole_attempt_timeout' for r in records) == 2
    assert sum(r['status'] == 'not_started_deadline' for r in records) == 4
    for row in records:
        if 'pid' in row:
            with pytest.raises(ProcessLookupError):
                os.kill(row['pid'], 0)


def test_scheduler_success_and_exclusive_journal(tmp_path):
    specs, contexts = scheduler_fixture(tmp_path)
    records = driver.execute_tasks(specs, contexts, tmp_path, '', workers=2,
        deadline=time.monotonic() + 3, worker_fn=complete_worker)
    assert len(records) == 4 and all(r['status'] == 'complete' for r in records)
    with pytest.raises(FileExistsError):
        driver.execute_tasks(specs, contexts, tmp_path, '', workers=2,
                             deadline=time.monotonic() + 3, worker_fn=complete_worker)


def test_interrupt_during_launch_still_joins_the_registered_child(tmp_path, monkeypatch):
    specs, contexts = scheduler_fixture(tmp_path, 2)
    real_context = driver.mp.get_context('fork')
    launched = []

    class LaunchContext:
        @staticmethod
        def Process(*args, **kwargs):
            process = real_context.Process(*args, **kwargs)
            real_start = process.start

            def interrupting_start():
                real_start()
                launched.append(process.pid)
                os.kill(os.getpid(), signal.SIGINT)

            process.start = interrupting_start
            return process

    monkeypatch.setattr(driver.mp, 'get_context', lambda method: LaunchContext())
    with pytest.raises(KeyboardInterrupt):
        driver.execute_tasks(specs, contexts, tmp_path, '', workers=2,
            deadline=time.monotonic() + 3, worker_fn=sleeping_worker)
    assert len(launched) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(launched[0], 0)
    records = [json.loads(s) for s in (tmp_path / 'journal.jsonl').read_text().splitlines()]
    assert [r['status'] for r in records] == ['interrupted', 'not_started_interruption']


def test_cleanup_accepts_a_never_started_owned_process():
    process = driver.mp.get_context('fork').Process(target=lambda: None)
    driver.terminate_owned(process)


def test_waveform_disagreement_is_not_hidden_by_equal_task_errors():
    times = np.arange(3001) * .1
    a = np.column_stack([np.sin(times), np.cos(times), np.sin(2 * times)])
    b = -a
    comparison = driver.array_comparison(a, b, times)
    assert comparison['tail_activation_rms'] > .1
    assert comparison['activation_waveform_agreement'] is False
    assert driver.array_comparison(a, a, times)['activation_waveform_agreement'] is True


@pytest.mark.parametrize('mutated', [dict(rmse=.1, phase=.2, windows=[.1], cadence=True),
                                   dict(rmse=.1, phase=0., windows=[.9], cadence=True),
                                   dict(rmse=.1, phase=0., windows=[.1], cadence=False)])
def test_replay_checks_phase_windows_and_cadence_even_when_aggregate_matches(mutated):
    saved = dict(rmse=.1, phase=0., windows=[.1], cadence=True)
    with pytest.raises(ValueError, match='Scoring'):
        driver.compare_scoring(saved, mutated)


def test_source_guard_rejects_dirty_source_before_network(monkeypatch):
    calls = []

    def fake(*args):
        calls.append(args)
        return '?? uncommitted.py'

    monkeypatch.setattr(driver, 'git', fake)
    with pytest.raises(ValueError, match='clean'):
        driver.source_guard({}, driver.PLAN)
    assert calls == [('status', '--porcelain', '--untracked-files=all')]


def test_source_guard_requires_live_exact_pushed_head(monkeypatch):
    import hashlib
    content = b'committed bytes'
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(driver, 'sha', lambda path: digest)
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda *a, **k: content)

    def fake_git(*args):
        if args[0] == 'status':
            return ''
        if args[0] == 'rev-parse':
            return 'a' * 40
        if args[0] == 'symbolic-ref':
            return 'codex/synthetic'
        if args[0] == 'ls-remote':
            return 'b' * 40 + '\trefs/heads/codex/synthetic'
        raise AssertionError(args)

    monkeypatch.setattr(driver, 'git', fake_git)
    with pytest.raises(ValueError, match='pushed'):
        driver.source_guard({}, driver.PLAN)


@pytest.mark.parametrize('field,value', [('workers', 8), ('metric_stability_atol', .01),
                                       ('planned_traces', 2775), ('paid_compute', True)])
def test_frozen_plan_rejects_budget_tolerance_and_census_changes(field, value):
    plan = json.loads(driver.PLAN.read_text())
    plan[field] = value
    with pytest.raises(ValueError):
        driver.validate(plan)


def test_owned_compiler_timeout_does_not_leave_parent_process():
    with pytest.raises(Exception):
        driver.checked_command([sys.executable, '-c', 'import time; time.sleep(10)'], .03)


def test_summary_missing_shared_action_keeps_policy_undefined_and_independent_pairs(tmp_path):
    rows, decisions = old_fixture()
    plan = dict(**driver.select_cases(rows, decisions), methods=driver.METHODS, scope='synthetic')
    path = tmp_path / 'synthetic.npz'
    driver.atomic_npz(path, times=np.arange(3001) * .1, states=np.zeros((3001, 3)))
    records = []
    for task in driver.tasks(plan):
        failed = task['case']['key'] == ['test', 'test', 675, .001, 4] and task['method']['id'] == 'rk4-0.00125'
        records.append(dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
            status='task_timeout' if failed else 'complete', rmse=None if failed else .1,
            raw='synthetic.npz'))
    summary = driver.summarize(plan, records, tmp_path)
    assert summary['all_complete'] is False
    fine = summary['fixed_policy_sensitivity']['rk4-0.00125']
    for policy in ['baseline', 'geometry', 'min_preview']:
        assert fine[policy]['mean_rmse'] is None
    assert fine['shuffled_geometry']['mean_rmse'] == .1
    assert fine['baseline_minus_geometry']['mean'] is None
    bad = next(c for c in summary['details'] if c['key'] == ['test', 'test', 675, .001, 4])
    assert bad['d2'] is None and bad['rmse_stable_at_this_resolution'] is None
    assert bad['waveform_comparisons']['dop853-1e-11_vs_dop853-1e-12']['rmse_difference'] == 0
    assert bad['independent_rmse_agreement'] is None
    assert len(summary['fixed_action_sensitivity']) == 4


def test_summary_rejects_incomplete_or_duplicate_terminal_inventory(tmp_path):
    rows, decisions = old_fixture()
    plan = dict(**driver.select_cases(rows, decisions), methods=driver.METHODS, scope='synthetic')
    records = [dict(id=t['id'], status='failed', rmse=None) for t in driver.tasks(plan)]
    with pytest.raises(ValueError, match='census'):
        driver.summarize(plan, records[:-1], tmp_path)
    with pytest.raises(ValueError, match='census'):
        driver.summarize(plan, records + records[:1], tmp_path)
