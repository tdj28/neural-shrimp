"""EXP-010 synthetic/data-only tests; no new research trajectories."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_preview_refinement as driver


def fixture():
    old_plan = dict(anchors=[dict(role=r, id=r) for r in ['development', 'test']],
                    phases=[675], radii=[.001])
    rows = []
    for role in ['development', 'test']:
        for c in range(18):
            score = c / 100.
            rows.append(dict(role=role, anchor=role, phase=675, radius=.001, candidate=c,
                id=f'{role}-c{c}', block=role, raw=f'previews/{role}-{c}.npz',
                status='complete', preview_score=score, metrics=dict(aligned_rms=dict(aggregate=score))))
    decisions = [dict(anchor='test', phase=675, radius=.001, block='test', status='complete',
        policies={p: dict(status='complete', candidate=8 if p == 'geometry' else 0)
                  for p in driver.runtime.POLICIES})]
    return rows, decisions, old_plan


def small_plan():
    return dict(**driver.select_census(*fixture()), methods=driver.METHODS, scope='synthetic')


def test_full_candidate_product_and_old_only_sentinels():
    rows, decisions, old_plan = fixture()
    p = driver.select_census(rows, decisions, old_plan)
    assert p['counts'] == dict(cases=36, independent=4, selected_unique=2, assignments=4, contexts=2)
    assert {c['key'][-1] for c in p['census'] if c['independent_labels']} == {8, 17}
    assert driver.select_census(list(reversed(rows)), decisions, old_plan) == p
    assert all(s['median_index'] == 8 for s in p['independent_strata'])


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nonfinite', 'failed', 'boolean'])
def test_incomplete_original_labels_cannot_be_replaced(mutation):
    rows, decisions, old_plan = fixture()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'failed':
        rows[0]['status'] = 'failed'
    else:
        rows[0]['preview_score'] = float('nan') if mutation == 'nonfinite' else True
    with pytest.raises(ValueError):
        driver.select_census(rows, decisions, old_plan)


def test_real_prior_preview_census_and_sentinels_without_integration():
    result = driver.select_census(*driver.read_old())
    assert result['counts'] == dict(cases=3456, independent=16, selected_unique=220, assignments=384, contexts=192)
    assert result['cohort_sha256'] == '09ff668f80159af17c9740b7c92428eb752013f0bdd9c7cc19b81460fb52ba68'
    assert all(s['size'] == 432 and s['median_index'] == 215 for s in result['independent_strata'])
    assert sum(bool(c['independent_labels']) for c in result['census']) == 16


def context():
    return dict(times=np.arange(751) * .1, initial=np.zeros(3), delta=np.zeros(2),
                alpha=1., gamma=10., reference=np.zeros((2700, 3)), period=37., epoch=9.25, old_phase=.1)


def test_entire75_interval_and_original_epoch_enter_score(monkeypatch):
    c = context()
    seen = {}

    def fake(reference, times, activation, period, **kwargs):
        seen.update(times=times, activation=activation, period=period, kwargs=kwargs)
        return dict(aligned_rms=dict(aggregate=.1))

    monkeypatch.setattr(driver, 'waveform_errors', fake)
    result = driver.score(c, c['times'], np.zeros((751, 3)))
    assert result['rmse'] == .1
    assert len(seen['times']) == 751 and seen['times'][0] == 9.25 and seen['times'][-1] == 84.25
    assert seen['kwargs'] == dict(phase_grid=128)
    with pytest.raises(ValueError):
        driver.score(c, c['times'][1:], np.zeros((750, 3)))


@pytest.mark.parametrize('method', driver.METHODS)
def test_new_worker_uses75_not300_with_synthetic_solvers(tmp_path, monkeypatch, method):
    for name in ['records', 'traces']:
        (tmp_path / name).mkdir()
    c = context()
    seen = {}

    def fake_native(fn, alpha, gamma, initial, dt, **kw):
        seen.update(duration=kw['duration'], sample_dt=kw['sample_dt'], dt=dt)
        return c['times'], np.zeros((751, 3))

    def fake_adaptive(fn, span, initial, **kw):
        seen.update(span=span, **kw)
        return SimpleNamespace(t=c['times'], y=np.zeros((3, 751)), success=True, status=0,
                               message='synthetic', nfev=0)

    monkeypatch.setattr(driver, 'integrate_trace', fake_native)
    monkeypatch.setattr(driver.runtime, 'load_native', lambda path: None)
    monkeypatch.setattr(driver, 'solve_ivp', fake_adaptive)
    task = dict(id='synthetic', case=dict(id='case'), method=method)
    driver.worker(task, c, tmp_path, '', 0)
    result = json.loads((tmp_path / 'records/synthetic.json').read_text())
    assert result['status'] == 'complete'
    if method['kind'] == 'RK4':
        assert seen['duration'] == 75. and seen['sample_dt'] == .1
        assert result['solver']['steps'] == round(75 / method['dt'])
    else:
        assert seen['span'] == (0., 75.) and seen['t_eval'].shape == (751,)
        assert seen['rtol'] == method['rtol'] and seen['max_step'] == method['max_step']


def test_ranking_keeps_exact_ties_and_candidate_duplicates():
    result = driver.ranking([.1] * 18)
    assert result['order'] == list(range(18)) and result['argmin'] == 0
    assert result['top_four'] == [0, 1, 2, 3]
    assert result['best_second_gap'] == result['fourth_fifth_gap'] == 0
    assert result['signed_axes_argmin'] == 0
    assert driver.ranking([.1] * 17 + [None]) is None


def synthetic_records(plan, tmp_path):
    driver.runtime.atomic_npz(tmp_path / 'synthetic.npz', times=np.arange(751) * .1, states=np.zeros((751, 3)))
    return [dict(id=t['id'], case_id=t['case']['id'], method=t['method']['id'], status='complete',
                 rmse=t['case']['old_rmse'], raw='synthetic.npz') for t in driver.tasks(plan)]


def test_mandatory_independent_failure_gates_full_qualification(tmp_path):
    p = small_plan()
    records = synthetic_records(p, tmp_path)
    failed = next(r for r in records if r['case_id'] == 'test-c8' and r['method'] == 'dop853-1e-12')
    failed.update(status='task_timeout', rmse=None)
    s = driver.summarize(p, records, tmp_path)
    assert s['qualification']['native_all_required_complete'] is True
    assert s['qualification']['all_required_methods_complete'] is False
    assert s['qualification']['full_scalar_qualification'] is None
    assert s['original_selected_unique']['native_all_required_complete'] is True
    assert s['original_selected_unique']['all_required_methods_complete'] is False
    assert s['original_selected_unique']['full_scalar_qualification'] is None
    group = s['role_radius_groups']['test-r0.001-all']
    assert group['native_all_required_complete'] is True and group['full_group_scalar_qualification'] is None


def test_missing_native_label_makes_full_context_rankings_undefined(tmp_path):
    p = small_plan()
    records = synthetic_records(p, tmp_path)
    failed = next(r for r in records if r['case_id'] == 'test-c17' and r['method'] == 'rk4-0.00125')
    failed.update(status='failed', rmse=None)
    s = driver.summarize(p, records, tmp_path)
    test = next(x for x in s['context_rankings'] if x['context'][0] == 'test')
    assert test['rankings']['rk4-0.00125'] is None
    assert test['comparisons']['old_0.005_vs_rk4-0.00125'] is None
    assert s['qualification']['all_native_scalar_flags_pass'] is None


def test_ranking_flip_can_have_tiny_minimum_error_change(tmp_path):
    p = small_plan()
    for case in p['census']:
        if case['id'] in ['test-c0', 'test-c1']:
            case['old_rmse'] = .01 + (1e-8 if case['id'] == 'test-c1' else 0)
    records = synthetic_records(p, tmp_path)
    for r in records:
        if r['case_id'] == 'test-c1' and r['method'] == 'rk4-0.00125':
            r['rmse'] = .01 - 1e-8
    s = driver.summarize(p, records, tmp_path)
    c = next(x for x in s['context_rankings'] if x['context'][0] == 'test')['comparisons']['old_0.005_vs_rk4-0.00125']
    assert c['argmin_changed'] is True
    assert c['minimum_error_change_above_numerical_tolerance'] is False
    assert c['empirical_max_absolute_score_change'] == pytest.approx(2e-8)


def test_terminal_identity_mapping_rejects_a_valid_but_wrong_case():
    p = small_plan()
    rows = [dict(id=t['id'], case_id=t['case']['id'], method=t['method']['id'], status='failed') for t in driver.tasks(p)]
    rows[0]['case_id'] = rows[-1]['case_id']
    with pytest.raises(ValueError, match='case/method'):
        driver.validate_terminal_records(p, rows)


@pytest.mark.parametrize('field', ['method', 'raw', 'solver'])
def test_terminal_identity_and_solver_contract_cannot_alias_other_jobs(field):
    p = small_plan()
    rows = [dict(id=t['id'], case_id=t['case']['id'], method=t['method']['id'], status='failed') for t in driver.tasks(p)]
    row = rows[0]
    row.update(status='complete', raw=f'traces/{row["id"]}.npz',
               solver=dict(method='RK4', dt=.0025, steps=30000))
    if field == 'method':
        row['method'] = 'rk4-0.00125'
    elif field == 'raw':
        row['raw'] = 'traces/another-valid-job.npz'
    else:
        row['solver']['steps'] = 120000
    with pytest.raises(ValueError):
        driver.validate_terminal_records(p, rows)


def test_complete_attempt_requires_summary_and_planned_records(tmp_path):
    row = dict(id='case--method', case_id='case', method='method', status='complete', raw='traces/case--method.npz')
    with pytest.raises(ValueError, match='summary'):
        driver.require_completed_artifacts([row], tmp_path, {row['raw']})
    with pytest.raises(ValueError, match='trace/record'):
        driver.require_completed_artifacts([row], tmp_path, {'summary.json', row['raw']})


def test_final_exp010_output_segment_is_resolved_before_write(tmp_path):
    (tmp_path / 'EXP-010').symlink_to(driver.ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match='inside'):
        driver.external_attempt(tmp_path, 'attempt-001')


def test_native_waveform_comparison_uses_whole_preview():
    a = np.zeros((751, 3)); b = a.copy(); b[0, 0] = 1.
    assert driver.waveform_comparison(a, b)['activation_waveform_agreement'] is False
