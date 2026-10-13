"""Tiny synthetic output fixtures only; no producer/data/model imports."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'codex/huginn-independent-replay/comparison.py'
spec = importlib.util.spec_from_file_location('huginn_comparison', PATH)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def fixtures_a(count=3, boot=5):
    common = {'requested': count, 'cohort': count, 'minimum_cohort': math.ceil(.75*count),
              'setup_failed': False, 'dispositions': {
                  str(q): {'status': 'ok', 'residual_F': 1e-6,
                           'arnoldi_converged': {'LM': True, 'SR': False, 'LR': True}, 'eigenvalues': 3}
                  for q in range(count)},
              'primary': {'statistic': .45, 'per_arm': dict(zip(c.A_ARMS, [.3, .4, .5, .6])),
                          'ci95': [.2, .7], 'undefined_draws': 0, 'success': False},
              'answer_changes_descriptive': {'detected_unstable_changed': 1, 'detected_unstable_total': 2,
                  'no_unstable_mode_detected_changed': 3, 'no_unstable_mode_detected_total': 10},
              'share_detected_unstable_by_arm': {a: .25 for a in c.A_ARMS}}
    p, i = deepcopy(common), deepcopy(common)
    p['args'] = {'command': 'items', 'spectra': 'synthetic', 'rows': 'synthetic'}
    i['cohort_indices'] = list(range(count))
    i['all_cohort_cells'] = [{'question': q, 'lambda': float(a), 'rho': .9,
                             'hash': 'synthetic', 'residual': 1e-5, 'changes': False}
                            for q in range(count) for a in c.A_ARMS]
    i['bootstrap_statistics'] = [.45] * boot
    return p, i


def fixtures_b(count=3):
    pair = {'difference': -.1, 'ci95': [-.2, .03], 'lower_one_sided_95': -.15,
            'mcnemar_less_p': .25, 'lost': 2, 'gained': 1}
    metric = {'accuracy': {a: .5 for a in c.B_ARMS}, **{a: deepcopy(pair) for a in c.B_ARMS[1:]}}
    contrast = {'arm': '1.20', 'difference': -.1, 'lost': 2, 'gained': 1, 'p': .25}
    common = {'n': count, 'metrics': {'correct': deepcopy(metric), 'correct_strict': deepcopy(metric)},
              'B1_decline': {**contrast, 'pass': False},
              'B3_noninferior': {'arm': '1.05', 'difference': -.1, 'lower_one_sided_95': -.15, 'margin': -.04, 'pass': False},
              'primary_success': False, 'B2_descriptive': {**contrast, 'arm': '1.15'},
              **{k: {a: .5 for a in c.B_ARMS} for k in ('cap_rate', 'tokens_median', 'strict_format_share')}}
    p, i = deepcopy(common), deepcopy(common)
    p['args'] = {'command': 'generation', 'rows': ['synthetic']}
    i['all_rows_retained'] = [{'arm': a, 'index': q, 'correct': q == 1,
                              'correct_strict': False, 'new_tokens': 10}
                             for a in c.B_ARMS for q in range(count)]
    return p, i


def compare(part, p, i):
    return c.compare_a(p, i, expected_count=3, expected_boot=5) if part == 'A' else c.compare_b(p, i, expected_count=3)


@pytest.mark.parametrize('part,fixture', [('A', fixtures_a), ('B', fixtures_b)])
def test_every_ordinary_field_compared_and_extras_retained(part, fixture):
    p, i = fixture(); before = deepcopy((p, i)); r = compare(part, p, i)
    assert r['comparison_passed'] and r['exact_result_match']
    assert not r['differences'] and not r['schema_errors']
    assert r['producer_only']['args'] == p['args']
    extras = c.A_EXTRA if part == 'A' else c.B_EXTRA
    assert set(r['independent_only']) == set(extras)
    assert all(r['independent_only'][k] == i[k] for k in extras)
    direct = c.A_DIRECT if part == 'A' else c.B_DIRECT
    assert all(any(row['path'].split('/')[1] == key for row in r['checks']) for key in direct)
    assert (p, i) == before
    # Report owns a copy, so later caller mutation cannot change released rows.
    p['args']['command'] = 'changed'
    assert r['producer_only']['args']['command'] != 'changed'


@pytest.mark.parametrize('delta,passed', [(0., True), (1e-13, True), (-1e-13, True), (1e-12, True), (1.0001e-12, False), (1e-6, False)])
def test_absolute_tolerance_without_rounding_or_relative_allowance(delta, passed):
    p, i = fixtures_b(); p['B2_descriptive']['difference'] = 0.; i['B2_descriptive']['difference'] = delta
    r = compare('B', p, i)
    assert r['comparison_passed'] is passed
    found = [v for v in r['differences'] if v['path'] == '/B2_descriptive/difference']
    assert len(found) == bool(delta)
    if delta:
        assert found[0]['independent_minus_producer'] == delta
        assert found[0]['status'] == ('within_tolerance' if passed else 'mismatch')
    p['B2_descriptive']['difference'] = 1e8; i['B2_descriptive']['difference'] = 1e8 + 1e-6
    assert not compare('B', p, i)['comparison_passed']


def test_all_nonconclusion_differences_survive():
    p, i = fixtures_b()
    i['metrics']['correct_strict']['1.15']['ci95'][1] += 1e-4
    i['tokens_median']['1.0'] += 1
    i['cap_rate']['1.20'] += 1e-13
    r = compare('B', p, i)
    assert not r['comparison_passed'] and p['primary_success'] == i['primary_success']
    assert {v['path'] for v in r['differences']} == {
        '/metrics/correct_strict/1.15/ci95/1', '/tokens_median/1.0', '/cap_rate/1.20'}


@pytest.mark.parametrize('bad', [True, 3.0, '3', -1])
def test_count_aliases_malformed_even_when_both_sides_match(bad):
    p, i = fixtures_b(); p['n'] = i['n'] = bad
    r = compare('B', p, i)
    assert not r['comparison_passed'] and len(r['schema_errors']) == 2


@pytest.mark.parametrize('bad', [0, 1., 'false', None])
def test_boolean_decisions_are_exact_typed(bad):
    p, i = fixtures_b(); p['primary_success'] = i['primary_success'] = bad
    assert not compare('B', p, i)['comparison_passed']


@pytest.mark.parametrize('part,fixture', [('A', fixtures_a), ('B', fixtures_b)])
def test_ordinary_missing_and_extra_fields_all_fail(part, fixture):
    p, i = fixture()
    del p['primary' if part == 'A' else 'B1_decline']
    p['new_result'] = 7
    i['unmapped_diagnostic'] = 9
    r = compare(part, p, i)
    assert not r['comparison_passed']
    assert {row['reason'] for row in r['schema_errors']} >= {'missing field', 'unexpected field'}
    assert any(row['path'] == '/unmapped_diagnostic' for row in r['schema_errors'])


@pytest.mark.parametrize('bad', [None, 'nan', math.inf, -math.inf, True])
def test_malformed_statistic_not_an_undefined_alias(bad):
    p, i = fixtures_a(); p['primary']['statistic'] = i['primary']['statistic'] = bad
    assert not compare('A', p, i)['comparison_passed']


@pytest.mark.parametrize('position', [0, 1, 2, 3])
def test_source_short_circuit_undefined_first_middle_last(position):
    p, i = fixtures_a()
    p['primary']['statistic'] = i['primary']['statistic'] = float('nan')
    i['primary']['per_arm'][c.A_ARMS[position]] = float('nan')
    p['primary']['per_arm'] = {a: p['primary']['per_arm'][a] for a in c.A_ARMS[:position]}
    p['primary']['ci95'] = i['primary']['ci95'] = [float('nan')] * 2
    r = compare('A', p, i)
    assert r['comparison_passed'] and not r['exact_result_match']
    assert len(r['representation_differences']) == 4 - position
    assert all(v['status'] == 'producer_source_short_circuit' for v in r['differences'])
    assert sum(v['first_undefined_arm'] for v in r['differences']) == 1
    assert any(v['status'] == 'undefined_match' for v in r['checks'])
    assert len(r['independent_only']['bootstrap_statistics']) == 5


@pytest.mark.parametrize('case', ['nonprefix', 'defined_omission', 'finite_statistic', 'nan_appended'])
def test_illicit_per_arm_omissions_or_values_fail(case):
    p, i = fixtures_a(); p['primary']['statistic'] = i['primary']['statistic'] = float('nan')
    if case == 'nonprefix':
        del p['primary']['per_arm']['1.05']
        i['primary']['per_arm']['1.05'] = float('nan')
    elif case == 'defined_omission':
        del p['primary']['per_arm']['1.1']
    elif case == 'finite_statistic':
        del p['primary']['per_arm']['1.1']; i['primary']['per_arm']['1.1'] = float('nan')
        p['primary']['statistic'] = .4
    else:
        p['primary']['per_arm']['1.1'] = i['primary']['per_arm']['1.1'] = float('nan')
    assert not compare('A', p, i)['comparison_passed']


@pytest.mark.parametrize('left,right,passed', [(None, None, True), (float('nan'), float('nan'), True),
    (math.inf, math.inf, True), (-math.inf, -math.inf, True), (None, float('nan'), False), (math.inf, -math.inf, False)])
def test_excluded_residual_undefined_kinds_are_explicit(left, right, passed):
    p, i = fixtures_a()
    p['dispositions']['0']['residual_F'] = left; i['dispositions']['0']['residual_F'] = right
    for result in (p, i):
        result['dispositions']['0']['status'] = 'no fixed point'
        result['dispositions']['0']['eigenvalues'] = 0
        result['cohort'] = 2
        result['setup_failed'] = True
    i['cohort_indices'] = [1, 2]
    i['all_cohort_cells'] = i['all_cohort_cells'][4:]
    r = compare('A', p, i)
    assert r['comparison_passed'] is passed
    row = next(v for v in r['checks'] if v['path'] == '/dispositions/0/residual_F')
    assert row['producer_class'] == c._class(left) and row['independent_class'] == c._class(right)


@pytest.mark.parametrize('field', ['dispositions', 'cohort_indices', 'all_cohort_cells', 'bootstrap_statistics'])
def test_a_indices_dispositions_and_diagnostic_coverage(field):
    p, i = fixtures_a()
    if field == 'dispositions':
        del p[field]['1']
    elif field == 'cohort_indices':
        i[field][0] = 1
    elif field == 'all_cohort_cells':
        i[field][0]['question'] = 1
    else:
        i[field].pop()
    assert not compare('A', p, i)['comparison_passed']


def test_disposition_decision_and_arnoldi_changes_are_exact():
    p, i = fixtures_a(); i['dispositions']['1']['arnoldi_converged']['SR'] = True
    i['dispositions']['2']['status'] = 'no fixed point'; i['primary']['success'] = True
    r = compare('A', p, i)
    assert not r['comparison_passed']
    assert {v['path'] for v in r['differences']} >= {
        '/dispositions/1/arnoldi_converged', '/dispositions/2/status', '/primary/success'}


@pytest.mark.parametrize('part,fixture', [('A', fixtures_a), ('B', fixtures_b)])
def test_args_are_required_object_but_not_compared_to_absent_replay_args(part, fixture):
    p, i = fixture(); del p['args']; assert not compare(part, p, i)['comparison_passed']
    p['args'] = 'wrong'; assert not compare(part, p, i)['comparison_passed']
    p['args'] = {'deliberately': 'different path labels are admitted by caller'}
    assert compare(part, p, i)['comparison_passed']


@pytest.mark.parametrize('bad', [float('nan'), math.inf, None, True])
def test_b_all_numerics_must_be_finite(bad):
    p, i = fixtures_b(); p['metrics']['correct_strict']['1.15']['ci95'][0] = bad
    assert not compare('B', p, i)['comparison_passed']


def test_b_all_diagnostic_indices_and_rows_retained_without_claimed_producer_comparison():
    p, i = fixtures_b(); i['all_rows_retained'][0]['index'] = 1
    assert not compare('B', p, i)['comparison_passed']
    p, i = fixtures_b(); i['all_rows_retained'][0]['correct'] = True
    r = compare('B', p, i)
    assert r['comparison_passed']
    assert r['independent_only']['all_rows_retained'][0]['correct'] is True
    assert not any('/all_rows_retained' in row['path'] for row in r['checks'])


def test_top_level_malformed_returns_all_available_errors_without_silent_pass():
    p, i = fixtures_b()
    for a, b in [(None, i), (p, []), ({}, {})]:
        r = compare('B', a, b)
        assert not r['comparison_passed'] and r['schema_errors']


def test_count_and_boot_overrides_are_positive_exact_integers():
    p, i = fixtures_a()
    for count in (True, 3., 0, -1):
        with pytest.raises(ValueError):
            c.compare_a(p, i, expected_count=count)
    with pytest.raises(ValueError):
        c.compare_a(p, i, expected_boot=0)


def leaf_paths(value, path=()):
    if type(value) is dict:
        for key, child in value.items():
            yield from leaf_paths(child, path + (key,))
    elif type(value) is list:
        for j, child in enumerate(value):
            yield from leaf_paths(child, path + (j,))
    else:
        yield path, value


@pytest.mark.parametrize('part,fixture', [('A', fixtures_a), ('B', fixtures_b)])
def test_each_declared_producer_result_leaf_individually_detected(part, fixture):
    original, i = fixture()
    checked = 0
    for path, value in leaf_paths(original):
        if path[0] == 'args':
            continue
        p = deepcopy(original); node = p
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = not value if type(value) is bool else value + 1 if type(value) in (int, float) else value + '_changed'
        r = compare(part, p, i)
        assert not r['comparison_passed'], path
        assert r['differences'] or r['schema_errors'], path
        checked += 1
    assert checked >= 30


@pytest.mark.parametrize('residual', [None, float('nan'), math.inf, -1e-7, .1])
def test_matching_nonqualifying_residual_cannot_be_called_an_ok_disposition(residual):
    p, i = fixtures_a()
    p['dispositions']['0']['residual_F'] = i['dispositions']['0']['residual_F'] = residual
    r = compare('A', p, i)
    assert not r['comparison_passed']
    assert any('inconsistent' in row['reason'] for row in r['schema_errors'])
