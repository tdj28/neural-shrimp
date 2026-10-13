"""Synthetic-only tests: no data/model/producer module imports."""
from pathlib import Path
import copy
import importlib.util
import math

import numpy as np
import pytest
from scipy.stats import binom, rankdata, spearmanr

PATH = Path(__file__).resolve().parents[1] / 'codex/huginn-independent-replay/statistics.py'
spec = importlib.util.spec_from_file_location('independent_huginn_statistics', PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def a_fixture(count=20):
    spectra, rows = {}, []
    for q in range(count):
        digest = f'{q:064x}'
        spectra[q] = {'fixed_point_residual_F': 1e-6, 'token_ids_sha256': digest,
                      'arnoldi_converged': {'LM': True, 'SR': False, 'LR': True},
                      'spectra_of_F': {'LM': [[.70 + q * .009, .03]], 'SR': [[-.3, .01]],
                                       'LR': [[.70 + q * .009, .03]]}}
        for arm in s.A_ARMS:
            endpoint = [float(q + 1) * arm, 2.0]
            rows.append({'index': q, 'lam': arm, 'split': 'test', 'seed': q, 'gold': 0,
                         'gold_ids_sha256': digest, 'native_residual_rows': [endpoint[:] for _ in range(32)],
                         'by_steps': {str(k): {'loglik': [-1., -2.] if k < 32 else [-3., -2.]}
                                      for k in range(24, 33)}})
    return spectra, rows


def b_fixture(count=20):
    return [{'index': q, 'arm': a, 'lam': float(a), 'split': 'test', 'steps': 32,
             'prompt_style': 'cot8', 'seed': q,
             'correct': bool(q % 3 == 0 or (a == '1.05' and q % 5 == 0)),
             'correct_strict': bool(q % 3 == 0), 'new_tokens': 256 if q % 4 == 0 else 17 + q,
             'strict_pred': '1' if q % 3 == 0 else None}
            for a in s.B_ARMS for q in range(count)]


@pytest.mark.parametrize('values', [[2, 1, 2, 5, 1], [0, 0, 0], [-4, 1, 0, 1], [3]])
def test_manual_tie_ranks_match_independent_library(values):
    np.testing.assert_array_equal(s.average_ranks(values), rankdata(values))


def test_rank_correlation_bootstrap_multisets():
    x = np.array([0, 1, 1, 3, 7.]); y = np.array([2, 0, 0, 8, 5.])
    idx = np.array([4, 4, 0, 1, 1, 1, 3])
    assert s.rank_correlation(x[idx], y[idx]) == pytest.approx(spearmanr(x[idx], y[idx]).statistic, abs=1e-15)
    assert math.isnan(s.rank_correlation(np.ones(5), y))


@pytest.mark.parametrize('lost,gained', [(0, 0), (3, 1), (1, 3), (200, 0), (100, 100)])
def test_exact_tail_matches_binomial_reference(lost, gained):
    base = np.array([True] * lost + [False] * gained, dtype=bool)
    other = ~base
    p, b, c = s.exact_paired_decline(base, other)
    expected = binom.sf(lost - 1, lost + gained, .5) if lost + gained else 1.
    assert (b, c) == (lost, gained)
    assert p == pytest.approx(expected, rel=1e-13, abs=1e-60)


def test_a_frozen_rng_and_multiset_bootstrap_matches_scipy():
    spectra, rows = a_fixture()
    result = s.analyze_a(spectra, rows, count=20, boot=40)
    records = result['all_cohort_cells']; rng = np.random.default_rng(0); reference = []
    for _ in range(40):
        questions = rng.choice(list(range(20)), 20)
        arms = []
        for arm in s.A_ARMS:
            r = {v['question']: v for v in records if v['lambda'] == arm}
            arms.append(spearmanr([r[q]['rho'] for q in questions], [r[q]['residual'] for q in questions]).statistic)
        reference.append(np.mean(arms))
    np.testing.assert_allclose(result['bootstrap_statistics'], reference, atol=1e-14, rtol=0)
    np.testing.assert_allclose(result['primary']['ci95'], np.percentile(reference, [2.5, 97.5]), atol=1e-14)
    assert result['primary']['success'] is True
    assert result['cohort_indices'] == list(range(20))
    assert len(result['dispositions']) == 20 and len(records) == 80
    assert sum(result['answer_changes_descriptive'][k] for k in ['detected_unstable_changed', 'no_unstable_mode_detected_changed']) == 80


def test_same_pcg64_indices_as_choice_of_noncontiguous_cohort():
    cohort = np.array([0, 1, 4, 6, 12, 14, 17])
    left, right = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(10):
        np.testing.assert_array_equal(left.choice(cohort, len(cohort)), cohort[right.integers(len(cohort), size=len(cohort))])


def test_all_arm_hash_and_common_cohort():
    spectra, rows = a_fixture()
    rows[5 * 4 + 1]['gold_ids_sha256'] = 'different'
    del rows[7 * 4 + 3]['gold_ids_sha256']
    result = s.analyze_a(spectra, rows, count=20, boot=40)
    assert result['cohort'] == 18 and 5 not in result['cohort_indices'] and 7 not in result['cohort_indices']
    assert all(result['dispositions'][str(q)]['status'].startswith('token ids differ') for q in [5, 7])
    assert all(r['question'] not in (5, 7) for r in result['all_cohort_cells'])


@pytest.mark.parametrize('edit', ['missing', 'duplicate', 'extra', 'seed', 'split', 'steps', 'trace', 'negative', 'nonfinite'])
def test_a_invalid_panel_refused(edit):
    spectra, rows = a_fixture()
    if edit == 'missing': rows.pop()
    elif edit == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif edit == 'extra': rows[0]['index'] = 100
    elif edit == 'seed': rows[0]['seed'] = 1
    elif edit == 'split': rows[0]['split'] = 'train'
    elif edit == 'steps': del rows[0]['by_steps']['31']
    elif edit == 'trace': rows[0]['native_residual_rows'].pop()
    elif edit == 'negative': rows[0]['native_residual_rows'][-1][0] = -1
    else: rows[0]['native_residual_rows'][-1][0] = math.inf
    with pytest.raises((ValueError, KeyError)):
        s.analyze_a(spectra, rows, count=20, boot=2)


def test_a_retains_all_dispositions_and_failed_setup():
    spectra, rows = a_fixture()
    spectra[0] = None; spectra[1]['fixed_point_residual_F'] = 1e-2
    spectra[2]['spectra_of_F'] = {}; spectra[3].pop('token_ids_sha256')
    spectra[4]['spectra_of_F'] = {'LM': [[float('inf'), 0]]}
    spectra[5] = None
    result = s.analyze_a(spectra, rows, count=20, boot=10)
    assert result['cohort'] == 14 and result['minimum_cohort'] == 15
    assert result['setup_failed'] and not result['primary']['success']
    assert result['dispositions']['0']['status'] == 'missing'
    assert result['dispositions']['1']['status'] == 'no fixed point'
    assert result['dispositions']['2']['status'] == 'empty or nonfinite spectrum'
    assert result['dispositions']['3']['status'] == 'no token-id record'
    assert result['dispositions']['3']['eigenvalues'] == 0
    assert result['dispositions']['4']['status'] == 'empty or nonfinite spectrum'


def test_one_constant_arm_means_no_dropped_arm_and_undefined_all_draws():
    spectra, rows = a_fixture()
    for row in rows:
        if row['lam'] == s.A_ARMS[2]: row['native_residual_rows'][-1][0] = 0.
    result = s.analyze_a(spectra, rows, count=20, boot=25)
    assert math.isnan(result['primary']['statistic']) and len(result['primary']['per_arm']) == 4
    assert result['primary']['undefined_draws'] == 25 and not result['primary']['success']


def test_b_paired_bootstrap_and_every_row():
    rows = b_fixture(); result = s.analyze_b(rows, count=20, boot=100)
    base = np.array([r['correct'] for r in rows if r['arm'] == '1.0'])
    other = np.array([r['correct'] for r in rows if r['arm'] == '1.05'])
    d = other.astype(float) - base.astype(float)
    idx = np.random.default_rng(0).integers(0, 20, size=(100, 20))
    expected = d[idx].mean(axis=1)
    np.testing.assert_array_equal(result['metrics']['correct']['1.05']['ci95'], np.percentile(expected, [2.5, 97.5]))
    assert result['metrics']['correct']['1.05']['lower_one_sided_95'] == np.percentile(expected, 5)
    assert len(result['all_rows_retained']) == 80
    assert result['cap_rate']['1.0'] == .25
    assert result['B1_decline']['pass'] is False and result['primary_success'] is False


@pytest.mark.parametrize('edit', ['missing', 'duplicate', 'extra', 'seed', 'split', 'steps', 'prompt', 'arm', 'boolean', 'length'])
def test_b_invalid_or_prefix_data_refused(edit):
    rows = b_fixture()
    if edit == 'missing': rows.pop()
    elif edit == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif edit == 'extra': rows[0]['index'] = 100
    elif edit == 'seed': rows[0]['seed'] = 1
    elif edit == 'split': rows[0]['split'] = 'train'
    elif edit == 'steps': rows[0]['steps'] = 31
    elif edit == 'prompt': rows[0]['prompt_style'] = 'zero'
    elif edit == 'arm': rows[0]['lam'] = 1.2
    elif edit == 'boolean': rows[0]['correct'] = 'false'
    else: rows[0]['new_tokens'] = 257
    with pytest.raises(ValueError): s.analyze_b(rows, count=20, boot=2)


def test_b_exact_boundary_is_not_noninferior(monkeypatch):
    original = s.pair_summary
    def boundary(base, other, **kw):
        result = original(base, other, **kw); result['lower_one_sided_95'] = -.04
        return result
    monkeypatch.setattr(s, 'pair_summary', boundary)
    assert not s.analyze_b(b_fixture(), count=20, boot=2)['B3_noninferior']['pass']
