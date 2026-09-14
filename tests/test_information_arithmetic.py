"""Synthetic algebra checks only; no research outcome reads or integrations."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import diagnose_information_arithmetic as arithmetic


def test_projection_removes_phase_component_and_handles_zero_velocity():
    rng = np.random.default_rng(72)
    s = rng.normal(size=(9, 3, 2))
    v = rng.normal(size=(9, 3))
    residual, coefficient = arithmetic.projected_columns(s, v)
    np.testing.assert_allclose(v.ravel() @ residual, 0, atol=3e-14)
    np.testing.assert_allclose(residual + v.ravel()[:, None]*coefficient, s.reshape(-1, 2))
    residual, coefficient = arithmetic.projected_columns(s, np.zeros_like(v))
    np.testing.assert_array_equal(residual, s.reshape(-1, 2))
    np.testing.assert_array_equal(coefficient, [0., 0.])


def test_directional_norm_avoids_gram_cancellation():
    common = 1e8*np.arange(1, 7.)
    offset = np.array([0., 1., -1., 2., -2., .5])
    s = np.column_stack((common, common+offset)).reshape(2, 3, 2)
    context = arithmetic.gram_context(s, np.zeros((2, 3)))
    result = arithmetic.directional_diagnostic(context, [1., -1.])
    expected = float(np.mean(offset**2))
    assert result['extended_reference_quadratic'] == pytest.approx(expected, rel=1e-14)
    assert result['direct_residual_quadratic'] == pytest.approx(expected, rel=1e-14)
    assert result['direct_rmse_reference_error'] < 1e-12
    assert result['frozen_rmse_reference_error'] > 1e-3
    assert result['directional_cancellation_ratio'] > 1e15
    # Keep the original failing path visible; never relabel it as repaired.
    assert result['gram_formation_error']+result['quadratic_evaluation_error'] > .1


def test_well_conditioned_gram_and_direct_methods_agree():
    rng = np.random.default_rng(421)
    context = arithmetic.gram_context(rng.normal(size=(10,3,2)), rng.normal(size=(10,3)))
    result = arithmetic.directional_diagnostic(context, [.001, -.003])
    assert result['frozen_clamped_rmse'] == pytest.approx(result['direct_residual_rmse'], rel=1e-14)
    assert result['frozen_clamped_rmse'] == pytest.approx(result['extended_reference_rmse'], rel=1e-14)


@pytest.mark.parametrize('s,v', [
    (np.zeros((0,3,2)), np.zeros((0,3))),
    (np.zeros((2,3,2)), np.zeros((2,2))),
    (np.full((2,3,2), np.nan), np.zeros((2,3))),
])
def test_invalid_projection_inputs_are_rejected(s, v):
    with pytest.raises(ValueError):
        arithmetic.projected_columns(s, v)


def test_three_ridge_algorithms_match_closed_form_solution():
    x = np.array([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]])
    y = x @ [1.,2.]
    fit = arithmetic.solver_grid(x, y, ['one']*4, lambdas=(.1, 1.))
    for item in fit['results']:
        expected = np.array([1.,2.])/np.sqrt(2)/(1+item['penalty'])
        for result in item['solvers'].values():
            np.testing.assert_allclose(result['coefficients'], expected, rtol=1e-14, atol=1e-14)
            assert result['relative_stationarity'] < 1e-14
        assert item['regularized_normal_condition_number'] == pytest.approx(1.)


def test_ridge_records_ill_conditioning_without_threshold_relaxation():
    rng = np.random.default_rng(482)
    x0 = rng.normal(size=70)
    x = np.column_stack((x0, x0+1e-10*rng.normal(size=70), np.ones(70)))
    y = .3*x0+.01*rng.normal(size=70)
    result = arithmetic.solver_grid(x, y, ['a']*30+['b']*40, lambdas=(1e-6,))['results'][0]
    assert result['regularized_normal_condition_number'] > 1e6
    assert result['augmented_design_condition_number']**2 == pytest.approx(
        result['regularized_normal_condition_number'])
    assert set(result['solvers']) == set(arithmetic.SOLVERS)
    assert result['differences']['svd_vs_augmented_qr']['max_training_prediction_difference'] < 1e-10
    assert result['solvers']['svd']['stationarity_norm_extended'] < 1e-12


def test_all_fold_penalties_and_saved_coefficient_failures_are_retained():
    rng = np.random.default_rng(920)
    x = rng.normal(size=(48, 4))
    y = .3+x @ [.1, .05, -.02, .03]+rng.normal(scale=.003, size=48)
    groups = [f'g{i//12}' for i in range(48)]
    saved = arithmetic.model._fit_one(x, y, groups, ['a','b','c','d'])
    saved['coefficients'] = saved['coefficients'].copy()
    saved['coefficients'][0] += .01  # A synthetic saved failure must survive.
    report = arithmetic.ridge_diagnostic(x, y, groups, saved)
    assert len(report['folds']) == 4
    assert all(len(fold['grid']) == 5 for fold in report['folds'])
    assert len(report['final_grid']) == 5
    assert set(report['cv_mse']) == set(arithmetic.SOLVERS)
    for solver in arithmetic.SOLVERS:
        assert report['comparison_to_saved'][solver]['choice_matches_saved']
        assert not report['comparison_to_saved'][solver]['frozen_coefficient_comparison']['passed']


@pytest.mark.parametrize('change', ['nan', 'rows', 'lambda'])
def test_invalid_ridge_inputs_are_rejected(change):
    x, y, groups, penalties = np.eye(3), np.ones(3), ['a']*3, (1.,)
    if change == 'nan':
        x[0,0] = np.nan
    elif change == 'rows':
        groups = ['a']
    else:
        penalties = (0.,)
    with pytest.raises(ValueError):
        arithmetic.solver_grid(x, y, groups, penalties)


def test_changed_receipt_blocks_all_arithmetic(monkeypatch, tmp_path):
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip('Diagnostic deliberately unavailable without wider longdouble')
    monkeypatch.setattr(arithmetic.bindings, 'bound_inputs', lambda _: (_ for _ in ()).throw(
        ValueError('Original receipt binding changed')))
    with pytest.raises(ValueError, match='receipt binding'):
        arithmetic.diagnose(tmp_path)


def test_cli_rejects_output_inside_immutable_attempt_before_any_data_read(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['diagnose', str(tmp_path), '--out', str(tmp_path/'result.json'),
                                   '--expected-source', '0'*40])
    with pytest.raises(SystemExit) as error:
        arithmetic.main()
    assert error.value.code == 2
    assert not (tmp_path/'result.json').exists()


def test_cli_preserves_existing_report(monkeypatch, tmp_path):
    output = tmp_path/'prior.json'
    output.write_text('original')
    monkeypatch.setattr(sys, 'argv', ['diagnose', str(tmp_path/'attempt'), '--out', str(output),
                                   '--expected-source', '0'*40])
    with pytest.raises(SystemExit):
        arithmetic.main()
    assert output.read_text() == 'original'


def test_cli_rejects_repository_report(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['diagnose', str(tmp_path/'attempt'), '--out',
        str(arithmetic.ROOT/'new-report.json'), '--expected-source', '0'*40])
    with pytest.raises(SystemExit) as error:
        arithmetic.main()
    assert error.value.code == 2


def test_cli_requires_pushed_source_ref_before_data_read(monkeypatch, tmp_path):
    output = tmp_path/'report.json'
    monkeypatch.setattr(sys, 'argv', ['diagnose', str(tmp_path/'attempt'), '--out', str(output),
                                   '--expected-source', '0'*40])
    monkeypatch.setattr(arithmetic.subprocess, 'check_output', lambda cmd, **kwargs:
        '0'*40 if cmd[1]=='rev-parse' else '')
    monkeypatch.setattr(arithmetic, 'diagnose', lambda _: pytest.fail('Data must not be read'))
    with pytest.raises(SystemExit) as error:
        arithmetic.main()
    assert error.value.code == 2
    assert not output.exists()


def test_cli_retains_failed_attempt_provenance(monkeypatch, tmp_path):
    output = tmp_path/'report.json'
    monkeypatch.setattr(sys, 'argv', ['diagnose', str(tmp_path/'attempt'), '--out', str(output),
                                   '--expected-source', '0'*40])
    def git(cmd, **kwargs):
        return {'rev-parse': '0'*40, 'status': '', 'for-each-ref': 'refs/remotes/origin/diagnosis\n'}[cmd[1]]
    monkeypatch.setattr(arithmetic.subprocess, 'check_output', git)
    monkeypatch.setattr(arithmetic.platform, 'platform', lambda: 'synthetic-platform')
    def failed(_):
        assert json.loads(output.read_text())['diagnostic_status'] == 'started'
        raise ValueError('synthetic failed input')
    monkeypatch.setattr(arithmetic, 'diagnose', failed)
    with pytest.raises(SystemExit) as error:
        arithmetic.main()
    assert error.value.code == 2
    report = json.loads(output.read_text())
    assert report['diagnostic_status'] == 'failed'
    assert report['error'] == "ValueError('synthetic failed input')"
    assert report['diagnostic_source'] == '0'*40
    assert report['containing_origin_remote_refs'] == ['refs/remotes/origin/diagnosis']
