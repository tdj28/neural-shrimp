"""Synthetic failure-preservation tests; no integrations or new outcomes."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import audit_information_platform as audit


def fixture():
    packet = dict(status='complete', packet_sha256='saved', models={
        p:dict(coefficients=[.2, .3], chosen_lambda=1e-6, chosen_lambda_index=0)
        for p in audit.LEARNERS})
    scores = (np.arange(18)*.01+.1).tolist()
    decisions = [dict(anchor='a', block='b', phase=675, radius=.016,
        status='complete', candidate_ids=list(range(18)), model_packet_sha256='saved',
        required_preview_calls={p:18 for p in audit.POLICIES}, preview_scores=scores,
        predictions={p:scores.copy() for p in audit.LEARNERS},
        policies={p:dict(status='complete', candidate=0, selection_score=.1) for p in audit.POLICIES})]
    return packet, decisions


def test_collects_every_failure_and_keeps_unchanged_tolerance():
    left = dict(coefficients=[.2, .3, .4], index=2, flag=True)
    right = dict(coefficients=[.2+1e-4, .3-1e-4, .4], index=3, flag=1)
    result = audit.compare_all(left, right)
    assert result['failure_count'] == 4
    assert {f['path'] for f in result['failures']} == {
        'model.coefficients[0]', 'model.coefficients[1]', 'model.index', 'model.flag'}
    assert result['rtol'] == 1e-7 and result['atol'] == 1e-9
    assert result['float_comparisons'] == 3 and not result['passed']
    for failure in result['failures'][:2]:
        assert failure['error_over_allowance'] > 1


def test_recorded_linux_feature_difference_stays_failed():
    saved, rebuilt = .0036248908761150602, .0036248908761290386
    result = audit.compare_all([saved], [rebuilt], rtol=1e-12, atol=1e-14, path='base')
    assert result['failure_count'] == 1
    failure = result['failures'][0]
    assert failure['absolute_difference'] == abs(saved-rebuilt)
    assert failure['allowed_difference'] == 1e-14+1e-12*abs(rebuilt)
    assert failure['absolute_difference'] > failure['allowed_difference']


def test_coefficient_failure_can_finish_diagnostic_but_never_passes_frozen_check():
    packet, decisions = fixture()
    refit = copy.deepcopy(packet)
    refit['models']['baseline']['coefficients'][1] += 1e-4
    result = audit.assess_packets(packet, refit, decisions, copy.deepcopy(decisions), copy.deepcopy(decisions))
    assert result['diagnostic_exit_code'] == 0
    assert not result['strict_model_replay_passed']
    assert result['diagnostic_status'] == 'completed_with_frozen_coefficient_failures'
    assert result['coefficient_failure_counts'] == {'baseline':1}
    assert result['model_comparison']['failure_count'] == 1
    assert result['decisions']['refitted_actions_unchanged']
    assert result['decisions']['candidate_prediction_count'] == 54


@pytest.mark.parametrize('change', ['penalty', 'action', 'saved_prediction', 'nonfinite', 'metadata'])
def test_critical_platform_changes_cannot_exit_successfully(change):
    packet, sealed = fixture()
    refit, saved_decisions, new_decisions = copy.deepcopy(packet), copy.deepcopy(sealed), copy.deepcopy(sealed)
    if change == 'penalty':
        refit['models']['baseline'].update(chosen_lambda=.01, chosen_lambda_index=2)
    elif change == 'action':
        new_decisions[0]['policies']['baseline']['candidate'] = 1
        new_decisions[0]['predictions']['baseline'][1] = .01
    elif change == 'saved_prediction':
        saved_decisions[0]['predictions']['baseline'][0] += .01
    elif change == 'nonfinite':
        new_decisions[0]['predictions']['geometry'][5] = float('nan')
    else:
        new_decisions[0]['anchor'] = 'another'
    result = audit.assess_packets(packet, refit, sealed, saved_decisions, new_decisions)
    assert result['diagnostic_exit_code'] == 2 and result['critical_failures']


def test_refitted_prediction_differences_and_near_ties_are_retained_without_new_threshold():
    _, sealed = fixture()
    new = copy.deepcopy(sealed)
    # Preserve action ordering but perturb scores more than the saved-model
    # replay tolerance: this is quantified, not relabeled a tolerance pass.
    new[0]['predictions']['baseline'][0] += .0001
    result = audit.compare_decisions(sealed, copy.deepcopy(sealed), new)
    values = result['conditions'][0]['learners']['baseline']
    assert values['max_refitted_prediction_difference'] == pytest.approx(.0001)
    assert values['sealed_best_to_second_gap'] == pytest.approx(.01)
    assert result['refitted_prediction_tolerance'] is None
    assert len(values['sealed_scores']) == len(values['refitted_model_platform_scores']) == 18
    assert result['refitted_actions_unchanged']


def test_raw_binding_failure_is_reported_and_cannot_trigger_fit(monkeypatch, tmp_path):
    def fail(_):
        raise ValueError('Raw artifact binding changed: injected')
    monkeypatch.setattr(audit, 'bound_inputs', fail)
    monkeypatch.setattr(audit.driver, 'fit_development', lambda *a: pytest.fail('No fit after binding failure'))
    result = audit.audit(tmp_path)
    assert not result['raw_bindings_passed']
    assert result['diagnostic_exit_code'] == 2 and result['diagnostic_status'] == 'diagnostic_error'
    assert 'Raw artifact binding changed' in result['error']
    assert result['frozen_gate_not_replaced'] and not result['full_frozen_verifier_run']


def test_changed_receipt_is_rejected_before_loading_features(tmp_path):
    (tmp_path/'receipt.json').write_text('{"status":"complete"}')
    with pytest.raises(ValueError, match='Original receipt binding changed'):
        audit.bound_inputs(tmp_path)
