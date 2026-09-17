"""Synthetic EXP-018 -> EXP-019 -> unchanged EXP-016 scientific API tests.

Every numerical kernel is explicitly injected. The nine-maximum oscillator has
an analytic flow and tangent; native-ABI callbacks below return that exact flow,
not RK4 approximations. These tests establish interfaces, event operations,
serialization, scoring and missingness semantics, not neural solver accuracy.
No historical target packet, native library, fitting or network is accessed.
"""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import periodic_reference_refinement as refine
import refined_two_write_context as adapter
import run_phase_memory_recovery as codec
import two_write_acquisition as acquire
import two_write_evaluation as evaluate
import test_periodic_reference_refinement_integration as oscillator


def exact_states(initial, elapsed):
    """Vectorized closed flow, separate from the producer's numerical solvers."""
    times = np.asarray(elapsed, dtype=np.float64)
    xy = initial[:2]
    decay = np.exp(-2 * oscillator.K * times)
    scale = 1 / np.sqrt(xy @ xy + (1 - xy @ xy) * decay)
    c, s = np.cos(times), np.sin(times)
    states = np.empty((len(times), 3), dtype=np.float64)
    states[:, 0] = scale * (c * xy[0] - s * xy[1])
    states[:, 1] = scale * (s * xy[0] + c * xy[1])
    wave = np.real((states[:, 0] + 1j * states[:, 1]) ** 9)
    states[:, 2] = wave + np.exp(-oscillator.LAMBDA * times) * (
        initial[2] - oscillator.wave(xy))
    states[times == 0] = initial
    return states


class ExactFiniteKernels:
    """Both explicit acquisition interfaces, with a counted synthetic call log."""
    def __init__(self):
        self.native_calls = []
        self.adaptive_calls = []

    def native(self, alpha, gamma, initial, dt, steps, stride, count,
               output, final, progress):
        assert (alpha, gamma) == (0., 0.)
        x = np.ctypeslib.as_array(initial, shape=(3,)).copy()
        elapsed = np.arange(count, dtype=np.float64) * (stride * dt)
        # The caller clamps only its last recorded time. Compute the same
        # declared segment endpoint here, independent of the stride product.
        elapsed[-1] = steps * dt
        values = exact_states(x, elapsed)
        np.ctypeslib.as_array(output, shape=(3 * count,)).reshape(count, 3)[:] = values
        np.ctypeslib.as_array(final, shape=(3,))[:] = values[-1]
        np.ctypeslib.as_array(progress, shape=(2,))[:] = [steps, count]
        self.native_calls.append(dict(initial=x, dt=dt, steps=steps, samples=count))
        return count

    def adaptive(self, fun, span, initial, *, t_eval, **kwargs):
        assert initial.shape == (3,) and kwargs['method'] == 'DOP853'
        assert kwargs['rtol'] == 1e-13 and kwargs['atol'] == 1e-15
        values = exact_states(initial, t_eval - span[0])
        fun(span[0], initial)
        fun(span[1], values[-1])
        self.adaptive_calls.append(dict(initial=initial.copy(), span=span,
            samples=len(t_eval), method=kwargs['method']))
        return SimpleNamespace(t=t_eval.copy(), y=values.T, success=True,
            message='injected exact nine-maximum oscillator; no adaptive solve',
            nfev=2, njev=0, nlu=0)


def roundtrip(directory, name, packet):
    record = codec.save_packet(directory, name, packet)
    restored = codec.load_packet(record['path'])
    assert codec.packet_digest(restored) == codec.packet_digest(packet)
    return record, restored


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp('refined-two-write-exact-oscillator')
    phases = evaluate.phase_census()  # Fixed regression fractions, no target data.
    old = oscillator.reference()
    shots = {method: refine.refine_cycle(old, method, rhs=oscillator.rhs,
        jacobian=oscillator.jacobian, solver=oscillator.exact_solver)
        for method in refine.METHODS}
    pair = refine.crossverify_reference(old, shots, phases, rhs=oscillator.rhs,
        jacobian=oscillator.jacobian, solver=oscillator.exact_solver)
    assert pair['complete'] and pair['qualified']
    pair_record, pair = roundtrip(directory, 'refined-reference-pair', pair)
    source = dict(experiment_id='EXP-018', source_commit='a' * 40,
        plan_sha256='b' * 64, receipt_sha256='c' * 64,
        reference_id=pair['id'], packet_sha256=pair_record['sha256'],
        packet_semantic_sha256=pair_record['semantic_sha256'])
    context = adapter.prepare_context(pair, phases[0], source_binding=source,
        rhs=oscillator.rhs)
    _, context = roundtrip(directory, 'new-context', context)
    reference_calls = []

    def tangent_solver(fun, span, initial, **kwargs):
        assert initial.shape == (12,)  # No adjoint or finite-state ambiguity.
        reference_calls.append(dict(initial=initial.copy(), span=span,
            method=kwargs['method'], samples=len(kwargs['t_eval'])))
        return oscillator.exact_solver(fun, span, initial, **kwargs)

    references = {}
    for method in acquire.TANGENT_METHODS:
        packet = acquire.acquire_tangent(context, method, rhs=oscillator.rhs,
            jacobian=oscillator.jacobian, solver=tangent_solver)
        _, references[method] = roundtrip(directory, 'fresh-reference-' + method, packet)
    physical = acquire.seal_context(context, references['dop853'], rhs=oscillator.rhs)
    seal = evaluate.seal_reference(physical, references['dop853'], references['rk45'])
    assert seal['complete'] and seal['qualified'], seal.get('error', seal['gates'])
    _, prepared = roundtrip(directory, 'sealed-readouts', dict(acquisition=physical, evaluation=seal))
    physical, seal = prepared['acquisition'], prepared['evaluation']
    histories = evaluate.history_census(phases)
    selected = [h for h in histories if h['context_id'] == context['id']]
    kernels = ExactFiniteKernels()
    scores, comparisons, trace_records = {}, [], {}
    examples = {}
    for history in selected:
        packets = {}
        for task in evaluate.task_census([history]):
            raw = acquire.acquire_trace(task, physical, native_fn=kernels.native,
                rhs=oscillator.rhs, solver=kernels.adaptive)
            record, saved = roundtrip(directory, task['id'], raw)
            trace_records[task['id']] = record
            score = evaluate.score_trace(task, saved, seal)
            assert score['complete'] and score['qualified'], score
            _, score = roundtrip(directory, task['id'] + '-score', score)
            replay = evaluate.score_trace(task, saved, seal)
            assert codec.packet_digest(score) == codec.packet_digest(replay)
            scores[task['id']] = score
            packets[task['method']] = saved
        for spec in evaluate.comparison_census([history]):
            left, right = spec['left_method'], spec['right_method']
            comparison = evaluate.compare_traces(history, packets[left], packets[right], seal,
                left_score=scores[spec['left_task_id']], right_score=scores[spec['right_task_id']],
                methods=(left, right))
            assert comparison['complete'] and comparison['qualified'], comparison
            comparisons.append(comparison)
        if history['first_sign'] == 1 and history['second_sign'] == -1:
            examples[history['arm']] = packets['rk4-fine']
    summary = evaluate.summarize(histories, scores, comparisons, {context['id']: seal})
    _, summary = roundtrip(directory, 'partial-study-summary', summary)
    return dict(directory=directory, phases=phases, context=context, physical=physical,
        seal=seal, source=source, references=references, reference_calls=reference_calls,
        histories=histories, selected=selected, scores=scores, comparisons=comparisons,
        trace_records=trace_records, examples=examples, kernels=kernels, summary=summary)


def test_closed_flow_matches_independent_scalar_fixture():
    initial = np.array([.97, .14, .83], dtype=np.float64)
    times = np.array([0., .03, .7, 2.4, 8 * oscillator.PERIOD])
    expected = np.asarray([oscillator.exact_flow(initial, float(t))[0] for t in times])
    np.testing.assert_allclose(exact_states(initial, times), expected, rtol=2e-14, atol=2e-14)


def test_refined_pair_to_fresh_reference_seal_keeps_canonical_state_and_gradient(pipeline):
    p = pipeline
    assert p['context']['provenance']['source_binding'] == p['source']
    assert p['context']['new_normalizations'] == 0
    assert p['context']['provenance']['eight_period_reference_qualified'] is False
    assert 'template_velocity' not in p['context']
    assert len(p['reference_calls']) == 4
    for method in acquire.TANGENT_METHODS:
        packet = p['references'][method]
        np.testing.assert_array_equal(packet['pre']['initial'], p['context']['initial'])
        np.testing.assert_array_equal(packet['post']['initial'], packet['pre']['states'][-1])
        for segment, count in [('pre', 4097), ('post', 28673)]:
            assert packet[segment]['states'].shape == (count, 3)
            np.testing.assert_array_equal(packet[segment]['phi'][0], np.eye(3))
    np.testing.assert_array_equal(p['physical']['event_phase_gradient'], p['context']['phase_gradient'])
    assert p['seal']['qualified'] and all(p['seal']['gates'].values())
    assert set(p['seal']['windows']) == {'immediate', 'k2', 'k4', 'k8'}


def test_all_twenty_seven_histories_use_full_grids_and_truthful_injected_interfaces(pipeline):
    p = pipeline
    assert len(p['selected']) == 27 and len(p['scores']) == 81
    assert len(p['trace_records']) == 81 and len(p['comparisons']) == 54
    assert len(p['kernels'].native_calls) == 108
    assert len(p['kernels'].adaptive_calls) == 54
    assert {x['samples'] for x in p['kernels'].native_calls} == {4097, 28673}
    for spec in evaluate.comparison_census(p['selected']):
        comparison = next(row for row in p['comparisons'] if row['id'] == spec['id'])
        assert comparison['qualified'] and all(comparison['gates'].values())
        for segment in ('pre', 'post'):
            assert comparison['segments'][segment]['state']['max_abs'] < 1e-12
            assert comparison['segments'][segment]['activation']['max_abs'] < 1e-12


def test_retention_then_second_write_and_unchanged_pre_event_history(pipeline):
    p = pipeline
    before = p['examples']['identity']['pre']['states']
    for arm, packet in p['examples'].items():
        np.testing.assert_array_equal(packet['pre']['states'], before)
        event = packet['event']
        assert event['operation_order'] == 'retention_then_second_write'
        H = {'identity': np.eye(3), 'timing': p['physical']['Q'], 'transverse': p['physical']['P']}[arm]
        expected = (event['before'].copy() if arm == 'identity' else
            p['physical']['reference_at_event'] + H @ (event['before'] - p['physical']['reference_at_event']))
        np.testing.assert_array_equal(event['after_retention'], expected)
        np.testing.assert_array_equal(event['after'], expected - evaluate.EPSILON * p['physical']['direction'])
        np.testing.assert_array_equal(packet['post']['states'][0], event['after'])
    np.testing.assert_array_equal(p['examples']['identity']['event']['before'],
        p['examples']['identity']['event']['after_retention'])


def test_fixed_zero_write_baseline_and_independent_gaussian_probability(pipeline):
    p = pipeline
    for score in p['scores'].values():
        task = score['task']
        for window, observed in score['windows'].items():
            primary = observed['primary']
            expected = .5 if task['first_sign'] == 0 else float(ndtr(
                task['first_sign'] * primary['standardized_mean']))
            assert primary['probability_correct'] == pytest.approx(expected, rel=1e-12, abs=1e-14)
            assert observed['activation_displacement'].shape == (96,)
            assert observed['derivative']['available']
            if task['first_sign'] == task['second_sign'] == 0:
                assert observed['zero_zero_drift_pass'] is True
            else:
                assert observed['zero_zero_drift_pass'] is None
        # All arms/second signs have the same pre-event pulse and readout.
        match = next(s for s in p['scores'].values() if s['task']['first_sign'] == task['first_sign']
            and s['task']['arm'] == 'identity' and s['task']['second_sign'] == 0
            and s['task']['method'] == task['method'])
        np.testing.assert_array_equal(score['windows']['immediate']['activation_displacement'],
            match['windows']['immediate']['activation_displacement'])
    for window in p['seal']['windows'].values():
        assert window['baseline'].shape == (32, 3) and window['h'].shape == (96,)


def test_single_synthetic_context_does_not_create_a_study_estimate(pipeline):
    summary = pipeline['summary']
    assert not summary['complete'] and not summary['qualified']
    assert summary['counts']['scores_complete'] == 81
    assert summary['counts']['comparisons_complete'] == 54
    assert len(summary['scores']) == 2700 and len(summary['comparisons']) == 1404
    assert len(summary['contexts']) == 48 and len(summary['balanced_accuracies']) == 576
    assert summary['contexts']['core--p00']['qualified']
    assert sum(c['qualified'] for c in summary['contexts'].values()) == 1
    assert summary['primary'] is None
    assert all(row['means'] is None for row in summary['settings'].values())
    assert summary['primary_failure_bounds']['edge_minus_core_percentage_points'] == [-100., 100.]


def test_missing_saved_method_retains_failure_identity_and_planned_denominator(pipeline):
    p = pipeline
    history = p['selected'][0]
    task = evaluate.task_census([history])[0]
    scores = dict(p['scores'])
    scores.pop(task['id'])
    comparisons = [c for c in p['comparisons'] if c['left_task_id'] != task['id']]
    summary = evaluate.summarize(p['histories'], scores, comparisons, {p['context']['id']: p['seal']})
    missing = next(s for s in summary['scores'] if s['task_id'] == task['id'])
    assert missing['status'] == 'unavailable' and not missing['qualified']
    assert summary['counts']['scores_complete'] == 80
    assert len(summary['scores']) == 2700 and len(summary['comparisons']) == 1404
    assert not summary['contexts'][history['context_id']]['qualified']


def test_saved_packet_direction_identity_cannot_be_reinterpreted(pipeline):
    p = pipeline
    packet = copy.deepcopy(p['examples']['identity'])
    packet['event']['after_retention'][0] += 1e-5
    score = evaluate.score_trace(packet['task'], packet, p['seal'])
    assert not score['complete'] and not score['qualified']
    assert 'retention' in score['error']
