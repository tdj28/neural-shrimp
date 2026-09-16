"""EXP-012 synthetic contracts only: no CUDA context or research integration."""
from __future__ import annotations

import copy
from concurrent.futures import Future
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_gpu_preview_qualification as driver


class InlinePool:
    def map(self, function, values):
        return map(function, values)

    def submit(self, function, value):
        future = Future()
        try:
            future.set_result(function(value))
        except BaseException as error:
            future.set_exception(error)
        return future


def batch_fixture(n=18, dt=.0025):
    ids = [f'synthetic-{i:02d}' for i in range(n)]
    contexts = {identity: dict(alpha=1.+i/100., gamma=10.+i/100.,
        initial=np.array([i/100., -.1, .2]), delta=np.array([.001, 0.]))
        for i, identity in enumerate(ids)}
    initial = np.array([contexts[i]['initial'] for i in ids])
    states = np.repeat(initial[:, None, :], 751, axis=1)
    states[:, 1:, 0] += np.linspace(0., .01, 750)
    arrays = dict(times=np.arange(751)*.1, states=states,
        statuses=np.full(n, 751, dtype=np.int32), final=states[:, -1].copy(),
        progress=np.tile([round(75/dt), 751], (n, 1)).astype(np.int64),
        alpha=np.array([contexts[i]['alpha'] for i in ids]),
        gamma=np.array([contexts[i]['gamma'] for i in ids]), initial=initial,
        delta=np.array([contexts[i]['delta'] for i in ids]))
    census = [dict(id=identity, key=['development', 'synthetic-origin', 675, .001, i])
              for i, identity in enumerate(ids)]
    scores = [full_score(i/100.) for i in range(n)]
    return ids, contexts, arrays, census, scores


def metric_fixture():
    return dict(rmse=.1, old_fixed_phase_rmse=.11, metrics=dict(
        aligned_rms=dict(aggregate=.1, components=[.08, .10, .12]),
        synchronous_rms=dict(aggregate=.11, components=[.09, .11, .13]),
        phase_offset_cycles=.125,
        phase_search=dict(method='synthetic', grid_bins=128,
                          refinement_success=True, refinement_evaluations=5),
        period=37., sample_count=751,
        windows=[dict(index=0, first_sample_time=0., last_sample_time=37.5,
                      aligned_rms=dict(aggregate=.1, components=[.08, .10, .12]))]))


def full_score(value):
    result = metric_fixture()
    result['rmse'] = result['metrics']['aligned_rms']['aggregate'] = value
    return result


def test_schedule_has_exact_matched_pairs_layouts_and_integration_count():
    jobs = driver.jobs()
    assert len(jobs) == len({j['id'] for j in jobs}) == 54
    assert sum(j['size'] for j in jobs) == 64188
    assert [j['arm'] for j in jobs[:2]] == ['cpu', 'gpu']
    assert all(j['kind'] == 'cold' and j['size'] == 18 and j['dt'] == .0025
               for j in jobs[:2])
    warm = [j for j in jobs if j['kind'] == 'warm']
    assert len(warm) == 48
    expected = [(dt, size, rep, arm)
                for dt in [.0025, .00125] for size in [18, 144, 576, 3456]
                for rep in range(3)
                for arm in (['cpu', 'gpu'] if rep % 2 == 0 else ['gpu', 'cpu'])]
    assert [(j['dt'], j['size'], j['rep'], j['arm']) for j in warm] == expected
    assert [(j['dt'], j['layout'], j['size']) for j in jobs if j['kind'] == 'layout'] == [
        (.0025, 'reverse', 3456), (.0025, 'chunked', 3456),
        (.00125, 'reverse', 3456), (.00125, 'chunked', 3456)]
    assert divmod(3456, driver.CONTRACT['chunk_size']) == (13, 115)
    assert driver.CONTRACT['deployment_size'] == 3456
    assert driver.CONTRACT['deployment_dt'] == .00125


def test_bitwise_checks_sign_dtype_shape_and_bits():
    x = np.array([0., 1.], dtype=np.float64)
    assert driver.bitwise(x, x.copy())
    assert not driver.bitwise(x, np.array([-0., 1.]))
    assert not driver.bitwise(x, x.astype(np.float32))
    assert not driver.bitwise(x, x.reshape(1, 2))
    y = x.copy()
    y[1] = np.nextafter(1., np.inf)
    assert not driver.bitwise(x, y)


@pytest.mark.parametrize('dt', [.0025, .00125])
def test_array_contract_accepts_exact_complete_synthetic_identity(dt):
    ids, contexts, arrays, _, _ = batch_fixture(dt=dt)
    driver.validate_arrays(arrays, ids, contexts, dt)


@pytest.mark.parametrize('mutation', [
    'float32', 'status_dtype', 'progress_dtype', 'state_shape', 'nan', 'inf',
    'initial', 'first_row', 'final', 'status', 'progress', 'time', 'alpha',
    'gamma', 'delta', 'missing_field', 'extra_field', 'duplicate_ids', 'swapped_ids'])
def test_array_contract_rejects_incomplete_or_rebound_output(mutation):
    ids, contexts, arrays, _, _ = batch_fixture()
    if mutation == 'float32':
        arrays['states'] = arrays['states'].astype(np.float32)
    elif mutation == 'status_dtype':
        arrays['statuses'] = arrays['statuses'].astype(np.int64)
    elif mutation == 'progress_dtype':
        arrays['progress'] = arrays['progress'].astype(np.int32)
    elif mutation == 'state_shape':
        arrays['states'] = arrays['states'][:, :-1]
    elif mutation in ('nan', 'inf'):
        arrays['states'][0, 8, 0] = float(mutation)
    elif mutation == 'initial':
        arrays['initial'][0, 0] += .01
    elif mutation == 'first_row':
        arrays['states'][0, 0, 0] += .01
    elif mutation == 'final':
        arrays['final'][0, 0] += .01
    elif mutation == 'status':
        arrays['statuses'][0] = -2
    elif mutation == 'progress':
        arrays['progress'][0, 0] -= 1
    elif mutation == 'time':
        arrays['times'][9] = np.nextafter(arrays['times'][9], np.inf)
    elif mutation in ('alpha', 'gamma', 'delta'):
        arrays[mutation].flat[0] += .01
    elif mutation == 'missing_field':
        del arrays['final']
    elif mutation == 'extra_field':
        arrays['unplanned'] = np.array([1.])
    elif mutation == 'duplicate_ids':
        ids[1] = ids[0]
    elif mutation == 'swapped_ids':
        ids[0], ids[1] = ids[1], ids[0]
    with pytest.raises((ValueError, KeyError)):
        driver.validate_arrays(arrays, ids, contexts, .0025)


def test_metric_parity_is_circular_for_phase_but_absolute_for_each_error():
    original = metric_fixture()
    changed = copy.deepcopy(original)
    changed['metrics']['phase_offset_cycles'] += 1.+5e-9
    changed['metrics']['windows'][0]['aligned_rms']['components'][2] += 5e-11
    changed['old_fixed_phase_rmse'] += 5e-11
    difference = driver.metric_differences(original, changed)
    assert len(difference) == 3 and all(d['passed'] for d in difference)
    changed['metrics']['windows'][0]['aligned_rms']['components'][2] += 2e-10
    assert not all(d['passed'] for d in driver.metric_differences(original, changed))


@pytest.mark.parametrize('mutation', ['phase', 'period', 'window_identity', 'metadata', 'missing'])
def test_metric_parity_rejects_changed_contract_or_excess_phase(mutation):
    original = metric_fixture()
    changed = copy.deepcopy(original)
    if mutation == 'phase':
        changed['metrics']['phase_offset_cycles'] += 2e-8
    elif mutation == 'period':
        changed['metrics']['period'] += 1e-12
    elif mutation == 'window_identity':
        changed['metrics']['windows'][0]['index'] = 1
    elif mutation == 'metadata':
        changed['metrics']['phase_search']['grid_bins'] = 64
    else:
        del changed['metrics']['windows']
    assert not all(d['passed'] for d in driver.metric_differences(original, changed))


def test_phase_evaluation_count_is_retained_as_diagnostic_only():
    original = metric_fixture()
    changed = copy.deepcopy(original)
    changed['metrics']['phase_search']['refinement_evaluations'] = 8
    difference = driver.metric_differences(original, changed)
    assert len(difference) == 1
    assert difference[0]['passed'] and difference[0]['diagnostic']
    assert difference[0]['a'] == 5 and difference[0]['b'] == 8


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_equal_nonfinite_metric_leaves_never_pass(value):
    original = metric_fixture()
    original['metrics']['windows'][0]['aligned_rms']['aggregate'] = value
    assert not all(d['passed'] for d in driver.metric_differences(original, copy.deepcopy(original)))


def test_category_booleans_cannot_alias_integer_metadata():
    original = metric_fixture()
    changed = copy.deepcopy(original)
    changed['metrics']['phase_search']['refinement_success'] = 1
    assert not all(d['passed'] for d in driver.metric_differences(original, changed))


def test_full_ranking_uses_candidate_ties_and_declared_controls():
    _, _, _, census, _ = batch_fixture()
    ranked = driver.rank_choices(census, [dict(rmse=.1) for _ in census])[0]
    assert ranked['ranked'] == list(range(18))
    assert ranked['minimum'] == 0 and ranked['top4'] == [0, 1, 2, 3]
    assert ranked['controls'] == dict(axes=0, diagonal=2, local_coverage=0)


@pytest.mark.parametrize('mutation', ['short_scores', 'missing_candidate', 'duplicate_candidate',
                                     'nan_score', 'inf_score', 'bool_score'])
def test_ranking_requires_complete_finite_eighteen_entry_contexts(mutation):
    _, _, _, census, scores = batch_fixture()
    if mutation == 'short_scores':
        scores.pop()
    elif mutation == 'missing_candidate':
        census.pop(); scores.pop()
    elif mutation == 'duplicate_candidate':
        census[-1]['key'][-1] = 0
    else:
        scores[0]['rmse'] = {'nan_score': float('nan'), 'inf_score': float('inf'), 'bool_score': True}[mutation]
    with pytest.raises(ValueError):
        driver.rank_choices(census, scores)


def test_cross_device_gate_detects_waveform_and_decision_failures_separately():
    _, _, cpu, census, scores = batch_fixture()
    gpu = {k: v.copy() for k, v in cpu.items()}
    assert driver.compare_batches(cpu, gpu, scores, copy.deepcopy(scores), census)['passed']
    gpu['states'][0, 10, 0] += 1e-6
    result = driver.compare_batches(cpu, gpu, scores, copy.deepcopy(scores), census)
    assert not result['passed'] and not result['state_pass']
    gpu = {k: v.copy() for k, v in cpu.items()}
    tied = [full_score(.1) for _ in census]
    shifted = copy.deepcopy(tied)
    shifted[1]['rmse'] -= 1e-12
    shifted[1]['metrics']['aligned_rms']['aggregate'] = shifted[1]['rmse']
    result = driver.compare_batches(cpu, gpu, tied, shifted, census)
    assert result['state_pass'] and result['metrics_pass']
    assert not result['ranking_pass'] and not result['passed']


def test_write_batch_replays_saved_arrays_and_every_score_leaf(tmp_path, monkeypatch):
    ids, contexts, arrays, _, _ = batch_fixture(n=2)
    seen = []

    def scorer(context, times, states):
        seen.append((context['alpha'], times.copy(), states.copy()))
        result = metric_fixture()
        result['metrics']['windows'][0]['aligned_rms']['aggregate'] = float(states[20, 0])
        return result

    monkeypatch.setattr(driver.preview, 'score', scorer)
    scores = [scorer(contexts[i], arrays['times'], arrays['states'][n]) for n, i in enumerate(ids)]
    seen.clear()
    artifacts = driver.write_batch(tmp_path, arrays, scores, {'synthetic': True}, contexts, ids, InlinePool())
    assert len(seen) == 2
    assert all(np.array_equal(item[1], arrays['times']) for item in seen)
    assert all(np.array_equal(item[2], arrays['states'][i]) for i, item in enumerate(seen))
    assert artifacts['raw'] == driver.runtime.binding(tmp_path/'traces.npz', root=tmp_path)
    assert artifacts['scores'] == driver.runtime.binding(tmp_path/'scores.json', root=tmp_path)


def test_saved_window_mismatch_rejected_and_outputs_preserved(tmp_path, monkeypatch):
    ids, contexts, arrays, _, _ = batch_fixture(n=1)
    monkeypatch.setattr(driver.preview, 'score', lambda *_: metric_fixture())
    saved = metric_fixture()
    saved['metrics']['windows'][0]['aligned_rms']['components'][1] += .001
    with pytest.raises(ValueError):
        driver.write_batch(tmp_path, arrays, [saved], {}, contexts, ids, InlinePool())
    assert (tmp_path/'traces.npz').is_file() and (tmp_path/'scores.json').is_file()
    assert json.loads((tmp_path/'scores.json').read_text())['scores'] == [saved]


def test_readback_detects_changed_raw_bytes_before_acceptance(tmp_path, monkeypatch):
    ids, contexts, arrays, _, _ = batch_fixture(n=1)
    original = driver.runtime.atomic_npz

    def corrupt(path, **values):
        values = {k: v.copy() for k, v in values.items()}
        values['states'][0, 1, 0] += .01
        return original(path, **values)

    monkeypatch.setattr(driver.runtime, 'atomic_npz', corrupt)
    with pytest.raises(ValueError, match='readback'):
        driver.write_batch(tmp_path, arrays, [metric_fixture()], {}, contexts, ids, InlinePool())


def test_receive_returns_ready_message_without_waiting():
    process = SimpleNamespace(pid=42, is_alive=lambda: True)
    connection = SimpleNamespace(poll=lambda _: True, recv=lambda: {'status': 'ready'})
    assert driver.receive(process, connection, driver.time.monotonic()+1) == {'status': 'ready'}


def test_receive_detects_owned_worker_exit():
    process = SimpleNamespace(pid=42, exitcode=17, is_alive=lambda: False)
    connection = SimpleNamespace(poll=lambda _: False)
    with pytest.raises(RuntimeError, match='42 exited 17'):
        driver.receive(process, connection, driver.time.monotonic()+1)


def test_receive_honors_already_expired_deadline():
    process = SimpleNamespace(pid=42, is_alive=lambda: True)
    connection = SimpleNamespace(poll=lambda _: pytest.fail('Expired deadline must not poll'))
    with pytest.raises(TimeoutError):
        driver.receive(process, connection, driver.time.monotonic()-1)


def test_failed_pool_map_drains_every_future_and_retains_each_result():
    seen = []

    class DelayedPool:
        def submit(self, function, value):
            def result():
                seen.append(value)
                return function(value)
            return SimpleNamespace(result=result)

    def function(value):
        if value != 1:
            raise RuntimeError(f'failure-{value}')
        return 'kept'

    with pytest.raises(driver.BatchFailure) as caught:
        driver.drained_map(DelayedPool(), function, range(3))
    assert seen == [0, 1, 2]
    assert caught.value.results == [None, 'kept', None]
    assert [e['index'] for e in caught.value.errors] == [0, 2]


def test_cpu_scoring_failure_preserves_other_cases_and_unscored_trace(tmp_path, monkeypatch):
    ids, contexts, arrays, _, _ = batch_fixture(n=3)

    def integrate(library, alpha, gamma, initial, dt, **kwargs):
        i = int(round((alpha-1.)*100))
        return arrays['times'].copy(), arrays['states'][i].copy()

    scored = []

    def score(context, times, states):
        i = int(round((context['alpha']-1.)*100))
        scored.append(i)
        if i == 1:
            raise ValueError('synthetic scorer failure')
        return metric_fixture()

    monkeypatch.setattr(driver, 'integrate_trace', integrate)
    monkeypatch.setattr(driver.preview, 'score', score)
    with pytest.raises(driver.BatchFailure):
        driver.produce(dict(size=3, dt=.0025, arm='cpu', layout='canonical'),
                       contexts, ids, None, InlinePool(), tmp_path)
    assert scored == [0, 1, 2]
    assert (tmp_path/'completed-0.npz').is_file()
    assert (tmp_path/'completed-2.npz').is_file()
    with np.load(tmp_path/'unscored-1.npz', allow_pickle=False) as retained:
        assert np.array_equal(retained['states'], arrays['states'][1])
    assert json.loads((tmp_path/'case-failures.json').read_text())[0]['index'] == 1


def test_later_gpu_chunk_failure_preserves_earlier_completed_chunk(tmp_path, monkeypatch):
    ids, contexts, arrays, _, _ = batch_fixture(n=3)
    calls = []

    class FakeGpu:
        def integrate_batch(self, alpha, gamma, initial, dt, **kwargs):
            calls.append(alpha.copy())
            if len(calls) == 2:
                raise driver.cuda.CudaError('synthetic later chunk failure')
            return dict(times=arrays['times'].copy(), states=arrays['states'][:1].copy(),
                        statuses=arrays['statuses'][:1].copy(), final=arrays['final'][:1].copy(),
                        progress=arrays['progress'][:1].copy(), timings={}, memory={}, device={})

    monkeypatch.setitem(driver.CONTRACT, 'chunk_size', 1)
    with pytest.raises(driver.cuda.CudaError):
        driver.produce(dict(size=3, dt=.0025, arm='gpu', layout='chunked'),
                       contexts, ids, FakeGpu(), InlinePool(), tmp_path)
    assert len(calls) == 2
    with np.load(tmp_path/'completed-chunk-0.npz', allow_pickle=False) as retained:
        assert retained['indices'].tolist() == [0]
        assert np.array_equal(retained['states'], arrays['states'][:1])


def test_gpu_scoring_failure_keeps_complete_returned_waveforms(tmp_path, monkeypatch):
    ids, contexts, arrays, _, _ = batch_fixture(n=3)

    class FakeGpu:
        def integrate_batch(self, *args, **kwargs):
            return dict(**{k: arrays[k].copy() for k in ('times', 'states', 'statuses', 'final', 'progress')},
                        timings={}, memory={}, device={})

    def fail(*args):
        raise ValueError('synthetic scoring failure')

    monkeypatch.setattr(driver.preview, 'score', fail)
    with pytest.raises(driver.BatchFailure):
        driver.produce(dict(size=3, dt=.0025, arm='gpu', layout='canonical'),
                       contexts, ids, FakeGpu(), InlinePool(), tmp_path)
    with np.load(tmp_path/'unscored.npz', allow_pickle=False) as retained:
        assert np.array_equal(retained['states'], arrays['states'])
        assert np.array_equal(retained['initial'], arrays['initial'])
        assert np.array_equal(retained['progress'], arrays['progress'])


def persisted_batch(tmp_path, monkeypatch):
    ids, contexts, arrays, census, scores = batch_fixture()
    score_by_alpha = {contexts[i]['alpha']: scores[n] for n, i in enumerate(ids)}
    monkeypatch.setattr(driver.preview, 'score', lambda context, *_: copy.deepcopy(score_by_alpha[context['alpha']]))
    artifacts = driver.write_batch(tmp_path, arrays, scores, {}, contexts, ids, InlinePool())
    job = driver.jobs()[0]
    record = dict(**job, status='complete', ids=ids, artifacts=artifacts,
                  end_to_end_seconds=1., pid=123)
    driver.runtime.write(tmp_path/'record.json', record)
    return record, job, arrays, census, scores


def test_load_batch_requires_exact_job_and_reads_complete_bound_packet(tmp_path, monkeypatch):
    _, job, arrays, census, scores = persisted_batch(tmp_path, monkeypatch)
    actual_arrays, actual_scores = driver.load_batch(tmp_path, expected_job=job, census=census)
    assert actual_scores == scores
    assert all(driver.bitwise(actual_arrays[k], arrays[k]) for k in arrays)


@pytest.mark.parametrize('mutation', [
    'swapped_ids', 'duplicate_ids', 'wrong_arm', 'wrong_dt', 'score_count',
    'score_nonfinite', 'missing_score_field', 'aggregate_binding',
    'state_dtype', 'state_shape', 'artifact_alias', 'missing_artifact'])
def test_load_batch_rejects_internally_rehashed_wrong_identity_or_output(tmp_path, monkeypatch, mutation):
    record, job, arrays, census, _ = persisted_batch(tmp_path, monkeypatch)
    payload = json.loads((tmp_path/'scores.json').read_text())
    if mutation == 'swapped_ids':
        record['ids'][0], record['ids'][1] = record['ids'][1], record['ids'][0]
    elif mutation == 'duplicate_ids':
        record['ids'][1] = record['ids'][0]
    elif mutation == 'wrong_arm':
        record['arm'] = 'gpu'
    elif mutation == 'wrong_dt':
        record['dt'] = .00125
    elif mutation == 'score_count':
        payload['scores'].pop()
    elif mutation == 'score_nonfinite':
        payload['scores'][0]['metrics']['windows'][0]['aligned_rms']['aggregate'] = float('inf')
    elif mutation == 'missing_score_field':
        del payload['scores'][0]['metrics']
    elif mutation == 'aggregate_binding':
        payload['scores'][0]['rmse'] += .001
    elif mutation in ('state_dtype', 'state_shape'):
        arrays['states'] = arrays['states'].astype(np.float32) if mutation == 'state_dtype' else arrays['states'][:, :-1]
        driver.runtime.atomic_npz(tmp_path/'traces.npz', **arrays)
        record['artifacts']['raw'] = driver.runtime.binding(tmp_path/'traces.npz', root=tmp_path)
        payload['raw'] = record['artifacts']['raw']
    elif mutation == 'artifact_alias':
        (tmp_path/'aliased.npz').write_bytes((tmp_path/'traces.npz').read_bytes())
        record['artifacts']['raw'] = driver.runtime.binding(tmp_path/'aliased.npz', root=tmp_path)
        payload['raw'] = record['artifacts']['raw']
    else:
        del record['artifacts']['raw']
    # Rehash the deliberately malformed packet so these test structure/identity,
    # not merely an easy stale-hash rejection.
    (tmp_path/'scores.json').write_text(json.dumps(payload))
    record['artifacts']['scores'] = driver.runtime.binding(tmp_path/'scores.json', root=tmp_path)
    driver.runtime.write(tmp_path/'record.json', record)
    with pytest.raises((ValueError, KeyError, IndexError)):
        driver.load_batch(tmp_path, expected_job=job, census=census)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_task'])
def test_audit_records_census_failure_without_any_research_inputs(tmp_path, monkeypatch, mutation):
    census = [dict(id=f'g{g:03d}-c{c:02d}', key=['development', f'g{g:03d}', 675, .001, c])
              for g in range(192) for c in range(18)]
    plan = dict(jobs=driver.jobs(), census=census)
    records = [dict(**job, status='not_started', error='synthetic unavailable') for job in driver.jobs()]
    if mutation == 'missing':
        records.pop()
    elif mutation == 'duplicate':
        records[-1] = copy.deepcopy(records[0])
    else:
        records[0]['arm'] = 'gpu'
    # The fake label root is deliberately empty; no old label is read.
    monkeypatch.setattr(driver, 'LABEL_ROOT', tmp_path/'absent-reference')
    result = driver.audit(plan, tmp_path, records)
    assert not result['complete'] and not result['all_parity_pass']
    assert not result['gpu_deployment_qualified']
    errors = [i for i in result['terminal_census_issues'] if i['location'] != 'old_reference/0.0025'
              and i['location'] != 'old_reference/0.00125']
    assert errors
    text = ' '.join(i['error'] for i in errors)
    assert ('Missing' in text or '54' in text) if mutation == 'missing' else (
        'duplicate' in text if mutation == 'duplicate' else 'task fields' in text)
    assert not (tmp_path/'absent-reference').exists()
