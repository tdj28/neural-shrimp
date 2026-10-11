"""Synthetic EXP-013 owned-worker and durable failure contracts; no ODEs."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import fresh_preview_runtime as work


def specs(count):
    return [dict(id=f'task-{i}', kind='synthetic', case=dict(id=f'case-{i}'),
        method=dict(id='synthetic-method')) for i in range(count)]


def contexts(tasks, **fields):
    return {t['case']['id']: dict(fields) for t in tasks}


def terminal(task, status='complete', **fields):
    return dict(id=task['id'], kind=task['kind'], case_id=task['case']['id'],
        method=task['method']['id'], status=status, pid=os.getpid(), rmse=None, **fields)


def complete_worker(task, context, directory, libraries, parent_pid):
    assert os.getppid() == parent_pid
    work.write(Path(directory) / 'records' / f'{task["id"]}.json', terminal(task))


def hang_worker(task, context, directory, libraries, parent_pid):
    dest = Path(directory) / 'payloads' / task['id']
    dest.mkdir()
    work.write(dest / 'partial.json', dict(id=task['id'], pid=os.getpid(), acquired_prefix=[1, 2, 3]))
    time.sleep(10)


def exit_worker(task, context, directory, libraries, parent_pid):
    dest = Path(directory) / 'payloads' / task['id']
    dest.mkdir()
    np.savez_compressed(dest / 'partial.npz', prefix=np.arange(7, dtype=float))
    os._exit(23)


def malformed_worker(task, context, directory, libraries, parent_pid):
    path = Path(directory) / 'records' / f'{task["id"]}.json'
    if context['mutation'] == 'invalid_json':
        path.write_text('{partial')
    elif context['mutation'] == 'no_record':
        return
    else:
        value = terminal(task)
        value[context['mutation']] = 'wrong-identity'
        work.write(path, value)


def passed_certificate():
    phase = np.linspace(0., 2*np.pi, 2701)
    states = np.column_stack((np.sin(phase), np.cos(phase), np.sin(2*phase)))
    return dict(verified=True, attracting=True, primitive=True, peak_count=9, expected_peak_count=9,
        alpha=1.065, gamma=10.3, period=23., transverse_spectral_radius=.3,
        gates=dict.fromkeys(('closure', 'phase_condition', 'section_is_maximum', 'nonstationary',
            'expected_peak_count', 'primitive', 'neutral_mode', 'transverse_stability'), True),
        orbit_times=np.linspace(0., 23., 2701).tolist(), orbit_states=states.tolist(),
        orbit_activations=np.tanh(states).tolist(), section_state=states[0].tolist())


def read(path):
    return json.loads(Path(path).read_text())


def assert_cleanup(directory):
    cleanup = read(directory / 'cleanup.json')
    assert cleanup['all_registered_workers_joined'] is True
    assert cleanup['all_original_identities_absent'] is True
    for item in cleanup['workers']:
        assert item['original_identity_absent'] is True
        assert work.identity(item['pid']) != item['identity'] or item['identity'] is None
        assert not any(child.pid == item['pid'] for child in mp.active_children())


def test_success_has_exact_terminal_census_journal_and_owned_cleanup(tmp_path):
    tasks = specs(6); directory = tmp_path / 'success'
    rows = work.execute(tasks, contexts(tasks), directory, {}, workers=4,
        deadline=time.monotonic()+5, per_task=2, worker_fn=complete_worker)
    assert len(rows) == 6 and {r['id'] for r in rows} == {t['id'] for t in tasks}
    assert {r['status'] for r in rows} == {'complete'}
    assert read(directory / 'terminal-records.json') == rows
    assert [json.loads(line) for line in (directory / 'journal.jsonl').read_text().splitlines()] == rows
    assert_cleanup(directory)
    with pytest.raises(FileExistsError):
        work.execute(tasks, contexts(tasks), directory, {}, workers=1,
            deadline=time.monotonic()+5, per_task=2, worker_fn=complete_worker)


@pytest.mark.parametrize('workers', [0, -1, 5, 100])
def test_worker_cap_rejects_out_of_scope_parallelism(tmp_path, workers):
    tasks = specs(1)
    with pytest.raises(ValueError, match='Worker cap'):
        work.execute(tasks, contexts(tasks), tmp_path / 'bad', {}, workers=workers,
            deadline=time.monotonic()+5, per_task=2, worker_fn=complete_worker)
    assert not (tmp_path / 'bad').exists()


def test_per_task_timeout_keeps_prefixes_and_full_denominator(tmp_path):
    tasks = specs(3); directory = tmp_path / 'timeout'
    rows = work.execute(tasks, contexts(tasks), directory, {}, workers=2,
        deadline=time.monotonic()+5, per_task=.08, worker_fn=hang_worker)
    assert len(rows) == 3 and {r['status'] for r in rows} == {'task_timeout'}
    for task in tasks:
        partial = read(directory / 'payloads' / task['id'] / 'partial.json')
        assert partial['acquired_prefix'] == [1, 2, 3]
        assert read(directory / 'records' / f'{task["id"]}-termination.json')['status'] == 'task_timeout'
    assert_cleanup(directory)


def test_whole_deadline_keeps_active_and_not_started_tasks(tmp_path):
    tasks = specs(7); directory = tmp_path / 'deadline'
    rows = work.execute(tasks, contexts(tasks), directory, {}, workers=2,
        deadline=time.monotonic()+.08, per_task=5, worker_fn=hang_worker)
    assert len(rows) == 7
    assert sum(r['status'] == 'whole_attempt_timeout' for r in rows) == 2
    assert sum(r['status'] == 'not_started_deadline' for r in rows) == 5
    assert_cleanup(directory)


def test_already_expired_stage_starts_no_process_but_keeps_every_task(tmp_path):
    tasks = specs(4); directory = tmp_path / 'expired'
    rows = work.execute(tasks, contexts(tasks), directory, {}, workers=4,
        deadline=time.monotonic()-1, per_task=2, worker_fn=complete_worker)
    assert len(rows) == 4 and {r['status'] for r in rows} == {'not_started_deadline'}
    assert not (directory / 'processes.jsonl').exists()
    assert_cleanup(directory)


def test_worker_exit_preserves_raw_prefix_and_status(tmp_path):
    tasks = specs(2); directory = tmp_path / 'exit'
    rows = work.execute(tasks, contexts(tasks), directory, {}, workers=2,
        deadline=time.monotonic()+5, per_task=2, worker_fn=exit_worker)
    assert len(rows) == 2 and {r['status'] for r in rows} == {'worker_failed'}
    assert {r['exitcode'] for r in rows} == {23}
    for task in tasks:
        with np.load(directory / 'payloads' / task['id'] / 'partial.npz') as raw:
            np.testing.assert_array_equal(raw['prefix'], np.arange(7, dtype=float))
    assert_cleanup(directory)


@pytest.mark.parametrize('mutation, status', [('invalid_json', 'invalid_worker_record'),
    ('no_record', 'worker_failed'), ('id', 'invalid_worker_identity'),
    ('case_id', 'invalid_worker_identity'), ('method', 'invalid_worker_identity'),
    ('kind', 'invalid_worker_identity'), ('pid', 'invalid_worker_identity')])
def test_bad_worker_record_never_relabels_or_discards_a_planned_task(tmp_path, mutation, status):
    tasks = specs(2); directory = tmp_path / mutation
    rows = work.execute(tasks, contexts(tasks, mutation=mutation), directory, {}, workers=2,
        deadline=time.monotonic()+5, per_task=2, worker_fn=malformed_worker)
    assert len(rows) == 2 and {r['status'] for r in rows} == {status}
    assert {r['id'] for r in rows} == {t['id'] for t in tasks}
    assert {r['case_id'] for r in rows} == {t['case']['id'] for t in tasks}
    assert_cleanup(directory)


def test_interrupt_immediately_after_start_joins_child_and_keeps_pending(tmp_path, monkeypatch):
    tasks = specs(3); directory = tmp_path / 'interrupted'
    real = mp.get_context('fork')
    owned = []

    class InterruptedContext:
        def Process(self, *args, **kwargs):
            child = real.Process(*args, **kwargs)
            original_start = child.start
            def start():
                original_start(); owned.append(child)
                raise KeyboardInterrupt('synthetic launch boundary')
            child.start = start
            return child

    monkeypatch.setattr(work.mp, 'get_context', lambda _: InterruptedContext())
    with pytest.raises(KeyboardInterrupt):
        work.execute(tasks, contexts(tasks), directory, {}, workers=2,
            deadline=time.monotonic()+5, per_task=2, worker_fn=hang_worker)
    rows = read(directory / 'terminal-records.json')
    assert len(rows) == 3
    assert [r['status'] for r in rows] == ['interrupted', 'not_started_interruption', 'not_started_interruption']
    assert owned and all(not child.is_alive() for child in owned)
    assert_cleanup(directory)


def test_launch_masks_every_producer_interrupt_including_alarm(tmp_path, monkeypatch):
    tasks = specs(1); directory = tmp_path / 'mask'
    real = mp.get_context('fork')
    captured = []

    class InspectContext:
        def Process(self, *args, **kwargs):
            child = real.Process(*args, **kwargs)
            original_start = child.start
            def start():
                captured.append(signal.pthread_sigmask(signal.SIG_BLOCK, set()))
                original_start()
            child.start = start
            return child

    monkeypatch.setattr(work.mp, 'get_context', lambda _: InspectContext())
    work.execute(tasks, contexts(tasks), directory, {}, workers=1,
        deadline=time.monotonic()+5, per_task=2, worker_fn=complete_worker)
    assert captured and {signal.SIGINT, signal.SIGTERM, signal.SIGALRM} <= captured[0]
    assert_cleanup(directory)


def test_start_failure_before_pid_assignment_keeps_full_census(tmp_path, monkeypatch):
    tasks = specs(2); directory = tmp_path / 'start-failure'
    real = mp.get_context('fork')

    class FailedContext:
        def Process(self, *args, **kwargs):
            child = real.Process(*args, **kwargs)
            def start():
                raise OSError('synthetic fork unavailable')
            child.start = start
            return child

    monkeypatch.setattr(work.mp, 'get_context', lambda _: FailedContext())
    with pytest.raises(OSError, match='fork unavailable'):
        work.execute(tasks, contexts(tasks), directory, {}, workers=1,
            deadline=time.monotonic()+5, per_task=2, worker_fn=complete_worker)
    rows = read(directory / 'terminal-records.json')
    assert len(rows) == 2
    assert [r['status'] for r in rows] == ['interrupted', 'not_started_interruption']
    assert_cleanup(directory)


@pytest.mark.parametrize('error_type', [BrokenPipeError, KeyboardInterrupt])
def test_progress_transport_failure_or_interrupt_never_duplicates_finalized_task(tmp_path, monkeypatch, error_type):
    tasks = specs(101); directory = tmp_path / 'progress-failure'
    def fail_print(*args, **kwargs):
        raise error_type('synthetic progress failure after terminal persistence')
    monkeypatch.setattr(work, 'print', fail_print, raising=False)
    arguments = dict(workers=4, deadline=time.monotonic()+10, per_task=2, worker_fn=complete_worker)
    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            work.execute(tasks, contexts(tasks), directory, {}, **arguments)
    else:
        work.execute(tasks, contexts(tasks), directory, {}, **arguments)
    rows = read(directory / 'terminal-records.json')
    assert len(rows) == len({row['id'] for row in rows}) == len(tasks)
    assert {row['id'] for row in rows} == {task['id'] for task in tasks}
    journal = [json.loads(line) for line in (directory / 'journal.jsonl').read_text().splitlines()]
    assert len(journal) == len({row['id'] for row in journal}) == len(tasks)
    assert_cleanup(directory)


def test_certificate_checker_accepts_original_representation():
    assert work.certificate_ok(passed_certificate()) is True


def test_certificate_checker_binds_reference_to_the_expected_origin():
    certificate = passed_certificate()
    assert work.certificate_ok(certificate, dict(alpha=1.065, gamma=10.3)) is True
    assert work.certificate_ok(certificate, dict(alpha=1.0651, gamma=10.3)) is False


@pytest.mark.parametrize('mutation', ['initial', 'times', 'shape', 'nonfinite', 'float32'])
def test_preview_score_rejects_wrong_initial_time_or_precision_before_scoring(monkeypatch, mutation):
    initial = np.array([0., .25, .5]); states = np.broadcast_to(initial, (751, 3)).copy()
    times = work.TIMES.copy()
    if mutation == 'initial': states[0, 0] += .01
    elif mutation == 'times': times[1] += 1e-12
    elif mutation == 'shape': states = states[:-1]
    elif mutation == 'nonfinite': states[10, 1] = np.nan
    else: states = states.astype(np.float32)
    monkeypatch.setattr(work, 'waveform_errors', lambda *args, **kwargs:
        pytest.fail('Invalid input reached waveform scoring'))
    with pytest.raises(ValueError):
        work.score(dict(initial=initial), times, states)


@pytest.mark.parametrize('mutation', ['failed_gate', 'renamed_gate', 'wrong_expected_peaks',
    'nonuniform_times', 'missing_endpoint', 'nan_state', 'wrong_activation', 'wrong_section',
    'nonattracting', 'wrong_period', 'nonprimitive', 'missing_field'])
def test_certificate_checker_rejects_changed_scientific_contract(mutation):
    cert = passed_certificate()
    if mutation == 'failed_gate': cert['gates']['closure'] = False
    elif mutation == 'renamed_gate': cert['gates']['invented'] = cert['gates'].pop('closure')
    elif mutation == 'wrong_expected_peaks': cert['expected_peak_count'] = 8
    elif mutation == 'nonuniform_times': cert['orbit_times'][1] += 1e-8
    elif mutation == 'missing_endpoint': cert['orbit_states'].pop()
    elif mutation == 'nan_state': cert['orbit_states'][1][0] = float('nan')
    elif mutation == 'wrong_activation': cert['orbit_activations'][1][0] += 1e-5
    elif mutation == 'wrong_section': cert['section_state'][0] += .01
    elif mutation == 'nonattracting': cert['transverse_spectral_radius'] = 1.
    elif mutation == 'wrong_period': cert['period'] = -1.
    elif mutation == 'nonprimitive': cert['primitive'] = False
    else: del cert['orbit_times']
    assert work.certificate_ok(cert) is False


def test_worker_persists_native_failure_payload_without_integration(tmp_path, monkeypatch):
    directory = tmp_path / 'stage'; (directory / 'payloads').mkdir(parents=True)
    (directory / 'records').mkdir()
    task = dict(id='synthetic-preview', kind='preview', case=dict(id='synthetic-case'), method=work.METHODS[0])
    context = dict(alpha=1.065, gamma=10.3, initial=np.array([.1, .2, .3]))

    class FailedTrace(RuntimeError):
        payload = dict(times=np.array([0., .1]), states=np.array([[.1, .2, .3], [.2, .3, .4]]), native_status=-1)

    def fail(*args, **kwargs):
        raise FailedTrace('synthetic native failure')

    monkeypatch.setattr(work.legacy, 'load_native', lambda _: object())
    monkeypatch.setattr(work, 'integrate_trace', fail)
    work.worker(task, context, directory, {'trace': {'path': 'never-read'}}, os.getpid())
    row = read(directory / 'records' / 'synthetic-preview.json')
    assert row['status'] == 'failed' and 'synthetic native failure' in row['error']
    with np.load(directory / 'payloads' / task['id'] / 'partial-failure.npz') as raw:
        np.testing.assert_array_equal(raw['states'], FailedTrace.payload['states'])
        assert raw['native_status'] == -1


def test_worker_keeps_full_raw_trace_when_later_scoring_fails(tmp_path, monkeypatch):
    directory = tmp_path / 'stage'; (directory / 'payloads').mkdir(parents=True)
    (directory / 'records').mkdir()
    task = dict(id='synthetic-preview', kind='preview', case=dict(id='synthetic-case'), method=work.METHODS[0])
    initial = np.array([.1, .2, .3]); states = np.broadcast_to(initial, (751, 3)).copy()
    context = dict(alpha=1.065, gamma=10.3, initial=initial, delta=np.array([.001, 0.]))
    monkeypatch.setattr(work.legacy, 'load_native', lambda _: object())
    monkeypatch.setattr(work, 'integrate_trace', lambda *a, **k: (work.TIMES.copy(), states.copy()))
    def fail_score(*args):
        raise ValueError('synthetic score failure')
    monkeypatch.setattr(work, 'score', fail_score)
    work.worker(task, context, directory, {'trace': {'path': 'never-read'}}, os.getpid())
    row = read(directory / 'records' / 'synthetic-preview.json')
    assert row['status'] == 'failed' and 'synthetic score failure' in row['error']
    path = directory / row['raw']
    assert work.legacy.sha(path) == row['raw_sha256'] and path.stat().st_size == row['raw_bytes']
    with np.load(path) as raw:
        np.testing.assert_array_equal(raw['states'], states)


def test_exclusive_json_writer_preserves_first_evidence(tmp_path):
    path = tmp_path / 'evidence.json'
    work.write(path, dict(first=True, unavailable=float('nan')))
    with pytest.raises(FileExistsError):
        work.write(path, dict(first=False))
    assert read(path) == dict(first=True, unavailable=None)


def test_compiler_failure_has_durable_attempt_record_without_launching_compiler(tmp_path, monkeypatch):
    def fail(command, timeout):
        assert timeout > 0
        raise RuntimeError('synthetic compiler unavailable')
    monkeypatch.setattr(work, 'checked_command', fail)
    directory = tmp_path / 'build'
    with pytest.raises(RuntimeError, match='compiler unavailable'):
        work.build_libraries(directory, time.monotonic()+5)
    record = read(directory / 'baseline-build.json')
    assert record['status'] == 'failed'
    assert 'synthetic compiler unavailable' in record['error']
    assert record['source']['path'] == 'native/hopfield.cpp'
    assert record['wall_seconds'] >= 0
    assert not (directory / 'trace-build.json').exists()


def test_owned_command_masks_launch_interrupts_and_joins_on_timeout(monkeypatch):
    original = subprocess.Popen
    launched, masks = [], []
    def capture(*args, **kwargs):
        masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, set()))
        process = original(*args, **kwargs)
        launched.append(process)
        return process
    monkeypatch.setattr(work.subprocess, 'Popen', capture)
    with pytest.raises(subprocess.TimeoutExpired):
        work.checked_command([sys.executable, '-c', 'import time; time.sleep(10)'], .05)
    assert {signal.SIGINT, signal.SIGTERM, signal.SIGALRM} <= masks[0]
    assert launched[0].poll() is not None
    assert work.identity(launched[0].pid) is None


def test_owned_command_records_nonzero_stderr_and_preserves_success_stdout():
    assert work.checked_command([sys.executable, '-c', 'print("synthetic command")'], 2) == 'synthetic command\n'
    with pytest.raises(RuntimeError, match='synthetic command failure'):
        work.checked_command([sys.executable, '-c',
            'import sys; sys.stderr.write("synthetic command failure"); sys.exit(7)'], 2)


def test_baseline_failure_retains_earlier_native_output(tmp_path, monkeypatch):
    directory = tmp_path / 'stage'; (directory / 'payloads').mkdir(parents=True)
    (directory / 'records').mkdir()
    task = dict(id='synthetic-baseline', kind='baseline', case=dict(id='synthetic-origin'),
        method=dict(id='paired-baseline-certificate'))
    def fail_baseline(fn, anchor, plan, dest):
        prefix = dest / 'baselines' / anchor['id']; prefix.mkdir(parents=True)
        np.savez_compressed(prefix / 'dt-0.01.npz', final=np.array([1., 2., 3.]))
        raise ValueError('synthetic second baseline failure')
    monkeypatch.setattr(work, 'load_baseline', lambda _: object())
    monkeypatch.setattr(work.baseline_recipe, 'baseline', fail_baseline)
    work.worker(task, dict(anchor=dict(id='synthetic-origin')), directory,
        dict(baseline=dict(path='never-read')), os.getpid())
    row = read(directory / 'records' / 'synthetic-baseline.json')
    assert row['status'] == 'failed' and 'second baseline failure' in row['error']
    path = directory / 'payloads' / task['id'] / 'baselines' / 'synthetic-origin' / 'dt-0.01.npz'
    with np.load(path) as raw:
        np.testing.assert_array_equal(raw['final'], [1., 2., 3.])


def test_inventory_includes_failed_and_partial_files_and_excludes_only_receipt(tmp_path):
    (tmp_path / 'payloads').mkdir()
    (tmp_path / 'payloads' / 'partial.bin').write_bytes(b'partial')
    work.write(tmp_path / 'failure.json', {'status': 'failed'})
    work.write(tmp_path / 'receipt.json', {'excluded': True})
    items = work.inventory(tmp_path)
    assert {item['path'] for item in items} == {'failure.json', 'payloads/partial.bin'}
    assert all(item['bytes'] == (tmp_path / item['path']).stat().st_size for item in items)
