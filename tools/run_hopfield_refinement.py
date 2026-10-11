#!/usr/bin/env python3
"""EXP-009: prospectively bound refinement of already observed discrepancies.

prepare/validate inspect old results only. run alone creates new target traces.
All new raw evidence lives outside the source checkout; no attempt is resumed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
for _name in THREAD_ENV:
    os.environ[_name] = '1'

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from hopfield_diagnostics import hopfield_rhs
from hopfield_drift_metrics import periodic_interpolate, waveform_errors
from hopfield_drift_probe import integrate_trace

PLAN = ROOT / 'experiments/hopfield/EXP-009-plan.json'
OLD = ROOT / 'evidence/EXP-008/attempt-001'
DEFAULT_DATA_ROOT = Path('/data2/neural-shrimp-data/raw')
OLD_RECEIPT_SHA = 'ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb'
COHORT_SHA = '3d222c46b0c6c446e1b1c55b3442911a34aa3055c72b22bcec8bc153f28ecdad'
RADII = [0.00025, 0.001, 0.004, 0.016]
POLICIES = ['baseline', 'geometry', 'shuffled_geometry', 'min_preview']
METHODS = [dict(id='rk4-0.0025', kind='RK4', dt=.0025),
           dict(id='rk4-0.00125', kind='RK4', dt=.00125),
           dict(id='dop853-1e-11', kind='DOP853', rtol=1e-11, atol=1e-13, max_step=.05),
           dict(id='dop853-1e-12', kind='DOP853', rtol=1e-12, atol=1e-14, max_step=.025)]
SOURCE_PATHS = ['tools/run_hopfield_refinement.py', 'tools/hopfield_diagnostics.py',
                'tools/hopfield_drift_metrics.py', 'tools/hopfield_drift_probe.py',
                'native/hopfield.cpp', 'native/hopfield_drift.cpp',
                'tests/test_hopfield_refinement.py', 'docs/EXP-009-protocol.md',
                'pyproject.toml', 'requirements.lock', 'uv.lock']


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path, root=ROOT):
    path = Path(path)
    return dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('xb') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False).encode() + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def key(row):
    return (row['role'], row['anchor'], row['phase'], row['radius'], row['candidate'])


def case_id(k):
    return f'{k[1]}-p{k[2]}-r{k[3]:g}-c{k[4]}'


def read_prior(old=OLD):
    rows = []
    for role in ('development', 'test'):
        rows.extend(json.loads(line) for line in (old / f'{role}-targets.jsonl').read_text().splitlines())
    return rows, json.loads((old / 'test-decisions.json').read_text())


def select_cases(rows, decisions):
    """Outcome-aware but deterministic old-data selection, with no integration."""
    pairs = defaultdict(dict)
    for row in rows:
        k, dt = key(row), row['dt']
        if dt not in (.01, .005) or dt in pairs[k]:
            raise ValueError('Duplicate case/step or unexpected step')
        value = row.get('rmse')
        if (row.get('status') != 'complete' or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2):
            raise ValueError('Incomplete or invalid old outcome')
        pairs[k][dt] = row
    if any(set(pair) != {.01, .005} for pair in pairs.values()):
        raise ValueError('Missing old step')
    delta = {k: abs(pair[.01]['rmse'] - pair[.005]['rmse']) for k, pair in pairs.items()}
    sensitive = {k for k in pairs if delta[k] > .001}
    selected = defaultdict(set)
    assignments = []
    seen_decisions = set()
    for row in decisions:
        context = (row['anchor'], row['phase'], row['radius'])
        if context in seen_decisions or row.get('status') != 'complete':
            raise ValueError('Duplicate or incomplete old decision')
        seen_decisions.add(context)
        if set(row['policies']) != set(POLICIES):
            raise ValueError('Missing frozen policy')
        for policy in POLICIES:
            choice = row['policies'][policy]
            k = ('test', *context, choice['candidate'])
            if choice.get('status') != 'complete' or k not in pairs:
                raise ValueError('Invalid frozen selection')
            selected[policy].add(k)
            assignments.append(dict(policy=policy, key=list(k), block=row['block']))
    controls, unmatched = set(), []
    for context in sorted({k[:-1] for k in sensitive}):
        eligible = [k for k in pairs if k[:-1] == context
                    and k[-1] in [0, 4, 8, 12, 16, 17] and delta[k] <= .001]
        if eligible:
            controls.add(min(eligible, key=lambda k: (delta[k], k[-1])))
        else:
            unmatched.append(list(context))
    zeros = {k for k in pairs if k[-1] == -1}
    union = sensitive | controls | zeros | set().union(*selected.values())
    sentinels = defaultdict(list)

    def addmax(label, pool):
        if pool:
            sentinels[min(pool, key=lambda k: (-delta[k], k))].append(label)

    empty_strata = []
    for role in ('development', 'test'):
        for radius in RADII:
            for label, cohort in [('sensitive', sensitive), ('stable_control', controls)]:
                pool = [k for k in cohort if k[0] == role and k[3] == radius]
                name = f'{label}-{role}-{radius:g}'
                addmax(name, pool)
                if not pool:
                    empty_strata.append(name)
        addmax(f'sham-{role}', [k for k in zeros if k[0] == role])
    for policy in POLICIES:
        addmax(f'selected-{policy}', selected[policy])
    census = []
    for k in sorted(pairs):
        memberships = (['sensitive'] if k in sensitive else []) + (['stable_control'] if k in controls else [])
        memberships += ['zero_dose'] if k in zeros else []
        memberships += [f'selected-{policy}' for policy in POLICIES if k in selected[policy]]
        census.append(dict(key=list(k), id=case_id(k), selected=k in union,
                           reasons=memberships or ['outside_frozen_refinement_cohort'],
                           old_delta=delta[k], old_coarse_rmse=pairs[k][.01]['rmse'],
                           old_fine_rmse=pairs[k][.005]['rmse'],
                           old_coarse_raw=pairs[k][.01]['raw'], old_fine_raw=pairs[k][.005]['raw'],
                           independent_labels=sentinels.get(k, [])))
    return dict(census=census, assignments=assignments, unmatched_controls=unmatched,
                empty_independent_strata=empty_strata,
                counts=dict(prior_pairs=len(pairs), sensitive=len(sensitive),
                            selected_unique=len(set().union(*selected.values())),
                            stable_controls=len(controls), zero_dose=len(zeros), cohort=len(union),
                            independent=len(sentinels), assignments=len(assignments)),
                cohort_sha256=hashlib.sha256(canonical([list(k) for k in sorted(union)])).hexdigest())


def build_plan():
    if sha(OLD / 'receipt.json') != OLD_RECEIPT_SHA:
        raise ValueError('Original EXP-008 receipt differs')
    rows, decisions = read_prior()
    selection = select_cases(rows, decisions)
    receipt = json.loads((OLD / 'receipt.json').read_text())
    indexed = {r['path']: r for r in receipt['artifacts']}
    old_paths = {'receipt.json', 'manifest.json', 'test-decisions.json',
                 'development-targets.jsonl', 'test-targets.jsonl',
                 'development-baselines.json', 'test-baselines.json'}
    for case in selection['census']:
        if case['selected']:
            old_paths.update([case['old_coarse_raw'], case['old_fine_raw']])
    inputs = []
    for name in sorted(old_paths):
        item = binding(OLD / name)
        if name != 'receipt.json':
            expected = indexed[name]
            if (item['sha256'], item['bytes']) != (expected['sha256'], expected['bytes']):
                raise ValueError(f'Old receipt binding failed: {name}')
        inputs.append(item)
    plan = dict(experiment_id='EXP-009', date='2026-09-14',
                status='Post-run known-case numerical refinement; execution freeze precedes new outcomes',
                old_attempt='evidence/EXP-008/attempt-001', old_receipt_sha256=OLD_RECEIPT_SHA,
                **selection, methods=METHODS, duration=300., sample_dt=.1, tail_start_exclusive=75.,
                phase_grid=128, workers=4, thread_limit=1, wall_clock_ceiling_seconds=3600,
                per_task_ceiling_seconds=60, output_reserve_bytes=2 * 1024**3,
                planned_native_traces=2738, planned_independent_traces=38, planned_traces=2776,
                metric_stability_atol=1e-4, waveform_rms_atol=1e-3, waveform_max_atol=1e-2,
                required_runtime_packages={'numpy': '2.5.3', 'scipy': '1.18.1'},
                source_files=[binding(ROOT / p) for p in SOURCE_PATHS], source_artifacts=inputs,
                default_data_root=str(DEFAULT_DATA_ROOT), paid_compute=False, paid_review=False,
                scope='Incomplete prior-selected candidate cohort. No refitting, new action selection, '
                      'finite-candidate oracle, all-candidate prediction MSE, population convergence or new utility confirmation.')
    validate(plan)
    return plan


def validate(plan, *, check_bindings=False, reconstruct=False):
    expected = dict(prior_pairs=3504, sensitive=1012, selected_unique=220,
                    stable_controls=108, zero_dose=48, cohort=1369, independent=19, assignments=384)
    if plan['experiment_id'] != 'EXP-009' or plan['counts'] != expected or plan['cohort_sha256'] != COHORT_SHA:
        raise ValueError('Unexpected refinement census')
    constants = dict(methods=METHODS, duration=300., sample_dt=.1, tail_start_exclusive=75.,
                     phase_grid=128, workers=4, thread_limit=1, wall_clock_ceiling_seconds=3600,
                     per_task_ceiling_seconds=60, output_reserve_bytes=2 * 1024**3,
                     planned_native_traces=2738, planned_independent_traces=38, planned_traces=2776,
                     metric_stability_atol=1e-4, waveform_rms_atol=1e-3, waveform_max_atol=1e-2,
                     required_runtime_packages={'numpy': '2.5.3', 'scipy': '1.18.1'},
                     paid_compute=False, paid_review=False, old_receipt_sha256=OLD_RECEIPT_SHA)
    for name, value in constants.items():
        if plan[name] != value:
            raise ValueError(f'Frozen contract differs: {name}')
    chosen = [r for r in plan['census'] if r['selected']]
    if len(plan['census']) != 3504 or len(chosen) != 1369 or len({r['id'] for r in plan['census']}) != 3504:
        raise ValueError('Duplicate or incomplete census')
    if hashlib.sha256(canonical([r['key'] for r in chosen])).hexdigest() != COHORT_SHA:
        raise ValueError('Cohort keys changed')
    if len(tasks(plan)) != 2776 or len(plan['assignments']) != 384:
        raise ValueError('Task or policy-assignment census differs')
    if [r['path'] for r in plan['source_files']] != SOURCE_PATHS:
        raise ValueError('Incomplete executable source binding')
    if check_bindings:
        for item in plan['source_files'] + plan['source_artifacts']:
            path = ROOT / item['path']
            if path.resolve().is_relative_to(ROOT) is False or binding(path) != item:
                raise ValueError(f'Source/input binding differs: {item["path"]}')
    if reconstruct and plan != build_plan():
        raise ValueError('Plan differs from independently reconstructed old-data selection')
    return len(chosen)


def tasks(plan):
    return [dict(id=f'{case["id"]}--{method["id"]}', case=case, method=method)
            for case in plan['census'] if case['selected']
            for method in plan['methods'] if method['kind'] == 'RK4' or case['independent_labels']]


def external_attempt(data_root, attempt):
    root = Path(data_root).expanduser()
    if not root.is_absolute() or root.resolve().is_relative_to(ROOT):
        raise ValueError('Raw output root must be absolute and outside the source checkout')
    if not attempt.startswith('attempt-') or not attempt[8:].isdigit():
        raise ValueError('Attempt name must be attempt-NNN')
    path = (root / 'EXP-009' / attempt).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError('Raw output cannot resolve inside the source checkout')
    return path


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def source_guard(plan, plan_path):
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Tracked and untracked source must be clean before target outcomes')
    source = git('rev-parse', 'HEAD')
    for relative in [*SOURCE_PATHS, str(Path(plan_path).resolve().relative_to(ROOT))]:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{source}:{relative}'])
        if hashlib.sha256(content).hexdigest() != sha(ROOT / relative):
            raise ValueError(f'Uncommitted source dependency: {relative}')
    branch = git('symbolic-ref', '--short', 'HEAD')
    remote = git('ls-remote', '--exit-code', 'origin', f'refs/heads/{branch}')
    if remote.split()[0] != source:
        raise ValueError('Exact source/plan commit must be pushed before target outcomes')
    return dict(source_commit=source, pushed_branch=branch, remote_head_verified_at=now())


def checked_command(command, timeout):
    """Kill the owned compiler process group on timeout; retain stderr upstream."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        out, err = process.communicate(timeout=timeout)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f'Command failed ({process.returncode}): {err}')
    return out


def build_native(directory, deadline):
    directory.mkdir()
    library = directory / 'hopfield_drift.so'
    compiler = os.environ.get('CXX', 'c++')
    flags = ['-O3', '-std=c++17', '-fPIC', '-shared']
    checked_command([compiler, *flags, str(ROOT / 'native/hopfield_drift.cpp'), '-o', str(library)],
                    max(.01, min(60., deadline - time.monotonic())))
    return dict(compiler=checked_command([compiler, '--version'], 5.).splitlines()[0],
                flags=flags, library_sha256=sha(library), path=str(library))


def load_native(library):
    fn = ctypes.CDLL(str(library)).hopfield_trace
    ptr = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [ctypes.c_double, ctypes.c_double, ptr, ctypes.c_double,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr, ptr, ptr]
    fn.restype = ctypes.c_int
    return fn


def case_context(case, old, baselines, row_lookup):
    row = row_lookup[tuple(case['key'])][.005]
    coarse = row_lookup[tuple(case['key'])][.01]
    with np.load(old / case['old_fine_raw'], allow_pickle=False) as raw:
        initial = raw['initial'].copy()
        times = raw['times'].copy()
        alpha, gamma = float(raw['alpha']), float(raw['gamma'])
        delta = raw['delta'].copy()
        if (raw['dt'] != .005 or not np.array_equal(initial, raw['states'][0])
                or alpha != row['alpha'] or gamma != row['gamma']
                or not np.array_equal(delta, row['delta'])):
            raise ValueError('Saved fine initial/parameter convention differs')
    with np.load(old / case['old_coarse_raw'], allow_pickle=False) as raw:
        if (not np.array_equal(raw['initial'], initial) or not np.array_equal(raw['times'], times)
                or float(raw['alpha']) != alpha or float(raw['gamma']) != gamma):
            raise ValueError('Old paired initial/parameters/times differ')
    role, anchor, phase, _, _ = case['key']
    certificate = baselines[role][anchor]['certificate']
    reference = np.asarray(certificate['orbit_activations'], dtype=np.float64)[:-1]
    epoch = phase * certificate['period'] / 2700
    if (reference.shape != (2700, 3) or not np.array_equal(initial, certificate['orbit_states'][phase])
            or row['phase_epoch'] != epoch or coarse['phase_epoch'] != epoch
            or times.shape != (3001,) or not np.array_equal(times, np.arange(3001) * .1)):
        raise ValueError('Original reference/epoch/sample convention differs')
    return dict(alpha=alpha, gamma=gamma, initial=initial, times=times, delta=delta,
                reference=reference, period=certificate['period'], epoch=epoch,
                old_phase=row['tail_metrics']['phase_offset_cycles'])


def score_trace(context, times, states):
    if (not np.array_equal(times, context['times']) or states.shape != (3001, 3)
            or not np.isfinite(states).all() or not np.array_equal(states[0], context['initial'])):
        raise ValueError('Incomplete or altered waveform sample convention')
    mask = times > 75
    activation = np.tanh(states[mask])
    metrics = waveform_errors(context['reference'], times[mask] + context['epoch'], activation,
                              context['period'], phase_grid=128)
    fixed_reference = periodic_interpolate(context['reference'],
        (times[mask] + context['epoch']) / context['period'] + context['old_phase'])
    return dict(rmse=metrics['aligned_rms']['aggregate'], tail_metrics=metrics,
                old_fixed_phase_rmse=float(np.sqrt(np.mean((activation - fixed_reference)**2))))


def worker(task, context, directory, library, parent_pid):
    """One owned numerical task; native and adaptive failures cannot orphan jobs."""
    start = time.monotonic()
    raw = directory / 'traces' / f'{task["id"]}.npz'
    result = dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
                  status='failed', rmse=None, pid=os.getpid(), started_at=now())
    try:
        method = task['method']
        if method['kind'] == 'RK4':
            times, states = integrate_trace(load_native(library), context['alpha'], context['gamma'],
                context['initial'], method['dt'], duration=300., sample_dt=.1,
                failure_path=raw.with_name(raw.stem + '-failure.npz'))
            result['solver'] = dict(method='RK4', dt=method['dt'], steps=round(300 / method['dt']))
        else:
            solution = solve_ivp(lambda t, x: hopfield_rhs(t, x, context['alpha'], context['gamma']),
                (0., 300.), context['initial'], method='DOP853', t_eval=context['times'],
                rtol=method['rtol'], atol=method['atol'], max_step=method['max_step'])
            times, states = solution.t, solution.y.T
            result['solver'] = dict(**method, success=solution.success, status=solution.status,
                                    message=solution.message, nfev=solution.nfev)
            if not solution.success or states.shape != (3001, 3) or not np.isfinite(states).all():
                atomic_npz(raw.with_name(raw.stem + '-failure.npz'), times=times, states=states,
                           requested_times=context['times'], initial=context['initial'])
                raise ValueError('Incomplete independent solve')
        atomic_npz(raw, times=times, states=states, initial=context['initial'], delta=context['delta'],
                   alpha=context['alpha'], gamma=context['gamma'])
        result.update(score_trace(context, times, states), status='complete', raw=str(raw.relative_to(directory)),
                      raw_sha256=sha(raw), raw_bytes=raw.stat().st_size)
    except BaseException as error:
        result['error'] = repr(error)
    result.update(ended_at=now(), wall_seconds=time.monotonic() - start,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(directory / 'records' / f'{task["id"]}.json', result)


def owned_entry(worker_fn, task, context, directory, library, parent_pid):
    """Every child, including synthetic fixtures, gets the same signal lifecycle."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if sys.platform.startswith('linux'):
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            os._exit(125)
        if os.getppid() != parent_pid:
            os._exit(125)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    worker_fn(task, context, directory, library, parent_pid)


def terminate_owned(process):
    if process.pid is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(timeout=1.)
    if process.is_alive():
        process.kill()
        process.join()


def execute_tasks(specs, contexts, directory, library, *, workers, deadline, per_task=60., worker_fn=worker):
    """Bounded process scheduler; every requested ID gets a terminal status."""
    if not 1 <= workers <= 4:
        raise ValueError('Worker cap is 1..4')
    context = mp.get_context('fork')
    pending = iter(specs)
    active, results = {}, []
    exhausted = False
    journal = (directory / 'journal.jsonl').open('x')

    def record(task, result):
        journal.write(json.dumps(result, allow_nan=False) + '\n')
        journal.flush()
        os.fsync(journal.fileno())
        results.append(result)
        if len(results) % 100 == 0:
            print(json.dumps(dict(stage='refinement', terminal_records=len(results),
                                  required=len(specs))), flush=True)

    try:
        while active or not exhausted:
            expired = time.monotonic() >= deadline
            while not exhausted and len(active) < workers and not expired:
                task = next(pending, None)
                if task is None:
                    exhausted = True
                    break
                process = context.Process(target=owned_entry,
                    args=(worker_fn, task, contexts[task['case']['id']], directory, library, os.getpid()))
                # Register ownership before starting. Mask delivery during fork/start,
                # where multiprocessing may not yet expose the already-created PID.
                active[task['id']] = (process, task, time.monotonic())
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                try:
                    process.start()
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
                with (directory / 'processes.jsonl').open('a') as log:
                    log.write(json.dumps(dict(task_id=task['id'], pid=process.pid, started_at=now())) + '\n')
            for ident, (process, task, started) in list(active.items()):
                timed_out = expired or time.monotonic() - started >= per_task
                if process.is_alive() and not timed_out:
                    continue
                if timed_out:
                    terminate_owned(process)
                else:
                    process.join()
                path = directory / 'records' / f'{ident}.json'
                if path.exists() and not timed_out:
                    result = json.loads(path.read_text())
                else:
                    result = dict(id=ident, case_id=task['case']['id'], method=task['method']['id'],
                        status='whole_attempt_timeout' if expired else 'task_timeout' if timed_out else 'worker_failed',
                        rmse=None, pid=process.pid, exitcode=process.exitcode, ended_at=now(),
                        note='Any files written before termination are retained; unavailable in-process samples are not invented.')
                    write(directory / 'records' / f'{ident}-termination.json', result)
                record(task, result)
                del active[ident]
            if expired and not exhausted:
                for task in pending:
                    record(task, dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
                                      status='not_started_deadline', rmse=None))
                exhausted = True
            if active:
                time.sleep(.01)
    except BaseException:
        for process, task, _ in active.values():
            terminate_owned(process)
            record(task, dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
                             status='interrupted', rmse=None, pid=process.pid))
        for task in pending:
            record(task, dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
                             status='not_started_interruption', rmse=None))
        raise
    finally:
        for process, _, _ in active.values():
            terminate_owned(process)
        journal.close()
    return results


def array_comparison(a, b, times):
    answer = {}
    for name, mask in [('whole', np.ones(len(times), dtype=bool)), ('tail', times > 75)]:
        for observable, x, y in [('state', a, b), ('activation', np.tanh(a), np.tanh(b))]:
            difference = x[mask] - y[mask]
            answer[f'{name}_{observable}_rms'] = float(np.sqrt(np.mean(difference**2)))
            answer[f'{name}_{observable}_max'] = float(np.max(np.abs(difference)))
    answer['activation_waveform_agreement'] = (answer['tail_activation_rms'] <= .001
                                               and answer['tail_activation_max'] <= .01)
    return answer


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None,
                maximum=float(np.max(a)) if len(a) else None)


def compare_scoring(saved, replay, path='score'):
    """Check every saved score/phase/window/cadence field, including categories."""
    if isinstance(saved, dict):
        if not isinstance(replay, dict) or set(saved) != set(replay):
            raise ValueError(f'Scoring keys differ: {path}')
        for name in saved:
            compare_scoring(saved[name], replay[name], f'{path}.{name}')
    elif isinstance(saved, list):
        if not isinstance(replay, list) or len(saved) != len(replay):
            raise ValueError(f'Scoring list differs: {path}')
        for i, (a, b) in enumerate(zip(saved, replay)):
            compare_scoring(a, b, f'{path}[{i}]')
    elif isinstance(saved, float):
        if isinstance(replay, bool) or not isinstance(replay, (int, float)) or not np.isclose(
                saved, replay, rtol=1e-12, atol=1e-14):
            raise ValueError(f'Scoring number differs: {path}')
    elif type(saved) is not type(replay) or saved != replay:
        raise ValueError(f'Scoring category differs: {path}')


def summarize(plan, records, directory):
    indexed = {r['id']: r for r in records}
    expected = {r['id'] for r in tasks(plan)}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError('Incomplete/duplicate/extra terminal record census')
    details = []
    for case in plan['census']:
        if not case['selected']:
            continue
        panel = dict(id=case['id'], key=case['key'], reasons=case['reasons'],
                     independent_labels=case['independent_labels'], old_delta=case['old_delta'],
                     old_coarse_rmse=case['old_coarse_rmse'], old_fine_rmse=case['old_fine_rmse'])
        methods = METHODS if case['independent_labels'] else METHODS[:2]
        rr = {m['id']: indexed[f'{case["id"]}--{m["id"]}'] for m in methods}
        panel['new_scores'] = {name: dict(status=r['status'], rmse=r.get('rmse')) for name, r in rr.items()}
        pair = [rr[m['id']] for m in METHODS[:2]]
        if all(r['status'] == 'complete' for r in pair):
            d1, d2 = abs(case['old_fine_rmse'] - pair[0]['rmse']), abs(pair[0]['rmse'] - pair[1]['rmse'])
            panel.update(d1=d1, d2=d2, refinement_ratio=d2 / d1 if d1 else None,
                         threshold_gate=d2 <= 1e-4, nonincreasing_gate=d2 <= d1,
                         rmse_stable_at_this_resolution=d2 <= 1e-4 and d2 <= d1,
                         finest_minus_old_fine=pair[1]['rmse'] - case['old_fine_rmse'])
        else:
            panel.update(d1=None, d2=None, rmse_stable_at_this_resolution=None)
        arrays = {}
        for name, row in rr.items():
            if row['status'] == 'complete':
                with np.load(directory / row['raw'], allow_pickle=False) as raw:
                    arrays[name] = (raw['times'].copy(), raw['states'].copy())
        comparisons = [('rk4-0.0025', 'rk4-0.00125')]
        if case['independent_labels']:
            comparisons.extend([('rk4-0.00125', 'dop853-1e-11'),
                                ('rk4-0.00125', 'dop853-1e-12'),
                                ('dop853-1e-11', 'dop853-1e-12')])
        panel['waveform_comparisons'] = {}
        for a, b in comparisons:
            if a in arrays and b in arrays:
                if not np.array_equal(arrays[a][0], arrays[b][0]):
                    raise ValueError('New methods have different sample times')
                panel['waveform_comparisons'][f'{a}_vs_{b}'] = dict(
                    rmse_difference=abs(rr[a]['rmse'] - rr[b]['rmse']),
                    **array_comparison(arrays[a][1], arrays[b][1], arrays[a][0]))
            else:
                panel['waveform_comparisons'][f'{a}_vs_{b}'] = None
        if case['independent_labels']:
            independent = [panel['waveform_comparisons'][f'{a}_vs_{b}'] for a, b in comparisons[1:]]
            panel['independent_rmse_agreement'] = (all(r['rmse_difference'] <= 1e-4 for r in independent)
                if all(r is not None for r in independent) else None)
        details.append(panel)
    strata = {}
    for role in ('development', 'test'):
        for radius in [0., *RADII]:
            for group in ['cohort', 'sensitive', 'stable_control', 'zero_dose', *[f'selected-{p}' for p in POLICIES]]:
                group_rows = [r for r in details if r['key'][0] == role and r['key'][3] == radius
                              and (group == 'cohort' or group in r['reasons'])]
                if not group_rows:
                    continue
                strata[f'{role}-r{radius:g}-{group}'] = dict(required=len(group_rows),
                    undefined=sum(r['d2'] is None for r in group_rows),
                    stable=sum(r['rmse_stable_at_this_resolution'] is True for r in group_rows),
                    d2=distribution([r['d2'] for r in group_rows if r['d2'] is not None]))
    policies = {}
    cases_by_key = {tuple(c['key']): c for c in plan['census'] if c['selected']}
    action_sensitivity = []
    for assignment in plan['assignments']:
        case = cases_by_key[tuple(assignment['key'])]
        scores = {}
        for method in METHODS[:2]:
            row = indexed[f'{case["id"]}--{method["id"]}']
            scores[method['id']] = dict(status=row['status'], rmse=row.get('rmse'))
        action_sensitivity.append(dict(**assignment, case_id=case['id'],
            old_coarse_rmse=case['old_coarse_rmse'], old_fine_rmse=case['old_fine_rmse'], new_scores=scores))
    for method in METHODS[:2]:
        method_panel, block_values = {}, {}
        for policy in POLICIES:
            aa = [r for r in plan['assignments'] if r['policy'] == policy]
            values = [indexed[f'{case_id(tuple(a["key"]))}--{method["id"]}'] for a in aa]
            complete = all(r['status'] == 'complete' for r in values)
            blocks = {}
            for block in sorted({a['block'] for a in aa}):
                vv = [r for a, r in zip(aa, values) if a['block'] == block]
                blocks[block] = float(np.mean([r['rmse'] for r in vv])) if all(r['status'] == 'complete' for r in vv) else None
            method_panel[policy] = dict(required=len(values), complete=sum(r['status'] == 'complete' for r in values),
                mean_rmse=float(np.mean([r['rmse'] for r in values])) if complete else None, blocks=blocks)
        for block in method_panel['baseline']['blocks']:
            a, b = method_panel['baseline']['blocks'][block], method_panel['geometry']['blocks'][block]
            block_values[block] = a - b if a is not None and b is not None else None
        method_panel['baseline_minus_geometry'] = dict(blocks=block_values,
            mean=float(np.mean(list(block_values.values()))) if all(v is not None for v in block_values.values()) else None)
        policies[method['id']] = method_panel
    return dict(experiment_id='EXP-009', status_counts=dict(Counter(r['status'] for r in records)),
                required=len(expected), observed=len(records), all_complete=all(r['status'] == 'complete' for r in records),
                cohort_cases=len(details), details=details, strata=strata, fixed_policy_sensitivity=policies,
                fixed_action_sensitivity=action_sensitivity,
                scope=plan['scope'], independent_cases=sum(bool(r['independent_labels']) for r in details),
                rmse_stable=sum(r['rmse_stable_at_this_resolution'] is True for r in details),
                rmse_unstable=sum(r['rmse_stable_at_this_resolution'] is False for r in details),
                rmse_undefined=sum(r['rmse_stable_at_this_resolution'] is None for r in details))


def run(plan_path, data_root, attempt):
    plan = json.loads(Path(plan_path).read_text())
    validate(plan, check_bindings=True, reconstruct=True)
    for name, version in plan['required_runtime_packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Runtime package differs: {name}')
    if sys.platform != 'linux':
        raise ValueError('This bounded fork/PDEATHSIG runner is qualified for Linux only')
    freeze = source_guard(plan, plan_path)
    directory = external_attempt(data_root, attempt)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if os.statvfs(directory.parent).f_bavail * os.statvfs(directory.parent).f_frsize < plan['output_reserve_bytes']:
        raise ValueError('Insufficient external output reserve')
    directory.mkdir(exist_ok=False)
    for name in ['traces', 'records']:
        (directory / name).mkdir()
    shutil.copyfile(plan_path, directory / 'manifest.json')
    start = time.monotonic()
    deadline = start + plan['wall_clock_ceiling_seconds']
    started = dict(experiment_id='EXP-009', **freeze, manifest_sha256=sha(plan_path), started_at=now(),
        status='running', planned_traces=plan['planned_traces'], platform=platform.platform(), python=sys.version,
        packages={n: importlib.metadata.version(n) for n in plan['required_runtime_packages']},
        numpy_build_config=np.__config__.CONFIG, cpu_count=os.cpu_count(), load_average=list(os.getloadavg()),
        thread_environment={n: os.environ[n] for n in THREAD_ENV}, workers=plan['workers'], parent_pid=os.getpid(),
        paid_compute=False, paid_review=False)
    write(directory / 'started.json', started)
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    error = None
    try:
        info = build_native(directory / 'build', deadline)
        write(directory / 'build.json', info)
        rows, _ = read_prior()
        lookup = defaultdict(dict)
        for row in rows:
            lookup[key(row)][row['dt']] = row
        baselines = {r: json.loads((OLD / f'{r}-baselines.json').read_text()) for r in ('development', 'test')}
        contexts = {c['id']: case_context(c, OLD, baselines, lookup) for c in plan['census'] if c['selected']}
        records = execute_tasks(tasks(plan), contexts, directory, info['path'], workers=plan['workers'],
                                deadline=deadline, per_task=plan['per_task_ceiling_seconds'])
        write(directory / 'summary.json', summarize(plan, records, directory))
    except BaseException as exc:
        error = repr(exc)
        write(directory / 'failure.json', dict(status='failed', error=error, ended_at=now()))
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        records = ([json.loads(line) for line in (directory / 'journal.jsonl').read_text().splitlines()]
                   if (directory / 'journal.jsonl').exists() else [])
        observed = {r['id'] for r in records}
        for task in tasks(plan):
            if task['id'] not in observed:
                records.append(dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
                                    status='not_started_infrastructure_failure', rmse=None))
        write(directory / 'terminal-records.json', records)
        artifacts = [binding(f, directory) for f in sorted(directory.rglob('*')) if f.is_file() and f.name != 'receipt.json']
        write(directory / 'receipt.json', dict(**{k: v for k, v in started.items() if k != 'status'},
            status='complete' if error is None and all(r['status'] == 'complete' for r in records) else 'incomplete',
            ended_at=now(), wall_seconds=time.monotonic() - start, expected_records=len(tasks(plan)),
            observed_records=len(records), status_counts=dict(Counter(r['status'] for r in records)),
            error=error, artifacts=artifacts))
    verify(directory)
    return directory


def verify(directory):
    """Raw-array/scoring replay only: never integrate or refit a target."""
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('EXP-009 attempt must be external')
    receipt = json.loads((directory / 'receipt.json').read_text())
    paths = set()
    for item in receipt['artifacts']:
        path = directory / item['path']
        if path.resolve().is_relative_to(directory) is False or binding(path, directory) != item:
            raise ValueError(f'Raw binding differs: {item["path"]}')
        if item['path'] in paths:
            raise ValueError('Duplicate receipt path')
        paths.add(item['path'])
    actual = {str(f.relative_to(directory)) for f in directory.rglob('*') if f.is_file() and f.name != 'receipt.json'}
    if paths != actual:
        raise ValueError('Raw receipt inventory incomplete')
    plan = json.loads((directory / 'manifest.json').read_text())
    validate(plan, check_bindings=True)
    if sha(directory / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Frozen manifest bytes differ')
    records = json.loads((directory / 'terminal-records.json').read_text())
    expected = {r['id'] for r in tasks(plan)}
    if len(records) != len(expected) or {r['id'] for r in records} != expected:
        raise ValueError('Terminal census differs')
    statuses = dict(Counter(r['status'] for r in records))
    if (receipt['expected_records'] != len(expected) or receipt['observed_records'] != len(records)
            or receipt['status_counts'] != statuses
            or (receipt['status'] == 'complete') != all(r['status'] == 'complete' for r in records)):
        raise ValueError('Receipt completion census differs')
    rows, _ = read_prior()
    lookup = defaultdict(dict)
    for row in rows:
        lookup[key(row)][row['dt']] = row
    baselines = {r: json.loads((OLD / f'{r}-baselines.json').read_text()) for r in ('development', 'test')}
    cases = {c['id']: c for c in plan['census'] if c['selected']}
    for row in records:
        if row['status'] != 'complete':
            continue
        path = directory / row['raw']
        if sha(path) != row['raw_sha256'] or path.stat().st_size != row['raw_bytes']:
            raise ValueError('Record/raw binding differs')
        context = case_context(cases[row['case_id']], OLD, baselines, lookup)
        with np.load(path, allow_pickle=False) as raw:
            if (float(raw['alpha']) != context['alpha'] or float(raw['gamma']) != context['gamma']
                    or not np.array_equal(raw['delta'], context['delta'])
                    or not np.array_equal(raw['initial'], context['initial'])):
                raise ValueError('Recorded target parameters or initial state differ')
            replay = score_trace(context, raw['times'], raw['states'])
        compare_scoring({name: row[name] for name in replay}, replay, row['id'])
    if (directory / 'summary.json').exists():
        if json.loads((directory / 'summary.json').read_text()) != summarize(plan, records, directory):
            raise ValueError('Derived refinement summary differs')
    return dict(status='verified_saved_evidence', records=len(records), artifacts=len(paths),
                all_complete=all(r['status'] == 'complete' for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate', 'run', 'verify'])
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--attempt', default='attempt-001')
    args = parser.parse_args()
    if args.action == 'prepare':
        plan = build_plan()
        write(args.plan, plan)
        print(json.dumps(dict(status='prepared_from_old_data_only', **plan['counts'])))
    elif args.action == 'validate':
        plan = json.loads(args.plan.read_text())
        validate(plan, check_bindings=True, reconstruct=True)
        print(json.dumps(dict(status='validated_without_integration', **plan['counts'])))
    elif args.action == 'run':
        print(run(args.plan, args.data_root, args.attempt))
    else:
        print(json.dumps(verify(external_attempt(args.data_root, args.attempt))))


if __name__ == '__main__':
    main()
