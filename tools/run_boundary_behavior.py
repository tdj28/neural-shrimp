#!/usr/bin/env python3
"""Frozen EXP-025 acquisition, saved-data analysis and hash verification."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time
import traceback

import numpy as np
from scipy.integrate import solve_ivp
import scipy

import run_hopfield as base
from hopfield_drift_probe import build_trace, integrate_trace
from hopfield_diagnostics import hopfield_rhs
import boundary_behavior as assay

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'experiments/hopfield/EXP-025-plan.json'
SOURCE = ['tools/boundary_behavior.py', 'tools/run_boundary_behavior.py',
          'tools/run_hopfield.py', 'tools/hopfield_diagnostics.py',
          'tools/hopfield_drift_probe.py', 'native/hopfield.cpp', 'native/hopfield_drift.cpp',
          'experiments/hopfield/EXP-025-plan.json', 'docs/EXP-025-protocol.md',
          'tests/test_boundary_behavior.py', 'reports/EXP-025/audit_saved.py']


def validate(p):
    assert p['experiment_id'] == 'EXP-025'
    assert len(p['points']) == 26 and len({(x['alpha'], x['gamma']) for x in p['points']}) == 26
    assert p['initial_states'] == [[.1, .1, .1], [.2, -.1, .3]]
    assert p['train_contexts_per_start'] == 32 and p['test_contexts_per_start'] == 64
    assert p['amplitudes'] == [.01, .1] and p['task_dt'] == [.0025, .00125]
    assert p['windows'] == [[20, 25], [55, 60]]
    assert p['workers'] == 6 and p['wall_seconds'] == 1800
    assert p['geometry']['dt'] == [.01, .005]
    assert p['paid_compute'] is False
    return p


def context_indices(p):
    # Draw without replacement from a fixed autonomous observation tape.
    # Different initializations use independent seed streams, shared across parameters.
    return np.array([np.random.default_rng(p['context_seed']+j).choice(
        40001, 96, replace=False) for j in range(2)])


def trace(fn, alpha, gamma, state, dt, duration, sample_dt=1.):
    return integrate_trace(fn, alpha, gamma, state, dt, duration=duration, sample_dt=sample_dt)[1]


def two_pulse(fn, alpha, gamma, initial, amplitude, signs, dt):
    first = np.array(initial, copy=True)
    first[0] += amplitude*signs[0]
    a = trace(fn, alpha, gamma, first, dt, 20.)
    second = a[-1].copy()
    second[0] += amplitude*signs[1]
    b = trace(fn, alpha, gamma, second, dt, 40.)
    # t=20 records immediately AFTER the second pulse, identically in every regime.
    return np.concatenate((a[:-1], b))


def independent_pulse(alpha, gamma, initial, amplitude, signs):
    rhs = lambda t, x: hopfield_rhs(t, x, alpha, gamma)
    first = np.array(initial, copy=True)
    first[0] += amplitude*signs[0]
    a = solve_ivp(rhs, (0, 20), first, method='DOP853', rtol=2e-12, atol=2e-14,
                  max_step=.025, t_eval=np.arange(21.))
    assert a.success
    second = a.y[:, -1].copy()
    second[0] += amplitude*signs[1]
    b = solve_ivp(rhs, (0, 40), second, method='DOP853', rtol=2e-12, atol=2e-14,
                  max_step=.025, t_eval=np.arange(41.))
    assert b.success
    return np.concatenate((a.y.T[:-1], b.y.T))


def geometry(index, point, p, directory, fn):
    path = directory/f'point-{index:02d}'
    path.mkdir()
    g = p['geometry']
    starts = []
    final_states = []
    for j, initial in enumerate(p['initial_states']):
        cells = []
        for dt in g['dt']:
            z = base.cell(fn, point['alpha'], point['gamma'], initial, g, dt=dt,
                          failure_path=path/f'geometry-{j}-{dt}-failure.npz')
            final, events, stats, n, period, residual, duration = z
            np.savez_compressed(path/f'geometry-{j}-{dt}.npz', initial=initial, final=final,
                                events=events, stats=stats, n=n, period=period,
                                residual=residual, duration=duration)
            cells.append(z)
        q = cells[0][4] if cells[0][4] == cells[1][4] else 0
        cert = dict(verified=False, attracting=False, reason='No agreed candidate period')
        if q:
            fine = cells[-1]
            cert = base.certify(point['alpha'], point['gamma'],
                                fine[1][fine[3]-q, 1:], fine[6], int(q))
        status = 'unresolved'
        if cert.get('verified') and cert.get('attracting'):
            status = 'periodic'
        elif all(z[4] == 0 and z[2][1] > .005 for z in cells):
            status = 'irregular-supported'
        starts.append(dict(status=status, periods=[z[4] for z in cells],
                           ftle=[z[2][1] for z in cells], certificate=cert))
        final_states.append(cells[-1][0])
    status = starts[0]['status'] if starts[0]['status'] == starts[1]['status'] else 'mixed-or-unresolved'
    out = dict(index=index, **point, status=status, starts=starts)
    base.write(path/'geometry.json', out)
    np.save(path/'terminal_states.npy', final_states, allow_pickle=False)
    print(json.dumps(dict(stage='geometry', point=index, status=status)), flush=True)
    return out


def owned_job(function, index, *args):
    """Retain each worker's failure, including native partial payloads."""
    directory = args[2]
    try:
        return function(index, *args)
    except BaseException as exc:
        path = directory/f'point-{index:02d}'
        path.mkdir(exist_ok=True)
        if hasattr(exc, 'payload'):
            np.savez_compressed(path/f'{function.__name__}-failure-payload.npz', **exc.payload)
        base.write(path/f'{function.__name__}-failure.json',
                   dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


def acquire(index, point, p, directory, fn, deadline):
    path = directory/f'point-{index:02d}'
    def check():
        if time.monotonic() > deadline:
            raise TimeoutError('Frozen acquisition deadline')
    check()
    terminals = np.load(path/'terminal_states.npy')
    indices = context_indices(p)
    tapes = [trace(fn, point['alpha'], point['gamma'], state, .005, 2000., .05)
             for state in terminals]
    contexts = np.concatenate([tapes[j][indices[j]] for j in range(2)])
    np.savez_compressed(path/'contexts.npz', indices=indices, contexts=contexts, tapes=np.array(tapes))
    for k, dt in enumerate(p['task_dt']):
        for ai, amplitude in enumerate(p['amplitudes']):
            # Partial progress is durable after each context, including partial failures.
            target = path/f'traces-{k}-{ai}.npy'
            data = np.lib.format.open_memmap(target, mode='w+', dtype='float64', shape=(192, 4, 61, 3))
            data[:] = np.nan
            for c, state in enumerate(contexts):
                check()
                for si, signs in enumerate(assay.SIGNS):
                    data[c, si] = two_pulse(fn, point['alpha'], point['gamma'], state,
                                            amplitude, signs, dt)
                data.flush()
                base.write(path/'progress.json', dict(method=k, amplitude=ai, completed_contexts=c+1))
            del data
        # Same tiny initial uncertainty, no input; separate behavioral diagnostic.
        repeat = np.lib.format.open_memmap(path/f'repeat-{k}.npy', mode='w+', dtype='float64',
                                          shape=(192, 2, 61, 3))
        repeat[:] = np.nan
        for c, state in enumerate(contexts):
            check()
            for j in range(2):
                repeat[c, j] = trace(fn, point['alpha'], point['gamma'],
                    state+j*1e-6*np.ones(3)/np.sqrt(3), dt, 60.)
            repeat.flush()
        del repeat
    check()
    independent = np.array([[independent_pulse(point['alpha'], point['gamma'],
        contexts[c], .1, signs) for signs in assay.SIGNS] for c in [0, 96]])
    np.save(path/'independent.npy', independent, allow_pickle=False)
    print(json.dumps(dict(stage='assay', point=index, complete=True)), flush=True)


def analyze(directory, p):
    train = np.r_[0:32, 96:128]
    test = np.r_[32:96, 128:192]
    reports, losses, classes = [], [[], []], []
    for i, point in enumerate(p['points']):
        path = directory/f'point-{i:02d}'
        geo = json.loads((path/'geometry.json').read_text())
        classes.append(geo['status'])
        item = dict(index=i, **point, status=geo['status'], assays=[])
        for ai, amplitude in enumerate(p['amplitudes']):
            arrays = [np.load(path/f'traces-{k}-{ai}.npy', mmap_mode='r') for k in range(2)]
            assert all(np.isfinite(a).all() for a in arrays)
            for wi, window in enumerate(p['windows']):
                for quadratic in [False, True]:
                    scores = [assay.score(a[train], a[test], window, p['ridge'], quadratic) for a in arrays]
                    delta = float(np.max(np.abs(scores[0]['r2']-scores[1]['r2'])))
                    row = dict(amplitude=amplitude, window=window, quadratic=quadratic,
                               r2=[s['r2'] for s in scores], accuracy=[s['accuracy'] for s in scores],
                               r2_step_discrepancy=delta)
                    item['assays'].append(row)
                    for k, s in enumerate(scores):
                        np.savez_compressed(path/f'readout-{k}-{ai}-{wi}-{int(quadratic)}.npz', **s)
                    if ai == 1 and wi == 1 and not quadratic:
                        for k in range(2):
                            losses[k].append(scores[k]['context_loss'][:, 0])
            item.setdefault('max_activation_step_difference', []).append(
                float(np.max(np.abs(np.tanh(arrays[0])-np.tanh(arrays[1])))))
        independent = np.load(path/'independent.npy')
        fine = np.load(path/'traces-1-1.npy', mmap_mode='r')[[0, 96]]
        item['independent_activation_difference'] = float(np.max(np.abs(np.tanh(fine)-np.tanh(independent))))
        repetitions = []
        for k in range(2):
            raw = np.load(path/f'repeat-{k}.npy')
            separation = np.linalg.norm(np.tanh(raw[:, 0])-np.tanh(raw[:, 1]), axis=-1)
            repetitions.append(dict(final_mean=float(separation[test, -1].mean()),
                                    final_median=float(np.median(separation[test, -1])),
                                    final_max=float(separation[test, -1].max())))
        item['repeatability'] = repetitions
        item['qualified'] = (max(item['max_activation_step_difference']) <= p['activation_gate']
            and item['independent_activation_difference'] <= p['activation_gate']
            and max(a['r2_step_discrepancy'] for a in item['assays']) <= p['score_gate'])
        reports.append(item)
    decisions = [assay.compare_boundaries(p['points'], classes, loss, p['minimum_effect']) for loss in losses]
    # No dropping failed settings: the primary decision is gated on the entire census.
    qualified = all(x['qualified'] for x in reports)
    decision = decisions[-1]['decision'] if qualified and decisions[0]['decision'] == decisions[1]['decision'] else 'unqualified numerical comparison'
    summary = dict(experiment_id='EXP-025', qualified=qualified, decision=decision,
                   method_decisions=decisions, points=reports)
    base.write(directory/'analysis.json', summary)
    return summary


def inventory(directory):
    return [dict(path=str(f.relative_to(directory)), sha256=base.sha(f), bytes=f.stat().st_size)
            for f in sorted(directory.rglob('*')) if f.is_file() and f.name != 'receipt.json']


def run(directory):
    p = validate(json.loads(PLAN.read_text()))
    assert not base.git('status', '--porcelain'), 'Clean source required'
    source = base.git('rev-parse', 'HEAD')
    assert base.git('branch', '-r', '--contains', source), 'Pushed source required'
    assert all(os.environ.get(k) == '1' for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'])
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PLAN, directory/'manifest.json')
    started = time.monotonic()
    deadline = started+p['wall_seconds']
    meta = dict(source_commit=source, plan_sha256=base.sha(PLAN), started_at=base.now(), pid=os.getpid(),
                source_files={f:base.sha(ROOT/f) for f in SOURCE}, platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, workers=p['workers'],
                threads={k:os.environ[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
    base.write(directory/'start.json', meta)
    status = 'failed'
    try:
        engine, engine_info = base.build_engine(directory/'build')
        trace_fn, trace_info = build_trace(directory/'build')
        base.write(directory/'build.json', dict(engine=engine_info, trace=trace_info))
        with ThreadPoolExecutor(max_workers=p['workers']) as pool:
            futures = [pool.submit(owned_job, geometry, i, point, p, directory, engine)
                       for i, point in enumerate(p['points'])]
            geometries = [f.result() for f in futures]
        base.write(directory/'geometry-census.json', geometries)
        # Geometry census is complete and immutable before any task endpoint is acquired.
        with ThreadPoolExecutor(max_workers=p['workers']) as pool:
            futures = [pool.submit(owned_job, acquire, i, point, p, directory, trace_fn, deadline)
                       for i, point in enumerate(p['points'])]
            for future in as_completed(futures):
                future.result()
        result = analyze(directory, p)
        status = 'complete'
        print(json.dumps(dict(decision=result['decision'], qualified=result['qualified'])), flush=True)
    except BaseException as exc:
        base.write(directory/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        base.write(directory/'receipt.json', dict(**meta, status=status, finished_at=base.now(),
            wall_seconds=time.monotonic()-started, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artifacts=inventory(directory)))


def verify(directory):
    receipt = json.loads((directory/'receipt.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['plan_sha256'] == base.sha(directory/'manifest.json')
    for a in receipt['artifacts']:
        path = directory/a['path']
        assert path.stat().st_size == a['bytes'] and base.sha(path) == a['sha256'], a['path']
    p = validate(json.loads((directory/'manifest.json').read_text()))
    assert len(list(directory.glob('point-*/geometry.json'))) == len(p['points'])
    assert len(list(directory.glob('point-*/traces-*.npy'))) == len(p['points'])*4
    for f in directory.glob('point-*/contexts.npz'):
        with np.load(f) as z:
            np.testing.assert_array_equal(z['indices'], context_indices(p))
            np.testing.assert_array_equal(z['contexts'], np.concatenate([z['tapes'][j][z['indices'][j]] for j in range(2)]))
    print(json.dumps(dict(hash_verified=True, artifacts=len(receipt['artifacts']), source=receipt['source_commit'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['validate', 'run', 'verify'])
    parser.add_argument('directory', type=Path, nargs='?')
    args = parser.parse_args()
    if args.action == 'validate':
        validate(json.loads(PLAN.read_text()))
        print(json.dumps(dict(validated=True, points=26, contexts=192, signs=4)))
    elif args.action == 'run':
        run(args.directory)
    else:
        verify(args.directory)
