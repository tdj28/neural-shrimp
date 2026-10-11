#!/usr/bin/env python3
"""EXP-010 complete short-preview numerical qualification; no learning or control."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
# Sets numerical thread caps before loading NumPy/SciPy and supplies the frozen
# Linux worker lifecycle. No EXP-009 globals or files are changed by this module.
import run_hopfield_refinement as runtime
import numpy as np
from scipy.integrate import solve_ivp
from hopfield_diagnostics import hopfield_rhs
from hopfield_drift_metrics import periodic_interpolate, waveform_errors
from hopfield_drift_probe import integrate_trace

PLAN = ROOT / 'experiments/hopfield/EXP-010-plan.json'
OLD = runtime.OLD
METHODS = runtime.METHODS
RADII = runtime.RADII
SOURCE_PATHS = ['tools/run_preview_refinement.py', 'tests/test_preview_refinement.py',
                'docs/EXP-010-protocol.md', *runtime.SOURCE_PATHS]
CONTRACT = dict(duration=75., sample_dt=.1, phase_grid=128, workers=4, thread_limit=1,
                per_task_ceiling_seconds=60, wall_clock_ceiling_seconds=3600,
                output_reserve_bytes=2 * 1024**3, methods=METHODS,
                planned_native_traces=6912, planned_independent_traces=32, planned_traces=6944,
                metric_stability_atol=1e-4, waveform_rms_atol=1e-3, waveform_max_atol=1e-2,
                required_runtime_packages={'numpy': '2.5.3', 'scipy': '1.18.1'},
                paid_compute=False, paid_review=False)


def key(row):
    return (row['role'], row['anchor'], row['phase'], row['radius'], row['candidate'])


def read_old():
    rows = []
    for role in ['development', 'test']:
        rows.extend(json.loads(line) for line in (OLD / f'{role}-previews.jsonl').read_text().splitlines())
    return rows, json.loads((OLD / 'test-decisions.json').read_text()), json.loads((OLD / 'manifest.json').read_text())


def select_census(rows, decisions, old_plan):
    expected = {(a['role'], a['id'], ph, radius, candidate) for a in old_plan['anchors']
                for ph in old_plan['phases'] for radius in old_plan['radii'] for candidate in range(18)}
    indexed = {}
    for row in rows:
        k = key(row)
        if k in indexed:
            raise ValueError('Duplicate original preview')
        v = row.get('preview_score')
        if (row.get('status') != 'complete' or isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 <= v <= 2
                or row['metrics']['aligned_rms']['aggregate'] != v):
            raise ValueError('Missing or invalid original preview label')
        indexed[k] = row
    if set(indexed) != expected:
        raise ValueError('Original full candidate product is incomplete or contains extras')
    selected, assignments, seen_decisions = defaultdict(list), [], set()
    expected_contexts = {k[:-1] for k in expected if k[0] == 'test'}
    for row in decisions:
        context = ('test', row['anchor'], row['phase'], row['radius'])
        if context in seen_decisions or row['status'] != 'complete' or set(row['policies']) != set(runtime.POLICIES):
            raise ValueError('Original policy-decision census differs')
        seen_decisions.add(context)
        for policy in runtime.POLICIES:
            chosen = row['policies'][policy]
            k = (*context, chosen['candidate'])
            if chosen['status'] != 'complete' or k not in indexed:
                raise ValueError('Original selected preview missing')
            selected[k].append(policy)
            assignments.append(dict(key=list(k), policy=policy, block=row['block']))
    if seen_decisions != expected_contexts:
        raise ValueError('Original decisions do not cover every test context')
    sentinels = defaultdict(list)
    strata = []
    for role in ['development', 'test']:
        for radius in old_plan['radii']:
            pool = [k for k in indexed if k[0] == role and k[3] == radius]
            ordered = sorted(pool, key=lambda k: (indexed[k]['preview_score'], k))
            if not ordered:
                raise ValueError('Independent sentinel stratum empty')
            high = min(pool, key=lambda k: (-indexed[k]['preview_score'], k))
            median_index = (len(ordered) - 1) // 2
            median = ordered[median_index]
            for label, chosen in [('maximum', high), ('lower_median', median)]:
                sentinels[chosen].append(f'{role}-r{radius:g}-{label}')
            strata.append(dict(role=role, radius=radius, size=len(ordered), median_index=median_index,
                               maximum_key=list(high), lower_median_key=list(median)))
    census = [dict(id=indexed[k]['id'], key=list(k), block=indexed[k]['block'], old_raw=indexed[k]['raw'],
                   old_rmse=indexed[k]['preview_score'], original_selected_policies=selected.get(k, []),
                   independent_labels=sentinels.get(k, [])) for k in sorted(indexed)]
    return dict(census=census, assignments=assignments, independent_strata=strata,
                cohort_sha256=hashlib.sha256(runtime.canonical([c['key'] for c in census])).hexdigest(),
                counts=dict(cases=len(census), independent=len(sentinels), selected_unique=len(selected),
                            assignments=len(assignments), contexts=len({k[:-1] for k in indexed})))


def build_plan():
    if runtime.sha(OLD / 'receipt.json') != runtime.OLD_RECEIPT_SHA:
        raise ValueError('Old EXP-008 receipt differs')
    rows, decisions, old_plan = read_old()
    selection = select_census(rows, decisions, old_plan)
    receipt = json.loads((OLD / 'receipt.json').read_text())
    old_index = {r['path']: r for r in receipt['artifacts']}
    paths = {'receipt.json', 'manifest.json', 'test-decisions.json', 'development-baselines.json',
             'test-baselines.json', 'development-previews.jsonl', 'test-previews.jsonl'}
    paths.update(c['old_raw'] for c in selection['census'])
    bindings = []
    for name in sorted(paths):
        b = runtime.binding(OLD / name)
        if name != 'receipt.json' and (b['sha256'], b['bytes']) != (old_index[name]['sha256'], old_index[name]['bytes']):
            raise ValueError(f'Original input binding differs: {name}')
        bindings.append(b)
    plan = dict(experiment_id='EXP-010', date='2026-09-14',
        status='Post-run qualification of the complete original 75-unit preview-label set; no new utility confirmation',
        **CONTRACT, **selection, old_attempt=str(OLD.relative_to(ROOT)), old_receipt_sha256=runtime.OLD_RECEIPT_SHA,
        source_files=[runtime.binding(ROOT / path) for path in SOURCE_PATHS], source_artifacts=bindings,
        default_data_root=str(runtime.DEFAULT_DATA_ROOT),
        scope='All original 75-unit previews only. No features, fit, new action selection, '
              '300-unit task qualification, compute-savings claim or new four-preview utility test.')
    validate(plan)
    return plan


def tasks(plan):
    return [dict(id=f'{case["id"]}--{method["id"]}', case=case, method=method)
            for case in plan['census'] for method in plan['methods']
            if method['kind'] == 'RK4' or case['independent_labels']]


def validate(plan, *, check_bindings=False, reconstruct=False):
    if plan['experiment_id'] != 'EXP-010' or plan['counts'] != dict(
            cases=3456, independent=16, selected_unique=220, assignments=384, contexts=192):
        raise ValueError('Unexpected preview qualification census')
    for name, value in CONTRACT.items():
        if plan[name] != value:
            raise ValueError(f'Frozen contract differs: {name}')
    if (len(plan['census']) != 3456 or len({c['id'] for c in plan['census']}) != 3456
            or len(tasks(plan)) != 6944 or len(plan['assignments']) != 384
            or hashlib.sha256(runtime.canonical([c['key'] for c in plan['census']])).hexdigest() != plan['cohort_sha256']):
        raise ValueError('Duplicate, missing or changed case identity')
    if [r['path'] for r in plan['source_files']] != SOURCE_PATHS:
        raise ValueError('Executable dependency binding incomplete')
    if check_bindings:
        for item in plan['source_files'] + plan['source_artifacts']:
            path = ROOT / item['path']
            if not path.resolve().is_relative_to(ROOT) or runtime.binding(path) != item:
                raise ValueError(f'Source/input binding differs: {item["path"]}')
    if reconstruct and plan != build_plan():
        raise ValueError('Old-data census or sentinel reconstruction differs')
    return plan['counts']


def load_contexts(plan):
    rows, _, _ = read_old()
    indexed = {key(r): r for r in rows}
    baselines = {role: json.loads((OLD / f'{role}-baselines.json').read_text()) for role in ['development', 'test']}
    contexts = {}
    for case in plan['census']:
        row = indexed[tuple(case['key'])]
        role, anchor, phase, _, _ = case['key']
        cert = baselines[role][anchor]['certificate']
        with np.load(OLD / case['old_raw'], allow_pickle=False) as raw:
            initial, times, delta = raw['initial'].copy(), raw['times'].copy(), raw['delta'].copy()
            alpha, gamma = float(raw['alpha']), float(raw['gamma'])
            if (float(raw['dt']) != .005 or not np.array_equal(initial, raw['states'][0])
                    or not np.array_equal(initial, cert['orbit_states'][phase])
                    or not np.array_equal(times, np.arange(751) * .1)
                    or (alpha, gamma) != (row['alpha'], row['gamma']) or not np.array_equal(delta, row['delta'])):
                raise ValueError(f'Original initial/destination/sample convention differs: {case["id"]}')
        reference = np.asarray(cert['orbit_activations'], dtype=float)[:-1]
        if reference.shape != (2700, 3):
            raise ValueError('Original reference sample count differs')
        contexts[case['id']] = dict(initial=initial, times=times, delta=delta, alpha=alpha, gamma=gamma,
            reference=reference, period=cert['period'], epoch=phase * cert['period'] / 2700,
            old_phase=row['metrics']['phase_offset_cycles'])
    return contexts


def score(context, times, states):
    if (not np.array_equal(times, context['times']) or states.shape != (751, 3)
            or not np.isfinite(states).all() or not np.array_equal(states[0], context['initial'])):
        raise ValueError('Incomplete or changed preview sample convention')
    activation = np.tanh(states)
    metrics = waveform_errors(context['reference'], times + context['epoch'], activation,
                              context['period'], phase_grid=128)
    fixed_ref = periodic_interpolate(context['reference'],
        (times + context['epoch']) / context['period'] + context['old_phase'])
    return dict(rmse=metrics['aligned_rms']['aggregate'], metrics=metrics,
                old_fixed_phase_rmse=float(np.sqrt(np.mean((activation - fixed_ref)**2))))


def worker(task, context, directory, library, parent_pid):
    """Runs under EXP-009's unchanged owned_entry and bounded scheduler."""
    start = time.monotonic()
    path = directory / 'traces' / f'{task["id"]}.npz'
    row = dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
               status='failed', rmse=None, pid=os.getpid(), started_at=runtime.now())
    try:
        method = task['method']
        if method['kind'] == 'RK4':
            times, states = integrate_trace(runtime.load_native(library), context['alpha'], context['gamma'],
                context['initial'], method['dt'], duration=75., sample_dt=.1,
                failure_path=path.with_name(path.stem + '-failure.npz'))
            row['solver'] = dict(method='RK4', dt=method['dt'], steps=round(75 / method['dt']))
        else:
            result = solve_ivp(lambda t, x: hopfield_rhs(t, x, context['alpha'], context['gamma']),
                (0., 75.), context['initial'], method='DOP853', t_eval=context['times'],
                rtol=method['rtol'], atol=method['atol'], max_step=method['max_step'])
            times, states = result.t, result.y.T
            row['solver'] = dict(**method, success=result.success, status=result.status,
                                 message=result.message, nfev=result.nfev)
            if not result.success or states.shape != (751, 3) or not np.isfinite(states).all():
                runtime.atomic_npz(path.with_name(path.stem + '-failure.npz'), times=times, states=states,
                                   initial=context['initial'], requested_times=context['times'])
                raise ValueError('Incomplete independent preview solve')
        runtime.atomic_npz(path, times=times, states=states, initial=context['initial'],
                           alpha=context['alpha'], gamma=context['gamma'], delta=context['delta'])
        row.update(score(context, times, states), status='complete', raw=str(path.relative_to(directory)),
                   raw_sha256=runtime.sha(path), raw_bytes=path.stat().st_size)
    except BaseException as error:
        row['error'] = repr(error)
    row.update(ended_at=runtime.now(), wall_seconds=time.monotonic() - start,
               max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    runtime.write(directory / 'records' / f'{task["id"]}.json', row)


def waveform_comparison(a, b):
    result = {}
    for name, x, y in [('state', a, b), ('activation', np.tanh(a), np.tanh(b))]:
        difference = x - y
        result[f'{name}_rms'] = float(np.sqrt(np.mean(difference**2)))
        result[f'{name}_max'] = float(np.max(np.abs(difference)))
    result['activation_waveform_agreement'] = result['activation_rms'] <= .001 and result['activation_max'] <= .01
    return result


def ranking(scores):
    if len(scores) != 18 or any(v is None or not math.isfinite(v) for v in scores):
        return None
    order = sorted(range(18), key=lambda c: (scores[c], c))
    axes = min([0, 4, 8, 12], key=lambda c: (scores[c], c))
    return dict(order=order, argmin=order[0], best_second_gap=scores[order[1]] - scores[order[0]],
                fourth_fifth_gap=scores[order[4]] - scores[order[3]], top_four=order[:4],
                signed_axes_argmin=axes, signed_axes_minimum=scores[axes])


def summarize(plan, records, directory):
    indexed = {row['id']: row for row in records}
    if len(indexed) != len(records) or set(indexed) != {t['id'] for t in tasks(plan)}:
        raise ValueError('Terminal preview census differs')
    details = []
    contexts = defaultdict(list)
    for case in plan['census']:
        methods = METHODS if case['independent_labels'] else METHODS[:2]
        rr = {m['id']: indexed[f'{case["id"]}--{m["id"]}'] for m in methods}
        detail = dict(**case, new_scores={n: dict(status=r['status'], rmse=r.get('rmse')) for n, r in rr.items()})
        first, second = rr['rk4-0.0025'], rr['rk4-0.00125']
        if first['status'] == second['status'] == 'complete':
            d1, d2 = abs(case['old_rmse'] - first['rmse']), abs(first['rmse'] - second['rmse'])
            detail.update(d1=d1, d2=d2, threshold_gate=d2 <= 1e-4, nonincreasing_gate=d2 <= d1,
                rmse_stable_at_this_resolution=d2 <= 1e-4 and d2 <= d1,
                finest_minus_old=second['rmse'] - case['old_rmse'])
        else:
            detail.update(d1=None, d2=None, threshold_gate=None, nonincreasing_gate=None,
                          rmse_stable_at_this_resolution=None, finest_minus_old=None)
        arrays = {}
        for name, row in rr.items():
            if row['status'] == 'complete':
                with np.load(directory / row['raw'], allow_pickle=False) as raw:
                    arrays[name] = (raw['times'].copy(), raw['states'].copy())
        pairs = [('rk4-0.0025', 'rk4-0.00125')]
        if case['independent_labels']:
            pairs += [('rk4-0.00125', 'dop853-1e-11'), ('rk4-0.00125', 'dop853-1e-12'),
                      ('dop853-1e-11', 'dop853-1e-12')]
        comparisons = {}
        for a, b in pairs:
            name = f'{a}_vs_{b}'
            comparisons[name] = None
            if a in arrays and b in arrays:
                if not np.array_equal(arrays[a][0], arrays[b][0]):
                    raise ValueError('Independent preview sample times differ')
                comparisons[name] = dict(rmse_difference=abs(rr[a]['rmse'] - rr[b]['rmse']),
                                         **waveform_comparison(arrays[a][1], arrays[b][1]))
        detail['waveform_comparisons'] = comparisons
        detail['all_required_methods_complete'] = all(r['status'] == 'complete' for r in rr.values())
        if case['independent_labels']:
            panel = [comparisons[f'{a}_vs_{b}'] for a, b in pairs[1:]]
            detail['independent_rmse_agreement'] = (all(v['rmse_difference'] <= 1e-4 for v in panel)
                                                   if all(v is not None for v in panel) else None)
        detail['full_case_scalar_qualification'] = (bool(detail['rmse_stable_at_this_resolution'])
            and detail.get('independent_rmse_agreement', True) if detail['all_required_methods_complete'] else None)
        details.append(detail)
        contexts[tuple(case['key'][:-1])].append(detail)
    strata = {}
    for role in ['development', 'test']:
        for radius in RADII:
            for group in ['all', 'original_selected_unique']:
                members = [d for d in details if d['key'][0] == role and d['key'][3] == radius
                           and (group == 'all' or d['original_selected_policies'])]
                if not members:
                    continue
                complete = [d for d in members if d['d2'] is not None]
                waveform = [d['waveform_comparisons']['rk4-0.0025_vs_rk4-0.00125'] for d in members]
                all_methods = all(v['status'] == 'complete' for d in members for v in d['new_scores'].values())
                strata[f'{role}-r{radius:g}-{group}'] = dict(required=len(members),
                    native_complete=len(complete), native_all_required_complete=len(complete) == len(members),
                    all_required_methods_complete=all_methods,
                    full_group_scalar_qualification=all(d['full_case_scalar_qualification'] for d in members) if all_methods else None,
                    scalar_stable=sum(d['rmse_stable_at_this_resolution'] is True for d in members),
                    scalar_unresolved=sum(d['rmse_stable_at_this_resolution'] is False for d in members),
                    waveform_agree=sum(w is not None and w['activation_waveform_agreement'] for w in waveform),
                    d2_completed_only=runtime.distribution([d['d2'] for d in complete]),
                    absolute_old_to_finest_completed_only=runtime.distribution([abs(d['finest_minus_old']) for d in complete]))
    ranks = []
    for context, members in sorted(contexts.items()):
        members.sort(key=lambda d: d['key'][-1])
        if [d['key'][-1] for d in members] != list(range(18)):
            raise ValueError('Ranking context lost candidate IDs or duplicates')
        values = {'old_0.005': [d['old_rmse'] for d in members]}
        for method in METHODS[:2]:
            values[method['id']] = [d['new_scores'][method['id']]['rmse']
                if d['new_scores'][method['id']]['status'] == 'complete' else None for d in members]
        ranked = {m: ranking(v) for m, v in values.items()}
        comparisons = {}
        for a, b in [('old_0.005', 'rk4-0.0025'), ('old_0.005', 'rk4-0.00125'), ('rk4-0.0025', 'rk4-0.00125')]:
            comparison = None
            if ranked[a] is not None and ranked[b] is not None:
                deltas = [y - x for x, y in zip(values[a], values[b])]
                envelope = max(abs(v) for v in deltas)
                minimum_change = values[b][ranked[b]['argmin']] - values[a][ranked[a]['argmin']]
                comparison = dict(argmin_changed=ranked[a]['argmin'] != ranked[b]['argmin'],
                    top_four_membership_changed=set(ranked[a]['top_four']) != set(ranked[b]['top_four']),
                    ranking_changed=ranked[a]['order'] != ranked[b]['order'],
                    signed_axes_argmin_changed=ranked[a]['signed_axes_argmin'] != ranked[b]['signed_axes_argmin'],
                    signed_candidate_deltas=deltas, empirical_max_absolute_score_change=envelope,
                    old_best_second_gap_exceeds_twice_change=ranked[a]['best_second_gap'] > 2 * envelope,
                    old_fourth_fifth_gap_exceeds_twice_change=ranked[a]['fourth_fifth_gap'] > 2 * envelope,
                    signed_minimum_error_change=minimum_change,
                    minimum_error_change_above_numerical_tolerance=abs(minimum_change) > 1e-4,
                    signed_axes_minimum_error_change=ranked[b]['signed_axes_minimum'] - ranked[a]['signed_axes_minimum'],
                    absolute_axes_minimum_error_change=abs(ranked[b]['signed_axes_minimum'] - ranked[a]['signed_axes_minimum']),
                    envelope_scope='Observed discrepancy across these two evaluated levels, not a bound on exact-solution error or task noninferiority.')
            comparisons[f'{a}_vs_{b}'] = comparison
        ranks.append(dict(context=list(context), candidate_ids=list(range(18)), scores=values,
                          rankings=ranked, comparisons=comparisons))
    selected = [d for d in details if d['original_selected_policies']]
    native_complete = all(d['d2'] is not None for d in details)
    native_waves = [d['waveform_comparisons']['rk4-0.0025_vs_rk4-0.00125'] for d in details]
    return dict(experiment_id='EXP-010', required=len(records), observed=len(records),
        status_counts=dict(Counter(r['status'] for r in records)), all_complete=all(r['status'] == 'complete' for r in records),
        cases=len(details), scalar_stable=sum(d['rmse_stable_at_this_resolution'] is True for d in details),
        scalar_unresolved=sum(d['rmse_stable_at_this_resolution'] is False for d in details),
        scalar_undefined=sum(d['rmse_stable_at_this_resolution'] is None for d in details),
        qualification=dict(native_all_required_complete=native_complete,
            all_required_methods_complete=all(r['status'] == 'complete' for r in records),
            full_scalar_qualification=all(d['full_case_scalar_qualification'] for d in details)
                if all(d['all_required_methods_complete'] for d in details) else None,
            all_native_scalar_flags_pass=all(d['rmse_stable_at_this_resolution'] for d in details) if native_complete else None,
            all_native_waveform_flags_pass=all(w['activation_waveform_agreement'] for w in native_waves) if native_complete else None),
        original_selected_unique=dict(required=len(selected),
            native_complete=sum(d['d2'] is not None for d in selected),
            native_all_required_complete=all(d['d2'] is not None for d in selected),
            all_required_methods_complete=all(v['status'] == 'complete' for d in selected for v in d['new_scores'].values()),
            full_scalar_qualification=all(d['full_case_scalar_qualification'] for d in selected)
                if all(d['all_required_methods_complete'] for d in selected) else None,
            scalar_stable=sum(d['rmse_stable_at_this_resolution'] is True for d in selected)),
        details=details, role_radius_groups=strata, context_rankings=ranks, scope=plan['scope'])


def external_attempt(data_root, attempt):
    # Reuse the same external-root check without changing the frozen helper.
    checked = runtime.external_attempt(data_root, attempt)
    path = (checked.parent.parent / 'EXP-010' / checked.name).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError('EXP-010 output cannot resolve inside the source checkout')
    return path


def source_guard(plan_path):
    if runtime.git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('All tracked/untracked source must be clean before preview outcomes')
    source = runtime.git('rev-parse', 'HEAD')
    for path in [*SOURCE_PATHS, str(Path(plan_path).resolve().relative_to(ROOT))]:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{source}:{path}'])
        if hashlib.sha256(content).hexdigest() != runtime.sha(ROOT / path):
            raise ValueError(f'Source or plan not committed: {path}')
    branch = runtime.git('symbolic-ref', '--short', 'HEAD')
    remote = runtime.git('ls-remote', '--exit-code', 'origin', f'refs/heads/{branch}')
    if remote.split()[0] != source:
        raise ValueError('Exact preview source and plan must be pushed')
    return dict(source_commit=source, pushed_branch=branch, remote_head_verified_at=runtime.now())


def run(plan_path, data_root, attempt):
    plan = json.loads(Path(plan_path).read_text())
    validate(plan, check_bindings=True, reconstruct=True)
    if sys.platform != 'linux':
        raise ValueError('Frozen owned-worker lifecycle is qualified for Linux')
    for name, version in plan['required_runtime_packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Runtime package differs: {name}')
    freeze = source_guard(plan_path)
    directory = external_attempt(data_root, attempt)
    directory.parent.mkdir(parents=True, exist_ok=True)
    space = os.statvfs(directory.parent)
    if space.f_bavail * space.f_frsize < plan['output_reserve_bytes']:
        raise ValueError('External output reserve unavailable')
    directory.mkdir(exist_ok=False)
    for name in ['traces', 'records']:
        (directory / name).mkdir()
    shutil.copyfile(plan_path, directory / 'manifest.json')
    started = dict(experiment_id='EXP-010', **freeze, manifest_sha256=runtime.sha(plan_path),
        started_at=runtime.now(), parent_pid=os.getpid(), planned_traces=6944,
        platform=platform.platform(), python=sys.version, numpy_build_config=np.__config__.CONFIG,
        packages={name: importlib.metadata.version(name) for name in plan['required_runtime_packages']},
        thread_environment={name: os.environ[name] for name in runtime.THREAD_ENV},
        cpu_count=os.cpu_count(), load_average=list(os.getloadavg()), workers=4, paid_compute=False, paid_review=False)
    runtime.write(directory / 'started.json', started)
    begin = time.monotonic()
    deadline = begin + plan['wall_clock_ceiling_seconds']
    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    error = None
    try:
        build = runtime.build_native(directory / 'build', deadline)
        runtime.write(directory / 'build.json', build)
        contexts = load_contexts(plan)
        records = runtime.execute_tasks(tasks(plan), contexts, directory, build['path'], workers=4,
            deadline=deadline, per_task=plan['per_task_ceiling_seconds'], worker_fn=worker)
        runtime.write(directory / 'summary.json', summarize(plan, records, directory))
    except BaseException as exc:
        error = repr(exc)
        runtime.write(directory / 'failure.json', dict(error=error, ended_at=runtime.now()))
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        records = ([json.loads(line) for line in (directory / 'journal.jsonl').read_text().splitlines()]
                   if (directory / 'journal.jsonl').exists() else [])
        observed = {r['id'] for r in records}
        for task in tasks(plan):
            if task['id'] not in observed:
                records.append(dict(id=task['id'], case_id=task['case']['id'], method=task['method']['id'],
                                    status='not_started_infrastructure_failure', rmse=None))
        runtime.write(directory / 'terminal-records.json', records)
        inventory = [runtime.binding(f, directory) for f in sorted(directory.rglob('*')) if f.is_file() and f.name != 'receipt.json']
        runtime.write(directory / 'receipt.json', dict(**started, ended_at=runtime.now(),
            wall_seconds=time.monotonic() - begin, expected_records=6944, observed_records=len(records),
            status='complete' if error is None and all(r['status'] == 'complete' for r in records) else 'incomplete',
            status_counts=dict(Counter(r['status'] for r in records)), error=error, artifacts=inventory))
    verify(directory)
    return directory


def validate_terminal_records(plan, records):
    expected = {task['id']: task for task in tasks(plan)}
    if len(records) != len(expected) or {r['id'] for r in records} != set(expected):
        raise ValueError('Terminal preview identity census differs')
    for row in records:
        task = expected[row['id']]
        if row.get('case_id') != task['case']['id'] or row.get('method') != task['method']['id']:
            raise ValueError('Terminal ID does not bind its planned case/method')
        if row['status'] == 'complete' and row.get('raw') != f'traces/{row["id"]}.npz':
            raise ValueError('Complete row does not bind its planned raw path')
        if row['status'] == 'complete':
            method = task['method']
            solver = row.get('solver', {})
            if method['kind'] == 'RK4':
                if solver != dict(method='RK4', dt=method['dt'], steps=round(75 / method['dt'])):
                    raise ValueError('Recorded RK4 solver differs from planned method')
            elif (set(solver) != {*method, 'success', 'status', 'message', 'nfev'}
                    or any(solver.get(k) != v for k, v in method.items())
                    or solver['success'] is not True or solver['status'] != 0
                    or type(solver['nfev']) is not int or solver['nfev'] <= 0):
                raise ValueError('Recorded independent solver differs from planned method')


def require_completed_artifacts(records, directory, receipt_paths):
    if all(row['status'] == 'complete' for row in records) and 'summary.json' not in receipt_paths:
        raise ValueError('Complete attempt requires a receipt-bound summary')
    for row in records:
        if row['status'] != 'complete':
            continue
        record = f'records/{row["id"]}.json'
        if row['raw'] not in receipt_paths or record not in receipt_paths:
            raise ValueError('Complete attempt lacks required trace/record artifacts')
        if json.loads((directory / record).read_text()) != row:
            raise ValueError('Per-trace record differs from terminal journal row')


def verify(directory):
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('Preview raw evidence must remain external')
    receipt = json.loads((directory / 'receipt.json').read_text())
    seen = set()
    for item in receipt['artifacts']:
        path = directory / item['path']
        if not path.resolve().is_relative_to(directory) or runtime.binding(path, directory) != item or item['path'] in seen:
            raise ValueError(f'Raw artifact binding differs: {item["path"]}')
        seen.add(item['path'])
    if seen != {str(f.relative_to(directory)) for f in directory.rglob('*') if f.is_file() and f.name != 'receipt.json'}:
        raise ValueError('Raw inventory differs')
    plan = json.loads((directory / 'manifest.json').read_text())
    validate(plan, check_bindings=True)
    if runtime.sha(directory / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Manifest bytes differ')
    records = json.loads((directory / 'terminal-records.json').read_text())
    validate_terminal_records(plan, records)
    require_completed_artifacts(records, directory, seen)
    if (len(records) != 6944 or len({r['id'] for r in records}) != 6944
            or {r['id'] for r in records} != {t['id'] for t in tasks(plan)}
            or receipt['expected_records'] != 6944 or receipt['observed_records'] != 6944
            or receipt['status_counts'] != dict(Counter(r['status'] for r in records))
            or (receipt['status'] == 'complete') != all(r['status'] == 'complete' for r in records)):
        raise ValueError('Terminal qualification census differs')
    contexts = load_contexts(plan)
    for row in records:
        if row['status'] != 'complete':
            continue
        path = directory / row['raw']
        if runtime.sha(path) != row['raw_sha256'] or path.stat().st_size != row['raw_bytes']:
            raise ValueError('Record/raw binding differs')
        context = contexts[row['case_id']]
        with np.load(path, allow_pickle=False) as raw:
            if (float(raw['alpha']) != context['alpha'] or float(raw['gamma']) != context['gamma']
                    or not np.array_equal(raw['initial'], context['initial']) or not np.array_equal(raw['delta'], context['delta'])):
                raise ValueError('Realized preview inputs differ')
            replay = score(context, raw['times'], raw['states'])
        runtime.compare_scoring({name: row[name] for name in replay}, replay, row['id'])
    if (directory / 'summary.json').exists() and json.loads((directory / 'summary.json').read_text()) != summarize(plan, records, directory):
        raise ValueError('Preview qualification summary differs')
    return dict(status='verified_saved_preview_evidence', records=6944, artifacts=len(seen),
                all_complete=all(r['status'] == 'complete' for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'validate', 'run', 'verify'])
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--data-root', type=Path, default=runtime.DEFAULT_DATA_ROOT)
    parser.add_argument('--attempt', default='attempt-001')
    args = parser.parse_args()
    if args.action == 'prepare':
        plan = build_plan()
        runtime.write(args.plan, plan)
        print(json.dumps(dict(status='prepared_from_original_previews_only', **plan['counts'])))
    elif args.action == 'validate':
        plan = json.loads(args.plan.read_text())
        validate(plan, check_bindings=True, reconstruct=True)
        contexts = load_contexts(plan)
        print(json.dumps(dict(status='validated_without_integration', validated_case_inputs=len(contexts), **plan['counts'])))
    elif args.action == 'verify':
        print(json.dumps(verify(external_attempt(args.data_root, args.attempt))))
    else:
        directory = run(args.plan, args.data_root, args.attempt)
        complete = json.loads((directory / 'receipt.json').read_text())['status'] == 'complete'
        print(json.dumps(dict(directory=str(directory), complete=complete)))
        if not complete:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
