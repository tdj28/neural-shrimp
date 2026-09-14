#!/usr/bin/env python3
"""EXP008: sealed development fit, identical test actions, geometry ablation."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from numbers import Real
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import importlib.metadata
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import run_hopfield as base
from run_hopfield_drift import atomic_npz, baseline, assert_portable_equal, verify_wave_metrics
from run_hopfield_holdout import check_trace
from hopfield_drift_metrics import waveform_errors
from hopfield_drift_probe import build_trace, integrate_trace, local_parameter_sensitivity, phase_projected_gram
from hopfield_probe_selector import direction_candidates, score_probe
from hopfield_information_design import build_atlas_geometry, geometry_features, GEOMETRY_FEATURE_NAMES
from hopfield_information_features import base_features, BASE_FEATURE_NAMES
from hopfield_information_model import (fit_information_models, predict_information_models,
    select_candidate_actions, summarize_information_outcomes, packet_digest)

PLAN = ROOT/'experiments/hopfield/EXP-008-plan.json'
PHASES = [675, 1350]
RADII = [.00025, .001, .004, .016]
ROLES = ['development', 'test']


def validate(p, *, check_sources=False):
    assert p['experiment_id'] == 'EXP-008'
    assert p['phases'] == PHASES and p['radii'] == RADII and p['dt'] == [.01, .005]
    assert p['primary_dt'] == p['decision_dt'] == .005
    assert p['preview_duration'] == 75 and p['probe_duration'] == 300 and p['sample_dt'] == .1
    assert p['phase_grid'] == 128 and p['workers'] == 4 and p['wall_clock_ceiling_seconds'] == 3600
    assert p['initial_state'] == [.1, .1, .1] and p['burn_time'] == 10000 and p['sample_time'] == 2000
    assert p['event_capacity'] == 128 and p['max_peak_period'] == 32 and p['minimum_repeats'] == 4
    assert p['recurrence_atol'] == 2e-6 and p['recurrence_rtol'] == 2e-5
    assert p['candidate_count'] == 18 and p['shuffle_shift'] == 5
    assert p['bootstrap_samples'] == 10000 and p['bootstrap_seed'] == 20260913
    assert p['lambda_grid'] == [1e-6, 1e-4, 1e-2, 1, 100]
    assert p['required_runtime_packages'] == {'numpy':'2.5.3', 'scipy':'1.18.1'}
    assert p['base_feature_names'] == list(BASE_FEATURE_NAMES)
    assert p['geometry_feature_names'] == list(GEOMETRY_FEATURE_NAMES)
    assert p['paid_compute'] is False and p['paid_review'] is False
    assert len(p['anchors']) == 24 and len({a['id'] for a in p['anchors']}) == 24
    for role in ROLES:
        aa = [a for a in p['anchors'] if a['role'] == role]
        assert len(aa) == len({a['block_id'] for a in aa}) == 12
    assert p['planned_targets'] == 7008 and p['planned_previews'] == 3456
    if check_sources:
        for item in p['source_artifacts'] + p['source_files']:
            assert base.sha(ROOT/item['path']) == item['sha256'], item['path']
    return 7008


def anchors_for(p, role):
    return [a for a in p['anchors'] if a['role'] == role]


def condition_id(aid, phase, radius):
    return f'{aid}-p{phase}-r{radius:g}'


def feature_id(aid, phase, radius, candidate):
    return f'{condition_id(aid, phase, radius)}-c{candidate:02d}'


def specs(p, role):
    for a in anchors_for(p, role):
        for phase in p['phases']:
            for radius in [0., *p['radii']]:
                for candidate in ([-1] if radius == 0 else range(18)):
                    for dt in p['dt']:
                        yield dict(role=role, anchor=a['id'], block=a['block_id'], phase=phase,
                                   radius=radius, candidate=candidate, dt=dt)


def trial_id(s):
    stem = (f"{condition_id(s['anchor'], s['phase'], 0.)}-zero" if s['candidate'] == -1 else
            feature_id(s['anchor'], s['phase'], s['radius'], s['candidate']))
    return f"{stem}-dt{s['dt']:g}"


def candidates(local):
    rows = direction_candidates('uniform_16') + direction_candidates('local_signed', local_direction=local)
    return np.asarray([row['direction'] for row in rows])


def valid_rmse(value):
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and
            np.isfinite(value) and 0 <= value <= 2)


def expected_features(p, role):
    return [feature_id(a['id'], ph, r, c) for a in anchors_for(p, role)
            for ph in p['phases'] for r in p['radii'] for c in range(18)]


def shuffled_geometry(rows, p, role):
    lookup = {(r['anchor'], r['phase'], r['radius'], r['candidate']): r for r in rows}
    aa = anchors_for(p, role)
    donor = {a['id']: aa[(i+p['shuffle_shift']) % 12]['id'] for i, a in enumerate(aa)}
    return [lookup[(donor[r['anchor']], r['phase'], r['radius'], r['candidate'])]['geometry'] for r in rows]


def fit_development(rows, records, p):
    """Explicit role/census gates keep test records out of training."""
    if any(r['role'] != 'development' for r in [*rows, *records]):
        raise ValueError('Only development records may enter fitting')
    if [r['id'] for r in rows] != expected_features(p, 'development'):
        raise ValueError('Development feature census/order mismatch')
    keys = [trial_id(s) for s in specs(p, 'development')]
    if len(records) != len(keys) or {r['id'] for r in records} != set(keys):
        raise ValueError('Incomplete development target census')
    lookup = {r['id']: r for r in records}
    labels = [lookup[f"{r['id']}-dt0.005"].get('rmse') for r in rows]
    if (any(r['status'] != 'complete' for r in rows) or
            any(r['status'] != 'complete' or not valid_rmse(r.get('rmse')) for r in records)):
        labels = [None] * len(rows)
    return fit_information_models([r.get('base') for r in rows], [r.get('geometry') for r in rows],
        labels, [r['block'] for r in rows], shuffled_geometry_features=shuffled_geometry(rows, p, 'development'),
        base_feature_names=list(BASE_FEATURE_NAMES), geometry_feature_names=list(GEOMETRY_FEATURE_NAMES),
        row_ids=[r['id'] for r in rows], expected_row_ids=expected_features(p, 'development'),
        expected_group_ids=[a['block_id'] for a in anchors_for(p, 'development')],
        shuffle_metadata=dict(method='cyclic donor origin, matched phase/radius/candidate entry', shift=5))


def decide_test(rows, packet, p):
    """Only previews/features and the locked development packet enter decisions."""
    if any(r['role'] != 'test' for r in rows) or [r['id'] for r in rows] != expected_features(p, 'test'):
        raise ValueError('Test feature census/order mismatch')
    result = predict_information_models(packet, [r.get('base') for r in rows],
        [r.get('geometry') for r in rows], shuffled_geometry_features=shuffled_geometry(rows, p, 'test'),
        base_feature_names=list(BASE_FEATURE_NAMES), geometry_feature_names=list(GEOMETRY_FEATURE_NAMES))
    decisions = []
    for offset in range(0, len(rows), 18):
        part = rows[offset:offset+18]
        predictions = {**result, 'scores': {k: None if v is None else v[offset:offset+18]
                                           for k, v in result['scores'].items()}}
        choice = select_candidate_actions(predictions, [r.get('preview_score') for r in part])
        first = part[0]
        decisions.append(dict(anchor=first['anchor'], block=first['block'], phase=first['phase'],
                              radius=first['radius'], **choice))
    return decisions


def authorize_test_targets(directory):
    """The append-only target journal cannot exist until both seals are bound."""
    assert not (directory/'test-targets.jsonl').exists(), 'Test target phase already consumed'
    model_seal = json.loads((directory/'model-seal.json').read_text())
    decision_seal = json.loads((directory/'test-decision-seal.json').read_text())
    assert model_seal['model_sha256'] == base.sha(directory/'model.json')
    assert decision_seal['model_seal_sha256'] == base.sha(directory/'model-seal.json')
    for key, name in [('decisions_sha256', 'test-decisions.json'), ('features_sha256', 'test-features.json')]:
        assert decision_seal[key] == base.sha(directory/name)
    assert model_seal['sealed_at'] <= decision_seal['sealed_at']
    return dict(started_at=base.now(), model_seal_sha256=base.sha(directory/'model-seal.json'),
                decision_seal_sha256=base.sha(directory/'test-decision-seal.json'))


def prepare_context(trace_fn, anchor, cert, phase, p, directory):
    orbit = np.asarray(cert['orbit_states'])[:-1]
    initial = orbit[phase].copy()
    sens = local_parameter_sensitivity(anchor['alpha'], anchor['gamma'], initial, np.linspace(0, 300, 3001))
    selected = sens['times'] > 75
    gram = phase_projected_gram(sens['activation_sensitivity'][selected], sens['activation_velocity'][selected])
    stem = directory/'sensitivity'/f"{anchor['id']}-p{phase}"
    atomic_npz(stem.with_suffix('.npz'), **{k:v for k,v in sens.items() if isinstance(v, np.ndarray)},
               tail_gram=gram['gram'], tail_direction=gram['minimum_eigenvector'])
    base.write(stem.with_suffix('.json'), dict(wall_seconds=sens['wall_seconds'], nfev=sens['nfev'],
                                             solver=sens['solver'], tail_gram=gram))
    return dict(initial=initial, reference=np.asarray(cert['orbit_activations'])[:-1],
                T=cert['period'], epoch=phase*cert['period']/2700,
                gram=gram, directions=candidates(gram['minimum_eigenvector']), certificate=cert)


def preview_candidate(candidate, anchor, phase, radius, context, geometry, trace_fn, p, directory):
    fid = feature_id(anchor['id'], phase, radius, candidate)
    dest = directory/'previews'/f'{fid}.npz'
    row = dict(id=fid, role=anchor['role'], anchor=anchor['id'], block=anchor['block_id'],
               phase=phase, radius=radius, candidate=candidate, status='failed', preview_score=None,
               base=None, geometry=None, raw=str(dest.relative_to(directory)))
    started = time.perf_counter()
    try:
        if context is None:
            raise ValueError('Baseline or sensitivity context unavailable')
        direction = context['directions'][candidate]
        delta = radius*direction
        alpha, gamma = np.asarray([anchor['alpha'], anchor['gamma']])+delta
        row.update(direction=direction.tolist(), delta=delta.tolist(), alpha=alpha, gamma=gamma)
        geo = geometry_features(geometry, anchor, direction, radius)
        row.update(geometry=geo['values'], geometry_diagnostics=geo['diagnostics'])
        times, states = integrate_trace(trace_fn, alpha, gamma, context['initial'], p['decision_dt'],
            duration=75, sample_dt=.1, failure_path=dest.with_name(dest.stem+'-failure.npz'))
        atomic_npz(dest, times=times, states=states, initial=context['initial'], delta=delta,
                   alpha=alpha, gamma=gamma, dt=p['decision_dt'])
        metrics = score_probe(context['reference'], times, np.tanh(states), context['T'],
                              phase_epoch=context['epoch'], phase_grid=128)
        features = base_features(anchor, phase, direction, radius, context['certificate'], context['gram'], metrics)
        row.update(status='complete', base=features, preview_score=metrics['aligned_rms']['aggregate'], metrics=metrics)
    except Exception as error:
        row['error'] = repr(error)
    row['wall_seconds'] = time.perf_counter()-started
    return base.safe(row)


def collect_features(role, p, directory, fn, trace_fn, geometry, pool):
    rows, contexts, baselines = [], {}, {}
    with (directory/f'{role}-previews.jsonl').open('x') as journal:
        for anchor in anchors_for(p, role):
            try:
                result = baseline(fn, anchor, p, directory)
                if result['eligible'] and len(result['certificate']['orbit_states']) != 2701:
                    result.update(eligible=False, status='unexpected_orbit_length')
            except Exception as error:
                result = dict(anchor=anchor['id'], eligible=False, status='failed', error=repr(error))
            baselines[anchor['id']] = result
            base.write(directory/f'{role}-baselines.json', baselines)
            for phase in p['phases']:
                context = None
                if result['eligible']:
                    try:
                        context = prepare_context(trace_fn, anchor, result['certificate'], phase, p, directory)
                    except Exception as error:
                        stem = directory/'sensitivity'/f"{anchor['id']}-p{phase}-failure"
                        if hasattr(error, 'payload'):
                            atomic_npz(stem.with_suffix('.npz'), **error.payload)
                        base.write(stem.with_suffix('.json'), dict(status='failed', error=repr(error)))
                contexts[(anchor['id'], phase)] = context
                for radius in p['radii']:
                    batch = list(pool.map(lambda c: preview_candidate(c, anchor, phase, radius, context,
                                                                      geometry, trace_fn, p, directory), range(18)))
                    for row in batch:
                        rows.append(row); journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
            print(json.dumps(dict(stage=f'{role}_previews', anchor=anchor['id'], baseline_eligible=result['eligible'],
                                  completed_rows=len(rows))), flush=True)
    base.write(directory/f'{role}-features.json', rows)
    return rows, contexts


def target(spec, anchor, context, trace_fn, p, directory, binding):
    tid = trial_id(spec); dest = directory/'trials'/f'{tid}.npz'
    row = {**spec, **binding, 'id':tid, 'status':'failed', 'rmse':None, 'raw':str(dest.relative_to(directory))}
    started = time.perf_counter()
    try:
        if context is None:
            raise ValueError('Baseline or sensitivity context unavailable')
        direction = np.zeros(2) if spec['candidate'] == -1 else context['directions'][spec['candidate']]
        delta = spec['radius']*direction
        alpha, gamma = np.asarray([anchor['alpha'], anchor['gamma']])+delta
        row.update(direction=direction, delta=delta, alpha=alpha, gamma=gamma, phase_epoch=context['epoch'])
        times, states = integrate_trace(trace_fn, alpha, gamma, context['initial'], spec['dt'], duration=300,
                                        sample_dt=.1, failure_path=dest.with_name(dest.stem+'-failure.npz'))
        atomic_npz(dest, times=times, states=states, initial=context['initial'], delta=delta,
                   alpha=alpha, gamma=gamma, dt=spec['dt'])
        mask = times > 75
        metrics = waveform_errors(context['reference'], times[mask]+context['epoch'], np.tanh(states[mask]),
                                  context['T'], phase_grid=128)
        row.update(status='complete', tail_metrics=metrics, rmse=metrics['aligned_rms']['aggregate'])
    except Exception as error:
        row['error'] = repr(error)
    row['wall_seconds'] = time.perf_counter()-started
    return base.safe(row)


def collect_targets(role, p, directory, contexts, trace_fn, pool, binding):
    if role == 'test':
        start = authorize_test_targets(directory)
        base.write(directory/'test-target-started.json', start)
        binding = {**binding, 'test_decision_seal_sha256': start['decision_seal_sha256']}
    aa = {a['id']:a for a in anchors_for(p, role)}
    records = []
    with (directory/f'{role}-targets.jsonl').open('x') as journal:
        for anchor in anchors_for(p, role):
            cases = [s for s in specs(p, role) if s['anchor'] == anchor['id']]
            for row in pool.map(lambda s: target(s, aa[s['anchor']], contexts[(s['anchor'], s['phase'])],
                                                trace_fn, p, directory, binding), cases):
                records.append(row); journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
            print(json.dumps(dict(stage=f'{role}_targets', anchor=anchor['id'], completed_rows=len(records))), flush=True)
    return records


def summarize(records, decisions, p):
    test_rows = [r for r in records if r['role'] == 'test' and r['candidate'] >= 0]
    result = summarize_information_outcomes(test_rows, decisions,
        block_ids=[a['block_id'] for a in anchors_for(p, 'test')], phases=PHASES, radii=RADII, dts=p['dt'])
    result['all_target_status_counts'] = dict(Counter(r['status'] for r in records))
    expected = {trial_id(s) for role in ROLES for s in specs(p, role)}
    complete = (len(records) == len(expected) == 7008 and {r['id'] for r in records} == expected
                and all(r['status'] == 'complete' and valid_rmse(r.get('rmse')) for r in records))
    result['primary_data_gate'] = dict(passed=complete, required_targets=7008,
        rule='Every development/test candidate and zero-sham record at both steps must complete; no success-only primary')
    if not complete:
        result['primary'] = dict(status='undefined_incomplete', reason='A required target or numerical sham is incomplete')
    result['zero'] = {}
    for role in ROLES:
        for dt in p['dt']:
            rr = [r for r in records if r['role'] == role and r['candidate'] == -1 and r['dt'] == dt]
            values = [r['rmse'] for r in rr if r['status'] == 'complete']
            result['zero'][f'{role}-{dt}'] = dict(planned=24, complete=len(values),
                mean=float(np.mean(values)) if len(values) == 24 else None,
                maximum=max(values) if len(values) == 24 else None)
    return base.safe(result)


def run(directory):
    p = json.loads(PLAN.read_text()); validate(p, check_sources=True)
    for name, version in p['required_runtime_packages'].items():
        assert importlib.metadata.version(name) == version, f'Runtime package version differs: {name}'
    assert not base.git('status', '--porcelain'), 'Source must be clean'
    source = base.git('rev-parse', 'HEAD')
    assert base.git('branch', '-r', '--contains', source), 'Push the complete source before outcomes'
    directory.mkdir(parents=True, exist_ok=False); shutil.copyfile(PLAN, directory/'manifest.json')
    started = dict(experiment_id='EXP-008', source_commit=source, manifest_sha256=base.sha(PLAN),
                   started_at=base.now(), status='running', planned_targets=7008, planned_previews=3456,
                   platform=platform.platform(), python=sys.version,
                   versions={name:importlib.metadata.version(name) for name in ['numpy','scipy']},
                   paid_compute=0, paid_review=0)
    base.write(directory/'started.json', started)
    clock_start = time.monotonic()
    def stop():
        base.write(directory/'timeout.json', dict(status='hard_timeout', ended_at=base.now(), source_commit=source))
        os._exit(124)
    watchdog = threading.Timer(3600, stop); watchdog.daemon=True; watchdog.start()
    try:
        for name in ['trials', 'previews', 'sensitivity']:
            (directory/name).mkdir()
        fn, info = base.build_engine(directory/'build'/'long'); base.write(directory/'long-build.json', info)
        trace_fn, info = build_trace(directory/'build'/'trace'); base.write(directory/'trace-build.json', info)
        grid = base.load_grid(ROOT/'evidence/EXP-004/attempt-001/focus')
        geometry = build_atlas_geometry(grid, p['selection']['core_index'])
        binding = {k:started[k] for k in ['source_commit', 'manifest_sha256']}
        with ThreadPoolExecutor(max_workers=4) as pool:
            dev_features, dev_contexts = collect_features('development', p, directory, fn, trace_fn, geometry, pool)
            dev_records = collect_targets('development', p, directory, dev_contexts, trace_fn, pool, binding)
            fit_start = time.perf_counter()
            packet = fit_development(dev_features, dev_records, p)
            base.write(directory/'model.json', packet)
            base.write(directory/'model-seal.json', dict(sealed_at=base.now(), model_sha256=base.sha(directory/'model.json'),
                development_targets_sha256=base.sha(directory/'development-targets.jsonl'),
                development_features_sha256=base.sha(directory/'development-features.json'),
                fit_seconds=time.perf_counter()-fit_start, source_commit=source))
            print(json.dumps(dict(stage='model_sealed', status=packet['status'])), flush=True)
            del dev_contexts
            test_features, test_contexts = collect_features('test', p, directory, fn, trace_fn, geometry, pool)
            packet = json.loads((directory/'model.json').read_text())
            decisions = decide_test(test_features, packet, p)
            base.write(directory/'test-decisions.json', decisions)
            base.write(directory/'test-decision-seal.json', dict(sealed_at=base.now(),
                model_seal_sha256=base.sha(directory/'model-seal.json'),
                features_sha256=base.sha(directory/'test-features.json'),
                decisions_sha256=base.sha(directory/'test-decisions.json')))
            test_records = collect_targets('test', p, directory, test_contexts, trace_fn, pool, binding)
        records = dev_records + test_records
        base.write(directory/'summary.json', summarize(records, decisions, p))
        artifacts = [f for f in sorted(directory.rglob('*')) if f.is_file() and 'build' not in f.relative_to(directory).parts]
        base.write(directory/'receipt.json', {**started, 'status':'complete', 'ended_at':base.now(),
            'wall_seconds':time.monotonic()-clock_start, 'artifacts':[dict(path=str(f.relative_to(directory)),
            sha256=base.sha(f), bytes=f.stat().st_size) for f in artifacts]})
        verify(directory)
    except BaseException as error:
        if hasattr(error, 'payload'):
            atomic_npz(directory/'unhandled-numerical-failure.npz', **error.payload)
        base.write(directory/'failure.json', dict(error=repr(error), ended_at=base.now(), source_commit=source,
                                                 wall_seconds=time.monotonic()-clock_start))
        raise
    finally:
        watchdog.cancel()


def compare_model_replay(saved, replayed, path='model'):
    """Prospective arithmetic tolerance for conditioned ridge solves.

    Raw model/source/input hashes stay exact. Standardized correlated columns
    and ridge lambda down to 1e-6 can amplify BLAS rounding. This tolerance
    applies only to numerical fit replay, never to measured waveform scores,
    cohort/feature identity, selected lambda index, or the saved action ID.
    """
    if isinstance(replayed, dict):
        assert isinstance(saved, dict) and set(saved) == set(replayed), path
        for key in replayed:
            compare_model_replay(saved[key], replayed[key], path+'.'+key)
    elif isinstance(replayed, list):
        assert isinstance(saved, list) and len(saved) == len(replayed), path
        for index, (a, b) in enumerate(zip(saved, replayed)):
            compare_model_replay(a, b, f'{path}[{index}]')
    elif isinstance(replayed, float):
        assert isinstance(saved, (int, float)) and not isinstance(saved, bool), path
        np.testing.assert_allclose(saved, replayed, rtol=1e-7, atol=1e-9, err_msg=path)
    else:
        assert type(saved) is type(replayed) and saved == replayed, path


def verify(directory):
    """Exact raw hashes, realized inputs, full phase metrics, fits and decisions."""
    p = json.loads((directory/'manifest.json').read_text()); validate(p)
    receipt = json.loads((directory/'receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['manifest_sha256'] == base.sha(directory/'manifest.json')
    assert not (directory/'failure.json').exists() and not (directory/'timeout.json').exists()
    paths = [item['path'] for item in receipt['artifacts']]; assert len(paths) == len(set(paths))
    for item in receipt['artifacts']:
        assert base.sha(directory/item['path']) == item['sha256'], item['path']
        assert (directory/item['path']).stat().st_size == item['bytes']
    for item in p['source_artifacts'] + p['source_files']:
        assert base.sha(ROOT/item['path']) == item['sha256'], item['path']
    geometry = build_atlas_geometry(base.load_grid(ROOT/'evidence/EXP-004/attempt-001/focus'), p['selection']['core_index'])
    aa = {a['id']:a for a in p['anchors']}
    all_records, features = [], {}
    for role in ROLES:
        rows = json.loads((directory/f'{role}-features.json').read_text()); features[role] = rows
        assert [r['id'] for r in rows] == expected_features(p, role)
        logged = [json.loads(line) for line in (directory/f'{role}-previews.jsonl').read_text().splitlines()]
        assert logged == rows
        baselines = json.loads((directory/f'{role}-baselines.json').read_text())
        assert set(baselines) == {a['id'] for a in anchors_for(p, role)}
        contexts = {}
        for aid, item in baselines.items():
            if not item['eligible']:
                continue
            assert all(s['period'] == 9 for s in item['steps'])
            cert = item['certificate']; assert cert['verified'] and cert['attracting']
            reference = np.asarray(cert['orbit_activations'])[:-1]
            orbit = np.asarray(cert['orbit_states'])[:-1]
            for phase in PHASES:
                path = directory/'sensitivity'/f'{aid}-p{phase}.npz'
                if not path.exists():
                    assert (directory/'sensitivity'/f'{aid}-p{phase}-failure.json').exists(); continue
                with np.load(path, allow_pickle=False) as raw:
                    assert np.array_equal(raw['states'][0], orbit[phase])
                    tail = raw['times'] > 75
                    gram = phase_projected_gram(raw['activation_sensitivity'][tail], raw['activation_velocity'][tail])
                    np.testing.assert_allclose(raw['tail_direction'], gram['minimum_eigenvector'], rtol=1e-12, atol=1e-14)
                    ds = candidates(raw['tail_direction'])
                contexts[(aid, phase)] = dict(initial=orbit[phase], reference=reference, T=cert['period'],
                    epoch=phase*cert['period']/2700, gram=gram, directions=ds, certificate=cert)
        for row in rows:
            if row['status'] != 'complete':
                continue
            context = contexts[(row['anchor'], row['phase'])]; a = aa[row['anchor']]
            direction = context['directions'][row['candidate']]
            np.testing.assert_array_equal(row['direction'], direction)
            geo = geometry_features(geometry, a, direction, row['radius'])
            assert_portable_equal(row['geometry'], geo['values'], 'geometry')
            with np.load(directory/row['raw'], allow_pickle=False) as raw:
                check_trace(raw, context['initial'], a, row['radius']*direction, .005, 75)
                verify_wave_metrics(row['metrics'], context['reference'], raw['times']+context['epoch'],
                                    np.tanh(raw['states']), context['T'], 128)
            values = base_features(a, row['phase'], direction, row['radius'], context['certificate'], context['gram'], row['metrics'])
            np.testing.assert_allclose(row['base'], values, rtol=1e-12, atol=1e-14)
            assert row['preview_score'] == row['metrics']['aligned_rms']['aggregate']
        records = [json.loads(line) for line in (directory/f'{role}-targets.jsonl').read_text().splitlines()]
        expected = {trial_id(s):s for s in specs(p, role)}
        assert len(records) == len(expected) and {r['id'] for r in records} == set(expected)
        for row in records:
            s = expected[row['id']]
            assert all(row[k] == v for k, v in s.items())
            assert row['source_commit'] == receipt['source_commit'] and row['manifest_sha256'] == receipt['manifest_sha256']
            if row['status'] != 'complete':
                continue
            context = contexts[(row['anchor'], row['phase'])]; a = aa[row['anchor']]
            direction = np.zeros(2) if row['candidate'] == -1 else context['directions'][row['candidate']]
            np.testing.assert_array_equal(row['direction'], direction)
            np.testing.assert_array_equal(row['delta'], row['radius']*direction)
            assert row['phase_epoch'] == context['epoch']
            with np.load(directory/row['raw'], allow_pickle=False) as raw:
                check_trace(raw, context['initial'], a, row['radius']*direction, row['dt'], 300)
                mask = raw['times'] > 75
                verify_wave_metrics(row['tail_metrics'], context['reference'], raw['times'][mask]+context['epoch'],
                                    np.tanh(raw['states'][mask]), context['T'], 128)
            assert row['rmse'] == row['tail_metrics']['aligned_rms']['aggregate']
        all_records.extend(records)
    packet = json.loads((directory/'model.json').read_text())
    assert packet['packet_sha256'] == packet_digest(packet)
    repeated = fit_development(features['development'], [r for r in all_records if r['role']=='development'], p)
    # Hashes bind saved fits exactly; portable arithmetic checks compare replayed fits excluding their digest.
    compare_model_replay({k:v for k,v in packet.items() if k!='packet_sha256'},
                         {k:v for k,v in repeated.items() if k!='packet_sha256'}, 'development fit')
    decisions = json.loads((directory/'test-decisions.json').read_text())
    assert_portable_equal(decisions, decide_test(features['test'], packet, p), 'test decisions')
    seal = json.loads((directory/'model-seal.json').read_text())
    dseal = json.loads((directory/'test-decision-seal.json').read_text())
    begin = json.loads((directory/'test-target-started.json').read_text())
    assert seal['model_sha256'] == base.sha(directory/'model.json')
    assert seal['development_targets_sha256'] == base.sha(directory/'development-targets.jsonl')
    assert seal['development_features_sha256'] == base.sha(directory/'development-features.json')
    assert dseal['model_seal_sha256'] == base.sha(directory/'model-seal.json')
    assert dseal['features_sha256'] == base.sha(directory/'test-features.json')
    assert dseal['decisions_sha256'] == base.sha(directory/'test-decisions.json')
    assert begin['model_seal_sha256'] == base.sha(directory/'model-seal.json')
    assert begin['decision_seal_sha256'] == base.sha(directory/'test-decision-seal.json')
    assert receipt['started_at'] <= seal['sealed_at'] <= dseal['sealed_at'] <= begin['started_at'] <= receipt['ended_at']
    assert all(r['test_decision_seal_sha256'] == begin['decision_seal_sha256'] for r in all_records if r['role']=='test')
    assert_portable_equal(json.loads((directory/'summary.json').read_text()), summarize(all_records, decisions, p), 'summary')
    print(json.dumps(dict(verified=True, targets=len(all_records), previews=sum(map(len, features.values())),
                          source=receipt['source_commit'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'run', 'verify']); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'validate':
        print(json.dumps(dict(valid=True, targets=validate(json.loads(PLAN.read_text()), check_sources=True))))
    elif args.output is None:
        parser.error('--output is required')
    elif args.action == 'run':
        run(args.output)
    else:
        verify(args.output)

if __name__ == '__main__':
    main()
