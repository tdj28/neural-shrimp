#!/usr/bin/env python3
"""Post-run EXP-008 regeneration from immutable trajectories, without integration.

The source-bound original numerical routines are deliberately reused. Unlike
the earlier saved-feature diagnostic, fitting consumes newly evaluated preview
metrics/features and development labels. This is a diagnostic on observed data,
not a successor experiment or a replacement for the frozen verifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np

import run_hopfield_information as driver
from hopfield_diagnostics import hopfield_rhs
from hopfield_drift_metrics import periodic_interpolate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '3c5604497ec57b19d62fbb595da7b190102f4b33'
MANIFEST = 'fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd'
RECEIPT = 'ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb'
LEARNERS = ('baseline', 'geometry', 'shuffled_geometry')
POLICIES = (*LEARNERS, 'min_preview')
ARITHMETIC = dict(rtol=1e-12, atol=1e-14)
MODEL_ARITHMETIC = dict(rtol=1e-7, atol=1e-9)


class AuditInterrupted(BaseException):
    """Bypass per-record exception handlers to retain a stopped audit's state."""


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def execution_provenance(expected_head):
    """Require a named, clean, already pushed audit source before execution.

    Remote-tracking containment is a local provenance check. The coordinating
    run plan must additionally verify the real remote ref with ls-remote.
    """
    if (not isinstance(expected_head, str) or len(expected_head) != 40
            or any(char not in '0123456789abcdef' for char in expected_head)):
        raise ValueError('An explicit full lowercase --expected-head SHA is required')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    actual = git('rev-parse', 'HEAD')
    if actual != expected_head:
        raise ValueError(f'Expected audit HEAD {expected_head}; actual HEAD {actual}')
    if git('status', '--porcelain'):
        raise ValueError('The audit source checkout must be clean before execution')
    refs = git('for-each-ref', '--contains='+expected_head, '--format=%(refname)', 'refs/remotes/').splitlines()
    if not refs:
        raise ValueError('Expected audit source has no remote-tracking containment; push before execution')
    return dict(audit_source_commit=actual, audit_source_tree=git('rev-parse', 'HEAD^{tree}'),
        containing_remote_tracking_refs=refs, clean_checkout=True,
        live_remote_check='The coordinating run record must retain a separate successful ls-remote check')


class Recorder:
    """Count every comparison and retain every failure, without first-error exit."""

    def __init__(self):
        self.failures = []
        self.counts = defaultdict(lambda: dict(exact=0, numeric=0,
            max_absolute_difference=0., failures=0))

    def fail(self, stage, path, kind, **details):
        self.failures.append(dict(stage=stage, path=path, kind=kind, **safe(details)))
        self.counts[stage]['failures'] += 1

    def check(self, condition, stage, path, **details):
        self.counts[stage]['exact'] += 1
        if not condition:
            self.fail(stage, path, 'identity', **details)
        return bool(condition)

    def compare(self, saved, regenerated, stage, path, *, rtol=1e-12, atol=1e-14):
        """Original asymmetric allclose rule; missing keys/items remain failures."""
        # Preserve numeric types until comparison. Serialization turns NaN/inf
        # into readable strings, but identical nonfinite inputs must not pass
        # by comparing those strings as exact identities.
        def native(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return list(value) if isinstance(value, tuple) else value
        left, right = native(saved), native(regenerated)
        if isinstance(right, dict):
            if not self.check(isinstance(left, dict), stage, path, expected='object'):
                return
            self.check(set(left) == set(right), stage, path + '.keys',
                saved=sorted(left), regenerated=sorted(right))
            for key in sorted(set(left) & set(right)):
                self.compare(left[key], right[key], stage, f'{path}.{key}', rtol=rtol, atol=atol)
        elif isinstance(right, list):
            if not self.check(isinstance(left, list), stage, path, expected='array'):
                return
            self.check(len(left) == len(right), stage, path + '.length',
                saved=len(left), regenerated=len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                self.compare(a, b, stage, f'{path}[{index}]', rtol=rtol, atol=atol)
        elif isinstance(right, float):
            self.counts[stage]['numeric'] += 1
            if (not isinstance(left, (int, float)) or isinstance(left, bool)
                    or not np.isfinite([left, right]).all()):
                self.fail(stage, path, 'nonfinite_or_type', saved=left, regenerated=right)
                return
            difference = abs(left - right)
            self.counts[stage]['max_absolute_difference'] = max(
                self.counts[stage]['max_absolute_difference'], difference)
            bound = atol + rtol * abs(right)
            if difference > bound:
                self.fail(stage, path, 'arithmetic', saved=left, regenerated=right,
                    absolute_difference=difference, allowed_difference=bound, rtol=rtol, atol=atol)
        else:
            self.check(type(left) is type(right) and left == right, stage, path,
                saved=left, regenerated=right)

    def exception(self, stage, path, error):
        self.fail(stage, path, 'unavailable', error=f'{type(error).__name__}: {error}')


def bounded_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe artifact path: {relative}')
    destination = root / path
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact escapes its root: {relative}')
    return destination


def bindings(attempt, log):
    """Enumerate byte/source binding failures before any scientific regeneration."""
    start = len(log.failures)
    try:
        log.check(sha(attempt/'receipt.json') == RECEIPT, 'bindings', 'receipt.sha256')
        receipt = read(attempt/'receipt.json')
        p = read(attempt/'manifest.json')
    except Exception as error:
        log.exception('bindings', 'receipt_or_manifest', error)
        return None
    log.check(receipt.get('status') == 'complete', 'bindings', 'receipt.status')
    log.check(receipt.get('source_commit') == SOURCE, 'bindings', 'source_commit')
    log.check(receipt.get('manifest_sha256') == MANIFEST == sha(attempt/'manifest.json'),
        'bindings', 'manifest.sha256')
    for marker in ('failure.json', 'timeout.json'):
        log.check(not (attempt/marker).exists(), 'bindings', marker)
    try:
        frozen = subprocess.check_output(['git', 'show',
            SOURCE + ':experiments/hopfield/EXP-008-plan.json'], cwd=ROOT)
        log.check(frozen == (attempt/'manifest.json').read_bytes(), 'bindings', 'git_manifest')
        driver.validate(p)
    except Exception as error:
        log.exception('bindings', 'source_plan', error)
    checked_bytes = 0
    artifacts = receipt.get('artifacts', [])
    paths = [entry.get('path') for entry in artifacts]
    log.check(len(artifacts) == 10650, 'bindings', 'artifact_count', observed=len(artifacts))
    log.check(len(paths) == len(set(paths)), 'bindings', 'duplicate_artifacts')
    for entry in artifacts:
        name = entry.get('path', '<missing>')
        try:
            path = bounded_path(attempt, name)
            size = path.stat().st_size
            log.check(size == entry['bytes'], 'bindings', name+'.bytes', observed=size)
            log.check(sha(path) == entry['sha256'], 'bindings', name+'.sha256')
            checked_bytes += size
        except Exception as error:
            log.exception('bindings', str(name), error)
    actual = {str(f.relative_to(attempt)) for f in attempt.rglob('*')
        if f.is_file() and 'build' not in f.relative_to(attempt).parts
        and f != attempt/'receipt.json'}
    log.check(actual == set(paths), 'bindings', 'artifact_census',
        missing=sorted(set(paths)-actual), extra=sorted(actual-set(paths)))
    for entry in p.get('source_artifacts', []) + p.get('source_files', []):
        name = entry.get('path', '<missing>')
        try:
            log.check(sha(bounded_path(ROOT, name)) == entry['sha256'],
                'bindings', 'source.'+name)
        except Exception as error:
            log.exception('bindings', 'source.'+str(name), error)
    if len(log.failures) != start:
        return None
    return p, receipt, dict(artifacts=len(artifacts), bytes=checked_bytes,
        old_inputs=len(p['source_artifacts']), source_files=len(p['source_files']))


def census(rows, expected, log, name, *, ordered=False):
    """Return unique expected rows only after recording every omission/duplicate."""
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('id'), str):
            log.fail('census', f'{name}[{index}]', 'malformed_row')
            continue
        grouped[row['id']].append(row)
    for key in sorted(set(grouped)-set(expected)):
        log.fail('census', name+'.'+key, 'unexpected_id')
    for key in expected:
        if len(grouped.get(key, [])) != 1:
            log.fail('census', name+'.'+key, 'missing_or_duplicate', count=len(grouped.get(key, [])))
    if ordered:
        log.check([r.get('id') for r in rows if isinstance(r, dict)] == list(expected),
            'census', name+'.order')
    return {key: grouped[key][0] for key in expected if len(grouped.get(key, [])) == 1}


def identity(row, expected, log, name):
    for key, value in expected.items():
        log.check(type(row.get(key)) is type(value) and row.get(key) == value,
            'identity', name+'.'+key, saved=row.get(key), expected=value)


def load_inputs(attempt, p, receipt, log):
    result = dict(features={}, records={}, baselines={})
    for role in driver.ROLES:
        rows = read(attempt/f'{role}-features.json')
        journal = [json.loads(s) for s in (attempt/f'{role}-previews.jsonl').read_text().splitlines()]
        log.check(rows == journal, 'identity', role+'.preview_journal')
        expected = driver.expected_features(p, role)
        result['features'][role] = census(rows, expected, log, role+'.features', ordered=True)
        for a in driver.anchors_for(p, role):
            for phase in p['phases']:
                for radius in p['radii']:
                    for candidate in range(p['candidate_count']):
                        key = driver.feature_id(a['id'], phase, radius, candidate)
                        if key in result['features'][role]:
                            identity(result['features'][role][key], dict(id=key, role=role,
                                anchor=a['id'], block=a['block_id'], phase=phase, radius=radius,
                                candidate=candidate, status='complete', raw=f'previews/{key}.npz'), log, key)
        records = [json.loads(s) for s in (attempt/f'{role}-targets.jsonl').read_text().splitlines()]
        specs = {driver.trial_id(s): s for s in driver.specs(p, role)}
        result['records'][role] = census(records, list(specs), log, role+'.targets')
        for key, row in result['records'][role].items():
            identity(row, dict(**specs[key], id=key, source_commit=SOURCE,
                manifest_sha256=MANIFEST, status='complete', raw=f'trials/{key}.npz'), log, key)
        result['baselines'][role] = read(attempt/f'{role}-baselines.json')
        log.check(set(result['baselines'][role]) == {a['id'] for a in driver.anchors_for(p, role)},
            'census', role+'.baselines')
    result['packet'] = read(attempt/'model.json')
    packet = result['packet']
    log.check(packet.get('status') == 'complete' and packet.get('packet_sha256') == driver.packet_digest(packet),
        'bindings', 'model_packet_digest')
    seal, decision_seal, started = [read(attempt/name) for name in
        ('model-seal.json', 'test-decision-seal.json', 'test-target-started.json')]
    for item, key, file in ((seal, 'model_sha256', 'model.json'),
        (seal, 'development_targets_sha256', 'development-targets.jsonl'),
        (seal, 'development_features_sha256', 'development-features.json'),
        (decision_seal, 'model_seal_sha256', 'model-seal.json'),
        (decision_seal, 'features_sha256', 'test-features.json'),
        (decision_seal, 'decisions_sha256', 'test-decisions.json'),
        (started, 'model_seal_sha256', 'model-seal.json'),
        (started, 'decision_seal_sha256', 'test-decision-seal.json')):
        log.check(item.get(key) == sha(attempt/file), 'bindings', 'seals.'+file+'.'+key)
    log.check(seal.get('source_commit') == SOURCE, 'bindings', 'model_seal.source')
    log.check(receipt['started_at'] <= seal['sealed_at'] <= decision_seal['sealed_at']
        <= started['started_at'] <= receipt['ended_at'], 'bindings', 'chronology')
    for key, row in result['records']['test'].items():
        log.check(row.get('test_decision_seal_sha256') == started['decision_seal_sha256'],
            'bindings', key+'.decision_seal')
    result['decisions'] = read(attempt/'test-decisions.json')
    expected_decisions = [(a['id'], a['block_id'], ph, radius)
        for a in driver.anchors_for(p, 'test') for ph in p['phases'] for radius in p['radii']]
    actual_decisions = [tuple(d.get(k) for k in ('anchor', 'block', 'phase', 'radius'))
        for d in result['decisions']]
    log.check(actual_decisions == expected_decisions, 'census', 'decisions.order')
    for i, decision in enumerate(result['decisions']):
        identity(decision, dict(status='complete', candidate_ids=list(range(18)),
            required_preview_calls={key:18 for key in POLICIES},
            model_packet_sha256=packet['packet_sha256']), log, f'decisions[{i}]')
    return result


def check_array(actual, expected, log, path):
    return log.check(np.array_equal(actual, expected), 'raw_identity', path,
        saved_shape=list(np.shape(actual)), expected_shape=list(np.shape(expected)))


def build_context(raw, meta, anchor, certificate, phase, log, name):
    """Regenerate variational readouts and Grams, retaining realized directions."""
    orbit = np.asarray(certificate['orbit_states'], dtype=float)
    reference = np.asarray(certificate['orbit_activations'], dtype=float)
    if (orbit.shape != (2701, 3) or reference.shape != orbit.shape
            or not np.isfinite(orbit).all() or not np.isfinite(reference).all()):
        raise ValueError('Incomplete/nonfinite endpoint-included reference orbit')
    if (not np.isfinite(certificate['period']) or certificate['period'] <= 0
            or not 0 < certificate['transverse_spectral_radius'] < 1):
        raise ValueError('Required positive period and attracting transverse radius are invalid')
    log.compare(reference, np.tanh(orbit), 'sensitivity', name+'.reference_activation', **ARITHMETIC)
    initial = orbit[phase]
    times, states, state_sens = (np.asarray(raw[key]) for key in ('times', 'states', 'state_sensitivity'))
    initial_failures = len(log.failures)
    check_array(times, np.linspace(0, 300, 3001), log, name+'.times')
    check_array(states[0], initial, log, name+'.initial')
    check_array(state_sens[0], np.zeros((3, 2)), log, name+'.zero_initial_sensitivity')
    if (states.shape != (3001, 3) or state_sens.shape != (3001, 3, 2)
            or not np.isfinite(states).all() or not np.isfinite(state_sens).all()
            or len(log.failures) != initial_failures):
        raise ValueError('Sensitivity raw identity, shape, or finite-value gate failed')
    derivative = 1 - np.tanh(states)**2
    sensitivity = derivative[:, :, None] * state_sens
    velocity = derivative * np.asarray([hopfield_rhs(t, x, anchor['alpha'], anchor['gamma'])
        for t, x in zip(times, states)])
    log.compare(raw['activation_sensitivity'], sensitivity, 'sensitivity', name+'.activation_sensitivity')
    log.compare(raw['activation_velocity'], velocity, 'sensitivity', name+'.activation_velocity')
    tail = times > 75
    gram = driver.phase_projected_gram(sensitivity[tail], velocity[tail])
    log.compare(raw['tail_gram'], gram['gram'], 'sensitivity', name+'.tail_gram')
    log.compare(raw['tail_direction'], gram['minimum_eigenvector'], 'sensitivity', name+'.tail_direction')
    log.compare(meta['tail_gram'], gram, 'sensitivity', name+'.tail_gram_metadata')
    # Frozen verify() constructs its candidates from the saved direction. A
    # fresh eigenvector does not authorize relabeling an already integrated
    # trajectory with a different delta. Both direction sets remain visible.
    realized = driver.candidates(raw['tail_direction'])
    regenerated = driver.candidates(gram['minimum_eigenvector'])
    context = dict(initial=initial, reference=reference[:-1], T=certificate['period'],
        epoch=phase*certificate['period']/2700, certificate=certificate,
        gram=gram, directions=realized)
    diagnostic = dict(id=name, gram=safe(gram),
        saved_local_direction=safe(raw['tail_direction']),
        regenerated_local_direction=safe(gram['minimum_eigenvector']),
        regenerated_directions=safe(regenerated), realized_directions=safe(realized),
        directions_bitwise_equal=bool(np.array_equal(realized, regenerated)),
        direction_max_absolute_difference=float(np.max(np.abs(realized-regenerated))))
    return context, diagnostic


def trace_inputs(raw, context, anchor, direction, radius, dt, duration, log, name):
    before = len(log.failures)
    delta = radius * direction
    for key, expected in (('initial', context['initial']), ('delta', delta),
        ('dt', dt), ('times', np.r_[np.arange(round(duration/.1))*.1, duration])):
        check_array(raw[key], expected, log, name+'.'+key)
    check_array([raw['alpha'], raw['gamma']],
        np.array([anchor['alpha'], anchor['gamma']])+delta, log, name+'.parameters')
    states, times = np.asarray(raw['states']), np.asarray(raw['times'])
    shape_ok = states.shape == (round(duration/.1)+1, 3)
    log.check(shape_ok and np.isfinite(states).all(), 'raw_identity', name+'.states')
    if shape_ok:
        check_array(states[0], context['initial'], log, name+'.first_state')
    if len(log.failures) != before:
        raise ValueError('Saved trajectory does not match its declared realized intervention')
    return times, np.tanh(states)


def _rms(errors):
    return dict(aggregate=min(2., float(np.sqrt(np.mean(errors**2)))),
        per_component=np.minimum(2., np.sqrt(np.mean(errors**2, axis=0))).tolist())


def regenerate_metrics(saved, reference, times, values, period, log, name, *, phase_grid=128):
    """Fresh optimization for refitting; separate unchanged frozen-phase check.

    Frozen verify_wave_metrics explicitly permits equivalent phase minimizers.
    It re-evaluates saved RMS at the saved phase with the original strict rule.
    We reproduce that rule while retaining the *fresh* phase for regeneration.
    Optimizer evaluation counts are recorded, not compared as exact identities.
    """
    fresh = driver.waveform_errors(reference, times, values, period, phase_grid=phase_grid)
    phase = saved['phase_offset_cycles']
    if (not isinstance(phase, (int, float)) or isinstance(phase, bool)
            or not np.isfinite(phase) or not -.5 <= phase < .5):
        raise ValueError('Saved phase is invalid')
    errors = values - periodic_interpolate(reference, times/period+phase)
    fixed = copy.deepcopy(fresh)
    fixed['phase_offset_cycles'] = phase
    fixed['aligned_rms'] = _rms(errors)
    for window in fixed['windows']:
        selected = (times >= window['first_sample_time']) & (times <= window['last_sample_time'])
        log.check(int(selected.sum()) == window['sample_count'], 'metrics', name+'.window_census')
        window['aligned_rms'] = _rms(errors[selected])
    phase_allowance = 4*(np.sqrt(2.2e-16)*(1+1/phase_grid)+1e-10/3)
    slopes = np.diff(np.vstack((reference, reference[:1])), axis=0)*len(reference)
    slope = float(np.sqrt(np.mean(slopes**2, axis=1)).max())
    allowance = slope*phase_allowance + 1e-14 + 1e-12*abs(fresh['aligned_rms']['aggregate'])
    discrepancy = abs(fixed['aligned_rms']['aggregate']-fresh['aligned_rms']['aggregate'])
    log.check(discrepancy <= allowance, 'metrics', name+'.optimizer_bound',
        difference=discrepancy, allowed_difference=allowance)
    old = copy.deepcopy(saved)
    evaluations = {}
    for label, metrics in (('saved', old), ('fresh', fixed)):
        count = metrics['phase_search'].pop('refinement_evaluations')
        log.check(type(count) is int and count > 0, 'metrics', name+'.'+label+'_optimizer_evaluations')
        evaluations[label] = count
    log.compare(old, fixed, 'metrics', name, **ARITHMETIC)
    diagnostic = dict(id=name, saved_phase=phase, regenerated_phase=fresh['phase_offset_cycles'],
        saved_rmse=saved['aligned_rms']['aggregate'], saved_phase_recomputed_rmse=fixed['aligned_rms']['aggregate'],
        regenerated_rmse=fresh['aligned_rms']['aggregate'], optimizer_allowance=allowance,
        optimizer_difference=discrepancy, optimizer_evaluations=evaluations,
        regenerated_metrics_sha256=digest(fresh))
    return fresh, diagnostic


def regenerate_preview(row, raw, anchor, context, geometry, log):
    key = row['id']
    direction = context['directions'][row['candidate']]
    identity(row, dict(direction=direction.tolist(), delta=(row['radius']*direction).tolist(),
        alpha=float(anchor['alpha']+row['radius']*direction[0]),
        gamma=float(anchor['gamma']+row['radius']*direction[1])), log, key)
    times, values = trace_inputs(raw, context, anchor, direction, row['radius'], .005, 75, log, key)
    metrics, diagnostic = regenerate_metrics(row['metrics'], context['reference'], times+context['epoch'],
        values, context['T'], log, key+'.metrics')
    geo = driver.geometry_features(geometry, anchor, direction, row['radius'])
    features = driver.base_features(anchor, row['phase'], direction, row['radius'],
        context['certificate'], context['gram'], metrics)
    log.compare(row['geometry'], geo['values'], 'geometry_features', key+'.values')
    log.compare(row['geometry_diagnostics'], geo['diagnostics'], 'geometry_features', key+'.diagnostics')
    log.compare(row['base'], features, 'regenerated_base_features', key+'.base')
    log.check(row['preview_score'] == row['metrics']['aligned_rms']['aggregate'],
        'identity', key+'.saved_score_identity')
    result = copy.deepcopy(row)
    result.update(base=safe(features), geometry=safe(geo['values']), geometry_diagnostics=geo['diagnostics'],
        metrics=metrics, preview_score=metrics['aligned_rms']['aggregate'])
    return result, diagnostic


def regenerate_target(row, raw, anchor, context, log):
    key = row['id']
    direction = np.zeros(2) if row['candidate'] == -1 else context['directions'][row['candidate']]
    identity(row, dict(direction=direction.tolist(), delta=(row['radius']*direction).tolist(),
        phase_epoch=context['epoch'], alpha=float(anchor['alpha']+row['radius']*direction[0]),
        gamma=float(anchor['gamma']+row['radius']*direction[1])), log, key)
    times, values = trace_inputs(raw, context, anchor, direction, row['radius'], row['dt'], 300, log, key)
    mask = times > 75
    log.check(int(mask.sum()) == 2250, 'raw_identity', key+'.tail_samples')
    metrics, diagnostic = regenerate_metrics(row['tail_metrics'], context['reference'],
        times[mask]+context['epoch'], values[mask], context['T'], log, key+'.metrics')
    log.check(row['rmse'] == row['tail_metrics']['aligned_rms']['aggregate'],
        'identity', key+'.saved_score_identity')
    result = {k:row[k] for k in ('id', 'role', 'anchor', 'block', 'phase', 'radius', 'candidate', 'dt')}
    result.update(status='complete', rmse=metrics['aligned_rms']['aggregate'])
    return result, diagnostic


def decision_effects(sealed, fresh, log, name):
    """All action/score differences; changed data digests are descriptive."""
    keys = ('anchor', 'block', 'phase', 'radius')
    old_ids = [tuple(d[k] for k in keys) for d in sealed]
    new_ids = [tuple(d[k] for k in keys) for d in fresh]
    log.check(old_ids == new_ids, 'decisions', name+'.census')
    if old_ids != new_ids:
        return dict(status='unavailable_census', conditions=[])
    conditions, changes = [], []
    for index, (a, b) in enumerate(zip(sealed, fresh)):
        item = {key:a[key] for key in keys}
        item['policies'] = {}
        for policy in POLICIES:
            old_choice, new_choice = a['policies'][policy], b['policies'][policy]
            scores_a = a['preview_scores'] if policy == 'min_preview' else a['predictions'][policy]
            scores_b = b['preview_scores'] if policy == 'min_preview' else b['predictions'][policy]
            old_scores, new_scores = np.asarray(scores_a), np.asarray(scores_b)
            if (old_scores.shape != (18,) or new_scores.shape != (18,)
                    or not np.isfinite(old_scores).all() or not np.isfinite(new_scores).all()):
                raise ValueError(f'Incomplete/nonfinite {name} predictions at {index}/{policy}')
            for label, scores, choice in (('sealed', old_scores, old_choice), ('regenerated', new_scores, new_choice)):
                log.check(choice['status'] == 'complete' and int(np.argmin(scores)) == choice['candidate'],
                    'decisions', f'{name}[{index}].{policy}.{label}_argmin')
                log.compare(choice['selection_score'], float(scores[choice['candidate']]),
                    'decisions', f'{name}[{index}].{policy}.{label}_selected_score')
            changed = old_choice['candidate'] != new_choice['candidate']
            if changed:
                changes.append(dict(condition=index, policy=policy,
                    sealed=old_choice['candidate'], regenerated=new_choice['candidate']))
            error = float(np.max(np.abs(old_scores-new_scores)))
            gap = float(np.sort(old_scores)[1]-np.sort(old_scores)[0])
            item['policies'][policy] = dict(sealed_candidate=old_choice['candidate'],
                regenerated_candidate=new_choice['candidate'], sealed_scores=scores_a, regenerated_scores=scores_b,
                max_absolute_score_difference=error, saved_best_to_second_gap=gap,
                gap_exceeds_twice_difference=gap > 2*error)
        conditions.append(item)
    return dict(status='complete', conditions=conditions, action_changes=changes,
        compared_learned_actions=len(conditions)*len(LEARNERS),
        compared_preview_actions=len(conditions), score_difference_pass_threshold=None)


def fit_and_assess(p, old, regenerated_features, regenerated_records, log):
    """Development labels only; no fallback to saved features or saved labels."""
    result = {}
    expected = {role:len(driver.expected_features(p, role)) for role in driver.ROLES}
    for role in driver.ROLES:
        if len(regenerated_features[role]) != expected[role]:
            return dict(status='unavailable_incomplete_features', role=role)
    dev_records = regenerated_records['development']
    if len(dev_records) != len(list(driver.specs(p, 'development'))):
        return dict(status='unavailable_incomplete_development_labels')
    # Saved-input decisions are a control for input-regeneration consequences.
    saved_features = {role:[old['features'][role][key] for key in driver.expected_features(p, role)]
        for role in driver.ROLES}
    packet = driver.fit_development(regenerated_features['development'], dev_records, p)
    result.update(status=packet['status'], regenerated_model=packet)
    if packet['status'] != 'complete':
        log.fail('fitting', 'regenerated_model', 'training_failed', error=packet.get('error'))
        return result
    # Input hashes should change whenever regeneration changes even one bit.
    # They are explicitly retained, not subjected to an impossible equality gate.
    old_packet = old['packet']
    result['data_hashes'] = dict(saved=old_packet['development_data_sha256'],
        regenerated=packet['development_data_sha256'])
    log.compare({k:v for k,v in old_packet.items() if k not in ('packet_sha256', 'development_data_sha256')},
        {k:v for k,v in packet.items() if k not in ('packet_sha256', 'development_data_sha256')},
        'regenerated_model', 'model', **MODEL_ARITHMETIC)
    result['penalties'] = {key:dict(saved=old_packet['models'][key]['chosen_lambda'],
        regenerated=packet['models'][key]['chosen_lambda'],
        saved_index=old_packet['models'][key]['chosen_lambda_index'],
        regenerated_index=packet['models'][key]['chosen_lambda_index']) for key in LEARNERS}
    unchanged = driver.decide_test(saved_features['test'], old_packet, p)
    log.compare(old['decisions'], unchanged, 'saved_input_decision_replay', 'decisions')
    input_only = driver.decide_test(regenerated_features['test'], old_packet, p)
    fresh = driver.decide_test(regenerated_features['test'], packet, p)
    result['saved_model_on_regenerated_inputs'] = decision_effects(old['decisions'], input_only, log, 'input_only')
    result['regenerated_model_on_regenerated_inputs'] = decision_effects(old['decisions'], fresh, log, 'full')
    result['regenerated_decisions'] = fresh
    all_records = regenerated_records['development'] + regenerated_records['test']
    if len(all_records) != p['planned_targets']:
        result['outcomes'] = dict(status='undefined_incomplete', received=len(all_records),
            required=p['planned_targets'], primary=None)
    else:
        result['outcomes'] = dict(status='complete',
            sealed_actions_saved_labels=driver.summarize(
                [row for role in driver.ROLES for row in old['records'][role].values()], old['decisions'], p),
            sealed_actions_regenerated_labels=driver.summarize(all_records, old['decisions'], p),
            regenerated_actions_regenerated_labels=driver.summarize(all_records, fresh, p))
    return result


def audit(attempt, *, expected_head, progress=None):
    started = time.perf_counter()
    log = Recorder()
    output = dict(schema_version=1, audit='EXP008 saved-trajectory regeneration',
        source_commit=SOURCE, manifest_sha256=MANIFEST, receipt_sha256=RECEIPT,
        auditor_sha256=sha(Path(__file__)), platform=platform.platform(), machine=platform.machine(),
        started_at=datetime.now(timezone.utc).isoformat(), expected_audit_head=expected_head,
        python=sys.version, packages={name:importlib.metadata.version(name) for name in ('numpy', 'scipy')},
        thread_environment={key:os.environ.get(key) for key in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        frozen_gate_not_replaced=True, full_frozen_verifier_run=False, new_integrations=0,
        research_status='Post-run development diagnostic of already observed trajectories',
        feature_action_scope='Regenerated metrics/Grams/features at the receipt-bound realized actions; fresh directions compared separately',
        arithmetic_rule=ARITHMETIC, model_arithmetic_rule=MODEL_ARITHMETIC,
        planned=dict(contexts=48, previews=3456, targets=7008, learned_actions=288),
        contexts=[], metrics=[], regenerated_features={}, regenerated_targets={},
        unavailable=[], status='incomplete', exit_code=2)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    output['numpy_build'] = buffer.getvalue()
    try:
        try:
            output['execution_provenance'] = execution_provenance(expected_head)
        except Exception as error:
            log.exception('execution_provenance', 'source_gate', error)
            output['status'] = 'source_gate_failure_no_regeneration'
            return output
        result = bindings(attempt, log)
        if result is None:
            output['status'] = 'binding_failure_no_regeneration'
            return output
        p, receipt, counts = result
        output['bindings'] = counts
        output['source_bindings'] = p['source_files']
        old = load_inputs(attempt, p, receipt, log)
        if log.failures:
            output['status'] = 'input_identity_failure_no_regeneration'
            return output
        geometry = driver.build_atlas_geometry(driver.base.load_grid(ROOT/'evidence/EXP-004/attempt-001/focus'),
            p['selection']['core_index'])
        features = {role:[] for role in driver.ROLES}
        records = {role:[] for role in driver.ROLES}
        output['regenerated_features'] = {role:[] for role in driver.ROLES}
        output['regenerated_targets'] = records
        for role in driver.ROLES:
            for anchor in driver.anchors_for(p, role):
                aid = anchor['id']
                contexts = {}
                try:
                    baseline = old['baselines'][role][aid]
                    cert = baseline['certificate']
                    if (not baseline['eligible'] or not cert['verified'] or not cert['attracting']
                            or [step['period'] for step in baseline['steps']] != [9, 9]):
                        raise ValueError('Required saved baseline certificate is ineligible')
                    for phase in p['phases']:
                        key = f'{aid}-p{phase}'
                        try:
                            path = attempt/'sensitivity'/f'{key}.npz'
                            with np.load(path, allow_pickle=False) as raw:
                                context, detail = build_context(raw, read(path.with_suffix('.json')), anchor,
                                    cert, phase, log, key)
                            contexts[phase] = context
                            output['contexts'].append(detail)
                        except Exception as error:
                            log.exception('contexts', key, error)
                            output['unavailable'].append(dict(kind='context', id=key))
                except Exception as error:
                    log.exception('baselines', aid, error)
                    output['unavailable'].append(dict(kind='baseline', id=aid))
                for key in driver.expected_features(p, role):
                    row = old['features'][role][key]
                    if row['anchor'] != aid:
                        continue
                    try:
                        with np.load(bounded_path(attempt, row['raw']), allow_pickle=False) as raw:
                            regenerated, metrics = regenerate_preview(row, raw, anchor,
                                contexts[row['phase']], geometry, log)
                        features[role].append(regenerated)
                        output['regenerated_features'][role].append({k:v for k,v in regenerated.items()
                            if k not in ('metrics', 'wall_seconds')})
                        output['metrics'].append(metrics)
                    except Exception as error:
                        log.exception('previews', key, error)
                        output['unavailable'].append(dict(kind='preview', id=key))
                for spec in driver.specs(p, role):
                    if spec['anchor'] != aid:
                        continue
                    key = driver.trial_id(spec)
                    row = old['records'][role][key]
                    try:
                        with np.load(bounded_path(attempt, row['raw']), allow_pickle=False) as raw:
                            regenerated, metrics = regenerate_target(row, raw, anchor,
                                contexts[row['phase']], log)
                        records[role].append(regenerated)
                        output['metrics'].append(metrics)
                    except Exception as error:
                        log.exception('targets', key, error)
                        output['unavailable'].append(dict(kind='target', id=key))
                if progress:
                    progress(dict(stage='regeneration', role=role, anchor=aid,
                        previews=len(features[role]), targets=len(records[role]), failures=len(log.failures)))
        output['received'] = dict(contexts=len(output['contexts']),
            previews=sum(map(len, features.values())), targets=sum(map(len, records.values())))
        try:
            if any(f['stage'] in ('identity', 'raw_identity', 'census', 'bindings') for f in log.failures):
                output['assessment'] = dict(status='unavailable_identity_failure')
            else:
                output['assessment'] = fit_and_assess(p, old, features, records, log)
        except Exception as error:
            log.exception('assessment', 'fit_and_assess', error)
            output['assessment'] = dict(status='unavailable_exception')
        complete = (output['received'] == {key:output['planned'][key] for key in ('contexts', 'previews', 'targets')}
            and output['assessment'].get('status') == 'complete')
        output['status'] = ('incomplete' if not complete else
            'completed_with_comparison_failures' if log.failures else 'completed_no_comparison_failures')
        output['exit_code'] = 2 if not complete else 1 if log.failures else 0
        output['postflight_execution_provenance'] = execution_provenance(expected_head)
        postflight_log = Recorder()
        postflight = bindings(attempt, postflight_log)
        output['postflight_bindings'] = dict(passed=postflight is not None,
            failures=postflight_log.failures)
        if postflight is None:
            for failure in postflight_log.failures:
                log.fail('postflight_bindings', failure['path'], failure['kind'], detail=failure)
            output.update(status='postflight_binding_failure', exit_code=2)
    except (Exception, KeyboardInterrupt, AuditInterrupted) as error:
        log.exception('audit', 'unexpected_error', error)
        output.update(status='incomplete_exception', exit_code=2)
    finally:
        output['received'] = dict(contexts=len(output['contexts']),
            previews=sum(map(len, output['regenerated_features'].values())),
            targets=sum(map(len, output['regenerated_targets'].values())))
        output['comparison_counts'] = dict(log.counts)
        output['failures'] = log.failures
        output['failure_count'] = len(log.failures)
        output['audit_wall_seconds'] = time.perf_counter()-started
        output['limits'] = [
            'No orbit shooting, baseline requalification, ODE reintegration, or new variational integration.',
            'Original frozen numerical routines are reused; this is not an independent-algorithm replication.',
            'A saved-trajectory conditional outcome is not an outcome for a changed regenerated direction.',
            'Fresh optimizer phases feed refitting; original saved phases separately retain the original strict replay rule.',
            'No corrected coefficient, label, phase, raw artifact, or original threshold is written back.',
            'Descriptive refitting on observed EXP-008 data is not fresh held-out evidence or a merge-gate replacement.']
    return safe(output)


def output_path(path, attempt):
    resolved = path.resolve()
    if resolved.is_relative_to(attempt.resolve()) or resolved.is_relative_to(ROOT.resolve()):
        raise ValueError('Output must be outside the entire repository and immutable attempt')
    if path.exists():
        raise ValueError('Output already exists; preserve every audit attempt')
    if not path.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path, nargs='?', default=ROOT/'evidence/EXP-008/attempt-001')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-head', required=True,
        help='Full SHA of the already pushed audit source; checked against a clean checkout')
    args = parser.parse_args()
    try:
        destination = output_path(args.out, args.attempt)
    except ValueError as error:
        parser.error(str(error))
    def interrupt(signum, frame):
        del frame
        raise AuditInterrupted(f'Regeneration interrupted by signal {signum}; partial results retained')
    signal.signal(signal.SIGTERM, interrupt)
    result = audit(args.attempt, expected_head=args.expected_head,
        progress=lambda value:print(json.dumps(value), flush=True))
    with destination.open('x') as stream:
        json.dump(safe(result), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], exit_code=result['exit_code'],
        failures=result['failure_count'], output=str(destination), sha256=sha(destination))), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    sys.exit(main())
