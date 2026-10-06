#!/usr/bin/env python3
"""Diagnose EXP008 platform replay without replacing its frozen failing gate.

Uses the original algebraic fit on receipt-bound saved development data. No
neural integration, metric optimization, new outcomes or source changes occur.
Coefficient-only failures are retained even when this diagnostic completes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import run_hopfield_information as driver

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '3c5604497ec57b19d62fbb595da7b190102f4b33'
MANIFEST = 'fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd'
RECEIPT = 'ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb'
LEARNERS = ('baseline', 'geometry', 'shuffled_geometry')
POLICIES = (*LEARNERS, 'min_preview')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def compare_all(saved, replayed, *, rtol=1e-7, atol=1e-9, path='model'):
    """Collect every mismatch using the frozen model comparison's semantics.

    The caller explicitly supplies the existing stricter arithmetic tolerance
    for saved-model decision replay. Nothing changes the frozen verifier.
    """
    report = dict(rtol=rtol, atol=atol, float_comparisons=0,
                  exact_comparisons=0, max_absolute_float_difference=0., failures=[])

    def fail(location, kind, left, right, **extra):
        report['failures'].append(dict(path=location, kind=kind,
            saved=safe(left), replayed=safe(right), **extra))

    def visit(left, right, location):
        if isinstance(right, dict):
            if not isinstance(left, dict):
                fail(location, 'type', left, right); return
            if set(left) != set(right):
                fail(location, 'keys', sorted(left), sorted(right))
            for key in sorted(set(left) & set(right)):
                visit(left[key], right[key], location+'.'+key)
        elif isinstance(right, list):
            if not isinstance(left, list):
                fail(location, 'type', left, right); return
            if len(left) != len(right):
                fail(location, 'length', len(left), len(right))
            for i, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f'{location}[{i}]')
        elif isinstance(right, float):
            report['float_comparisons'] += 1
            if not isinstance(left, (int, float)) or isinstance(left, bool):
                fail(location, 'type', left, right); return
            if not np.isfinite(left) or not np.isfinite(right):
                fail(location, 'nonfinite', left, right); return
            error = abs(left-right)
            allowed = atol+rtol*abs(right)
            report['max_absolute_float_difference'] = max(report['max_absolute_float_difference'], error)
            if error > allowed:
                fail(location, 'float_tolerance', left, right,
                    absolute_difference=error, allowed_difference=allowed,
                    error_over_allowance=error/allowed,
                    relative_difference=None if right == 0 else error/abs(right))
        else:
            report['exact_comparisons'] += 1
            if type(left) is not type(right) or left != right:
                fail(location, 'exact', left, right)

    visit(saved, replayed, path)
    report['passed'] = not report['failures']
    report['failure_count'] = len(report['failures'])
    return report


def bound_inputs(attempt):
    """Validate all immutable bindings before using development/test data."""
    require(sha(attempt/'receipt.json') == RECEIPT, 'Original receipt binding changed')
    receipt = read(attempt/'receipt.json')
    require(receipt.get('status') == 'complete', 'A completed receipt is required')
    require(not any((attempt/name).exists() for name in ['failure.json', 'timeout.json']),
            'Attempt failure or timeout exists')
    require(receipt['source_commit'] == SOURCE, 'Frozen experiment source changed')
    require(receipt['manifest_sha256'] == MANIFEST == sha(attempt/'manifest.json'),
            'Frozen manifest binding changed')
    p = read(attempt/'manifest.json')
    driver.validate(p, check_sources=True)
    require(sha(driver.PLAN) == MANIFEST, 'Repository plan differs from the frozen manifest')
    paths = [item['path'] for item in receipt['artifacts']]
    require(len(paths) == len(set(paths)), 'Duplicate receipt artifact')
    for item in receipt['artifacts']:
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe receipt path')
        file = attempt/relative
        require(file.is_file() and file.stat().st_size == item['bytes'] and sha(file) == item['sha256'],
                'Raw artifact binding changed: '+item['path'])
    features, records = {}, {}
    for role in driver.ROLES:
        features[role] = read(attempt/f'{role}-features.json')
        records[role] = [json.loads(line) for line in (attempt/f'{role}-targets.jsonl').read_text().splitlines()]
        require([r['id'] for r in features[role]] == driver.expected_features(p, role), 'Feature census/order changed')
        require(all(r['role'] == role and r['status'] == 'complete' for r in features[role]), 'Incomplete/mixed-role features')
        expected = {driver.trial_id(s): s for s in driver.specs(p, role)}
        require(len(records[role]) == len(expected) and {r['id'] for r in records[role]} == set(expected),
                'Target census changed')
        for row in records[role]:
            require(all(row[k] == v for k, v in expected[row['id']].items()), 'Target specification changed')
            require(row['status'] == 'complete' and driver.valid_rmse(row.get('rmse')), 'Required target incomplete')
            require(row['source_commit'] == SOURCE and row['manifest_sha256'] == MANIFEST, 'Target source binding changed')
    packet = read(attempt/'model.json')
    require(packet['status'] == 'complete' and packet['packet_sha256'] == driver.packet_digest(packet), 'Model packet binding changed')
    seal = read(attempt/'model-seal.json')
    dseal = read(attempt/'test-decision-seal.json')
    started = read(attempt/'test-target-started.json')
    bindings = [(seal, 'model_sha256', 'model.json'),
                (seal, 'development_targets_sha256', 'development-targets.jsonl'),
                (seal, 'development_features_sha256', 'development-features.json'),
                (dseal, 'model_seal_sha256', 'model-seal.json'),
                (dseal, 'features_sha256', 'test-features.json'),
                (dseal, 'decisions_sha256', 'test-decisions.json'),
                (started, 'model_seal_sha256', 'model-seal.json'),
                (started, 'decision_seal_sha256', 'test-decision-seal.json')]
    for item, key, name in bindings:
        require(item[key] == sha(attempt/name), 'Chronological seal changed: '+key)
    require(receipt['started_at'] <= seal['sealed_at'] <= dseal['sealed_at'] <= started['started_at'] <= receipt['ended_at'],
            'Model/decision/target chronology changed')
    require(all(r['test_decision_seal_sha256'] == started['decision_seal_sha256'] for r in records['test']),
            'A test target has another decision seal')
    return p, receipt, features, records, packet, read(attempt/'test-decisions.json')


def compare_decisions(sealed, saved_model_replay, refitted_model_replay):
    """Retain all scores and action gaps without inventing a refit-score gate."""
    strict = compare_all(sealed, saved_model_replay, rtol=1e-12, atol=1e-14, path='sealed_decisions')
    result = dict(saved_model_replay=strict, conditions=[], action_changes=[], critical_failures=[])
    if not strict['passed']:
        result['critical_failures'].append('Saved-model platform decisions fail existing arithmetic/identity replay')
    if len(sealed) != len(saved_model_replay) or len(sealed) != len(refitted_model_replay):
        result['critical_failures'].append('Decision census length differs')
    for i, (original, saved, refit) in enumerate(zip(sealed, saved_model_replay, refitted_model_replay)):
        item = {k: original[k] for k in ['anchor', 'block', 'phase', 'radius']}
        item['learners'] = {}
        for key in ['anchor', 'block', 'phase', 'radius', 'candidate_ids', 'required_preview_calls', 'preview_scores']:
            if original.get(key) != refit.get(key):
                result['critical_failures'].append(f'Condition {i}: refitted decision metadata differs: {key}')
        for policy in POLICIES:
            old_choice, new_choice = original['policies'][policy], refit['policies'][policy]
            if old_choice.get('status') != 'complete' or new_choice.get('status') != 'complete':
                result['critical_failures'].append(f'Condition {i}: incomplete selector {policy}')
            if old_choice.get('candidate') != new_choice.get('candidate'):
                result['action_changes'].append(dict(condition=i, policy=policy,
                    saved_candidate=old_choice.get('candidate'), refitted_candidate=new_choice.get('candidate')))
            if policy == 'min_preview':
                continue
            vectors = [np.asarray(d['predictions'][policy], dtype=float) for d in [original, saved, refit]]
            if any(v.shape != (18,) or not np.isfinite(v).all() for v in vectors):
                result['critical_failures'].append(f'Condition {i}: nonfinite/incomplete scores {policy}')
                continue
            old_scores, saved_scores, new_scores = vectors
            # An independent lowest-index argmin check catches stale action IDs.
            for label, scores, choice in [('sealed', old_scores, old_choice), ('refitted', new_scores, new_choice)]:
                if int(np.argmin(scores)) != choice.get('candidate'):
                    result['critical_failures'].append(f'Condition {i}: {label} argmin disagrees for {policy}')
            delta = float(np.max(np.abs(old_scores-new_scores)))
            gap = float(np.sort(old_scores)[1]-np.sort(old_scores)[0])
            item['learners'][policy] = dict(sealed_scores=old_scores.tolist(),
                saved_model_platform_scores=saved_scores.tolist(), refitted_model_platform_scores=new_scores.tolist(),
                sealed_candidate=old_choice.get('candidate'), refitted_candidate=new_choice.get('candidate'),
                max_saved_model_prediction_difference=float(np.max(np.abs(old_scores-saved_scores))),
                max_refitted_prediction_difference=delta, sealed_best_to_second_gap=gap,
                gap_exceeds_twice_refit_difference=gap > 2*delta,
                refitted_minimum=float(np.min(new_scores)), refitted_maximum=float(np.max(new_scores)))
        result['conditions'].append(item)
    if result['action_changes']:
        result['critical_failures'].append('At least one refitted policy selects another action')
    all_rows = [v for item in result['conditions'] for v in item['learners'].values()]
    result['learned_conditions'] = len(all_rows)
    result['candidate_prediction_count'] = 18*len(all_rows)
    result['max_refitted_prediction_difference'] = max((v['max_refitted_prediction_difference'] for v in all_rows), default=None)
    result['minimum_sealed_action_gap'] = min((v['sealed_best_to_second_gap'] for v in all_rows), default=None)
    result['conditions_with_gap_not_exceeding_twice_difference'] = sum(not v['gap_exceeds_twice_refit_difference'] for v in all_rows)
    result['refitted_actions_unchanged'] = not result['action_changes'] and not result['critical_failures']
    result['refitted_prediction_tolerance'] = None
    result['score_scope'] = 'All differences retained; no post-run numeric pass threshold for refitted versus sealed scores'
    return result


def replay_base_features(attempt, p, features):
    """Enumerate the frozen base-feature comparisons using saved preview metrics.

    Sensitivity Grams are reconstructed from stored variational arrays. The
    direction comes from the saved direction, as in the frozen verifier. This
    does not recompute the separately optimized preview-waveform metrics.
    """
    anchors = {a['id']: a for a in p['anchors']}
    result = dict(rows=0, feature_values=0, failures=[], direction_failures=[],
        max_absolute_difference=0., rtol=1e-12, atol=1e-14,
        scope='Recomputed sensitivity Grams and base features; saved preview metrics and certificates; not a full metric replay',
        model_fit_input='The separate model replay uses sealed SAVED features, not these reconstructed values')
    for role in driver.ROLES:
        baselines = read(attempt/f'{role}-baselines.json')
        require(set(baselines) == {a['id'] for a in driver.anchors_for(p, role)}, 'Baseline census differs')
        contexts = {}
        for aid, baseline in baselines.items():
            require(baseline['eligible'], 'Required baseline is ineligible')
            cert = baseline['certificate']
            require(cert['verified'] and cert['attracting'], 'Required baseline certificate failed')
            for phase in p['phases']:
                with np.load(attempt/'sensitivity'/f'{aid}-p{phase}.npz', allow_pickle=False) as raw:
                    tail = raw['times'] > 75
                    gram = driver.phase_projected_gram(raw['activation_sensitivity'][tail], raw['activation_velocity'][tail])
                    comparison = compare_all(raw['tail_direction'].tolist(), gram['minimum_eigenvector'].tolist(),
                        rtol=1e-12, atol=1e-14, path=f'{aid}.p{phase}.tail_direction')
                    result['direction_failures'].extend(comparison['failures'])
                    contexts[(aid, phase)] = (cert, gram, driver.candidates(raw['tail_direction']))
        for row in features[role]:
            cert, gram, directions = contexts[(row['anchor'], row['phase'])]
            direction = directions[row['candidate']]
            # Direction identities stay exact, as in frozen verify().
            require(np.array_equal(row['direction'], direction), 'Candidate direction identity differs: '+row['id'])
            rebuilt = driver.base_features(anchors[row['anchor']], row['phase'], direction, row['radius'], cert, gram, row['metrics'])
            comparison = compare_all(row['base'], np.asarray(rebuilt).tolist(),
                rtol=1e-12, atol=1e-14, path=row['id']+'.base')
            result['rows'] += 1
            result['feature_values'] += comparison['float_comparisons']
            result['max_absolute_difference'] = max(result['max_absolute_difference'], comparison['max_absolute_float_difference'])
            for failure in comparison['failures']:
                index = int(failure['path'].rsplit('[', 1)[1][:-1]) if failure['path'].endswith(']') else None
                result['failures'].append(dict(role=role, anchor=row['anchor'], phase=row['phase'],
                    radius=row['radius'], candidate=row['candidate'], feature_index=index,
                    feature_name=None if index is None else p['base_feature_names'][index], **failure))
    require(result['rows'] == 3456 and result['feature_values'] == 3456*51, 'Base-feature replay census incomplete')
    result['failure_count'] = len(result['failures'])
    result['passed'] = not result['failures'] and not result['direction_failures']
    return result


def assess_packets(packet, refitted, sealed, saved_model_decisions, refitted_decisions):
    comparison = compare_all({k:v for k,v in packet.items() if k != 'packet_sha256'},
                             {k:v for k,v in refitted.items() if k != 'packet_sha256'})
    penalties = []
    for policy in LEARNERS:
        a, b = packet['models'][policy], refitted['models'][policy]
        penalties.append(dict(policy=policy, saved_lambda=a['chosen_lambda'], refitted_lambda=b['chosen_lambda'],
            saved_index=a['chosen_lambda_index'], refitted_index=b['chosen_lambda_index'],
            identical=(type(a['chosen_lambda_index']) is type(b['chosen_lambda_index']) and
                       a['chosen_lambda_index'] == b['chosen_lambda_index'] and a['chosen_lambda'] == b['chosen_lambda'])))
    decisions = compare_decisions(sealed, saved_model_decisions, refitted_decisions)
    critical = list(decisions['critical_failures'])
    if not all(p['identical'] for p in penalties):
        critical.append('A development-selected penalty changed')
    noncoefficient = [f for f in comparison['failures']
                      if not ('.coefficients[' in f['path'] and f['kind'] == 'float_tolerance')]
    if noncoefficient:
        critical.append('Non-coefficient frozen model comparisons fail')
    coefficient_counts = Counter(f['path'].split('.models.')[1].split('.')[0]
        for f in comparison['failures'] if '.models.' in f['path'] and '.coefficients[' in f['path'])
    return dict(strict_model_replay_passed=comparison['passed'], model_comparison=comparison,
        coefficient_failure_counts=dict(coefficient_counts), penalties=penalties,
        decisions=decisions, critical_failures=critical,
        diagnostic_exit_code=2 if critical else 0,
        diagnostic_status=('critical_difference' if critical else
            'completed_with_frozen_coefficient_failures' if comparison['failures'] else 'completed_without_model_mismatch'))


def audit(attempt):
    started = time.perf_counter()
    report = dict(schema_version=1, audit='EXP008 platform algebra diagnostic',
        frozen_gate_not_replaced=True, full_frozen_verifier_run=False,
        frozen_source_commit=SOURCE, frozen_manifest_sha256=MANIFEST, frozen_receipt_sha256=RECEIPT,
        diagnostic_source_sha256=sha(Path(__file__)), attempt=str(attempt),
        platform=platform.platform(), machine=platform.machine(), python=sys.version,
        numpy_version=np.__version__, scipy_version=driver.importlib.metadata.version('scipy'),
        numerical_libraries=np.show_config(mode='dicts'),
        thread_environment={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'] if k in os.environ},
        integrations=0, new_targets=0, fitting_scope='Original same-algorithm algebraic replay, development labels only')
    try:
        p, receipt, features, records, packet, sealed = bound_inputs(attempt)
        report['raw_bindings_passed'] = True
        report['receipt_sha256'] = sha(attempt/'receipt.json')
        report['artifact_count'] = len(receipt['artifacts'])
        report['base_feature_replay'] = replay_base_features(attempt, p, features)
        report['strict_base_feature_replay_passed'] = report['base_feature_replay']['passed']
        refitted = driver.fit_development(features['development'], records['development'], p)
        require(refitted.get('status') == 'complete', 'Platform development fit failed')
        # Keep the entire local fit, including CV/scalers, so failed comparisons
        # and all predictions can be reconstructed without rerunning this fit.
        report['refitted_model'] = refitted
        saved_decisions = driver.decide_test(features['test'], packet, p)
        refitted_decisions = driver.decide_test(features['test'], refitted, p)
        report.update(assess_packets(packet, refitted, sealed, saved_decisions, refitted_decisions))
        if not report['strict_base_feature_replay_passed'] and not report['critical_failures']:
            report['diagnostic_status'] = 'completed_with_frozen_replay_failures'
        require(len(sealed) == 96 and report['decisions']['learned_conditions'] == 288 and
                report['decisions']['candidate_prediction_count'] == 5184, 'Incomplete required decision/prediction census')
    except Exception as error:
        report.setdefault('raw_bindings_passed', False)
        report.update(diagnostic_status='diagnostic_error', diagnostic_exit_code=2, error=repr(error))
    report['wall_seconds'] = time.perf_counter()-started
    report['interpretation'] = ('Diagnostic completion is not a passing frozen release gate. '
        'Every coefficient failure remains a failure; full frozen CI remains authoritative. '
        'Base-feature replay uses saved preview metrics; model refitting uses saved features. '
        'This diagnostic does not repeat waveform metric checks or establish general cross-platform reproducibility.')
    return safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    attempt, out = args.attempt.resolve(), args.out.resolve()
    if out.is_relative_to(attempt):
        parser.error('Diagnostic output must be outside the immutable attempt')
    result = audit(attempt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:result.get(key) for key in ['diagnostic_status','strict_base_feature_replay_passed','strict_model_replay_passed',
        'coefficient_failure_counts','diagnostic_exit_code','raw_bindings_passed','error']}))
    raise SystemExit(result['diagnostic_exit_code'])


if __name__ == '__main__':
    main()
