#!/usr/bin/env python3
"""Post-run EXP-012 audit: four canonical batches; no integration or model fit.

Uses the unchanged EXP-010 scorer and EXP-009 strict score comparator only.
Does not import the EXP-012 runner, CUDA backend, or its audit functions/flags.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
for name in THREAD_KEYS:
    os.environ[name] = '1'

import numpy as np

ROOT = Path('/data2/neural-shrimp')
SOURCE = 'c3a88b9db0a3e82b65d5d55a58d155b9a3275d5f'
PLAN_PATH = 'experiments/hopfield/EXP-012-plan.json'
RAW = Path('/data2/neural-shrimp-data/raw/EXP-012/attempt-001')
OLD = ROOT/'evidence/EXP-008/attempt-001'
QUALIFIED = Path('/data2/neural-shrimp-data/raw/EXP-010/attempt-001')
CONTROL = {'axes': [0, 4, 8, 12], 'diagonal': [2, 6, 10, 14],
           'local_coverage': [0, 8, 16, 17]}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def git_bytes(relative):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{SOURCE}:{relative}'])


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def ranks(census, scores):
    groups = defaultdict(list)
    if len(census) != len(scores):
        raise ValueError('Ranking census mismatch')
    for case, score in zip(census, scores, strict=True):
        value = score['rmse']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2:
            raise ValueError('Invalid scalar label')
        groups[tuple(case['key'][:-1])].append((float(value), case['key'][-1]))
    result = []
    for key, values in sorted(groups.items()):
        if len(values) != 18 or {c for _, c in values} != set(range(18)):
            raise ValueError('Incomplete candidate context')
        ordered = sorted(values, key=lambda item: (item[0], item[1]))
        candidate_scores = dict((c, value) for value, c in values)
        result.append(dict(key=list(key), candidate_order=[c for _, c in ordered],
            minimum=ordered[0][1], top4=[c for _, c in ordered[:4]],
            controls={name:min(ids, key=lambda c: (candidate_scores[c], c)) for name, ids in CONTROL.items()},
            scores=[candidate_scores[c] for c in range(18)],
            best_second_gap=ordered[1][0]-ordered[0][0],
            fourth_fifth_gap=ordered[4][0]-ordered[3][0]))
    return result


def metric_changes(a, b, path='score'):
    """Separate implementation of EXP-012 cross-device metric rules."""
    changes = []
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [dict(path=path, passed=False, reason='keys')]
        for key in a:
            changes.extend(metric_changes(a[key], b[key], path+'.'+key))
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return [dict(path=path, passed=False, reason='list shape')]
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            changes.extend(metric_changes(x, y, f'{path}[{i}]'))
    elif type(a) in (float, int) and not isinstance(a, bool):
        if type(b) not in (float, int) or not math.isfinite(a) or not math.isfinite(b):
            return [dict(path=path, passed=False, reason='numeric type/finite')]
        if type(a) is int:
            if type(b) is not int or a != b:
                diagnostic = path.endswith('phase_search.refinement_evaluations')
                changes.append(dict(path=path, a=a, b=b, passed=diagnostic, diagnostic=diagnostic))
        elif a != b:
            difference = abs(a-b)
            tolerance = 0.
            if path.endswith('phase_offset_cycles'):
                difference = abs((a-b+.5)%1.-.5)
                tolerance = 1e-8
            elif 'rms' in path or path.endswith('.rmse'):
                tolerance = 1e-10
            changes.append(dict(path=path, a=a, b=b, absolute_difference=difference,
                                tolerance=tolerance, passed=difference <= tolerance))
    elif type(a) is not type(b) or a != b:
        changes.append(dict(path=path, passed=False, reason='categorical value/type'))
    return changes


def wave_difference(a, b, ids, times):
    reports = {}
    for name, x, y in [('state', a, b), ('activation', np.tanh(a), np.tanh(b))]:
        signed = y-x
        absolute = np.abs(signed)
        maximum = float(absolute.max())
        index = np.unravel_index(int(absolute.argmax()), absolute.shape)
        tol = (1e-10 + 1e-10*np.maximum(np.abs(x), np.abs(y))) if name == 'state' else 1e-10
        reports[name] = dict(maximum_absolute_difference=maximum,
            rms_difference=float(np.sqrt(np.mean(signed**2))),
            per_component_rms=np.sqrt(np.mean(signed**2, axis=(0, 1))).tolist(),
            per_component_maximum=absolute.max(axis=(0, 1)).tolist(),
            maximum_location=dict(case_id=ids[index[0]], sample_index=int(index[1]),
                                  time=float(times[index[1]]), component=int(index[2])),
            threshold_violation_coordinates=int(np.count_nonzero(absolute > tol)),
            passed=bool(np.isfinite(signed).all() and np.all(absolute <= tol)))
    return reports


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.failures = []
        self.max_score_delta = 0.
        self.score_max_path = None

    def check(self, condition, label):
        self.counts['checks'] += 1
        if not bool(condition):
            self.failures.append(label)
        return bool(condition)

    def bound(self, path, expected, label):
        path = Path(path)
        self.check(path.is_file(), label+':exists')
        if path.is_file():
            self.check(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], label+':bytes/hash')

    def score_delta(self, saved, replay, path):
        if isinstance(saved, dict):
            for key in saved:
                self.score_delta(saved[key], replay[key], path+'.'+key)
        elif isinstance(saved, list):
            for i, (a, b) in enumerate(zip(saved, replay, strict=True)):
                self.score_delta(a, b, f'{path}[{i}]')
        elif type(saved) is float:
            delta = abs(saved-replay)
            if delta > self.max_score_delta:
                self.max_score_delta, self.score_max_path = delta, path

    def run(self, raw, workers):
        receipt, manifest = read(raw/'receipt.json'), read(raw/'manifest.json')
        self.check(receipt['source_commit'] == manifest['source_commit'] == SOURCE, 'fixed pushed source commit')
        self.check(receipt['status'] == 'complete', 'producer terminal receipt complete')
        inventory = receipt['artifacts']
        names = [item['path'] for item in inventory]
        self.check(len(names) == len(set(names)), 'receipt inventory unique')
        actual = {str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()}
        self.check(actual == set(names)|{'receipt.json'}, 'exact raw artifact census')
        for item in inventory:
            p = raw/item['path']
            self.check(p.resolve().is_relative_to(raw) and not p.is_symlink(), 'raw containment:'+item['path'])
            self.bound(p, item, 'raw artifact:'+item['path'])
            self.counts['artifact_files'] += 1
            self.counts['artifact_bytes'] += item['bytes']
        frozen_plan = git_bytes(PLAN_PATH)
        self.check(hashlib.sha256(frozen_plan).hexdigest() == receipt['plan_sha256'] == manifest['plan_sha256'], 'committed plan hash')
        plan = json.loads(frozen_plan)
        self.check(manifest['plan'] == plan, 'manifest contains exact committed plan')
        self.check(plan['duration'] == 75. and plan['sample_dt'] == .1 and plan['dts'] == [.0025, .00125], 'fixed integration endpoints/steps')
        for key, value in dict(state_atol=1e-10, state_rtol=1e-10, activation_atol=1e-10,
                               score_atol=1e-10, phase_atol=1e-8).items():
            self.check(plan[key] == value, 'fixed tolerance:'+key)
        self.check(plan['controls'] == CONTROL, 'fixed control IDs')
        for item in plan['source_files']:
            content = git_bytes(item['path'])
            self.check(len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256'], 'committed source:'+item['path'])
            self.bound(ROOT/item['path'], item, 'live unchanged source:'+item['path'])
            self.counts['source_files'] += 1
        self.bound(ROOT/plan['old_plan']['path'], plan['old_plan'], 'bound EXP010 plan')
        old_plan = read(ROOT/plan['old_plan']['path'])
        self.check(old_plan['census'] == plan['census'], 'complete EXP010 case census')
        for item in old_plan['source_artifacts']:
            self.bound(ROOT/item['path'], item, 'original input:'+item['path'])
            self.counts['old_input_files'] += 1
        old_receipt = read(OLD/'receipt.json')
        self.check(sha(OLD/'receipt.json') == old_plan['old_receipt_sha256'], 'original EXP008 receipt')
        old_inventory = {r['path']:r for r in old_receipt['artifacts']}
        for item in old_plan['source_artifacts']:
            p = ROOT/item['path']
            if p.resolve().is_relative_to(OLD) and p.name != 'receipt.json':
                member = old_inventory[str(p.relative_to(OLD))]
                self.check((item['sha256'],item['bytes']) == (member['sha256'],member['bytes']), 'original receipt membership:'+item['path'])
        self.bound(QUALIFIED/'receipt.json', plan['label_receipt'], 'EXP010 receipt')
        qualified_receipt = read(QUALIFIED/'receipt.json')
        qualified_index = {r['path']:r for r in qualified_receipt['artifacts']}
        census = plan['census']; ids = [c['id'] for c in census]
        self.check(len(census) == len(set(ids)) == 3456, '3456 distinct cases')
        self.check([c['key'] for c in census] == sorted(c['key'] for c in census), 'canonical case order')
        self.check(len(ranks(census, [dict(rmse=0.) for _ in ids])) == 192, '192 complete candidate contexts')
        terminal = read(raw/'terminal-records.json')
        by_job = {r['id']:r for r in terminal}
        self.check(len(terminal) == len(by_job) == 54 and set(by_job) == {j['id'] for j in plan['jobs']}, '54 exact terminal jobs')
        self.check(all(r['status'] == 'complete' for r in terminal), 'all required batches complete')
        # Independently rebuild original initial/reference context, without the producer loader.
        rows = {}
        for role in ('development', 'test'):
            for line in (OLD/f'{role}-previews.jsonl').read_text().splitlines():
                row = json.loads(line)
                rows[(row['role'], row['anchor'], row['phase'], row['radius'], row['candidate'])] = row
        baselines = {role:read(OLD/f'{role}-baselines.json') for role in ('development', 'test')}
        contexts = {}
        for case in census:
            key = case['key']; row = rows[tuple(key)]; certificate = baselines[key[0]][key[1]]['certificate']
            with np.load(OLD/case['old_raw'], allow_pickle=False) as original:
                initial = original['initial'].copy(); times = original['times'].copy()
                alpha, gamma = float(original['alpha']), float(original['gamma']); delta = original['delta'].copy()
                self.check(bits(times, np.arange(751)*.1) and bits(initial, original['states'][0]), case['id']+':old exact endpoint/grid')
                self.check(bits(initial, np.asarray(certificate['orbit_states'][key[2]], dtype=np.float64)), case['id']+':certificate initial phase')
                self.check((alpha, gamma) == (row['alpha'], row['gamma']) and bits(delta, np.asarray(row['delta'])), case['id']+':old destination/delta')
            reference = np.asarray(certificate['orbit_activations'], dtype=np.float64)[:-1]
            self.check(reference.shape == (2700,3) and np.isfinite(reference).all(), case['id']+':endpoint-excluded reference')
            self.check(certificate['verified'] and certificate['attracting'], case['id']+':saved reference certificate')
            contexts[case['id']] = dict(initial=initial, times=times, alpha=alpha, gamma=gamma, delta=delta,
                reference=reference, period=certificate['period'], epoch=key[2]*certificate['period']/2700,
                old_phase=row['metrics']['phase_offset_cycles'])
        self.counts['independently_reconstructed_contexts'] = len(contexts)
        if self.failures:
            raise ValueError('Input/provenance checks failed before scoring')
        # These exact unchanged functions are explicitly the replay target. No EXP012 imports.
        sys.path.insert(0, str(ROOT/'tools'))
        import run_preview_refinement as preview
        import run_hopfield_refinement as runtime
        results, comparisons = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for dt in [.0025, .00125]:
                batches = {}
                for arm in ['cpu', 'gpu']:
                    job_id = f'{arm}-dt{dt:g}-n3456-r0'; row = by_job[job_id]
                    expected = next(j for j in plan['jobs'] if j['id'] == job_id)
                    self.check(all(type(row[k]) is type(v) and row[k] == v for k,v in expected.items()), job_id+':planned job identity')
                    self.check(row['ids'] == ids, job_id+':exact ordered IDs')
                    directory = raw/'batches'/job_id; sealed = read(directory/'record.json')
                    self.check({k:v for k,v in row.items() if k != 'resources_before'} == sealed, job_id+':terminal/individual record')
                    self.check(set(row['artifacts']) == {'raw','scores'}, job_id+':artifact members')
                    for name, item in row['artifacts'].items():
                        self.check(item['path'] == ('traces.npz' if name == 'raw' else 'scores.json'), job_id+':artifact exact path')
                        self.bound(directory/item['path'], item, job_id+':'+name)
                    with np.load(directory/'traces.npz', allow_pickle=False) as saved:
                        arrays = {k:saved[k].copy() for k in saved.files}
                    scores_packet = read(directory/'scores.json'); scores = scores_packet['scores']
                    self.check(scores_packet['raw'] == row['artifacts']['raw'], job_id+':score/raw binding')
                    shape = dict(times=(751,), states=(3456,751,3), statuses=(3456,), final=(3456,3),
                        progress=(3456,2), alpha=(3456,), gamma=(3456,), initial=(3456,3), delta=(3456,2))
                    self.check(set(arrays) == set(shape), job_id+':array keys')
                    for name, expected_shape in shape.items():
                        value = arrays[name]
                        dtype = np.dtype('int32') if name == 'statuses' else np.dtype('int64') if name == 'progress' else np.dtype('float64')
                        self.check(value.shape == expected_shape and value.dtype == dtype and np.isfinite(value).all(), job_id+':array contract:'+name)
                    self.check(bits(arrays['times'], np.arange(751)*.1), job_id+':exact times')
                    self.check(np.all(arrays['statuses'] == 751) and np.all(arrays['progress'] == [round(75/dt),751]), job_id+':complete status/progress')
                    self.check(bits(arrays['initial'], arrays['states'][:,0]) and bits(arrays['final'], arrays['states'][:,-1]), job_id+':sample endpoints')
                    self.check(len(scores) == 3456, job_id+':score census')
                    for i, ident in enumerate(ids):
                        for name in ['alpha','gamma','initial','delta']:
                            self.check(bits(arrays[name][i], np.asarray(contexts[ident][name], dtype=np.float64)), job_id+':'+ident+':'+name)
                        self.check(set(scores[i]) == {'rmse','metrics','old_fixed_phase_rmse'} and scores[i]['rmse'] == scores[i]['metrics']['aligned_rms']['aggregate'], job_id+':'+ident+':score binding')
                    before = len(self.failures)
                    def replay(i):
                        try:
                            result = preview.score(contexts[ids[i]], arrays['times'], arrays['states'][i])
                            runtime.compare_scoring(scores[i], result)
                            return i, result, None
                        except Exception as error:
                            return i, None, repr(error)
                    begun = time.monotonic()
                    for i, replayed, error in pool.map(replay, range(3456)):
                        self.check(error is None, job_id+':strict complete score replay:'+ids[i]+(':'+error if error else ''))
                        if replayed is not None:
                            self.score_delta(scores[i], replayed, job_id+':'+ids[i])
                        self.counts['replayed_score_packets'] += 1
                        self.counts['replayed_score_windows'] += len(scores[i]['metrics']['windows'])
                    elapsed = time.monotonic()-begun
                    ranked = ranks(census, scores)
                    self.check(len(ranked) == 192, job_id+':ranking count')
                    results.append(dict(job_id=job_id, cases=3456, contexts=192, replay_seconds=elapsed,
                                        replay_passed=len(self.failures) == before))
                    batches[arm] = (arrays,scores,ranked)
                    print(json.dumps(dict(stage='canonical_replayed', **results[-1])), flush=True)
                # Independently require the new canonical CPU states and full metric tree
                # to reproduce receipt-bound EXP010 native evidence, not just the GPU arm.
                cpu, cpu_scores, cpu_rank = batches['cpu']; gpu, gpu_scores, gpu_rank = batches['gpu']
                old_fail_before = len(self.failures)
                for i, case in enumerate(census):
                    stem = f'{case["id"]}--rk4-{dt:g}'
                    for name in [f'traces/{stem}.npz', f'records/{stem}.json']:
                        self.bound(QUALIFIED/name, qualified_index[name], 'EXP010 member:'+name)
                        self.counts['qualified_members'] += 1
                    old_record = read(QUALIFIED/'records'/f'{stem}.json')
                    self.check((old_record['id'],old_record['case_id'],old_record['method'],old_record['status']) ==
                               (stem,case['id'],f'rk4-{dt:g}','complete'), 'EXP010 exact record:'+stem)
                    with np.load(QUALIFIED/'traces'/f'{stem}.npz', allow_pickle=False) as original:
                        for name in ['times','states','initial','alpha','gamma','delta']:
                            value = cpu[name] if name == 'times' else cpu[name][i]
                            self.check(bits(value, original[name]), stem+':CPU exact original '+name)
                    try:
                        runtime.compare_scoring({k:old_record[k] for k in cpu_scores[i]}, cpu_scores[i])
                        self.check(True, stem+':old complete CPU score tree')
                    except Exception as error:
                        self.check(False, stem+':old complete CPU score tree:'+repr(error))
                    self.counts['cpu_cases_vs_exp010'] += 1
                differences = wave_difference(cpu['states'],gpu['states'],ids,cpu['times'])
                metadata = {k:bits(cpu[k],gpu[k]) for k in cpu if k not in ('states','final')}
                metric_rows = []
                for case, a, b in zip(census,cpu_scores,gpu_scores,strict=True):
                    changed = metric_changes(a,b)
                    if changed:
                        metric_rows.append(dict(id=case['id'], changes=changed,
                                                passed=all(c['passed'] for c in changed)))
                rank_rows = []
                for a,b in zip(cpu_rank,gpu_rank,strict=True):
                    match = {key:a[key] == b[key] for key in ['key','candidate_order','minimum','top4','controls']}
                    rank_rows.append(dict(key=a['key'], passed=all(match.values()), exact_choices=match,
                        cpu=a,gpu=b,signed_gpu_minus_cpu=(np.array(b['scores'])-a['scores']).tolist()))
                comparison = dict(dt=dt, cases=3456, contexts=192, old_cpu_reference_passed=len(self.failures)==old_fail_before,
                    waveform=differences, exact_metadata=metadata,
                    metrics_passed=all(m['passed'] for m in metric_rows), metric_changes=metric_rows,
                    rankings_passed=all(r['passed'] for r in rank_rows), rankings=rank_rows)
                comparison['canonical_parity_passed'] = (all(metadata.values()) and all(v['passed'] for v in differences.values())
                    and comparison['metrics_passed'] and comparison['rankings_passed'])
                comparisons.append(comparison)
                self.counts['same_step_case_comparisons'] += 3456
                self.counts['independent_context_comparisons'] += 192
                del batches,cpu,gpu,cpu_scores,gpu_scores,cpu_rank,gpu_rank
        return dict(canonical_batches=results, comparisons=comparisons,
                    producer_receipt=binding(raw/'receipt.json'), committed_plan_sha256=receipt['plan_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=RAW)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    args=parser.parse_args()
    raw=args.raw.resolve(); output=args.output.resolve()
    if output.is_relative_to(raw) or output.is_relative_to(ROOT):
        raise ValueError('Audit destination must be separate from raw/source')
    if not (raw/'receipt.json').is_file():
        raise ValueError('Wait for completed benchmark receipt before auditing')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); audit=Audit(); result={}; exception=None
    try:
        result=audit.run(raw,args.workers)
    except BaseException as error:
        exception=dict(error=repr(error),traceback=traceback.format_exc())
    report=dict(schema='exp012-independent-canonical-score-audit-v1', source_commit=SOURCE,
        created_at=now(), invocation=sys.argv, python=sys.version, executable=sys.executable,
        script=binding(Path(__file__).resolve()), workers=args.workers,
        thread_environment={key:os.environ[key] for key in THREAD_KEYS},
        scope='Independent provenance/input reconstruction and threshold/ranking arithmetic. '
              'Explicit same-scorer full phase-search/tree replay for four canonical r0 full batches only; '
              'no integrations/refits/CUDA import, no performance/repeat/layout audit or host cleanup claim.',
        counts=dict(audit.counts), evidence_checks_passed=not audit.failures and exception is None,
        failures=audit.failures, exception=exception,
        maximum_score_replay_absolute_difference=audit.max_score_delta,
        maximum_score_replay_path=audit.score_max_path,wall_seconds=time.monotonic()-started,**result)
    report['canonical_gpu_parity_passed'] = (report['evidence_checks_passed'] and
        len(report.get('comparisons',[]))==2 and all(c['canonical_parity_passed'] for c in report['comparisons']))
    with (output/'independent-numerical-audit.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('evidence_checks_passed','canonical_gpu_parity_passed','counts','wall_seconds')}),flush=True)
    if not report['evidence_checks_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
