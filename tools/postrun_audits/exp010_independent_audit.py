"""Post-run independent saved-evidence audit; never imports producer/integrator."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path('/data2/neural-shrimp')
RAW = Path('/data2/neural-shrimp-data/raw/EXP-010/attempt-001')
OLD = ROOT / 'evidence/EXP-008/attempt-001'
SOURCE = '98632899646f3eefa292b8a9f4c1b5123ae37ba0'
failures = []
maxima = collections.defaultdict(float)
counts = collections.Counter()

def check(condition, label):
    counts['checks'] += 1
    if not condition:
        failures.append(label)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def binding(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))

def read(path):
    return json.loads(path.read_text())

def close(a, b, label, group):
    if isinstance(a, dict):
        check(isinstance(b, dict) and set(a) == set(b), label + ':keys')
        if isinstance(b, dict):
            for k in set(a) & set(b):
                close(a[k], b[k], f'{label}.{k}', group)
    elif isinstance(a, list):
        check(isinstance(b, list) and len(a) == len(b), label + ':length')
        if isinstance(b, list):
            for i, (x, y) in enumerate(zip(a, b)):
                close(x, y, f'{label}[{i}]', group)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        okay = isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(a) and math.isfinite(b)
        check(okay, label + ':finite numeric')
        if okay:
            maxima[group] = max(maxima[group], abs(a-b))
            check(abs(a-b) <= 1e-14 + 1e-12 * abs(b), label + ':tolerance')
    else:
        check(type(a) is type(b) and a == b, label + ':category')

def interpolate(reference, phase):
    # A separate interpolation implementation, using NumPy's linear interpolator.
    coordinate = np.remainder(phase, 1.) * len(reference)
    xp = np.arange(len(reference) + 1)
    cyclic = np.concatenate([reference, reference[:1]])
    return np.column_stack([np.interp(coordinate, xp, cyclic[:, j]) for j in range(3)])

def rms(error):
    square = np.square(error)
    return dict(aggregate=float(np.sqrt(np.sum(square)/square.size)),
                per_component=np.sqrt(np.sum(square, axis=0)/len(error)).tolist())

def metric_replay(row, arrays, case, oldrow, cert):
    ident = row['id']
    t, state = arrays['times'], arrays['states']
    expected_times = np.arange(751)*.1
    check(np.array_equal(t, expected_times), ident + ':times')
    check(state.shape == (751, 3) and np.isfinite(state).all(), ident + ':state shape/finite')
    with np.load(OLD/case['old_raw'], allow_pickle=False) as old:
        for field in ['initial','delta','alpha','gamma']:
            check(np.array_equal(arrays[field], old[field]), ident + ':' + field)
        check(np.array_equal(t, old['times']), ident + ':original times')
        check(np.array_equal(state[0], old['initial']), ident + ':initial sample')
    check(np.array_equal(arrays['initial'], cert['orbit_states'][case['key'][2]]), ident + ':certificate phase')
    reference = np.asarray(cert['orbit_activations'])[:-1]
    check(reference.shape == (2700, 3) and np.isfinite(reference).all(), ident + ':reference')
    period = cert['period']
    physical = t + case['key'][2] * period / 2700
    phases = physical / period
    activation = np.tanh(state)
    metrics = row['metrics']
    phase = metrics['phase_offset_cycles']
    aligned = activation - interpolate(reference, phases + phase)
    synchronous = activation - interpolate(reference, phases)
    close(metrics['aligned_rms'], rms(aligned), ident + '.aligned_rms', 'metric')
    close(metrics['synchronous_rms'], rms(synchronous), ident + '.synchronous_rms', 'metric')
    close(row['rmse'], rms(aligned)['aggregate'], ident + '.rmse', 'metric')
    fixed_old = activation - interpolate(reference, phases + oldrow['metrics']['phase_offset_cycles'])
    close(row['old_fixed_phase_rmse'], rms(fixed_old)['aggregate'], ident + '.old_fixed_phase_rmse', 'fixed_old_phase')
    number = max(1, math.ceil((physical[-1]-physical[0])/period))
    bins = np.minimum(np.floor((physical-physical[0])/period).astype(int), number-1)
    windows = []
    for i in range(number):
        mask = bins == i
        if mask.any():
            windows.append(dict(index=i, first_sample_time=float(physical[mask][0]),
                last_sample_time=float(physical[mask][-1]), sample_count=int(mask.sum()),
                aligned_rms=rms(aligned[mask]), synchronous_rms=rms(synchronous[mask])))
    close(metrics['windows'], windows, ident + '.windows', 'windows')
    check(metrics['period'] == period and metrics['window_cycles'] == 1., ident + ':period/window')
    check(metrics['phase_search']['coarse_bins'] == 128, ident + ':phase bins')
    check(metrics['sampling']['reference_samples_per_cycle'] == 2700, ident + ':sampling reference')
    counts['metric_traces'] += 1
    counts['metric_windows'] += len(windows)

def wave(a, b):
    result = {}
    for name, x, y in [('state', a, b), ('activation', np.tanh(a), np.tanh(b))]:
        difference = x-y
        result[name+'_rms'] = float(np.sqrt(np.sum(difference*difference)/difference.size))
        result[name+'_max'] = float(np.max(np.abs(difference)))
    result['activation_waveform_agreement'] = result['activation_rms'] <= .001 and result['activation_max'] <= .01
    return result

start = time.monotonic()
receipt = read(RAW/'receipt.json')
plan = read(RAW/'manifest.json')
records = read(RAW/'terminal-records.json')
summary = read(RAW/'summary.json')
check(receipt['source_commit'] == SOURCE, 'source commit')
check(receipt['manifest_sha256'] == digest(RAW/'manifest.json'), 'manifest hash')
check((RAW/'manifest.json').read_bytes() == subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':experiments/hopfield/EXP-010-plan.json']), 'manifest committed bytes')
listed = set()
for item in receipt['artifacts']:
    p = RAW/item['path']
    check(item['path'] not in listed and p.resolve().is_relative_to(RAW), 'artifact unique/contained '+item['path'])
    listed.add(item['path'])
    check(p.is_file(), 'artifact present '+item['path'])
    if p.is_file():
        check(p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], 'artifact binding '+item['path'])
        counts['artifact_bytes'] += item['bytes']
        counts['artifact_files'] += 1
check(listed == {str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file() and p.name != 'receipt.json'}, 'exact output file census')
old_receipt = read(OLD/'receipt.json')
old_bindings = {x['path']:x for x in old_receipt['artifacts']}
check(digest(OLD/'receipt.json') == plan['old_receipt_sha256'], 'original receipt')
for group in ['source_files','source_artifacts']:
    for item in plan[group]:
        p = ROOT/item['path']
        check(p.stat().st_size == item['bytes'] and digest(p) == item['sha256'], group+':'+item['path'])
        if group == 'source_files':
            committed = subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+item['path']])
            check(hashlib.sha256(committed).hexdigest() == item['sha256'] and len(committed) == item['bytes'], 'committed dependency:'+item['path'])
        elif p != OLD/'receipt.json':
            old = old_bindings[str(p.relative_to(OLD))]
            check((old['sha256'],old['bytes']) == (item['sha256'],item['bytes']), 'original receipt membership:'+item['path'])
        counts[group] += 1
rows = [json.loads(line) for role in ['development','test'] for line in (OLD/f'{role}-previews.jsonl').read_text().splitlines()]
key = lambda r: (r['role'],r['anchor'],r['phase'],r['radius'],r['candidate'])
oldrows = {key(r):r for r in rows}
oldplan = read(OLD/'manifest.json')
expected_keys = {(a['role'],a['id'],p,r,c) for a in oldplan['anchors'] for p in oldplan['phases'] for r in oldplan['radii'] for c in range(18)}
check(len(rows) == len(oldrows) == 3456 and set(oldrows) == expected_keys, 'complete original key product')
check(len(plan['census']) == 3456 and {tuple(c['key']) for c in plan['census']} == expected_keys, 'plan complete key product')
sentinels = collections.defaultdict(list)
for role in ['development','test']:
    for radius in oldplan['radii']:
        ordered = sorted([k for k in oldrows if k[0] == role and k[3] == radius], key=lambda k:(oldrows[k]['preview_score'],k))
        check(len(ordered) == 432, 'sentinel stratum')
        high = min(ordered, key=lambda k:(-oldrows[k]['preview_score'],k))
        for label,k in [('maximum',high),('lower_median',ordered[215])]:
            sentinels[k].append(f'{role}-r{radius:g}-{label}')
check(len(sentinels) == 16, 'sentinel count')
expected_tasks = {}
for c in plan['census']:
    old = oldrows[tuple(c['key'])]
    check(c['id'] == old['id'] and c['old_raw'] == old['raw'] and c['old_rmse'] == old['preview_score'] and c['block'] == old['block'], 'case original binding:'+c['id'])
    check(c['independent_labels'] == sentinels.get(tuple(c['key']),[]), 'sentinel original selection:'+c['id'])
    for m in plan['methods']:
        if m['kind'] == 'RK4' or c['independent_labels']:
            expected_tasks[f'{c["id"]}--{m["id"]}'] = (c,m)
check(len(expected_tasks) == 6944, 'expected task count')
indexed = {r['id']:r for r in records}
check(len(records) == len(indexed) == 6944 and set(indexed) == set(expected_tasks), 'terminal task census')
check(collections.Counter(r['status'] for r in records) == receipt['status_counts'], 'receipt statuses')
check(all(r['status'] == 'complete' for r in records), 'all required records complete')
journal = [json.loads(line) for line in (RAW/'journal.jsonl').read_text().splitlines()]
check(journal == records, 'append-only journal terminal equality')
baselines = {r:read(OLD/f'{r}-baselines.json') for r in ['development','test']}
details = {d['id']:d for d in summary['details']}
check(len(details) == len(summary['details']) == 3456, 'summary detail census')
for case in plan['census']:
    arrays = {}
    for method in plan['methods']:
        ident = f'{case["id"]}--{method["id"]}'
        if ident not in expected_tasks: continue
        row = indexed[ident]
        check(row['case_id'] == case['id'] and row['method'] == method['id'], ident+':record identity')
        if row['status'] != 'complete': continue
        check(row['raw'] == f'traces/{ident}.npz' and row['raw'] in listed, ident+':raw identity')
        check(read(RAW/f'records/{ident}.json') == row, ident+':record equality')
        p = RAW/row['raw']
        check(digest(p) == row['raw_sha256'] and p.stat().st_size == row['raw_bytes'], ident+':record raw binding')
        solver = row['solver']
        if method['kind'] == 'RK4':
            check(solver == dict(method='RK4',dt=method['dt'],steps=round(75/method['dt'])), ident+':native solver')
        else:
            check(all(solver[k] == v for k,v in method.items()) and solver['success'] is True and solver['status'] == 0 and solver['nfev'] > 0, ident+':independent solver')
        with np.load(p,allow_pickle=False) as a:
            arrays[method['id']] = {name:a[name].copy() for name in a.files}
        cert = baselines[case['key'][0]][case['key'][1]]['certificate']
        metric_replay(row, arrays[method['id']], case, oldrows[tuple(case['key'])], cert)
    for name, saved in details[case['id']]['waveform_comparisons'].items():
        a,b = name.split('_vs_')
        if a not in arrays or b not in arrays:
            check(saved is None, case['id']+':missing waveform pair')
            continue
        independently = dict(rmse_difference=abs(indexed[f'{case["id"]}--{a}']['rmse']-indexed[f'{case["id"]}--{b}']['rmse']), **wave(arrays[a]['states'],arrays[b]['states']))
        close(saved, independently, case['id']+':'+name, 'waveform')
        counts['waveform_pairs'] += 1
processes = [json.loads(line) for line in (RAW/'processes.jsonl').read_text().splitlines()]
check(len(processes) == 6944 and {p['task_id'] for p in processes} == set(expected_tasks), 'owned process census')
live = []
for p in processes:
    try: os.kill(p['pid'],0)
    except ProcessLookupError: pass
    else: live.append(p['pid'])
report = dict(scope='Independent post-run byte/census/saved-phase metric and waveform replay; no integration or phase reoptimization',
    source_commit=SOURCE, receipt=binding(RAW/'receipt.json'), manifest=binding(RAW/'manifest.json'),
    producer_summary=binding(RAW/'summary.json'), terminal_records=binding(RAW/'terminal-records.json'),
    audit_script=binding(Path(__file__)), counts=dict(counts), maximum_absolute_replay_discrepancy=dict(maxima),
    tolerance=dict(rtol=1e-12,atol=1e-14), owned_process_records=len(processes), pids_currently_existing=sorted(set(live)),
    failures=failures, elapsed_seconds=time.monotonic()-start)
Path('/tmp/exp010-independent-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
if failures: raise SystemExit(1)
