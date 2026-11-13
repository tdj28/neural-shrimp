"""CPU-only independent full-history and hash audit for FGR3."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, sha, write_json
from audit_fold_return_line import visits_from_mask, capture_trace
from audit_fold_global_return import independent_recurrence
from fold_return_surrogate import predict, rollout


def departures(history, centre, left, right, width, transverse_limit=.05):
    delta = history-centre
    v = delta@left
    transverse = np.linalg.norm(delta-v[:, None]*right, axis=1)
    visits = visits_from_mask((abs(v)<width)&(transverse<transverse_limit))
    events = []
    for j, visit in enumerate(visits):
        if visit['right_censored'] or visit['last_inside']-visit['entry']+2<20:
            continue
        next_entry = visits[j+1]['entry'] if j+1<len(visits) else len(history)
        candidates = np.arange(visit['last_inside']+2, min(next_entry, visit['last_inside']+34), 2)
        candidates = candidates[(v[candidates]>=3*width)&(transverse[candidates]<transverse_limit)]
        if len(candidates):
            t = int(candidates[0])
            events.append({'entry': visit['entry'], 'last_inside': visit['last_inside'],
                           'exit': t, 'v_over_U': float(v[t]/width),
                           'transverse': float(transverse[t])})
    return events, visits


def verify_manifest(run):
    run = Path(run); metadata = json.loads((run/'metadata.json').read_text())
    backup = Path(metadata['backup'])
    manifest = json.loads((run/'manifest.json').read_text())
    complete = json.loads((run/'complete.json').read_text())
    assert sha(run/'manifest.json') == complete['manifest_sha256'] == sha(backup/'manifest.json')
    total = 0
    for entry in manifest:
        for root in (run, backup):
            p = root/entry['path']
            assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], str(p)
        total += entry['bytes']
    assert metadata['plan_sha256'] == sha(run/'plan.json')
    assert json.loads((run/'precision-gate.json').read_text())['passed']
    import subprocess, hashlib
    for name, expected in metadata['runtime_sources'].items():
        content = subprocess.check_output(['git', 'show', f"{metadata['source_commit']}:tools/{name}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected
    for name, expected in json.loads((run/'parent-bindings.json').read_text()).items():
        assert sha(name) == expected
    plan = json.loads((run/'plan.json').read_text())
    committed_plan = json.loads(subprocess.check_output(
        ['git','show',f"{metadata['source_commit']}:docs/fold-return-autonomous-plan.json"], cwd=ROOT))
    assert plan == committed_plan
    parent = Path(plan['parent_curve'])
    parent_metadata = json.loads((parent/'metadata.json').read_text())
    for key in ('model_revision','prefix_ids','direction_sha256'):
        assert metadata[key] == parent_metadata[key], key
    with np.load(parent/'inputs.npz') as original, np.load(run/'inputs.npz') as copied:
        assert original.files == copied.files
        for key in original.files: assert np.array_equal(original[key],copied[key]),key
    return {'files': len(manifest), 'bytes_per_disk': total, 'manifest_sha256': sha(run/'manifest.json'),
            'backup': str(backup), 'source_commit': metadata['source_commit'],
            'plan_sha256': sha(run/'plan.json')}


def load_batch(batch, B=16):
    with np.load(batch/'initial.npz') as z:
        initial = z['states']; n = len(z['active_rows'])
    result = json.loads((batch/'result.json').read_text())
    arrays = [initial[None]]; expected = 1
    for path in sorted(batch.glob('updates-*.npz')):
        _, lo, hi = path.stem.split('-'); lo, hi = int(lo), int(hi)
        assert lo == expected
        with np.load(path) as z:
            x = z['states']; assert x.shape == (hi-lo+1, B, 896)
            assert np.isfinite(x).all() and np.isfinite(z['top_probabilities']).all()
            assert z['top_ids'].shape == z['top_probabilities'].shape == (len(x), B, 4)
            arrays.append(x)
        expected = hi+1
    history = np.concatenate(arrays)
    valid = np.array(result['valid_through'])
    assert len(valid) == B and all(valid[n:] == 0)
    for i in range(B):
        assert np.array_equal(history[valid[i]:, i], np.broadcast_to(history[valid[i], i], history[valid[i]:, i].shape))
    return history, result, valid, n


def score_route(predictions, outcomes):
    pairs = [(p, a) for p, a in zip(predictions, outcomes)
             if a['kind'] in ('return', 'reference_p3')]
    tp = sum(p['kind']==a['kind']=='reference_p3' for p, a in pairs)
    fp = sum(p['kind']=='reference_p3' and a['kind']=='return' for p, a in pairs)
    fn = sum(p['kind']!='reference_p3' and a['kind']=='reference_p3' for p, a in pairs)
    correct = sum(p['kind']==a['kind'] for p, a in pairs)
    return {'evaluated': len(pairs), 'correct': correct, 'capture_true_positive': tp,
            'capture_false_positive': fp, 'capture_false_negative': fn,
            'capture_precision': tp/(tp+fp) if tp+fp else None,
            'capture_recall': tp/(tp+fn) if tp+fn else None,
            'always_return_correct': sum(a['kind']=='return' for _, a in pairs)}


def audit(args):
    run, out, backup = map(Path, (args.run, args.out, args.backup))
    out.mkdir(parents=True, exist_ok=False); backup.mkdir(parents=True, exist_ok=False)
    binding = verify_manifest(run)
    plan = json.loads((run/'plan.json').read_text())
    parent = json.loads((Path(plan['parent_curve'])/'plan.json').read_text())
    with np.load(run/'inputs.npz') as z: data = {k: z[k] for k in z.files}
    rows = []; events_by_row = []; fresh = (run/'trajectories').exists()
    batches = [run/'trajectories'] if fresh else sorted(run.glob('round-*-batch-*'))
    for batch in batches:
        x, result, valid, n = load_batch(batch)
        for i in range(n):
            h = x[:valid[i]+1, i]
            events, visits = departures(h, data['centre'], data['left'], data['right'], parent['width'], parent['transverse_limit'])
            observed = result['events'][i]
            assert [e['exit'] for e in events] == [e['exit'] for e in observed], (batch.name, i)
            assert all(abs(a['v_over_U']-b['v_over_U'])<1e-10 for a,b in zip(events, observed))
            outcome = result['outcomes'][i]
            capture = capture_trace(h, data['reference_orbit']) if len(h)>=128 else {'first_128_match_start': None, 'insufficient_history': True}
            if outcome['kind'] == 'return':
                assert not fresh and events[0]['exit'] == valid[i]
            elif outcome['kind'] == 'reference_p3':
                assert capture['first_128_match_start'] is not None
            elif outcome['kind'] == 'other_recurrence_candidate':
                assert independent_recurrence(h)['lag'] == outcome['lag']
            else:
                assert valid[i] == (plan['validation'] if fresh else plan['training'])['max_updates']
            row = {'batch': batch.name, 'row': i, 'valid_through': int(valid[i]),
                   'outcome': outcome, 'capture': capture, 'visits': len(visits), 'departures': len(events)}
            rows.append(row); events_by_row.append(events)
        print(json.dumps({'audited_batch': batch.name, 'rows': len(rows)}), flush=True)
    summary = {'bindings': binding, 'rows': len(rows), 'all_saved_states_finite': True,
               'independent_departures_agree': True,
               'outcomes': {k: sum(r['outcome']['kind']==k for r in rows) for k in sorted({r['outcome']['kind'] for r in rows})}}
    if fresh:
        model = json.loads((run/'frozen-model.json').read_text())
        predictions = []; actual = []; errors = []; autonomous = []; first_predictions = []; first_actual = []
        for row, events in zip(rows, events_by_row):
            forecasts = {method: rollout(model, events[0]['v_over_U'], 128, method) for method in ('piecewise','nearest')} if events else {}
            autonomous.append({'row': row['row'], 'forecasts': forecasts, 'observed_events': events})
            for j, event in enumerate(events):
                p = predict(model, event['v_over_U'])
                if j+1<len(events):
                    a = {'kind': 'return', 'updates': events[j+1]['exit']-event['exit'],
                         'next_q': events[j+1]['v_over_U']}
                    if p['kind']=='return':
                        errors.append({'row': row['row'], 'position': p['next_q']-a['next_q'],
                                       'time': p['updates']-a['updates'], 'resolved': p['branch_resolved']})
                elif row['outcome']['kind']=='reference_p3':
                    a = {'kind': 'reference_p3'}
                else: continue
                predictions.append(p); actual.append(a)
                if j == 0:
                    first_predictions.append(p); first_actual.append(a)
        summary['one_cycle_route'] = score_route(predictions, actual)
        summary['first_departure_route'] = score_route(first_predictions, first_actual)
        equal_row_mse = [np.mean([r['position']**2 for r in errors if r['row']==i])
                         for i in sorted({r['row'] for r in errors})]
        summary['one_cycle_return_errors'] = {
            'n': len(errors), 'coordinate_RMSE_U': float(np.sqrt(np.mean([r['position']**2 for r in errors]))) if errors else None,
            'equal_trajectory_coordinate_RMSE_U': float(np.sqrt(np.mean(equal_row_mse))) if errors else None,
            'time_MAE_updates': float(np.mean([abs(r['time']) for r in errors])) if errors else None,
            'mixed_branch_predictions': sum(not r['resolved'] for r in errors)}
        summary['autonomous'] = {}
        for method in ('piecewise','nearest'):
            horizons = []
            for horizon in (1,2,4,8):
                position = []; timing = []; missed_return = 0; eligible_rows = 0
                for a in autonomous:
                    e = a['observed_events']; forecast = a['forecasts'].get(method,[])
                    if len(e) <= horizon: continue
                    eligible_rows += 1
                    if len(forecast)<horizon or forecast[horizon-1]['kind']!='return':
                        missed_return += 1; continue
                    position.append(forecast[horizon-1]['next_q']-e[horizon]['v_over_U'])
                    timing.append(forecast[horizon-1]['elapsed']-(e[horizon]['exit']-e[0]['exit']))
                horizons.append({'cycles':horizon,'observed_returning_rows':eligible_rows,
                                 'predicted_returning_rows':len(position),'premature_terminal_or_abstention':missed_return,
                                 'coordinate_RMSE_U':float(np.sqrt(np.mean(np.square(position)))) if position else None,
                                 'elapsed_MAE_updates':float(np.mean(np.abs(timing))) if timing else None})
            terminal = []
            for a in autonomous:
                e = a['observed_events']; forecast = a['forecasts'].get(method,[])
                row = rows[a['row']]
                if not e: continue
                captured = row['outcome']['kind']=='reference_p3'
                predicted = bool(forecast and forecast[-1]['kind']=='reference_p3')
                terminal.append({'row':a['row'],'observed_P3':captured,'predicted_P3':predicted,
                                 'observed_cycles_to_P3':len(e) if captured else None,
                                 'predicted_cycles_to_P3':len(forecast) if predicted else None,
                                 'observed_horizon':row['valid_through'],
                                 'predicted_horizon_cycles':len(forecast),
                                 'note':'Different finite horizons; terminal agreement is descriptive, not an uncensored fate accuracy.'})
            summary['autonomous'][method] = {'horizons':horizons,'terminal_comparison':terminal}
        summary['survival'] = [{'updates': t, 'not_yet_p3': sum(r['capture']['first_128_match_start'] is None or r['capture']['first_128_match_start']>t for r in rows),
                                'total': len(rows)} for t in plan['validation']['survival_horizons']]
        write_json(out/'autonomous.json', autonomous)
        write_json(out/'one-cycle-errors.json', errors)
    else:
        model = json.loads((run/'model.json').read_text())
        assert len(model['rows']) == len(rows) == plan['training']['maximum_points']
        summary['midpoint_development_checks'] = []
        for round_id in range(1, plan['training']['refinement_rounds']+1):
            previous = json.loads((run/f'round-{round_id-1:02d}-model.json').read_text())
            new = [r for r in model['rows'] if r['batch'].startswith(f'round-{round_id:02d}-')]
            predicted = [predict(previous, r['q']) for r in new]
            summary['midpoint_development_checks'].append({'round': round_id, **score_route(predicted, new)})
    write_json(out/'rows.json', rows); write_json(out/'events.json', events_by_row)
    write_json(out/'audit.json', summary)
    files = [backup_file(p, out, backup) for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out/'manifest.json', files); backup_file(out/'manifest.json', out, backup)
    print(json.dumps(summary, indent=2))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','out','backup'): p.add_argument('--'+name, required=True)
    audit(p.parse_args())
