"""FGR2: fixed-coordinate line scan with repeated returns and full-state capture.

All inference is local patched float64 CPU. Returning rows continue. Only
persistently matched reference cycles, other recurrence candidates, or the
declared horizon stop a row. Immutable shards are verified on two disks.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fold_global_return import backup_file, recurrence, sha, write_json
from fold_panel_data import channel_mask, coordinates, intervals


def reference_match(history, orbit, tolerance=1e-9, positions=128):
    """Check each state and the forward ordering of the reference cycle phases."""
    x, orbit = np.asarray(history), np.asarray(orbit)
    if len(x) < positions or x.ndim != 2 or orbit.ndim != 2 or x.shape[1] != orbit.shape[1]:
        raise ValueError('Matching full-state histories with sufficient positions required')
    if not np.isfinite(x).all() or not np.isfinite(orbit).all():
        raise ValueError('Nonfinite cycle evidence')
    distances = np.linalg.norm(x[-positions:, None] - orbit[None], axis=2)
    distances /= np.maximum(np.linalg.norm(orbit, axis=1), 1e-12)[None]
    phase = distances.argmin(axis=1)
    maximum = float(distances[np.arange(positions), phase].max())
    ordered = bool(np.all(np.diff(phase) % len(orbit) == 1))
    return {'passed': maximum <= tolerance and ordered,
            'max_relative_distance': maximum, 'phase_order_valid': ordered,
            'first_phase': int(phase[0]), 'positions': positions}


def channel_visits(v, transverse, width):
    """Extract persistent phase-aware visits without merging separate excursions."""
    mask = channel_mask(np.asarray(v), np.asarray(transverse), width)
    events = []
    for phase in (0, 1):
        for start, end in intervals(mask[phase::2]):
            if end - start + 1 < 3:
                continue
            lo, hi = 2*start + phase, 2*end + phase
            events.append({'entry': lo, 'last_inside': hi, 'phase': phase,
                           'readings': end-start+1, 'entry_v': float(v[lo]),
                           'entry_transverse': float(transverse[lo]),
                           'left_censored': lo <= 1, 'right_censored': hi >= len(v)-2})
    events.sort(key=lambda x: x['entry'])
    for a, b in zip(events, events[1:]):
        if b['entry'] <= a['last_inside']:
            raise ValueError('Overlapping phase visits need an explicit interpretation')
    return events


def prepare(args):
    prior, cycle, out = map(Path, (args.prior, args.cycle, args.out))
    out.mkdir(parents=True, exist_ok=False)
    old_plan = json.loads((prior/'plan.json').read_text())
    ver = json.loads((cycle/'verification.json').read_text())
    assert ver['run_manifest_sha256'] == sha(prior/'manifest.json')
    assert ver['plan_sha256'] == sha(prior/'plan.json')
    assert ver['status'] == 'numerically attracting primitive period-three orbit'
    assert old_plan['inputs_sha256'] == sha(prior/'inputs.npz')
    with np.load(prior/'inputs.npz') as z:
        old = {k:z[k] for k in z.files}
    with np.load(cycle/'orbit-and-monodromy.npz') as z:
        orbit = z['orbit']
    s = np.linspace(0., 1., 64)
    x0, x1 = old['seeds'][64], old['seeds'][77]
    seeds = x0[None] + s[:, None]*(x1-x0)[None]
    seeds[0], seeds[-1] = x0, x1  # Preserve the exact endpoint bytes.
    v, t = coordinates(seeds, old['centre'], old['left'], old['right'], 1.)
    assert np.ptp(v) < 1e-12 and not channel_mask(v, t, old_plan['width']).any()
    rows = [{'id':i, 's':float(q), 'split':'control' if i in (0,63) else
             ('test' if i % 2 else 'development')} for i,q in enumerate(s)]
    np.savez(out/'inputs.npz', seeds=seeds, s=s, reference_orbit=orbit,
             centre=old['centre'], left=old['left'], right=old['right'],
             replay_states=old['replay_states'], replay_F=old['replay_F'], replay_G=old['replay_G'])
    plan = {k:old_plan[k] for k in ('source','epsilon','width','transverse_limit')}
    plan.update({'name':'FGR2', 'status':'exploratory', 'rows':rows, 'max_updates':4096,
                 'threads':8, 'chunk_updates':128, 'maximum_wall_seconds':14400,
                 'initial_state_rule':'x(s)=FGR1_seed64+s*(FGR1_seed77-FGR1_seed64),64 evenly spaced points including endpoints',
                 'critical_coordinate_range': [float(v.min()),float(v.max())],
                 'prior_path':str(prior), 'prior_manifest_sha256':sha(prior/'manifest.json'),
                 'prior_inputs_sha256':sha(prior/'inputs.npz'), 'cycle_path':str(cycle),
                 'cycle_verification_sha256':sha(cycle/'verification.json'),
                 'cycle_arrays_sha256':sha(cycle/'orbit-and-monodromy.npz'),
                 'inputs_sha256':sha(out/'inputs.npz'),
                 'stop_rule':'Do not stop on return. Stop on 128 consecutive full-state matches to verified P3 at relative1e-9 with correct phase order, other recurrence through64 over128 positions, or4096 updates.',
                 'analysis':'All persistent channel visits; natural exits after >=20 updates in channel; first-return route versus later capture. Endpoint controls excluded from fresh heldout claims. Hold out complete initial-state rows, not individual visits.',
                 'limits':['Finite horizon does not prove permanent noncapture.','A line in state space is not a basin-volume sample.','No chaos claim from recurrence or long lifetime.']})
    write_json(out/'plan.json', plan)
    print(json.dumps({'rows':len(rows),'critical_coordinate_span':float(np.ptp(v)), 'source':str(out)},indent=2))


def run(args):
    from explore_llm_soft_loop import SoftLoop
    import explore_llm_phase_map as phase
    prepared, out, backup = map(Path, (args.prepared, args.out, args.backup))
    plan = json.loads((prepared/'plan.json').read_text())
    assert sha(prepared/'inputs.npz') == plan['inputs_sha256']
    out.mkdir(parents=True, exist_ok=False); backup.mkdir(parents=True, exist_ok=False)
    assert out.stat().st_dev != backup.stat().st_dev
    files = []
    for name in ('inputs.npz','plan.json'):
        shutil.copyfile(prepared/name,out/name)
        files.append(backup_file(out/name,out,backup))
    with np.load(out/'inputs.npz') as z:
        data = {k:z[k] for k in z.files}
    tic = time.monotonic()
    so = plan['source']; n = len(data['seeds'])
    loop = SoftLoop(so['model'],so['layer'],plan['threads'],so['prompt'],so['opener'],so['window'],'float64','cpu')
    torch = loop.torch
    direction = phase.load_vectors(so['vectors'])[so['concept']]
    loop.prepare(np.tile((so['alpha']+plan['epsilon'])*direction,(n,1)))
    T = np.full(n,10.**so['log10_temperature'])
    norms = [m for m in loop.model.modules() if type(m).__name__.endswith('RMSNorm')]
    rotaries = [m for m in loop.model.modules() if type(m).__name__.endswith('RotaryEmbedding')]
    assert loop.dtype == torch.float64 and norms and rotaries
    assert all(m.forward.__func__.__qualname__.startswith('keep_float64_norms.') for m in norms)
    assert all(m._inv_freq64.dtype == torch.float64 for m in rotaries)
    meta = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'runtime_sources':{p:sha(ROOT/'tools'/p) for p in ('fold_return_line.py','fold_global_return.py','fold_panel_data.py','explore_llm_soft_loop.py','explore_llm_attractors.py','explore_llm_phase_map.py')},
            'plan_sha256':sha(out/'plan.json'),'model_revision':loop.sm.engine.revision,
            'prefix_text':loop.prefix_text,'prefix_ids':loop.prefix_ids.cpu().tolist(),
            'direction_sha256':sha(phase.DATA/'explore'/so['vectors']/'vectors.npz'),
            'device':'cpu','dtype':str(loop.dtype),'threads':torch.get_num_threads(),
            'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'pid':os.getpid(),'backup':str(backup),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(out/'metadata.json',meta);files.append(backup_file(out/'metadata.json',out,backup))
    check = torch.tensor(np.tile(data['replay_states'],(n//8,1))[:,None],dtype=torch.float64)
    with torch.no_grad():
        y1,_ = loop.step(check,T); y2,_ = loop.step(y1,T)
    errors = [float(np.max(np.abs(y.numpy()[:,0]-np.tile(data[k],(n//8,1))))) for y,k in [(y1,'replay_F'),(y2,'replay_G')]]
    gate = {'one_update_max_abs':errors[0],'two_update_max_abs':errors[1],'tolerance':1e-11,'passed':max(errors)<1e-11}
    write_json(out/'replay-gate.json',gate);files.append(backup_file(out/'replay-gate.json',out,backup))
    assert gate['passed'], gate
    X = torch.tensor(data['seeds'][:,None],dtype=torch.float64)
    rolling = [data['seeds'].copy()]
    outcomes = [None]*n; valid = np.zeros(n,dtype=int)
    states, ids, probabilities = [],[],[]
    model_start = time.monotonic()
    def flush(step):
        lo = step-len(states)+1;path=out/f'updates-{lo:05d}-{step:05d}.npz'
        # A failed backup can revisit this flush; never rewrite its committed bytes.
        if not path.exists():
            with open(str(path)+'.partial','wb') as f:
                np.savez(f,states=np.array(states),top_ids=np.array(ids),top_probabilities=np.array(probabilities),valid_through=valid.copy())
                f.flush();os.fsync(f.fileno())
            os.replace(str(path)+'.partial',path)
        receipt=backup_file(path,out,backup)
        if receipt not in files:files.append(receipt)
        write_json(out/'retrieval.json',files)
        states.clear();ids.clear();probabilities.clear()
        status={'step':step,'p3_matched':sum(o is not None and o['kind']=='reference_p3' for o in outcomes),
                'other_recurrence':sum(o is not None and o['kind']=='other_recurrence_candidate' for o in outcomes),
                'active':sum(o is None for o in outcomes),'model_seconds':time.monotonic()-model_start,
                'backed_up_shards':sum(f['path'].startswith('updates-') for f in files)}
        write_json(out/'status.json',status);print(json.dumps(status),flush=True)
    try:
        for step in range(1,plan['max_updates']+1):
            active=np.array([o is None for o in outcomes])
            with torch.no_grad():
                nxt,p=loop.step(X,T);tp,ti=torch.topk(p,8,dim=1)
                X=torch.where(torch.as_tensor(active)[:,None,None],nxt,X)
            arr=X.numpy()[:,0].copy()
            if not np.isfinite(arr).all():raise ValueError('Nonfinite state')
            valid[active]=step;rolling.append(arr)
            if len(rolling)>193:rolling.pop(0)
            states.append(arr);ids.append(ti.numpy().copy());probabilities.append(tp.numpy().copy())
            if step>=256 and step%128==0:
                hist=np.array(rolling)
                for i in np.flatnonzero(active):
                    match=reference_match(hist[:,i],data['reference_orbit'])
                    if match['passed']:
                        outcomes[i]={'kind':'reference_p3','detected_update':step,**match}
                    else:
                        rec=recurrence(hist[:,i])
                        if rec:outcomes[i]={'kind':'other_recurrence_candidate','detected_update':step,**rec}
            finished=all(o is not None for o in outcomes)
            timeout=time.monotonic()-tic>=plan['maximum_wall_seconds']
            if step%plan['chunk_updates']==0 or finished or timeout or step==plan['max_updates']:
                flush(step)
            if finished or timeout:break
        outcomes=[o if o is not None else {'kind':'unresolved_at_horizon' if int(valid[i])==plan['max_updates'] else 'wall_time_limit','updates':int(valid[i])} for i,o in enumerate(outcomes)]
        write_json(out/'result.json',{'outcomes':outcomes,'valid_through':valid.tolist(),'wall_seconds':time.monotonic()-tic,'model_seconds':time.monotonic()-model_start})
        files.append(backup_file(out/'result.json',out,backup))
        write_json(out/'manifest.json',files);backup_file(out/'manifest.json',out,backup)
        write_json(out/'complete.json',{'manifest_sha256':sha(out/'manifest.json'),'backup_manifest_sha256':sha(backup/'manifest.json'),'finished':True})
        backup_file(out/'complete.json',out,backup)
    except BaseException as exc:
        # Preserve the current prefix if possible, then fail loudly.
        if states:
            try:flush(step)
            except Exception:pass
        write_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'step':int(valid.max())})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for name in ('prior','cycle','out'):a.add_argument('--'+name,required=True)
    a=sub.add_parser('run')
    for name in ('prepared','out','backup'):a.add_argument('--'+name,required=True)
    args=p.parse_args();{'prepare':prepare,'run':run}[args.command](args)
