"""Independently recompute FGR3 growth summaries from retained per-step logs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, sha, write_json
from audit_fold_return_followup import verify_manifest, load_batch
from audit_fold_global_return import independent_recurrence


def audit(args):
    run, out, backup = map(Path, (args.run, args.out, args.backup))
    out.mkdir(parents=True, exist_ok=False); backup.mkdir(parents=True, exist_ok=False)
    binding = verify_manifest(run)
    source = json.loads((run/'validation-binding.json').read_text())
    import subprocess, hashlib
    blob = subprocess.check_output(['git','show',f"{binding['source_commit']}:tools/fold_return_growth.py"],cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest() == source['growth_source_sha256']
    assert sha(Path(source['audit'])/'audit.json') == source['audit_sha256']
    assert sha(Path(source['audit'])/'rows.json') == source['rows_sha256']
    assert sha(Path(source['raw'])/'manifest.json') == source['manifest_sha256']
    histories, _, _, _ = load_batch(Path(source['raw'])/'trajectories')
    validation_rows = json.loads((Path(source['audit'])/'rows.json').read_text())
    plan = json.loads((run/'plan.json').read_text())
    result = json.loads((run/'result.json').read_text())
    summaries = []
    for window in result['windows']:
        name = window['name']; expected_start = 0; chunks = []; finite = []; relative = []
        primal_bound = 0.
        with np.load(run/(name+'-initial.npz')) as z:
            eligible = z['eligible_updates']
            assert np.max(abs(np.linalg.norm(z['directions'],axis=1)-1))<1e-12
        for path in sorted((run/name).glob('growth-*.npz')):
            _, lo, hi = path.stem.split('-'); lo, hi = int(lo), int(hi)
            assert lo == expected_start
            with np.load(path) as z:
                a = z['log_norm']; e = z['primal_max_abs']
                assert a.shape == e.shape == (hi-lo+1, plan['batch_size'])
                assert np.isfinite(a).all() and np.isfinite(e).all()
                mask = np.arange(lo,hi+1)[:,None] < eligible[None]
                if mask.any(): primal_bound = max(primal_bound,float(np.max(e[mask])))
                chunks.append(a)
                if len(z['offset_log_norm']):
                    finite.append(z['offset_log_norm']); relative.append(z['offset_relative_JVP_error'])
                assert np.array_equal(z['eligible_updates'],eligible)
            expected_start = hi+1
        logs = np.concatenate(chunks); assert len(logs) == max(eligible)
        assert primal_bound < 1e-11
        rows = []; alignment = 129 if window['start'] is None else plan['growth']['alignment_updates']
        length = 384 if window['start'] is None else plan['growth']['window_updates']
        offset = np.concatenate(finite); error = np.concatenate(relative)
        assert offset.shape == error.shape == (min(128,len(logs)),2,plan['batch_size'])
        assert np.isfinite(offset).all() and np.isfinite(error).all()
        for i, stop in enumerate(eligible):
            measured = logs[alignment:stop,i]; half = len(measured)//2
            row = {'row':i,'eligible_updates':int(stop),'measured_updates':len(measured),
                   'mean':float(np.mean(measured)) if len(measured) else None,
                   'first_half':float(np.mean(measured[:half])) if half else None,
                   'second_half':float(np.mean(measured[half:])) if half else None,
                   'full_planned_window':int(stop)==length}
            stored = window['rows'][i]
            for key in ('mean','first_half','second_half','full_planned_window','measured_updates'):
                assert row[key] == stored[key], (name,i,key)
            check_n = min(int(stop),len(offset))
            row['offset_checked_updates'] = check_n
            row['offset_mean_log_growth_difference'] = (np.mean(offset[:check_n,:,i]-logs[:check_n,i,None],axis=0).tolist() if check_n else None)
            row['offset_max_relative_JVP_error'] = np.max(error[:check_n,:,i],axis=0).tolist() if check_n else None
            if window['start'] is not None and stop>=192:
                start = window['start']
                row['final_window_recurrence_candidate'] = independent_recurrence(histories[start:start+stop+1,i])
            else:
                row['final_window_recurrence_candidate'] = None
            rows.append(row)
        summary = {'name':name,'start':window['start'],'primal_max_abs':primal_bound,'rows':rows}
        if window['start'] is None:
            parent = Path(plan['reference_cycle'])
            with np.load(parent/'orbit-and-monodromy.npz') as z:
                expected = float(np.log(max(abs(z['multipliers'])))/3)
            maximum = float(max(abs(r['mean']-expected) for r in rows))
            assert maximum<.005 and all(r['mean']<0 for r in rows)
            summary.update(expected_per_update=expected,max_control_error=maximum,control_passed=True)
        summaries.append(summary)
    data = [r for s in summaries if s['start'] is not None for r in s['rows'] if r['full_planned_window']]
    positive_rows = {r['row'] for r in data if r['first_half']>0 and r['second_half']>0
                     and r['final_window_recurrence_candidate'] is None}
    later_captured = sorted(i for i in positive_rows if validation_rows[i]['outcome']['kind']=='reference_p3')
    result = {'bindings':binding,'windows':summaries,
              'complete_network_windows':len(data),
              'positive_complete_windows':sum(r['mean']>0 for r in data),
              'positive_in_both_halves':sum(r['first_half']>0 and r['second_half']>0 for r in data),
              'distinct_rows_positive_both_halves_without_recurrence':sorted(positive_rows),
              'positive_rows_that_later_match_P3':later_captured,
              'limits':'Finite-time derivative growth at prespecified times, conditional on survival; no invariant-set or infinite-time chaos proof.'}
    write_json(out/'audit.json',result)
    files=[backup_file(p,out,backup) for p in out.iterdir() if p.is_file()]
    write_json(out/'manifest.json',files);backup_file(out/'manifest.json',out,backup)
    print(json.dumps({k:v for k,v in result.items() if k!='windows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','out','backup'):p.add_argument('--'+name,required=True)
    audit(p.parse_args())
