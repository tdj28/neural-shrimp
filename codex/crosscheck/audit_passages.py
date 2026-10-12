"""Read-only independent audit. No model or repository modules are imported."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from scipy.integrate import quad
root=Path('/data2/neural-shrimp-data/codex/task2-independent-crosscheck/passages-001')
refroot=Path('/data2/neural-shrimp-data/explore/llm-soft-loop')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
source_checks=[]
for expected in manifest['source']:
    path=root/'source'/expected['path']
    got=sha(path)
    source_checks.append(dict(path=str(path),sha256=got,passed=got==expected['sha256'] and path.stat().st_size==expected['bytes']))
ref_ids=json.loads((root/'source/codex/crosscheck/reference-inputs.json').read_text())
refchecks=[]
for expected in ref_ids:
    p=Path(expected['path'])
    if p.parent.name in ('normal-form-p2-t-0.115','passage-p2-t-0.115'):
        got=sha(p)
        refchecks.append(dict(path=str(p),sha256=got,passed=got==expected['sha256'] and p.stat().st_size==expected['bytes']))
nf=json.loads((refroot/'normal-form-p2-t-0.115/normal-form.json').read_text())
old=json.loads((refroot/'passage-p2-t-0.115/passage.json').read_text())
old_u=np.load(refroot/'passage-p2-t-0.115/u.npy',allow_pickle=False)
vec=np.load(refroot/'normal-form-p2-t-0.115/vectors.npz',allow_pickle=False)
U=old['U'];a=abs(nf['a']);b=abs(nf['b']);rows=[]; group_checks=[]
for groupfile in sorted((root/'run').glob('passage-group-*.json')):
    meta=json.loads(groupfile.read_text());npz=np.load(groupfile.with_suffix('.npz'),allow_pickle=False)
    cur=npz[meta['u']['array']]; transverse=npz[meta['transverse']['array']]
    groupchecks=dict(index=meta['index'],reported_steps=meta['G_evaluations'],trace_shape=list(cur.shape),
                     shape_matches=cur.shape==transverse.shape==(meta['G_evaluations'],3),
                     physical_alpha_matches=meta['alpha']==nf['source']['alpha']+meta['eps'],
                     expected_group=meta['eps']==old['args']['epsilons'][meta['index']])
    for j,r in enumerate(meta['rows']):
        col=r['column'];expected=old['rows'][col];series=cur[:,j]
        hits=np.flatnonzero(series>U);count=int(hits[0]+1) if len(hits) else None
        oldhits=np.flatnonzero(old_u[:,col]>U);oldcount=int(oldhits[0]+1) if len(oldhits) else None
        shared=min(count-1,oldcount-1) if count is not None and oldcount is not None else 0
        err=series[:shared]-old_u[:shared,col]
        entrance=0. if r['entrance']=='midpoint' else -U
        q=entrance
        for t in range(1,100001):
            q+=a*r['eps']+b*q*q
            if q>U:
                predicted=t;break
        else: predicted=None
        continuous,quadrature_error=quad(lambda u:1/(a*r['eps']+b*u*u),entrance,U,epsabs=1e-12,epsrel=1e-12)
        end=min(count,oldcount)
        through=np.max(np.abs(series[:end]-old_u[:end,col]))
        row=dict(column=col,eps=r['eps'],entrance=r['entrance'],measured=count,reference=oldcount,
                 count_difference=count-oldcount,discrete_prediction=predicted,measured_minus_prediction=count-predicted,
                 continuous_integral=continuous,measured_minus_integral=count-continuous,
                 shared_pre_exit=shared,max_absolute_pre_exit_error=float(np.max(np.abs(err))) if shared else 0.,
                 rms_pre_exit_error=float(np.sqrt(np.mean(err*err))) if shared else 0.,
                 max_error_through_first_exit=float(through),
                 previous_u=float(series[count-2]) if count>1 else entrance,
                 crossing_u=float(series[count-1]),
                 crossing_margin=float(series[count-1]-U),
                 pre_exit_margin=float(U-(series[count-2] if count>1 else entrance)),
                 largest_transverse_before_exit=float(np.max(transverse[:count,j])),
                 metadata_counts_match=count==r['measured_G']==expected['measured_G']==oldcount,
                 row_identity_matches=r['eps']==expected['eps'] and r['entrance']==expected['entrance'],
                 first_exit_strict=bool(count and np.all(series[:count-1]<=U) and series[count-1]>U),
                 no_post_exit_values=bool(np.isnan(series[count:]).all()),
                 retained_values_finite=bool(np.isfinite(series[:count]).all() and np.isfinite(transverse[:count,j]).all()),
                 quadratic_count_matches=predicted==r['predicted_discrete_G']==expected['predicted_discrete_G'],
                 continuous_matches=bool(abs(continuous-r['predicted_arctan_G'])<1e-10 and abs(continuous-expected['predicted_arctan_G'])<1e-10))
        row['passed']=all(row[k] for k in ['metadata_counts_match','row_identity_matches','first_exit_strict','no_post_exit_values','retained_values_finite','quadratic_count_matches','continuous_matches'])
        rows.append(row)
    groupchecks['stopping_rule_matches']=meta['G_evaluations']==max(r['measured_G'] for r in meta['rows'])
    groupchecks['passed']=all(groupchecks[k] for k in ['shape_matches','physical_alpha_matches','expected_group','stopping_rule_matches'])
    group_checks.append(groupchecks)
report=dict(source_commit=manifest['source_commit'],manifest_sha256=sha(root/'manifest.json'),
            source_checks=source_checks,reference_checks=refchecks,group_checks=group_checks,rows=rows,
            all18_present=len(rows)==18,all_checks_pass=all(x['passed'] for x in source_checks+refchecks+group_checks+rows),
            status=json.loads((root/'run/status.json').read_text()) if (root/'run/status.json').exists() else None,
            scope='Read-only audit of saved trajectories and recorded run source; no model execution, no fresh holdout, no global bottleneck or shrimp theorem.',
            coefficient_scope='Predictions use saved a,b,alpha_c,modes,U; independent coefficient recomputation is a separate run.')
if (root/'run/passages-summary.json').exists():
    summary=json.loads((root/'run/passages-summary.json').read_text())
    comparisons=summary['reference_comparison']
    for r,c in zip(rows,comparisons):
        if r['column']!=c['column'] or abs(r['max_absolute_pre_exit_error']-c['maximum_absolute_u_error'])>1e-18 or r['count_difference']!=c['count_difference']:
            raise RuntimeError('reported summary disagreement')
    report['summary_matches_independent_audit']=len(comparisons)==len(rows)
if (root/'receipt.json').exists():
    receipt=json.loads((root/'receipt.json').read_text())
    checks=[]
    for expected in receipt['artifacts']:
        path=root/expected['path'];got=sha(path)
        checks.append(dict(path=expected['path'],sha256=got,passed=got==expected['sha256'] and path.stat().st_size==expected['bytes']))
    expected_paths={r['path'] for r in receipt['artifacts']}
    actual_paths={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='receipt.json'}
    report.update(receipt_sha256=sha(root/'receipt.json'),receipt_checks=checks,
                  artifact_inventory_exact=expected_paths==actual_paths,receipt_exit_code=receipt['exit_code'],
                  receipt_timed_out=receipt['timed_out'])
    report['all_checks_pass'] &= all(r['passed'] for r in checks) and expected_paths==actual_paths
if (root/'run/passages-result.json').exists():
    result=json.loads((root/'run/passages-result.json').read_text())
    result_npz=np.load(root/'run/passages-result.npz',allow_pickle=False)
    fixed=result['fixed_inputs']
    def arr(ref): return result_npz[ref['array']]
    xc,rr,ll=vec['x_c'],-vec['r'],-vec['l']
    random=np.random.default_rng(0).normal(size=len(xc))
    random-=float(ll@random)/float(ll@rr)*rr; random/=np.linalg.norm(random)
    starts=np.stack([xc-U*rr,xc,xc-U*rr+U*random])
    fixed_checks=dict(center=np.array_equal(arr(fixed['x_c']),xc),
                      oriented_right=np.array_equal(arr(fixed['right']),rr),
                      oriented_left=np.array_equal(arr(fixed['left']),ll),
                      transverse=np.array_equal(arr(fixed['transverse']),random),
                      entrances=np.array_equal(arr(fixed['states']),starts),
                      width=fixed['U']==U,parameter_coefficient=fixed['a']==a,
                      quadratic_coefficient=fixed['b']==b)
    combined=arr(result['u'])
    for groupfile in sorted((root/'run').glob('passage-group-*.json')):
        group=json.loads(groupfile.read_text()); z=np.load(groupfile.with_suffix('.npz'),allow_pickle=False)
        j=group['index'];actual=z[group['u']['array']]
        fixed_checks['combined_group_'+str(j)]=np.array_equal(combined[:len(actual),j*3:j*3+3],actual,equal_nan=True)
    report['fixed_inputs_and_combined_traces_checks']=fixed_checks
    report['all_checks_pass'] &= all(fixed_checks.values())
Path('/tmp/codex-passages-independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('source_checks','reference_checks','rows','group_checks','receipt_checks')},indent=2))
for r in rows:
    print(f"{r['column']:2d} eps={r['eps']:.6g} {r['entrance']:16s} actual/reference={r['measured']}/{r['reference']} quad={r['discrete_prediction']} pre_err={r['max_absolute_pre_exit_error']:.3g} exit_err={r['max_error_through_first_exit']:.3g} passed={r['passed']}")
