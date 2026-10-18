"""Independent saved phase and inverse-proposal audit, no model calls."""
from pathlib import Path
import json,hashlib,sys,subprocess
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/certificate-ladder-task34';raw=Path('/data2/neural-shrimp-data/codex/certificate-ladder/phase-cascade-p3-001');sup=raw.with_name(raw.name+'-supervision')
sys.path.insert(0,str(R/'codex/certify-chaos'));import supervise as s
rc=json.loads((sup/'receipt.json').read_text());ad=json.loads((sup/'admission.json').read_text());assert rc['source_commit']==ad['source_commit']=='32c7e4126cd88c6e0c2030dcf1d287f356ee2db8'
assert rc['returncode']==0 and rc['failure']is None and rc['worker_reaped'] and rc['worker_group_session_absent']
for e in rc['output_files']:s.verify(e)
assert {str(p)for p in raw.rglob('*')if p.is_file()}=={e['path']for e in rc['output_files']}
for e in ad['source_blobs']:
 b=subprocess.check_output(['git','show',ad['source_commit']+':'+e['relative_path']],cwd=R);assert hashlib.sha256(b).hexdigest()==e['sha256']
for pid in [ad['supervisor']['pid'],rc['worker_pid']]:assert s.ident(pid)is None and not s.members(pid)
report=json.loads((raw/'report.json').read_text());assert len(report['rows'])==1;r=report['rows'][0];c=r['cell'];p=r['period'];assert c['name']=='cascade-p3' and p==3
folder=raw/c['name'];z=np.load(folder/'preconditioner.npz',allow_pickle=False);assert set(z.files)=={'A','B','centers','parameters','product','left_inverse_defect','right_inverse_defect'}
assert all(np.isfinite(z[k]).all() for k in z.files) and z['A'].shape==(3,896,896)
assert np.array_equal(z['centers'],np.load(c['states'],allow_pickle=False)[-p:]) and np.array_equal(z['parameters'],[c['alpha'],c['log10T']])
product=np.eye(896)
for i,phase in enumerate(r['phases']):
 a=np.load(folder/f'phase-{i}-matrix.npy',allow_pickle=False);assert np.array_equal(a,z['A'][i]);product=a@product
 v=np.load(folder/f'phase-{i}-residual.npz',allow_pickle=False);assert np.array_equal(v['center'],z['centers'][i]) and np.array_equal(v['residual'],v['image']-z['centers'][(i+1)%p])
 assert float(np.linalg.norm(v['residual'])/np.linalg.norm(z['centers'][(i+1)%p]))==phase['relative_l2_closure']<1e-10
assert np.array_equal(product,z['product'])
den=np.eye(896)-product;left=np.eye(896)-z['B']@den;right=np.eye(896)-den@z['B']
assert np.array_equal(left,z['left_inverse_defect']) and np.array_equal(right,z['right_inverse_defect'])
for k,a in [('left_inverse_defect_inf',left),('right_inverse_defect_inf',right),('product_norm_inf',product),('inverse_norm_inf',z['B'])]:assert float(np.linalg.norm(a,ord=np.inf))==r['diagnostics'][k]
assert float(np.linalg.cond(den))==r['diagnostics']['condition_2_estimate']
result=dict(receipt=s.pin(sup/'receipt.json'),source_commit=ad['source_commit'],source_blobs_checked=len(ad['source_blobs']),outputs_checked=len(rc['output_files']),host_closure=True,phases_checked=p,seven_proposal_arrays_checked=True,scientific_row=r,elapsed_seconds=rc['external_elapsed_seconds'],certificate_passed=False,scope='Every saved phase residual, ordered product, inverse-defect array and diagnostic checked; no model evaluation, rigorous inverse or derivative proof.')
(D/'phase-cascade-p3-001-completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(elapsed=result['elapsed_seconds'],phases=p,outputs=result['outputs_checked'])))
