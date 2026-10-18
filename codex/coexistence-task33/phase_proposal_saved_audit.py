import json,hashlib,subprocess,os
from pathlib import Path
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp'); raw=Path('/data2/neural-shrimp-data/codex/coexistence'); base=raw/'phase-proposal-001'; rp=raw/'phase-proposal-001-supervision/receipt.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads(rp.read_text()); plan=json.loads((R/'codex/coexistence-task33/phase-proposal-plan.json').read_text()); report=json.loads((base/'report.json').read_text())
assert r['returncode']==0 and r['failure'] is None and r['worker_reaped'] and r['worker_group_session_absent']
for x in r['output_files']:
 assert Path(x['path']).stat().st_size==x['bytes'] and sha(x['path'])==x['sha256']
for x in plan['source_files']:
 rel=str(Path(x['path']).relative_to(R)); blob=subprocess.check_output(['git','show',r['source_commit']+':'+rel],cwd=R)
 assert hashlib.sha256(blob).hexdigest()==x['sha256']==sha(x['path'])
ids={r['worker_pid'],r['supervisor']['pid']}
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  s=(q/'stat').read_text().rsplit(')',1)[1].split()
  assert not ({int(q.name),int(s[2]),int(s[3])}&ids),str(q)
 except (FileNotFoundError,PermissionError):pass
states=np.load(raw/'settle-001/states.npy'); checks=[]
for j,p in enumerate((2,3)):
 z=np.load(base/f'p{p}/preconditioner.npz'); I=np.eye(896); product=I.copy()
 assert set(z.files)=={'A','B','centers','parameters','product','left_inverse_defect','right_inverse_defect'}
 assert np.array_equal(z['centers'],states[-p:,j]);assert np.array_equal(z['parameters'],[.06468,-.148])
 for k in range(p):
  A=np.load(base/f'p{p}/phase-{k}-matrix.npy'); assert np.array_equal(A,z['A'][k]);product=A@product
  q=np.load(base/f'p{p}/phase-{k}-residual.npz');assert np.array_equal(q['center'],z['centers'][k]);assert np.array_equal(q['residual'],q['image']-z['centers'][(k+1)%p])
  assert float(np.linalg.norm(q['residual'])/np.linalg.norm(z['centers'][(k+1)%p]))==report['rows'][j]['phases'][k]['relative_l2_closure']
 assert np.array_equal(product,z['product']); assert np.array_equal(I-z['B']@(I-product),z['left_inverse_defect']);assert np.array_equal(I-(I-product)@z['B'],z['right_inverse_defect'])
 checks.append({'period':p,'all_phase_arrays_and_residuals_exact':True,'product_and_both_saved_defects_exact':True,'all_finite':all(np.isfinite(z[k]).all() for k in z.files)})
out={'source_commit':r['source_commit'],'receipt_sha256':sha(rp),'report_sha256':sha(base/'report.json'),'artifact_count':len(r['output_files']),'source_blobs_checked':len(plan['source_files']),'original_process_groups_absent':True,'session_reaped_exit':0,'external_seconds':r['external_elapsed_seconds'],'peak_child_kib':r['children_ru_maxrss_kib_at_reap'],'checks':checks,'rows':report['rows'],'scope':'Floating phase Jacobians and exact-dyadic inverse proposals; no interval enclosure or rigorous inverse claim.'}
Path('/tmp/task33-phase-completion.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
