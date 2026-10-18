"""Saved profile evidence audit; no model evaluation."""
from pathlib import Path
import json,hashlib,sys,subprocess
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/certificate-ladder-task34';RAW=Path('/data2/neural-shrimp-data/codex/certificate-ladder/derivative-profile-001');SUP=RAW.with_name(RAW.name+'-supervision')
sys.path.insert(0,str(R/'codex/certify-chaos'));import supervise as s
import torch
torch.set_num_threads(1)
def main():
 rc=json.loads((SUP/'receipt.json').read_text());ad=json.loads((SUP/'admission.json').read_text())
 assert rc['source_commit']==ad['source_commit']=='3f7c5c3084d3fbce69847898178ab3f0910b025c'
 assert rc['returncode']==0 and rc['failure']is None and rc['worker_reaped'] and rc['worker_group_session_absent']
 for e in rc['output_files']:s.verify(e)
 assert {str(p)for p in RAW.iterdir()if p.is_file()}=={e['path']for e in rc['output_files']}
 for e in ad['source_blobs']:
  b=subprocess.check_output(['git','show',ad['source_commit']+':'+e['relative_path']],cwd=R);assert hashlib.sha256(b).hexdigest()==e['sha256']
 for pid in [ad['supervisor']['pid'],rc['worker_pid']]:assert s.ident(pid)is None and not s.members(pid)
 card=json.loads((D/'derivative-profile-001-card.json').read_text());report=json.loads((RAW/'report.json').read_text());assert len(report['rows'])==5;rows=[]
 for c,row in zip(card['cells'],report['rows']):
  assert row['cell']==c and row['period']==c['period'] and row['state_index']==513-c['period'] and row['basis_columns']==list(range(8))
  z=np.load(RAW/(c['name']+'.npz'),allow_pickle=False)
  assert all(np.isfinite(z[k]).all() for k in z.files)
  assert np.array_equal(z['state'],np.load(c['states'],allow_pickle=False)[-c['period']])
  assert np.array_equal(z['basis_rows'],np.eye(896)[:8]) and np.array_equal(z['direction'],np.eye(896)[:8].sum(axis=0)/np.sqrt(8.))
  assert z['columns_as_rows'].shape==(8,896) and z['direct_jvp'].shape==(896,)
  batch=float((torch.from_numpy(z['columns_as_rows']).sum(dim=0)/np.sqrt(8.)-torch.from_numpy(z['direct_jvp'])).abs().max())
  assert batch==row['batch_linear_combination_vs_single_inf']
  assert [f['h']for f in row['fd_ladder']]==[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5]
  for f in row['fd_ladder']:
   h=f['h'];error=float(np.max(np.abs((z['plus_'+str(h)]-z['minus_'+str(h)])/(2*h)-z['direct_jvp'])))
   assert error==f['absolute_inf_error']
  rows.append(dict(name=c['name'],period=c['period'],batch_error=batch,fd_errors=[f['absolute_inf_error']for f in row['fd_ladder']],batch_seconds=row['batch_seconds'],dense_column_forecast_seconds=112*row['batch_seconds']))
 result=dict(source_commit=ad['source_commit'],receipt=s.pin(SUP/'receipt.json'),report=s.pin(RAW/'report.json'),outputs_checked=len(rc['output_files']),source_blobs_checked=len(ad['source_blobs']),host_closure=True,fd_rows_checked=35,batch_errors_checked=5,elapsed_seconds=rc['external_elapsed_seconds'],peak_kib=rc['children_ru_maxrss_kib_at_reap'],rows=rows,certificate_passed=False,scope='Saved finite composed derivative precision/cost diagnostics only; no dense spectrum or stability certificate.')
 (D/'derivative-profile-001-completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
