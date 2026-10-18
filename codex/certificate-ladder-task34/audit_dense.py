"""Independent saved dense numerical audit, without model evaluation."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
sys.path.insert(0,str(R/'codex/certify-chaos'));import supervise as s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);ap.add_argument('--source',required=True);ap.add_argument('--cell',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 raw=Path('/data2/neural-shrimp-data/codex/certificate-ladder')/a.attempt;sup=raw.with_name(raw.name+'-supervision')
 rc=json.loads((sup/'receipt.json').read_text());ad=json.loads((sup/'admission.json').read_text())
 assert rc['source_commit']==ad['source_commit']==a.source
 assert rc['returncode']==0 and rc['failure']is None and rc['worker_reaped'] and rc['worker_group_session_absent']
 for e in rc['output_files']:s.verify(e)
 assert {str(p)for p in raw.rglob('*')if p.is_file()}=={e['path']for e in rc['output_files']}
 for e in ad['source_blobs']:
  b=subprocess.check_output(['git','show',a.source+':'+e['relative_path']],cwd=R);assert hashlib.sha256(b).hexdigest()==e['sha256']
 for pid in [ad['supervisor']['pid'],rc['worker_pid']]:assert s.ident(pid)is None and not s.members(pid)
 report=json.loads((raw/'report.json').read_text());assert len(report['rows'])==1;row=report['rows'][0];folder=raw/a.cell
 assert row['cell']['name']==a.cell and row==json.loads((folder/'summary.json').read_text()) and row['columns']==896
 p=row['period'];assert row['state_index']==513-p
 history=np.load(row['cell']['states'],allow_pickle=False);assert history.shape==(513,896)
 m=np.load(folder/'monodromy.npy',allow_pickle=False)
 blocks=[np.load(folder/f'columns-{k:03d}.npy',allow_pickle=False)for k in range(0,896,8)]
 assert all(b.shape==(8,896) and np.isfinite(b).all()for b in blocks) and np.array_equal(m,np.concatenate(blocks).T)
 z=np.load(folder/'eigensystem.npz',allow_pickle=False);v=z['values'];U=z['vectors']
 assert v.shape==(896,) and U.shape==(896,896) and np.isfinite(v).all() and np.isfinite(U).all() and np.all(np.abs(v)[:-1]>=np.abs(v)[1:])
 residuals=np.max(np.abs(m@U-U*v[None,:]),axis=0)
 assert np.array_equal(residuals,z['residuals']) and float(residuals.max())==row['maximum_eigen_residual_inf']
 assert float(abs(v[0]))==row['spectral_radius'] and [float(v[0].real),float(v[0].imag)]==row['critical_multiplier'] and float(abs(v[1]))==row['next_modulus']
 assert row['numerically_inside_unit_circle']==bool(np.max(np.abs(v))<1)
 fdcount=0
 for check in row['critical_checks']:
  part=check['part'];direction=getattr(U[:,0],part);norm=float(np.linalg.norm(direction))
  if check['zero_component']:assert norm==0;continue
  z=np.load(folder/('critical-'+part+'.npz'),allow_pickle=False)
  assert all(np.isfinite(z[k]).all()for k in z.files) and np.array_equal(z['direction'],direction/norm)
  assert np.array_equal(z['dense'],m@z['direction'])
  assert float(np.max(np.abs(z['dense']-z['direct'])))==check['dense_vs_direct_inf']
  assert [f['h']for f in check['fd_ladder']]==[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5]
  for f in check['fd_ladder']:
   h=f['h'];err=float(np.max(np.abs((z['plus_'+str(h)]-z['minus_'+str(h)])/(2*h)-z['direct'])))
   assert err==f['absolute_inf_error'];fdcount+=1
 d=np.load(folder/'critical-direct-complex.npy',allow_pickle=False)
 assert np.isfinite(d).all() and float(np.max(np.abs(d-v[0]*U[:,0])))==row['critical_direct_eigen_residual_inf']
 cycle=history[-p:];assert row['adjacent_separations']==[float(np.max(np.abs(cycle[k]-cycle[(k+1)%p])))for k in range(p)]
 result=dict(receipt=s.pin(sup/'receipt.json'),source_commit=a.source,source_blobs_checked=len(ad['source_blobs']),outputs_checked=len(rc['output_files']),host_closure=True,columns_checked=896,eigen_residuals_checked=896,fd_rows_checked=fdcount,scientific_row=row,elapsed_seconds=rc['external_elapsed_seconds'],certificate_passed=False,scope='Saved composed matrix, complete eigensystem residuals, critical direct/FD comparisons and phase separation independently recomputed. AD and forward closure inherit reviewed original execution; no model rerun or interval certificate.')
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cell=a.cell,rho=row['spectral_radius'],fdrows=fdcount)))
if __name__=='__main__':main()
