"""Audit and visualize the fixed PCP1 attempt; refuses incomplete or altered evidence."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/data2/neural-shrimp-data/audits/prompt-control/pcp1-20261024-03')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text())
for n,h in m.items():assert sha(root/n)==h
r=json.loads((root/'result.json').read_text());assert r['success'] and len(r['rows'])==9
setup=json.loads((root/'setup.json').read_text());assert setup['source']=='0c27e40a88a0a2a607d8cd615d9976fa36f2e6c1'
assert setup['model_revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
assert setup['temperature']==2. and setup['alpha']==0. and setup['dtype']=='torch.float64'
assert len(setup['ids0'])==len(setup['ids1'])==37
assert sum(a!=b for a,b in zip(setup['ids0'],setup['ids1']))==1
rows=r['rows'];u=[];ys=[];states=[]
for i,row in enumerate(rows):
 assert row['s']==i/8
 assert row['converged'] and row['residual']<1e-11
 assert row['fd_relative_errors'][-1]<1e-6
 for spec in row['spectra']:assert max(spec['residuals'])<1e-6
 assert abs(row['spectra'][0]['radius']-row['spectra'][1]['radius'])<1e-6
 z=np.load(root/f'point-{i:02d}.npz')
 for key in z.files:assert z[key].dtype==np.float64 and z[key].shape==(1,1,896) and np.isfinite(z[key]).all()
 u.append(z['common_jvp']);ys.append(z['common_image']);states.append(z['state'])
 assert abs(row['common_jvp_relative_change']-np.linalg.norm(u[-1]-u[0])/np.linalg.norm(u[0]))<1e-10
 assert abs(row['common_map_relative_change']-np.linalg.norm(ys[-1]-ys[0])/np.linalg.norm(ys[0]))<1e-10
s=np.array([x['s'] for x in rows]);rho=np.array([x['spectra'][0]['radius'] for x in rows])
fig,ax=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
ax[0].plot(s,[100*x['common_map_relative_change'] for x in rows],'o-',color='#315c87')
ax[0].set(ylabel='Next embedding: relative change (%)',title='Same state, different next update')
ax[1].plot(s,[100*x['common_jvp_relative_change'] for x in rows],'s-',color='#b34d29')
ax[1].set(ylabel='JVP vector: relative change (%)',title='Same perturbation, different response')
ax[2].plot(s,rho,'o-',color='#167c80');ax[2].axhline(1,color='gray',ls='--',label='Stability boundary')
ax[2].set(ylabel='Estimated dominant multiplier modulus',title='Stability of the continued fixed point',ylim=(0,max(1.05,float(rho.max())*1.1)))
ax[2].legend(fontsize=9)
for a in ax:a.set_xlabel('Hello! (0) → Think! (1): soft-prompt fraction',fontsize=9);a.grid(alpha=.2)
fig.suptitle('One prompt token changes; length, temperature and steering stay fixed',fontsize=14)
fig.savefig(root/'prompt-control.png',dpi=160);plt.close(fig)
backup=Path('/archive/neural-shrimp-backups/20261024-pcp1-03');backup.mkdir(exist_ok=False)
for p in root.iterdir():
 if p.is_file():shutil.copy2(p,backup/p.name);assert sha(p)==sha(backup/p.name)
assert root.stat().st_dev!=backup.stat().st_dev
# Independent whole-file hashes are checked above; preserve failed setup too.
for attempt in ['pcp1-20261024-01','pcp1-20261024-02']:
 failed=root.parent/attempt;shutil.copytree(failed,backup/attempt)
 for p in failed.iterdir():
  if p.is_file():assert sha(p)==sha(backup/attempt/p.name)
receipt=dict(imported=json.loads((root/'resume.json').read_text()),source=setup['source'],raw=str(root),manifest_sha256=sha(root/'manifest.json'),figure=str(root/'prompt-control.png'),figure_sha256=sha(root/'prompt-control.png'),backup=str(backup),physical_devices={'source':'nvme1n1','backup':'sda'},checked_files=len(m),points=9,endpoint_map_relative_change=rows[-1]['common_map_relative_change'],endpoint_jvp_relative_change=rows[-1]['common_jvp_relative_change'],endpoint_directional_gain_ratio=float(np.linalg.norm(u[-1])/np.linalg.norm(u[0])),rho_min=float(rho.min()),rho_max=float(rho.max()),rho_start=float(rho[0]),rho_end=float(rho[-1]),max_fp_residual=max(x['residual'] for x in rows),max_eigen_residual=max(max(y['residuals']) for x in rows for y in x['spectra']),max_final_fd_error=max(x['fd_relative_errors'][-1] for x in rows),fixed_point_endpoint_relative_shift=float(np.linalg.norm(states[-1]-states[0])/np.linalg.norm(states[0])),endpoint_kv_relative_changes=rows[-1]['cache_relative_changes'],qualification='Nine exploratory soft-prompt settings; numerical fixed-point stability estimates, no interval certification. One common-state derivative direction is not full Jacobian distance; no isolated layer causal attribution.')
Path('docs/prompt-control-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
