"""Independent saved-history checks and compact evidence receipt for PCP1 follow-up."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('/data2/neural-shrimp-data/audits/prompt-control')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verified(name):
 root=base/name;m=json.loads((root/'manifest.json').read_text())
 for n,h in m.items():assert sha(root/n)==h
 r=json.loads((root/'result.json').read_text());assert r['success']
 return root,r
root,result=verified('pcp1-attractors-01');cross,c=verified('pcp1-crossing-01')
assert len(result['rows'])==3 and len(c['history'])==4
assert c['width']==.0078125
for r in c['history']:
 assert r['fixed_point_residual']<1e-11 and max(r['eigen_residuals'])<1e-6
 assert r['multiplier']<0 and abs(r['argument']-np.pi)<1e-8 and r['next_largest_modulus']<1
points=[]
for i,r in enumerate(result['rows']):
 assert r['s']==[.75,.875,1.][i]
 states=np.load(root/f'trajectory-{i}.npy');assert states.shape==(768,896) and states.dtype==np.float64 and np.isfinite(states).all()
 errors=[]
 for lag in range(1,65):
  errors.append(max(np.linalg.norm(states[j]-states[j-lag])/np.linalg.norm(states[j]) for j in range(640,768)))
 assert np.allclose(errors,r['recurrence_errors'],rtol=1e-12,atol=1e-14)
 period=next((k+1 for k,e in enumerate(errors) if e<1e-10),None);assert period==r['period_candidate']
 if period:
  assert r['closure']<1e-10 and all(v>1e-8 for v in r['proper_divisors'].values())
  assert r['composed_fd_errors'][-1]<1e-5 and r['composed_fd_errors'][-1]<r['composed_fd_errors'][0]/4
  for spec in r['spectra']:assert max(spec['residuals'])<1e-6
  assert abs(r['spectra'][0]['radius']-r['spectra'][1]['radius'])<1e-6
 else:
  z=np.load(root/f'growth-{i}.npz');g=z['growth'];assert g.shape==(512,2) and np.isfinite(g).all()
  assert np.allclose(g.mean(0),r['growth_means'])
 points.append(states[-128:])
# A fixed observed coordinate, selected by largest variance across all saved endpoints.
combined=np.concatenate(points);coordinate=int(np.argmax(np.var(combined,axis=0)))
fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
for i,(r,states) in enumerate(zip(result['rows'],points)):
 axs[i].plot(np.arange(32),states[-32:,coordinate],'o-',ms=3,color='#176f82')
 classification=r['classification'];period=r['period_candidate']
 axs[i].set(title=f"Prompt fraction {r['s']:g}\n"+(f'Period {period}; cycle multiplier modulus {r["spectra"][0]["radius"]:.3f}' if period else 'No qualified cycle'),xlabel='Final feedback iterations',ylabel=f'Feedback embedding coordinate {coordinate}')
 axs[i].grid(alpha=.2)
low=float(combined[:,coordinate].min());high=float(combined[:,coordinate].max());padding=.08*(high-low)
for ax in axs:ax.set_ylim(low-padding,high+padding)
fig.suptitle('What replaces the fixed point when the prompt changes?',fontsize=15)
fig.savefig(root/'prompt-attractors.png',dpi=170);plt.close(fig)
backup=Path('/archive/neural-shrimp-backups/20261024-pcp1-followup-01');backup.mkdir(exist_ok=False)
for source in [root,cross,base/'pcp1-crossing-input-01']:
 dest=backup/source.name;shutil.copytree(source,dest)
 for f in source.iterdir():
  if f.is_file():assert sha(f)==sha(dest/f.name)
assert root.stat().st_dev!=backup.stat().st_dev
receipt=dict(raw=str(root),manifest_sha256=sha(root/'manifest.json'),source=json.loads((root/'setup.json').read_text())['source'],crossing_raw=str(cross),crossing_manifest_sha256=sha(cross/'manifest.json'),crossing=c,rows=result['rows'],cycle_lyapunov=[float(np.log(r['spectra'][0]['radius'])/r['period_candidate']) if r['period_candidate'] else None for r in result['rows']],backup=str(backup),physical_devices={'source':'nvme1n1','backup':'sda'},figure=str(root/'prompt-attractors.png'),figure_sha256=sha(root/'prompt-attractors.png'),qualification='Independent full-state recurrence recomputation; numerical local cycle stability. No interval proof, global basin claim or proven chaos.')
Path('docs/prompt-control-followup-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'crossing_bracket':c['bracket'],'rows':[{k:r[k] for k in ['s','period_candidate','classification']} for r in result['rows']]}))
