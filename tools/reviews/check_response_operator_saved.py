"""Post-outcome producer saved-only terminal arithmetic; no model/solver replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
RAW=Path('/data2/neural-shrimp-data/explore/clark-response/operator-blocks-002')
def need(x,m):
 if not x:raise ValueError(m)
def entry(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return {'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def main(out,pin):
 start=time.monotonic();need(not out.exists(),'exclusive output');need(entry(RAW/'manifest.json')['sha256']==pin,'external inventory pin');inv=json.loads((RAW/'manifest.json').read_text())['files']
 need(set(inv)=={str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file() and p!=RAW/'manifest.json'},'exact inventory')
 for n,v in inv.items():
  p=RAW/n;need(not p.is_symlink() and p.resolve().is_relative_to(RAW.resolve()) and entry(p)==v and not p.stat().st_mode&0o222,'immutable identity '+n)
 launch=json.loads((RAW/'launch-end.json').read_text());need(launch['exit_code']==0 and launch['owned_group_empty'] and all(not Path('/proc',str(launch[k])).exists() for k in ('launcher_pid','child_pid')),'owned process exits')
 result=json.loads((RAW/'run/result.json').read_text());events={};count=0
 with (RAW/'run/events.jsonl').open() as f:
  for line in f:
   row=json.loads(line);count+=1;need(row['sequence']==count,'event order')
   if row['kind']=='terminal':events[row['slot']]=row
 need(count==result['events'],'complete event count');rows=[]
 for row in result['rows']:
  label=row['label'];e=events[label];p=RAW/'run'/e['array_bundle']['file'];need(entry(p)=={k:e['array_bundle'][k] for k in ('sha256','bytes')},'terminal bundle')
  with np.load(p,allow_pickle=False) as data:
   need(set(data.files)==set(e['arrays']),'all terminal arrays');a={k:data[k].copy() for k in data.files}
   for k,v in a.items():need(v.dtype.str==e['arrays'][k]['dtype'] and list(v.shape)==e['arrays'][k]['shape'],'array schema')
  maps=np.load(RAW/'run'/f'{label}-maps.npy',allow_pickle=False);oracle=np.load(RAW/'run'/f'{label}-saved-direct-column.npy',allow_pickle=False);period,n,_=maps.shape;steps=16*period;x=a['state'].reshape(steps+1,n)
  # Independent edge assembly of D^T(Dx-u)+eta^2*x, free endpoints.
  scale=np.exp(.05);edges=x[1:].copy();rhs=np.zeros_like(x)
  for t in range(steps):edges[t]-=scale*(maps[t%period]@x[t])
  edges[8*period,0]-=1;normal=.0001*x.copy()
  for t in range(steps):normal[t]-=scale*(maps[t%period].T@edges[t]);normal[t+1]+=edges[t]
  rhs[8*period]-=scale*maps[0].T[:,0];rhs[8*period+1,0]+=1
  rel=float(np.linalg.norm(normal)/np.linalg.norm(rhs));err=float(np.linalg.norm(x[8*period+1]-oracle))
  need(np.allclose(normal.ravel(),a['normal_residual'],rtol=0,atol=1e-14),'saved true normal residual')
  need(abs(rel-row['true_normal_relative_residual'])<1e-14 and abs(err-row['readout_error'])<1e-14,'reported endpoints')
  need(np.array_equal(x[8*period+1],np.asarray(row['readout_column'])) and e['counts']==row['counts'] and sum(e['counts'].values())==row['action_ceiling'],'saved readout and budget')
  need(not row['endpoint_pass'] and row['status']=='action_budget_exhausted','negative endpoint labels')
  rows.append({'label':label,'normal_relative_residual':rel,'readout_error':err,'phase_actions':sum(e['counts'].values()),'endpoint_pass':False})
 need(time.monotonic()-start<10,'10s saved-only cap');out.mkdir(parents=True,exist_ok=False)
 payload={'passed':True,'raw_manifest':pin,'files_verified':len(inv),'events_verified':count,'rows':rows,'elapsed_seconds':time.monotonic()-start,'scope':'Post-outcome producer-authored saved terminal arithmetic and inventory check; no new derivative, factorization, solver, training or target selection. Independent personnel audit remains pending.'}
 (out/'result.json').write_text(json.dumps(payload,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{'result.json':entry(out/'result.json')}},indent=2)+'\n')
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps(payload))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',required=True);a=p.parse_args();main(a.out,a.manifest)
