"""Prepared fixed512-step candidate settling; NOT reviewed for execution.
Finite closure/primitive diagnostics only; no stability or interval certificate.
"""
import argparse,json,hashlib,time,resource,sys
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 c=json.loads(Path(a.card).read_text());assert c['steps']==512 and c['alpha']==.06468 and c['log10T']==-.148
 assert c['keys']==['p2+_6','p3+_6']
 for e in c['inputs']:assert sha(e['path'])==e['sha256']
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 sys.path.insert(0,str(Path(c['repository'])/'codex/crosscheck'))
 import torch
 from qwen_map import ManualQwenSoftMap,PREFIX_IDS,PINNED_SHA256
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 t=time.monotonic();m=ManualQwenSoftMap.from_pinned_files();load=time.monotonic()-t
 seeds=np.load(c['seeds'],allow_pickle=False);x=torch.from_numpy(np.stack([seeds[k] for k in c['keys']]).astype(np.float64));assert x.shape==(2,896)
 t=time.monotonic();cache=m.prefix_cache(c['alpha']);prefix=time.monotonic()-t;states=[x.numpy().copy()];times=[]
 with torch.no_grad():
  for i in range(c['steps']):
   t=time.monotonic();x=m(x,c['alpha'],c['log10T'],cache);times.append(time.monotonic()-t)
   assert torch.isfinite(x).all() and torch.all(x.norm(dim=-1)>0)
   states.append(x.numpy().copy())
 states=np.array(states)
 np.save(out/'states.npy',states)
 diagnostics=[]
 for j,period in enumerate((2,3)):
  # Check every phase of the last two full cycles, not a selected best pair.
  tail=states[-2*period:,j,:]
  scale=np.maximum(1.0,np.max(np.abs(tail),axis=1))
  closure=np.max(np.abs(tail[period:]-tail[:period]),axis=1)/scale[period:]
  cycle=tail[-period:]
  separation=[float(np.max(np.abs(cycle[k]-cycle[(k+1)%period]))) for k in range(period)]
  diagnostics.append({'seed_key':c['keys'][j], 'proposed_period':period,
   'relative_inf_closure_each_phase':closure.tolist(),
   'adjacent_phase_inf_separation':separation,
   'all_phases_close_at_1e_minus_12':bool(np.all(closure<1e-12)),
   'all_adjacent_phases_separated_at_1e_minus_8':bool(min(separation)>1e-8),
   'scope':'Finite candidate screen only. For prime periods2/3 the only proper divisor is1; no exact root or stability claim.'})
 for e in c['inputs']:assert sha(e['path'])==e['sha256']
 report={'completed':True,'steps':512,'batch':2,'load_seconds':load,'prefix_seconds':prefix,'step_seconds':times,'total_step_seconds':sum(times),'elapsed_seconds':time.monotonic()-tic,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'prefix_ids':list(PREFIX_IDS),'assets':PINNED_SHA256,'state_shape':[513,2,896],'diagnostics':diagnostics,'scope':'Fixed finite settling candidate screen only; no root, stability, token-loop or certification claim.'}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
