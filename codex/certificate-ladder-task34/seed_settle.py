"""Task34 fixed finite candidate screens at the five accepted exact cells."""
import argparse,json,hashlib,time,resource,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diagnostics(states,period):
 if states.shape!=(513,896) or not np.isfinite(states).all():raise ValueError('complete finite513x896history required')
 if period not in (3,6,10,12):raise ValueError('unadmitted period')
 tail=states[-2*period:];cycle=tail[-period:]
 closure=np.max(np.abs(tail[period:]-tail[:period]),axis=1)/np.maximum(1,np.max(np.abs(tail[period:]),axis=1))
 divisors=[d for d in range(1,period)if period%d==0]
 separation={str(d):float(np.max(np.abs(cycle[0]-cycle[d])))for d in divisors}
 return dict(period=period,relative_inf_closure_each_phase=closure.tolist(),proper_divisor_phase0_separation=separation,finite_closure_pass=bool(np.all(closure<1e-12)),finite_primitive_screen_pass=all(v>1e-8 for v in separation.values()),scope='Finite proposal screen only; no exact root, multiplier, attraction, token or certificate claim.')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();c=json.loads(Path(a.card).read_text());assert c['steps']==512 and len(c['cells'])==5
 for e in c['inputs']:assert sha(e['path'])==e['sha256']
 out=Path(a.out);out.mkdir(exist_ok=False);sys.path.insert(0,str(Path(c['repository'])/'codex/crosscheck'))
 import torch
 from qwen_map import ManualQwenSoftMap,PREFIX_IDS,PINNED_SHA256
 torch.set_num_threads(1);torch.set_num_interop_threads(1);tic=time.monotonic();m=ManualQwenSoftMap.from_pinned_files();load=time.monotonic()-tic;reports=[]
 for cell in c['cells']:
  assert tuple(cell['alpha_ratio'])==cell['alpha'].as_integer_ratio() and tuple(cell['log10T_ratio'])==cell['log10T'].as_integer_ratio()
  seed=np.load(cell['seed_path'],allow_pickle=False);assert seed.shape==(896,) and np.isfinite(seed).all()
  x=torch.from_numpy(seed.astype(np.float64).reshape(1,896));assert x.norm()>0
  start=time.monotonic();cache=m.prefix_cache(cell['alpha']);states=[x.numpy()[0].copy()];times=[]
  with torch.no_grad():
   for i in range(512):
    t=time.monotonic();x=m(x,cell['alpha'],cell['log10T'],cache);times.append(time.monotonic()-t);assert torch.isfinite(x).all() and x.norm()>0;states.append(x.numpy()[0].copy())
  states=np.array(states);np.save(out/(cell['name']+'-states.npy'),states)
  row=dict(cell=cell,diagnostics=diagnostics(states,cell['period']),step_seconds=times,elapsed_seconds=time.monotonic()-start);reports.append(row);(out/(cell['name']+'-report.json')).write_text(json.dumps(row,indent=2)+'\n');print(cell['name']+' complete; all states retained',flush=True)
 for e in c['inputs']:assert sha(e['path'])==e['sha256']
 (out/'report.json').write_text(json.dumps(dict(completed=True,cells=reports,load_seconds=load,elapsed_seconds=time.monotonic()-tic,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,prefix_ids=list(PREFIX_IDS),assets=PINNED_SHA256,certificate_passed=False),indent=2)+'\n')
if __name__=='__main__':main()
