"""Task29 staged boundary investigation; finite labels, explicit unresolved gates."""
import argparse,importlib.util,json,sys,time,resource,hashlib
from pathlib import Path
import numpy as np

def recurrence(tail,max_period=128,tol=1e-8):
 assert tail.ndim==3 and len(tail)>=5*max_period and np.isfinite(tail).all()
 errors=[]
 for p in range(1,max_period+1):
  window=max(64,4*p);a=tail[-window:];norm=np.linalg.norm(a,axis=-1);assert np.all(norm>0)
  errors.append(np.max(np.linalg.norm(a-tail[-window-p:-p],axis=-1)/norm,axis=0))
 errors=np.stack(errors,axis=1)
 periods=[next((p+1 for p,e in enumerate(row) if e<tol),0) for row in errors]
 return periods,errors

def slope_control():
 # Stroboscopic saddle (u,v)->(1.2u,.5v), one iterate represents6 map steps.
 lam=1.2;p=6;dist=np.geomspace(1e-9,1e-3,16);times=[]
 for d in dist:
  u=d;v=.01;n=0
  while abs(u)<.1:
   u*=lam;v*=.5;n+=1
   assert n<1000
  times.append(p*n)
 slope,intercept=np.polyfit(np.log(dist),times,1);pred=-p/np.log(lam)
 assert abs(slope/pred-1)<.03
 # Finite period detection must distinguish fixed,period3 and no-return examples.
 t=np.arange(768);x=np.stack([np.ones(768),2+np.cos(2*np.pi*t/3),2+t*.001],axis=1)[...,None]
 periods,_=recurrence(x);assert periods==[1,3,0],periods
 return dict(multiplier=lam,period=p,distances=dist.tolist(),residence_steps=times,slope=float(slope),predicted=float(pred),relative_error=float(abs(slope/pred-1)),period_controls=periods,scope='Known linear stroboscopic saddle; rounding of integer exit time retained, not target evidence.')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card');ap.add_argument('--out');ap.add_argument('--controls',action='store_true');args=ap.parse_args()
 if args.controls:print(json.dumps(slope_control(),indent=2));return
 card=json.loads(Path(args.card).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 def write(name,v):(out/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 spec=importlib.util.spec_from_file_location('task29_map',card['map_source']);q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
 assert q.sha256(card['map_source'])==card['map_sha256']
 for e in card['inputs']:
  assert q.sha256(e['path'])==e['sha256'] and Path(e['path']).stat().st_size==e['bytes']
 start=time.monotonic();m=q.ManualQwenSoftMap.from_pinned_files();results=[];progress=(out/'progress.jsonl').open('x')
 write('metadata.json',dict(card=card,torch=torch.__version__,numpy=np.__version__,prefix_ids=q.PREFIX_IDS,load_seconds=time.monotonic()-start,controls=slope_control()))
 with torch.no_grad():
  for cell in card['cells']:
   alpha,lt=cell['alpha'],cell['log10T'];cache=m.prefix_cache(alpha);names=['fixed','band6',cell['stripe']]
   def seed(n):
    x=np.load(Path(card['seed_root'])/(n+'.npy'),allow_pickle=False);assert x.size==896 and np.isfinite(x).all();return torch.as_tensor(x.reshape(896),dtype=torch.float64)
   x=torch.stack([m.initial_state(),seed('band6'),seed(cell['stripe'])]);initial=x.numpy().copy();states=[];tokens=[];timings=[]
   for i in range(card['steps']):
    tick=time.monotonic();z=m.logits(x,alpha,cache);probs=torch.softmax(z/10**lt,dim=-1);x=probs@m.embedding
    assert torch.isfinite(x).all() and torch.all(x.norm(dim=-1)>0)
    timings.append(time.monotonic()-tick)
    if i>=card['steps']-768:states.append(x.numpy().copy());tokens.append(probs.argmax(-1).numpy())
    if (i+1)%100==0:progress.write(json.dumps(dict(alpha=alpha,log10T=lt,steps=i+1,elapsed=time.monotonic()-start))+'\n');progress.flush()
   tail=np.array(states);periods,errors=recurrence(tail);prev,prev_errors=recurrence(tail[:-32]);np.savez(out/(cell['id']+'.npz'),initial=initial,tail=tail,tokens=np.array(tokens),errors=errors,previous_errors=prev_errors)
   rows=[dict(start=n,period=p,previous_period=pp,qualified=p>0 and p==pp,relative_return=None if not p else float(errors[j,p-1]),token_loop=np.array(tokens)[-p:,j].tolist() if p else None) for j,(n,p,pp) in enumerate(zip(names,periods,prev))]
   results.append(dict(cell=cell,rows=rows,step_seconds_mean=float(np.mean(timings)),step_seconds_max=max(timings),scope='Seed qualification/cost only; common period or tokens alone do not assign family or prove stability.'))
   write('partial-'+cell['id']+'.json',results[-1])
 for e in card['inputs']:assert q.sha256(e['path'])==e['sha256']
 write('report.json',dict(completed=True,results=results,elapsed=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,next='Review seed gates before any segment/edge/Newton launch; failed coexistence remains unresolved.'))
if __name__=='__main__':main()
