#!/usr/bin/env python3
"""One new bounded saved-data check of Task39's newly disclosed close-up mask."""
import hashlib,json,os,resource,signal,time
from pathlib import Path
os.nice(10);resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(30,30));signal.alarm(45)
def memory_gate():
 d={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
 if d['MemAvailable']<16*1024**3:raise RuntimeError('Below16GiB shared available-memory floor')
memory_gate()
import numpy as np
O=Path('/tmp/task39-raw');P=Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64');out=O/'boundary-closeup-result-001.json'
assert not out.exists();start=time.monotonic();pins={}
def pin(p):
 p=Path(p);b=p.read_bytes();h=hashlib.sha256(b).hexdigest();pins[str(p)]={'bytes':len(b),'sha256':h};return h
prior_path=Path('/tmp/task38-geometry/geometry-curves-004.json');pin(prior_path);prior=json.loads(prior_path.read_text())['pins']
meta_path=O/'fig-p6-shrimp-assigned.json';pin(meta_path);meta=json.loads(meta_path.read_text());window=meta['boundary_gate_window'];assert len(window)==4
def admitted(p):
 h=pin(p);assert str(p) in prior and h==prior[str(p)]['sha256'],f'Original input changed: {p}'
 return p
z=np.load(admitted(P/'p6s/follow/follow.npz'),allow_pickle=False)
x,y,lab,per=z['alpha1'],z['log10_temperature'],z['label'],z['period'];scale=np.array([x[1]-x[0],y[1]-y[0]])
assert lab.shape==per.shape==(len(y),len(x));assert np.all(scale>0);assert np.isfinite(x).all() and np.isfinite(y).all()
segments={'fold':[],'flip':[]};inventory=[]
assert len(meta['curves'])==18 and len({c['curve'] for c in meta['curves']})==18
for c in meta['curves']:
 name=c['curve'];assert name.startswith(('p6s-','kite-'))
 p=P/('kite' if name.startswith('kite-') else 'p6s')/name/'curve.json';q=json.loads(admitted(p).read_text());assert q['kind'] in segments
 target=1 if q['kind']=='fold' else -1;accepted=[];rejected=[]
 for i,r in enumerate(q['points']):
  mu=r.get('critical_multiplier_refined',r.get('critical_multiplier'));mu=complex(*mu[:2]) if isinstance(mu,list) else complex(mu)
  ok=not r.get('rejected',False) and r.get('residual',1)<=1e-8 and abs(mu.real-target)<=1e-3 and abs(mu.imag)<=1e-3
  if ok:accepted.append([r['alpha'],r['log10_temperature']])
  else:rejected.append(i)
 assert len(accepted)==c['drawn'] and len(rejected)==c['failed_gate']
 a=np.array(accepted)/scale
 if len(a)>1:segments[q['kind']].extend(zip(a[:-1],a[1:]))
 inventory.append({'curve':name,'kind':q['kind'],'drawn':len(accepted),'excluded_indices':rejected})
seg={k:(np.array([a for a,b in v]),np.array([b-a for a,b in v])) for k,v in segments.items()};assert all(len(a) for a,d in seg.values())
rows={'flip':[],'fold':[]}
for i,j in zip(*np.where((lab==1)&(per==6))):
 for di,dj in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
  ii,jj=i+di,j+dj
  if not(0<=ii<len(y) and 0<=jj<len(x)):continue
  kind='flip' if lab[ii,jj]==1 and per[ii,jj]==12 else 'fold' if lab[ii,jj]==0 else None
  if kind is None:continue
  mid=np.array([(x[j]+x[jj])/2,(y[i]+y[ii])/2])
  if not(window[0]<=mid[0]<=window[1] and window[2]<=mid[1]<=window[3]):continue
  memory_gate();a,d=seg[kind];v=mid/scale-a;u=np.clip((v*d).sum(1)/np.maximum((d*d).sum(1),1e-30),0,1)
  distance=float(np.sqrt(((v-u[:,None]*d)**2).sum(1)).min());assert np.isfinite(distance)
  rows[kind].append({'cells':[[int(i),int(j)],[int(ii),int(jj)]],'midpoint':mid.tolist(),'distance_grid_steps':distance})
summary={}
for kind,r in rows.items():
 d=np.array([a['distance_grid_steps'] for a in r]);assert len(d)>0
 summary[kind]={'pairs':len(r),'within_1_step':int((d<=1).sum()),'within_2_steps':int((d<=2).sum()),'median_steps':float(np.median(d)),'max_steps':float(d.max())}
result={'schema':'task39-new-closeup-boundary-v1','article_commit':'4ca5b5e35bc766581a5e2d326858fcd0edc312bb','scope':'Newly disclosed close-up midpoint mask only; no old whole-domain replay','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'window':window,'scale':scale.tolist(),'plot_gate':{'residual_max':1e-8,'real_target_error_max':1e-3,'imaginary_max':1e-3,'exclude_rejected':True},'curve_inventory':inventory,'summary':summary,'all_pairs':rows,'pins':pins,'elapsed_seconds':time.monotonic()-start}
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'summary':summary,'elapsed_seconds':result['elapsed_seconds']}))
