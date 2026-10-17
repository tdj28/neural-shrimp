"""Independent saved-state arithmetic; no torch/model imports or trajectory rerun."""
from pathlib import Path
import json,hashlib,math,argparse,resource,time
import numpy as np

def pin(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=h.hexdigest())
def periods(x):
 assert x.ndim==3 and x.shape[0]==129 and np.isfinite(x).all();out=[]
 for j in range(x.shape[1]):
  scale=np.sqrt(np.sum(x[-64:,j]**2,axis=1));assert np.all(scale>0);es=[]
  for lag in range(1,33):es.append(float(np.max(np.sqrt(np.sum((x[-64:,j]-x[-64-lag:-lag,j])**2,axis=1))/scale)))
  out.append((next((k+1 for k,v in enumerate(es) if v<1e-8),0),es))
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.raw);dest=Path(a.out);dest.mkdir(exist_ok=False);start=time.monotonic()
 report=json.loads((root/'report.json').read_text());assert report['completed'];receipt=json.loads((root.parent/(root.name+'-supervision')/'receipt.json').read_text());assert receipt['returncode']==0 and receipt['failure'] is None and receipt['worker_reaped'] and receipt['worker_group_session_absent']
 for p in receipt['output_files']:
  q=pin(p['path']);assert all(q[k]==p[k] for k in ['bytes','sha256'])
 assert periods(np.load(root/'donor.npz')['tail'])[0][0]==2
 results=[]
 for entry in report['results']:
  alpha=entry['alpha'];v=json.loads((root/f'result-{alpha}.json').read_text());assert v==entry
  z=np.load(root/f'basins-{alpha}.npz');rt=np.load(root/f'references-{alpha}.npz')['tail'];tail=z['tail'];mid=z['mid'];assert [x[0] for x in periods(rt)]==v['reference_periods'];assert v['reference_periods'][1]==2;per=periods(tail);mp=periods(mid);stops=np.load(root/f'basins-{alpha}-stop-steps.npy');assert stops.shape==(323,) and np.all((stops>=192)&(stops<=1200))
  assert z['initial'].shape==(323,896) and z['single_ids'].shape==(128,) and z['mixture_ids'].shape==(128,8);assert np.allclose(z['mixture_weights'].sum(1),1,rtol=0,atol=1e-14) and np.all(z['mixture_weights']>0)
  for j,row in enumerate(v['rows']):
   assert row['index']==j and row['period']==per[j][0] and row['mid_period']==mp[j][0]
   assert np.allclose(per[j][1],z['period_errors'][j],rtol=1e-12,atol=1e-15)
   if stops[j]<1200:assert per[j][0]>0 and per[j][0]==mp[j][0]
   p=per[j][0];d=None
   if p in (2,3) and (p==2 or v['reference_periods'][0]==3):
    target=rt[-p:,1 if p==2 else 0];d=float(np.min(np.linalg.norm(target-tail[-1,j],axis=1))/np.linalg.norm(tail[-1,j]));label=('donor_period2' if p==2 else 'crescent_period3') if d<1e-6 else f'other_period{p}'
   else:label=f'other_period{p}' if p else 'no_return_up_to32'
   assert row['label']==label and (d==row['phase_distance'] or (d is not None and abs(d-row['phase_distance'])<1e-14))
   assert row['token_loop']==z['token_ids'][-max(1,p):,j].tolist()
  for arm,lo in [('single_embeddings',0),('top64_k8_mixtures',128)]:
   labels=[r['label'] for r in v['rows'][lo:lo+128]];assert set(labels)==set(v['fractions'][arm])
   for label,s in v['fractions'][arm].items():
    k=labels.count(label);assert s['count']==k and s['n']==128 and s['fraction']==k/128
    zz=1.959963984540054**2;center=(k/128+zz/256)/(1+zz/128);half=math.sqrt(zz*(k/128*(1-k/128)/128+zz/(4*128**2)))/(1+zz/128)
    assert np.allclose(s['wilson95'],[center-half,center+half],rtol=0,atol=1e-15)
  seg=[r['label'] for r in v['rows'][256:321]];assert seg==v['segment_labels'];changes=[j for j in range(64) if seg[j]!=seg[j+1]];assert len(changes)==len(v['coarse_transitions'])
  for j,b in zip(changes,v['coarse_transitions']):assert b==dict(lo=j/64,hi=(j+1)/64,left=seg[j],right=seg[j+1])
  rebuilt=[b.copy() for b in v['coarse_transitions'] if (b['left']=='donor_period2')!=(b['right']=='donor_period2')]
  if len(rebuilt)>16:rebuilt=[]
  assert len(rebuilt)==len(v['refined_binary_brackets'])
  for depth in range(8 if rebuilt else 0):
   bz=np.load(root/f'boundary-{alpha}-{depth}.npz');bps=periods(bz['tail']);rows=[x for x in v['bisection_history'] if x['depth']==depth];assert len(rows)==len(bps)
   for j,row in enumerate(rows):
    assert row['u']==bz['u'][j]==(rebuilt[j]['lo']+rebuilt[j]['hi'])/2 and row['period']==bps[j][0]
    dist=float(np.min(np.linalg.norm(bz['tail'][-1,j]-rt[-2:,1],axis=1))/np.linalg.norm(bz['tail'][-1,j]));assert abs(row['distance']-dist)<1e-14;assert row['donor']==(bps[j][0]==2 and dist<1e-6)
    if row['donor']==(rebuilt[j]['left']=='donor_period2'):rebuilt[j]['lo']=row['u']
    else:rebuilt[j]['hi']=row['u']
  assert rebuilt==v['refined_binary_brackets']
  for b in v['refined_binary_brackets']:assert 0<=b['lo']<b['hi']<=1 and b['hi']-b['lo']==1/16384
  initial=z['initial'][-2];d2=float(np.min(np.linalg.norm(rt[-2:,1]-initial,axis=1)));d3=float(np.min(np.linalg.norm(rt[-(v['reference_periods'][0] or 129):,0]-initial,axis=1)));assert abs(d2-v['proximity'][0]['min_distance_period2'])<1e-14 and abs(d3-v['proximity'][0]['min_distance_crescent_reference'])<1e-14
  nearest=[];confusion={};eligible=0;correct=0
  for j in range(256):
   initial=z['initial'][j];d2=float(np.min(np.linalg.norm(rt[-2:,1]-initial,axis=1)));dc=float(np.min(np.linalg.norm(rt[-(v['reference_periods'][0] or 129):,0]-initial,axis=1)));prediction='donor_period2' if d2<dc else 'crescent_period3';actual=v['rows'][j]['label'];qualifies=v['reference_periods'][0]==3 and actual in ('donor_period2','crescent_period3')
   nearest.append(dict(row=j,distance_donor=d2,distance_crescent_reference=dc,prediction=prediction,actual=actual,eligible=bool(qualifies)))
   if qualifies:
    eligible+=1;correct+=int(prediction==actual);key=prediction+' -> '+actual;confusion[key]=confusion.get(key,0)+1
  (dest/f'nearest-phase-{alpha}.json').write_text(json.dumps(nearest,indent=2)+'\n')
  results.append(dict(alpha=alpha,nearest_phase_heuristic=dict(eligible=eligible,correct=correct,accuracy=correct/eligible if eligible else None,confusion=confusion,excluded=256-eligible,scope='Post-design explanatory diagnostic on both256 sampled random starts; irregular/other outcomes retained outside binary score. No causality or out-of-sample claim.'),rows_checked=323,counts={arm:{k:x['count'] for k,x in values.items()} for arm,values in v['fractions'].items()},stop_min=int(stops.min()),stop_max=int(stops.max()),stop_mean=float(stops.mean()),at_ceiling=int(sum(stops==1200)),coarse_transitions=len(changes),refined_brackets=v['refined_binary_brackets'],proximity=v['proximity'],period_disagreements=v['mid_final_period_changes']))
  z.close();del tail,mid,rt
 result=dict(accepted=True,results=results,report=pin(root/'report.json'),receipt=pin(root.parent/(root.name+'-supervision')/'receipt.json'),elapsed=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits=['First16-iterate distances are trusted-source execution summaries; only step0 has independent saved-state reconstruction.','Finite sampled returns do not prove attraction, absence of other boundaries, fractality or chaos.','These are independent arithmetic checks of saved states, not independent model trajectories.'])
 (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
