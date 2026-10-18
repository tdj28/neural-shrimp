"""Task29 individual opposite-fate brackets and rebracketed edge trajectories.
No global-threshold, fractality or saddle claim is made by this stage.
"""
import argparse,importlib.util,json,sys,time,resource
from pathlib import Path
import numpy as np

def opposite_brackets(labels):
 return [j for j in range(len(labels)-1) if labels[j]>=0 and labels[j+1]>=0 and labels[j]!=labels[j+1]]

def bracket_update(lo,hi,mid,label,left,right):
 assert lo<mid<hi and left!=right and left>=0 and right>=0
 if label==left:return mid,hi
 if label==right:return lo,mid
 raise ValueError('unresolved/third fate; bracket cannot be updated')

def controls():
 labels=[0,0,-1,1,0,1];assert opposite_brackets(labels)==[3,4]
 lo,hi=0.,1.;boundary=np.pi/10
 for _ in range(24):
  mid=(lo+hi)/2;lo,hi=bracket_update(lo,hi,mid,int(mid>=boundary),0,1)
 assert lo<boundary<hi and hi-lo==2**-24
 for bad in [-1,2]:
  try:bracket_update(0.,1.,.5,bad,0,1)
  except ValueError:pass
  else:raise AssertionError('unresolved label accepted')
 return dict(passed=True,groups=4,boundary_bracket=[lo,hi],unresolved_retained=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card');ap.add_argument('--out');ap.add_argument('--controls',action='store_true');args=ap.parse_args()
 if args.controls:print(json.dumps(controls(),indent=2));return
 card=json.loads(Path(args.card).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(spec);sys.modules[name]=v;spec.loader.exec_module(v);return v
 q=load('edge_map',card['map_source']);assert q.sha256(card['map_source'])==card['map_sha256']
 for e in card['inputs']:assert q.sha256(e['path'])==e['sha256']
 model=q.ManualQwenSoftMap.from_pinned_files();a=card['alpha'];lt=card['log10T'];cache=model.prefix_cache(a);tic=time.monotonic();calls=0;ledger=[];events=(out/'events.jsonl').open('x')
 references=np.load(card['references'],allow_pickle=False)['tail'];periods=card['periods'];phases=[references[-p:,j] for p,j in zip(periods,card['reference_columns'])]
 assert len(phases)==2 and all(0<p<=128 for p in periods)
 def event(**v):events.write(json.dumps(dict(elapsed=time.monotonic()-tic,**v))+'\n');events.flush()
 def step(x):
  nonlocal calls
  calls+=len(x);assert calls<=card['max_map_rows'],'map-row budget exhausted'
  y=model(x,a,lt,cache);assert torch.isfinite(y).all() and torch.all(y.norm(dim=-1)>0);return y
 def match(tail):
  labels=[]
  for j in range(tail.shape[1]):
   found=[]
   for side,p in enumerate(periods):
    w=max(64,4*p);xx=tail[-w:,j];err=np.max(np.linalg.norm(xx-tail[-w-p:-p,j],axis=1)/np.linalg.norm(xx,axis=1));distance=np.min(np.linalg.norm(phases[side]-tail[-1,j],axis=1))/np.linalg.norm(tail[-1,j])
    if err<1e-8 and distance<1e-6:found.append(side)
   labels.append(found[0] if len(found)==1 else -1)
  return np.array(labels)
 def fate(initial,tag):
  x=torch.as_tensor(np.asarray(initial),dtype=torch.float64);count=len(x);active=np.arange(count);previous=np.full(count,-1);labels=np.full(count,-1);stops=np.full(count,card['fate_steps']);all_states=[x.numpy().copy()];last=[];minimum=max(160,5*max(periods)+32)
  # Fixed batch retains all trajectories; no dynamic-batch numerical differences.
  for n in range(1,card['fate_steps']+1):
   x=step(x);all_states.append(x.numpy().copy())
   if n>=minimum and n%32==0:
    lab=match(np.array(all_states[-(5*max(periods)+65):]));new=(labels<0)&(lab>=0)&(lab==previous)
    labels[new]=lab[new];stops[new]=n;previous=lab
    if np.all(labels>=0):break
  trajectory=np.array(all_states);np.save(out/(tag+'.npy'),trajectory)
  # Re-check every accepted fate against its actual stopping window, preserve later drift.
  final=match(trajectory[-(5*max(periods)+65):]);drift=(labels>=0)&(final!=labels);labels[drift]=-1
  row=dict(tag=tag,labels=labels.tolist(),stops=stops.tolist(),drift=drift.tolist(),steps=n,trajectory=tag+'.npy');ledger.append(row);event(stage='fate',**row);return labels,trajectory
 with torch.no_grad():
  left,right=phases[0][-1],phases[1][-1];u=np.linspace(0,1,card['coarse_points']);coarse=(1-u[:,None])*left+u[:,None]*right;lab,trajectory=fate(coarse,'coarse')
  eligible=opposite_brackets(lab)
  selected=eligible[:card['max_brackets']];results=[]
  for number,j in enumerate(selected):
   lo,hi=float(u[j]),float(u[j+1]);fl,fh=int(lab[j]),int(lab[j+1]);history=[];reason='refined';xl=coarse[j].copy();xh=coarse[j+1].copy()
   for depth in range(card['initial_bisections']):
    mid=(lo+hi)/2;xm=(1-mid)*left+mid*right;l,tr=fate(xm[None],f'b{number}-initial-{depth}');history.append(dict(u=mid,label=int(l[0])))
    if l[0]<0:reason='unresolved_midpoint';break
    lo,hi=bracket_update(lo,hi,mid,int(l[0]),fl,fh)
    if l[0]==fl:xl=xm.copy()
    else:xh=xm.copy()
   edge=[];rebrackets=[]
   if reason=='refined':
    for epoch in range(card['edge_epochs']):
     pair=torch.as_tensor(np.stack([xl,xh]),dtype=torch.float64);path=[]
     for n in range(card['edge_steps']):pair=step(pair);path.append(pair.numpy().copy())
     path=np.array(path);np.save(out/f'b{number}-edge-{epoch}.npy',path);edge.extend(path.mean(axis=1));xl,xh=path[-1,0].copy(),path[-1,1].copy();fates,_=fate(np.stack([xl,xh]),f'b{number}-epoch{epoch}-endpoints')
     if list(fates)!=[fl,fh]:reason='edge_endpoint_fate_lost';break
     depth=0
     while np.linalg.norm(xl-xh)>card['edge_width']*max(np.linalg.norm(xl),np.linalg.norm(xh)) and depth<card['rebracket_limit']:
      xm=(xl+xh)/2;l,tr=fate(xm[None],f'b{number}-epoch{epoch}-bisect{depth}');rebrackets.append(dict(epoch=epoch,depth=depth,label=int(l[0]),separation=float(np.linalg.norm(xl-xh))))
      if l[0]<0:reason='unresolved_edge_midpoint';break
      if l[0]==fl:xl=xm
      else:xh=xm
      depth+=1
     if reason!='refined':break
     if depth==card['rebracket_limit'] and np.linalg.norm(xl-xh)>card['edge_width']*max(np.linalg.norm(xl),np.linalg.norm(xh)):reason='rebracket_limit';break
   np.save(out/f'b{number}-edge-midpoints.npy',np.array(edge));results.append(dict(index=number,coarse_index=j,initial_bracket=[lo,hi],original_labels=[fl,fh],history=history,rebrackets=rebrackets,reason=reason,edge_points=len(edge)))
   event(stage='bracket_done',result=results[-1])
 result=dict(completed=True,coarse_labels=lab.tolist(),eligible_brackets=eligible,selected_brackets=selected,unselected_brackets=eligible[len(selected):],results=results,ledger=ledger,map_rows=calls,elapsed=time.monotonic()-tic,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Finite opposite-fate edge trajectories only. Midpoint averages are approximate; Newton, stability and residence-slope verification are separate stages. Unresolved labels cannot substitute for a basin.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
