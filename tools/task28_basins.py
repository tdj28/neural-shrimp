"""Exploratory finite-horizon basins; full-state returns, no fractality claim."""
import argparse,json,time,hashlib,sys,importlib.util,resource
from pathlib import Path
import numpy as np

def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def period(tail,tol=1e-8):
 # tail: time,batch,state. Require every return in final64 comparisons.
 ans=np.zeros(tail.shape[1],int);errs=[]
 for p in range(1,33):
  a=tail[-64:];b=tail[-64-p:-p];e=np.max(np.linalg.norm(a-b,axis=-1)/np.linalg.norm(a,axis=-1),axis=0)
  errs.append(e);ans[(ans==0)&(e<tol)]=p
 return ans,np.array(errs).T

def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(c-d),float(c+d)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();card=json.loads(Path(args.card).read_text());out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 spec=importlib.util.spec_from_file_location('basin_qwen',card['map_source']);q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
 assert q.sha256(card['map_source'])==card['map_sha256']
 start=time.monotonic();model=q.ManualQwenSoftMap.from_pinned_files();load=time.monotonic()-start
 tok=json.loads((q.SNAPSHOT/'tokenizer.json').read_text());vocab=tok['model']['vocab'];rev={v:k for k,v in vocab.items()};assert model.prefix_ids[-1].item()==198 and rev[198]=='Ċ'
 write(out/'metadata.json',dict(card=card,torch=torch.__version__,numpy=np.__version__,prefix_ids=q.PREFIX_IDS,last_prefix_token_id=198,last_prefix_token_bytelevel=rev[198],dtype='float64',device='cpu',norm='manual float64 mean/rsqrt',rotary='manual float64 frequencies and sin/cos',load_seconds=load))
 progress=(out/'progress.jsonl').open('x')
 def event(**kw):progress.write(json.dumps(dict(elapsed=time.monotonic()-start,**kw))+'\n');progress.flush()
 def step(x,a,c):
  z=model.logits(x,a,c);p=torch.softmax(z/(10.**-.156),dim=-1);return p@model.embedding,p
 def evolve(x,a,c,n,label,record=129):
  tail=[];mid=None;tokens=[];probs=[]
  for i in range(n):
   x,p=step(x,a,c)
   if i>=n-record:tail.append(x.numpy().copy());tokens.append(p.argmax(-1).numpy());probs.append(p.max(-1).values.numpy())
   if n//2-129<=i<n//2:
    if mid is None:mid=[]
    mid.append(x.numpy().copy())
   if (i+1)%100==0:event(stage=label,steps=i+1,total=n)
  return x,np.array(tail),np.array(tokens),np.array(probs),np.array(mid) if mid is not None else None
 with torch.no_grad():
  if card['mode']=='profile':
   a=.0642;c=model.prefix_cache(a);times={}
   for n in [2,16,323]:
    x=model.initial_state()[None].repeat(n,1);ts=[]
    for _ in range(4):
     tic=time.monotonic();x,p=step(x,a,c);ts.append(time.monotonic()-tic)
    times[str(n)]=ts
   # Independent central-difference consistency against automatic JVP, not a dtype assertion.
   x=model.initial_state();d=torch.arange(1,897,dtype=torch.float64);d/=d.norm()
   with torch.enable_grad():j=model.jvp(x,d,a,-.156,cache=c)
   fd=[]
   for h in [.01,.003,.001]:
    diff=(model(x+h*d,a,-.156,c)-model(x-h*d,a,-.156,c))/(2*h);fd.append({'h':h,'relative_jvp_error':float((diff-j).norm()/j.norm())})
   write(out/'report.json',dict(mode='profile',times=times,fd=fd,load_seconds=load,elapsed=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));return
  a=.0655;c=model.prefix_cache(a);donor,dt,_,_,_=evolve(model.initial_state()[None],a,c,1000,'donor')
  dp,de=period(dt);assert dp[0]==2,('donor not period2',dp)
  np.savez(out/'donor.npz',tail=dt,errors=de)
  all_results=[]
  for a in card['alphas']:
   c=model.prefix_cache(a);fixed=model.initial_state();ref,rt,rk,rp,_=evolve(torch.stack([fixed,donor[0]]),a,c,1200,f'references-{a}')
   periods,errors=period(rt);assert periods[1]==2,('target donor not2',a,periods)
   if a in [.0642,.0644]:assert periods[0]==3,('fixed crescent not3',a,periods)
   np.savez(out/f'references-{a}.npz',tail=rt,token_ids=rk,probabilities=rp,errors=errors)
   rng=np.random.default_rng(card['seed']);single=rng.integers(0,model.vocab,128);_,p0=step(fixed[None],a,c);top=torch.topk(p0[0],64).indices.numpy();mix_ids=np.array([rng.choice(top,8,replace=False) for _ in range(128)]);weights=rng.dirichlet(np.ones(8),128)
   random=torch.cat([model.embedding[single],(model.embedding[mix_ids]*torch.from_numpy(weights)[:,:,None]).sum(1)])
   # Same deterministic donor/crescent endpoint phases are saved, not chosen after basin outcomes.
   u=np.linspace(0,1,65);left=ref[1].clone();right=ref[0].clone();seg=(1-torch.from_numpy(u)[:,None])*left+torch.from_numpy(u)[:,None]*right
   starts=torch.cat([random,seg,torch.stack([fixed,donor[0]])]);assert len(starts)==323
   _,tail,keys,prob,mid=evolve(starts,a,c,card['steps'],f'basins-{a}')
   per,err=period(tail);mp,me=period(mid)
   np.savez(out/f'basins-{a}.npz',initial=starts.numpy(),tail=tail,mid=mid,token_ids=keys,probabilities=prob,period_errors=err,mid_errors=me,single_ids=single,mixture_ids=mix_ids,mixture_weights=weights,segment_u=u)
   labels=[]
   for i,p in enumerate(per):
    if p==2:
     dist=float(np.min(np.linalg.norm(tail[-1,i]-rt[-2:,1],axis=1))/np.linalg.norm(tail[-1,i]));lab='donor_period2' if dist<1e-6 else 'other_period2'
    elif p==3 and periods[0]==3:
     dist=float(np.min(np.linalg.norm(tail[-1,i]-rt[-3:,0],axis=1))/np.linalg.norm(tail[-1,i]));lab='crescent_period3' if dist<1e-6 else 'other_period3'
    else:dist=None;lab=f'other_period{p}' if p else 'no_return_up_to32'
    labels.append(dict(index=i,period=int(p),mid_period=int(mp[i]),label=lab,phase_distance=dist,token_loop=keys[-max(1,int(p)):,i].tolist(),bytelevel_loop=[rev.get(int(t),'<special>') for t in keys[-max(1,int(p)):,i]]))
   fractions={}
   for name,lo,hi in [('single_embeddings',0,128),('top64_k8_mixtures',128,256)]:
    labs=[r['label'] for r in labels[lo:hi]];fractions[name]={l:dict(count=labs.count(l),n=128,fraction=labs.count(l)/128,wilson95=wilson(labs.count(l),128)) for l in sorted(set(labs))}
   # Enumerate all changes on the coarse segment. Only binary donor/other transitions get refinement;
   # save all other label changes too. A finite grid cannot establish fractality or uniqueness.
   seglabs=[x['label'] for x in labels[256:321]];changes=[i for i in range(64) if seglabs[i]!=seglabs[i+1]]
   brackets=[dict(lo=float(u[i]),hi=float(u[i+1]),left=seglabs[i],right=seglabs[i+1]) for i in changes]
   active=[b.copy() for b in brackets if (b['left']=='donor_period2')!=(b['right']=='donor_period2')]
   if len(active)>16:event(stage='boundary-cap',cell=a,brackets=len(active));active=[]
   history=[]
   for depth in range(card['bisections'] if active else 0):
    us=np.array([(b['lo']+b['hi'])/2 for b in active]);x=(1-torch.from_numpy(us)[:,None])*left+torch.from_numpy(us)[:,None]*right
    _,bt,bk,bp,_=evolve(x,a,c,card['steps'],f'boundary-{a}-{depth}')
    bpds,be=period(bt)
    for i,b in enumerate(active):
     near=float(np.min(np.linalg.norm(bt[-1,i]-rt[-2:,1],axis=1))/np.linalg.norm(bt[-1,i]));isdonor=bpds[i]==2 and near<1e-6
     history.append(dict(depth=depth,u=float(us[i]),period=int(bpds[i]),donor=bool(isdonor),distance=near,errors=be[i].tolist()))
     if isdonor==(b['left']=='donor_period2'):b['lo']=float(us[i])
     else:b['hi']=float(us[i])
    np.savez(out/f'boundary-{a}-{depth}.npz',u=us,tail=bt,token_ids=bk,period_errors=be)
   # Compare fixed start and first16 iterates with all reference phase states.
   x=fixed[None];proximity=[]
   for t in range(17):
    d2=np.linalg.norm(rt[-2:,1]-x.numpy()[0],axis=1);dother=np.linalg.norm(rt[-(int(periods[0]) or 129):,0]-x.numpy()[0],axis=1)
    proximity.append(dict(step=t,min_distance_period2=float(d2.min()),min_distance_crescent_reference=float(dother.min()),closer='period2' if d2.min()<dother.min() else 'crescent_reference'))
    x,p=step(x,a,c)
   result=dict(alpha=a,log10T=-.156,reference_periods=periods.tolist(),rows=labels,fractions=fractions,segment_labels=seglabs,coarse_transitions=brackets,refined_binary_brackets=active,bisection_history=history,proximity=proximity,mid_final_period_changes=int(np.sum(per!=mp)),scope='Finite sampled basins only. No-return is unresolved/irregular, not chaos proof. Wilson intervals apply only to each IID random arm. Finite crossing observations do not prove fractality/uniqueness. Proximity is descriptive, not a causal explanation.')
   write(out/f'result-{a}.json',result);all_results.append(result);event(stage='cell_complete',alpha=a)
  write(out/'report.json',dict(results=all_results,elapsed=time.monotonic()-start,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,completed=True))
if __name__=='__main__':main()
