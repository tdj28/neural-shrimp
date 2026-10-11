"""Fixed first-order temporal correction to the adjacent-block mean prediction."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import numpy as np
import torch
import explore_tiny_lm_parity_order as P
N,S=P.N,P.S
from neural_shrimp import tiny_lm_block_first_order as H
import explore_tiny_lm_block_mean as M
import explore_tiny_lm_pair_blocks as B
need=H.need
BASE=Path('/data2/neural-shrimp-data/explore/tiny-lm-block-first-order')
PROFILE=BASE/'attempt001-profile';TARGET=BASE/'attempt001-target'
CARD='docs/plans/tiny-lm-block-first-order-inputs.json'
CAP=512*1024**2;RESERVE=4*1024**2
PARITY=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-parity-order/attempt001-target',source='e8030d33c47bb817b4ac9829951bbb6137315f57',outer='8e7b47c82b6c25db6873296c83e2a6589a1189e5dc8f3c3a8da2b7af2e69a012',worker='838bd1102ca3bcd005a10ace5c45bbb2e64902d12080203d1a1e7d5861dbb88f')
PARITY_AUDIT=dict(root='/data2/neural-shrimp-data/audits/tiny-lm-parity-order-independent/attempt001',source='b481b47224d41ccc17f73b5a3b6502e8b4e1bddc',outer='e114103e8fde5e925a028e9f9d34f4eec269fb8f01c7a6f16333fc6a4c19c7c2',worker='03e33ee5f82550563639b0d08dfcb49a45948446dbdd637444232e50f33ac586')
RESPONSE_PROFILE=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-response/attempt001-profile',source='b4c3d588da671f25a7fa22c6882e1cdcb42d2079',outer='b37b12abcafd1e6b4abd87caf7f8f7fb6b5acf9ee0fb89cde7a984e8bf9b17a7',worker='246cbba006bd85ade50a5060587e9ba5f0b56da5f8f35bfb86960273c738a84f')
PAIR={'root': '/data2/neural-shrimp-data/explore/tiny-lm-pair-blocks/attempt001-target', 'source': '1c91e14fb01276a5d764e8facc65ab6585f38606', 'outer': '03022d864990b299468c026a1b1bf9210a61a028bfae91f02a604fe7fb3cc5e6', 'worker': '615376d85ffcc62f6d2a24a0799544dab16c3e4a61b18cb4fae9f2e22fc19b58'}
PAIR_AUDIT={'root': '/data2/neural-shrimp-data/audits/tiny-lm-pair-blocks-independent/attempt001', 'source': '99e4efa111c97bc09c562dbbd30f5c90aecef1e8', 'outer': '906fc65ff04aee2d86aedf7bc1d63cd62a0016c7076412b4c8015f9b754896c5', 'worker': 'e6a22991e3bcddf5eabe74ecec5d3d8f3fd1b03083dddd27297a38a0f91b429f'}
MEAN=dict(root=str(M.TARGET),source='858fec5feaabd4c6eee31449445330d390500a01',outer='7f4fedc79c4b3ae16980e406cb768f9e32353eb48e1762d41fd7d08390c60fe5',worker='833baaa3d853a03b71576dd3f590bdc321c12efb42571dd78676cbc1377c40fe')
FILES=sorted(set(M.FILES+[CARD,'src/neural_shrimp/tiny_lm_block_first_order.py','tools/explore_tiny_lm_block_first_order.py','tools/launch_tiny_lm_block_first_order.py','tests/test_tiny_lm_block_first_order.py','tests/test_tiny_lm_block_first_order_driver.py','docs/plans/tiny-lm-block-first-order.md','docs/reviews/tiny-lm-block-mean-outcome.json','docs/reviews/tiny-lm-block-mean-independent.json']))
SETTINGS=dict(profile_counts=H.PROFILE_COUNTS,target_counts=H.TARGET_COUNTS,gates=H.GATES,blocks=8,powers=8,seeds=[28161,28162],roles=['original','quarter_step'],worker_seconds=dict(profile=60,target=1200),rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=CAP)

class Store(N.Store):
 def write(self,name,raw,closing=False):
  need(self.used+len(raw)<=CAP-(0 if closing else RESERVE),'512MiB raw/4MiB closing reserve');p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)

class Calls(N.Calls):
 def __init__(self,store,mode):
  super().__init__(store,mode,60 if mode=='profile'else 1200);self.cap=H.PROFILE_COUNTS if mode=='profile'else H.TARGET_COUNTS
  self.attempted=dict.fromkeys(self.cap,0);self.returned=dict.fromkeys(self.cap,0);self.times={k:[]for k in self.cap};self.return_bytes={k:[]for k in self.cap}

class Binding:
 def __init__(self,store,calls):self.store=store;self.calls=calls
 def tick(self):self.calls.check()
 def save(self,*a,**kw):self.store.json(*a,**kw)


def card_check(card,mode):
 need(card['schema']=='tiny-lm-block-first-order-v1'and card['settings']==SETTINGS and card['mean_producer']==MEAN,'literal fixed source design/authority')
 need(card['profile_root']==str(PROFILE)and card['target_root']==str(TARGET),'exclusive source paths')
 a=card['mean_authority'];need(isinstance(a,dict)and set(a)=={'root','source','outer','worker'}and all(isinstance(v,str)and v for v in a.values()),'sealed independent block-mean authority')
 need(card['status']==('profile_admitted'if mode=='profile'else'target_admitted')and card['source_review'],'mode-specific reviewed committed admission')
 if mode=='target':need(isinstance(card['profile'],dict)and set(card['profile'])=={'root','source','outer','worker'}and card['profile']['root']==str(PROFILE)and all(card['profile'].values()),'literal completed profile pins')


def history(spec,w):
 root=Path(spec['root']);S.inventory(root,spec['outer'],w.tick);S.inventory(root/'run',spec['worker'],w.tick);rt=S.historical(root,spec['source']);return root,rt


def metadata(store,calls):
 run,files,records,worst,pr,pf,pe=M.metadata(store,calls);w=Binding(store,calls);card=S.read(ROOT/CARD)
 mean,rt=history(MEAN,w);out=S.read(mean/'run/result.json');authority,art=history(card['mean_authority'],w);audit=S.read(authority/'run/result.json');binding=S.read(authority/'run/input-binding.json')
 need(out['passed']and out['attempted']==out['returned']=={'hvp':512}and out['event_pairs']==512 and len(out['cases'])==4 and all(v['qualified']for v in out['cases']),'complete fixed512 block-mean baseline')
 need(audit['passed']and audit['native_calls']==0 and audit['retained_hvp_checked']==512 and len(audit['cases'])==4 and all(v['qualified']for v in audit['cases'])and binding['producer']==MEAN,'literal complete independently qualified512 baseline')
 need(audit['descriptive']==out['descriptive']and out['descriptive']==dict(all_four_qualified=True,accurate_in_both_originals=False,tenfold_better_than_single_step_mean_in_both=True),'retain both prior accuracy failure and relative improvement')
 need(rt['torch']==torch.__version__ and rt['numpy']==np.__version__ and rt['sources']['src/neural_shrimp/tiny_lm.py']['sha256']==N.NATIVE_SHA,'same native model and libraries')
 costs=[]
 for i in range(1,513):
  x=S.read(mean/'run/events'/f'{i:05d}-input.json');y=S.read(mean/'run/events'/f'{i:05d}-return.json')
  need(x['operation']=='hvp'and x['index']==x['sequence']==i and all(y[k]==v for k,v in x.items())and set(y)==set(x)|{'returned','seconds'},'all actual512 saved block costs');costs.append(y['seconds'])
 profile=S.read(ROOT/M.CARD)['profile'];p=Path(profile['root']);S.inventory(p,profile['outer'],calls.check);S.inventory(p/'run',profile['worker'],calls.check);S.historical(p,profile['source'],2);po=S.read(p/'run/result.json')
 need(not po['passed']and 'error'not in po and po['attempted']==po['returned']=={'hvp':2}and po['target_arrays_unopened']and 900<po['forecast']['seconds']<1200,'retained original time-rejected complete profile')
 costs+=po['times']['hvp'];need(len(costs)==514 and all(np.isfinite(v)and v>0 for v in costs),'all prior new/cold costs');worst=max(worst,*costs)
 store.json('block-mean-binding.json',dict(producer=MEAN,independent=card['mean_authority'],profile=profile,all514_costs=costs,effective_max_hvp_seconds=worst))
 return run,files,records,worst,pr,pf,pe


def comparison_inputs(run,files,label,case,role):return M.comparison_inputs(run,files,label,case,role)


def profile_binding(card,store,calls,worst,identity):
 p,rt=history(card['profile'],Binding(store,calls));row=S.read(p/'run/result.json')
 need(row['passed']and row['mode']=='profile'and row['attempted']==row['returned']==H.PROFILE_COUNTS and row['event_pairs']==2 and row['target_arrays_unopened']and row['seconds']<60 and row['peak_rss_kib']<4*1024**2,'complete successful two-HVP profile')
 need(set(rt['sources'])==set(FILES),'same complete profiled closure')
 for n in FILES:
  if n!=CARD:need(rt['sources'][n]==identity[n],'same exact profiled closure '+n)
 need(rt['torch']==torch.__version__ and rt['numpy']==np.__version__,'same measured libraries')
 need(H.forecast(row['times']['hvp'],row['return_bytes']['hvp'],worst,row['preforecast_bytes'])==row['forecast'],'recompute all native/cold maxima and full512MiB cost')
 current=H.forecast(row['times']['hvp'],row['return_bytes']['hvp'],worst,store.used)
 need(current['seconds']<1200 and current['bytes']<CAP,'fixed1200s512MiB admission');store.json('profile-reference.json',dict(profile=card['profile'],original_forecast=row['forecast'],current_forecast=current));return current


def saved_mean(label,case,old,responses,slopes,rate,full_binding):
 run=Path(MEAN['root'])/'run';files=S.read(run/'manifest.json')['files'];row=S.read(run/label/'result.json');bind=S.read(run/label/'binding.json')
 need(bind==full_binding and bind['eta']==rate and bind['full_state']and bind['original_qualification'],'same entire consumed checkpoint/token/direction/rate binding')
 powers=S.alias(run,label+'/block-mean-powers',files);need(powers.shape==(9,15776)and np.array_equal(powers[0],case['direction']),'complete independently replayed powers')
 diagonal=[]
 for i in range(8):
  ends=S.alias(run,label+f'/power-{i}-block-ends',files);need(ends.shape==(8,15776)and np.array_equal(powers[i+1],np.sum(ends,axis=0)/8),'literal original mean recurrence');diagonal.append(ends[i])
 for key,value in old.items():need(np.array_equal(S.alias(run,label+'/baseline-'+key,files),value),'literal unchanged inherited baseline '+key)
 need(np.array_equal(S.alias(run,label+'/responses',files),np.asarray(responses))and S.read(run/label/'nonlinear-readout.json')['slopes']==slopes,'same full nonlinear/CE outcomes')
 comparison=M.H.compare(powers[-1],old,responses,slopes);need(comparison==row['comparison']and row['qualified'],'unchanged qualified block-mean baseline')
 return powers,np.asarray(diagonal),comparison


def main(args):
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ[k]=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source')
 card=S.read(ROOT/CARD);card_check(card,args.mode);need(args.out==(PROFILE if args.mode=='profile'else TARGET)/'run','exclusive worker root');source=N.execution();identity={n:N.entry(ROOT/n)for n in FILES}
 for n in FILES:need((ROOT/n).read_bytes()==N.git_bytes(source,n),'exact complete committed source')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);store=Store(args.out);calls=Calls(store,args.mode);out=dict(passed=False,mode=args.mode,cases=[])
 def stop(*_):raise TimeoutError('owned first-order deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,calls.seconds)
 try:
  store.json('runtime.json',dict(source_commit=source,pid=os.getpid(),sources=identity,torch=torch.__version__,numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),dtype='float64',device='cpu'));store.json('run-card.json',card)
  for n in FILES:store.write('source/'+n,(ROOT/n).read_bytes())
  run,files,records,worst,pr,pf,pe=metadata(store,calls)
  if args.mode=='target':profile_binding(card,store,calls,worst,identity)
  parent,pin=N.PARENTS[args.mode];N.inventory(parent,pin,legacy=True)
  for seed in([28160]if args.mode=='profile'else[28161,28162]):
   theta,x,y,eta,checkpoint=N.select(parent,seed,50 if args.mode=='profile'else 600);problem=N.LMProblem(N.TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001)
   def hvp(state,direction,j=None,k=None,phase=None):
    if j is not None:calls.label=label+f'/correction-{j}/block-{k}/phase-{phase}'
    return calls.call('hvp',lambda:problem.hvp(torch.as_tensor(state,dtype=torch.float64),torch.as_tensor(direction,dtype=torch.float64)).detach().numpy().copy(),state,direction)
   if args.mode=='profile':
    calls.label='seed-28160-profile';store.json(calls.label+'/binding.json',checkpoint);u=np.random.default_rng(28166).standard_normal(15776);u/=np.linalg.norm(u)
    for _ in range(2):need(np.isfinite(hvp(theta,u)).all(),'finite measured full-size HVP')
   else:
    for role in('original','quarter_step'):
     label=f'seed-{seed}-{role}';case,rate,binding=P.load_case(run,files,records,seed,role,theta,eta);full_binding=dict(**binding,checkpoint=checkpoint,eta=rate,full_state=True);store.json(label+'/binding.json',full_binding)
     old,responses,slopes,eps=comparison_inputs(run,files,label,case,role)
     for name,value in old.items():store.alias(label+'/baseline-'+name,value)
     store.alias(label+'/responses',np.asarray(responses));store.json(label+'/nonlinear-readout.json',dict(slopes=slopes,epsilons=eps))
     powers,diagonal,baseline=saved_mean(label,case,old,responses,slopes,rate,full_binding)
     store.alias(label+'/block-mean-powers',powers);store.alias(label+'/block-diagonal-actions',diagonal)
     pred=H.correct(case['reference'],powers,diagonal,rate,hvp,lambda n,v:store.alias(label+'/'+n,v),calls.check);store.alias(label+'/first-order-prediction',pred)
     comparison=M.H.compare(pred,old,responses,slopes);comparison['comparisons']['first_order']=comparison['comparisons'].pop('block_mean');comparison['gates']=H.GATES
     row=dict(seed=seed,rate_role=role,qualified=bool(comparison['qualified']and binding['original_qualification']),comparison=comparison,block_mean_comparison=baseline,inherited_response_qualified=binding['original_qualification']);out['cases'].append(row);store.json(label+'/result.json',row)
  need(calls.attempted==calls.returned==calls.cap,'all448 or profile2 native calls exactly')
  if args.mode=='profile':
   out['preforecast_bytes']=store.used;bound=H.forecast(calls.times['hvp'],calls.return_bytes['hvp'],worst,store.used);out.update(forecast=bound,passed=bool(bound['seconds']<1200 and bound['bytes']<CAP),target_arrays_unopened=True)
  else:out.update(passed=True,descriptive=H.descriptive(out['cases']),scope='Consumed full-state first derivative of the ordered block product about its constant mean path. Uses the whole observed path; epsilon=1 truncation, not an online forecast, isolated ordering effect, training intervention or geometry utility.')
 except BaseException as e:out.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-calls.started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted=calls.attempted,returned=calls.returned,times=calls.times,return_bytes=calls.return_bytes,event_pairs=calls.events);out['passed']=bool(out['passed']and out['seconds']<calls.seconds and out['peak_rss_kib']<4*1024**2);store.finish(out)
 print(json.dumps({k:out.get(k)for k in('passed','seconds','forecast','descriptive','error')}));return 0 if out['passed']else 2


def parser():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
