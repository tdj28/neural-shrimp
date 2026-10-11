"""Fixed adjacent-pair-preserving full-state pilot; separately admitted profile and target."""
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
from neural_shrimp import tiny_lm_pair_blocks as H
need=H.need
BASE=Path('/data2/neural-shrimp-data/explore/tiny-lm-pair-blocks')
PROFILE=BASE/'attempt001-profile';TARGET=BASE/'attempt001-target'
CARD='docs/plans/tiny-lm-pair-blocks-inputs.json'
CAP=192*1024**2;RESERVE=4*1024**2
PARITY=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-parity-order/attempt001-target',source='e8030d33c47bb817b4ac9829951bbb6137315f57',outer='8e7b47c82b6c25db6873296c83e2a6589a1189e5dc8f3c3a8da2b7af2e69a012',worker='838bd1102ca3bcd005a10ace5c45bbb2e64902d12080203d1a1e7d5861dbb88f')
PARITY_AUDIT=dict(root='/data2/neural-shrimp-data/audits/tiny-lm-parity-order-independent/attempt001',source='b481b47224d41ccc17f73b5a3b6502e8b4e1bddc',outer='e114103e8fde5e925a028e9f9d34f4eec269fb8f01c7a6f16333fc6a4c19c7c2',worker='03e33ee5f82550563639b0d08dfcb49a45948446dbdd637444232e50f33ac586')
RESPONSE_PROFILE=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-response/attempt001-profile',source='b4c3d588da671f25a7fa22c6882e1cdcb42d2079',outer='b37b12abcafd1e6b4abd87caf7f8f7fb6b5acf9ee0fb89cde7a984e8bf9b17a7',worker='246cbba006bd85ade50a5060587e9ba5f0b56da5f8f35bfb86960273c738a84f')
FILES=sorted(set(P.FILES+[CARD,'src/neural_shrimp/tiny_lm_pair_blocks.py','tools/explore_tiny_lm_pair_blocks.py','tools/launch_tiny_lm_pair_blocks.py','tests/test_tiny_lm_pair_blocks.py','tests/test_tiny_lm_pair_blocks_driver.py','docs/plans/tiny-lm-pair-blocks.md','docs/reviews/tiny-lm-parity-order-outcome.json']))
SETTINGS=dict(profile_counts=H.PROFILE_COUNTS,target_counts=H.TARGET_COUNTS,gates=H.GATES,orders={k:list(v)for k,v in H.ORDERS.items()},seeds=[28161,28162],roles=['original','quarter_step'],worker_seconds=dict(profile=60,target=240),rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=CAP)

class Store(N.Store):
 def write(self,name,raw,closing=False):
  need(self.used+len(raw)<=CAP-(0 if closing else RESERVE),'192MiB raw/4MiB closing reserve');p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)

class Calls(N.Calls):
 def __init__(self,store,mode):
  super().__init__(store,mode,60 if mode=='profile'else 240);self.cap=H.PROFILE_COUNTS if mode=='profile'else H.TARGET_COUNTS
  self.attempted=dict.fromkeys(self.cap,0);self.returned=dict.fromkeys(self.cap,0);self.times={k:[]for k in self.cap};self.return_bytes={k:[]for k in self.cap}

class Binding:
 def __init__(self,store,calls):self.store=store;self.calls=calls
 def tick(self):self.calls.check()
 def save(self,*a,**kw):self.store.json(*a,**kw)


def card_check(card,mode):
 need(card['schema']=='tiny-lm-pair-blocks-v1'and card['settings']==SETTINGS and card['parity_producer']==PARITY and card['parity_authority']==PARITY_AUDIT,'literal fixed source design/authority')
 need(card['profile_root']==str(PROFILE)and card['target_root']==str(TARGET),'exclusive source paths')
 need(card['status']==('profile_admitted'if mode=='profile'else'target_admitted')and card['source_review'],'mode-specific reviewed committed admission')
 if mode=='target':need(isinstance(card['profile'],dict)and set(card['profile'])=={'root','source','outer','worker'}and card['profile']['root']==str(PROFILE)and all(card['profile'].values()),'literal completed profile pins')


def history(spec,w):
 root=Path(spec['root']);S.inventory(root,spec['outer'],w.tick);S.inventory(root/'run',spec['worker'],w.tick);rt=S.historical(root,spec['source']);return root,rt


def parity_events(run,files):
 expected={f'events/{i:05d}-{kind}.json'for i in range(1,225)for kind in('input','return')};need({n for n in files if n.startswith('events/')}==expected,'all224 parity input/return pairs')
 result=[]
 for seed in(28161,28162):
  for role in('original','quarter_step'):
   for key in P.H.paths(role):
    for _ in range(16):
     i=len(result)+1;a=S.read(run/f'events/{i:05d}-input.json');b=S.read(run/f'events/{i:05d}-return.json')
     need(a['sequence']==a['index']==i and a['operation']=='hvp'and a['case']==f'seed-{seed}-{role}/'+key and set(b)==set(a)|{'returned','seconds'}and all(b[k]==v for k,v in a.items())and np.isfinite(b['seconds'])and b['seconds']>0,'exact parity case/order/event identity')
     result.append(b)
 return result


def metadata(store,calls):
 run,files,records,worst=P.metadata(store,calls);w=Binding(store,calls)
 pr,rt=history(PARITY,w);pf=S.read(pr/'run/manifest.json')['files'];parity_records=parity_events(pr/'run',pf);presult=S.read(pr/'run/result.json')
 need(presult['passed']and presult['attempted']==presult['returned']=={'hvp':224}and presult['event_pairs']==224 and len(presult['cases'])==4 and all(c['qualified']for c in presult['cases']),'complete four-case parity qualification')
 need(rt['torch']==torch.__version__ and rt['numpy']==np.__version__ and rt['sources']['src/neural_shrimp/tiny_lm.py']['sha256']==N.NATIVE_SHA,'literal reused parity native/runtime')
 ar,_=history(PARITY_AUDIT,w);audit=S.read(ar/'run/result.json');ib=S.read(ar/'run/input-binding.json')
 need(audit['passed']and audit['native_calls']==0 and audit['retained_hvp_checked']==224 and audit['old_reverse_hvp_checked']==32 and audit['descriptive_gate']is True and len(audit['cases'])==4 and all(c['qualified']for c in audit['cases'])and ib['producer']==PARITY and ib['mean']==P.MEAN_AUDIT and ib['ordering']==P.ORDER and ib['ordering_independent']==P.ORDER_AUDIT,'exact independently qualified P/R/Q/T authority')
 oldcard=S.read(ROOT/P.CARD);profile,prt=history(oldcard['profile'],w);cost=S.read(profile/'run/result.json');need(cost['passed']and cost['attempted']==cost['returned']=={'hvp':2},'both completed parity cold profile actions')
 response,rrt=history(RESPONSE_PROFILE,w);rc=S.read(response/'run/result.json');need(rc['passed']and rc['operation_attempted']==rc['operation_returned']==dict(ce=4,gradient=4,hvp=4)and rrt['source_identity']['src/neural_shrimp/tiny_lm.py']['sha256']==N.NATIVE_SHA and rrt['torch']==torch.__version__ and rrt['numpy']==np.__version__,'complete original same-model cold profile')
 # Histories were authenticated by P.metadata; retain every available actual
 # HVP timing as a conservative floor, including the smaller arm8 calls.
 response_times=[x['seconds']for x in S.read(S.RESPONSE/'run/events.json')if x['operation']=='hvp'];need(len(response_times)==256,'all original response HVP costs')
 two=next(root for root,_,_ in S.HISTORY if root.name=='attempt001'and root.parent.name=='tiny-lm-two-snapshot');two_times=[x['seconds']for x in S.read(two/'run/events.json')];need(len(two_times)==240,'all two-snapshot actual costs')
 order_times=[S.read(Path(P.ORDER['root'])/'run/operations'/f'hvp-{i:03d}-result.json')['seconds']for i in range(1,161)]
 upstream=dict(response_cold=[v for c in rc['profile_costs']for v in c['times']['hvp']],response_actual=response_times,two_snapshot_actual=two_times,ordering_actual=order_times,parity_successful_cold=cost['times']['hvp'],parity_actual=[r['seconds']for r in parity_records],mean_and_failed_parity_max=[worst])
 need(all(np.isfinite(x)and x>0 for v in upstream.values()for x in v),'all positive finite cold/actual costs');worst=max(x for v in upstream.values()for x in v)
 store.json('parity-binding.json',dict(producer=PARITY,independent=PARITY_AUDIT,response_profile=RESPONSE_PROFILE,parity_profile=oldcard['profile'],all_upstream_hvp_costs=upstream,effective_max_hvp_seconds=worst))
 return run,files,records,worst,pr/'run',pf,parity_records


def replay_saved_order(run,files,records,label,key,case,eta):
 trace=S.alias(run,label+'/'+key,files);need(trace.shape==(17,15776)and np.array_equal(trace[0],case['direction']),'full inherited response trace/direction')
 order=S.read(run/label/(key+'-order.json'));need(order['indices']==list(P.H.ORDERS[key]),'literal saved application order')
 rows=[r for r in records if r['case']==label+'/'+key];need(len(rows)==16,'entire16 retained actions')
 for i,(state_index,row)in enumerate(zip(P.H.ORDERS[key],rows)):
  state=S.load_object(run,row['state'],files);direction=S.load_object(run,row['direction'],files);action=S.load_object(run,row['returned'],files)
  need(np.array_equal(state,case['reference'][state_index])and np.array_equal(direction,trace[i])and np.array_equal(trace[i+1],trace[i]-eta*action),'all exact same-state/vector/action recurrence joins')
 return trace


def baseline(run,files,records,label,seed,role,case,rate,binding,checkpoint):
 oldbinding=S.read(run/label/'binding.json');need(oldbinding==dict(**binding,checkpoint=checkpoint,eta=rate,full_state=True),'same entire checkpoint/token/qualification genealogy')
 traces={k:replay_saved_order(run,files,records,label,k,case,rate)for k in P.H.paths(role)};replay=P.H.replay(traces['P'],case['ordered']);need(replay['passed'],'inherited literal chronological replay1/4/16')
 values={k:v[-1]for k,v in traces.items()}
 if role=='original':values['R']=P.original_reverse(seed,case,rate)
 even=S.alias(run,label+'/parity_even',files);need(np.array_equal(even,(values['Q']+values['T']-values['P']-values['R'])/2),'literal retained parity contrast')
 comparison=P.H.compare(case['direction'],values['P'],values['R'],values['Q'],values['T'],case['frozen'][16],case['readout_gradients'][2]);need(comparison==S.read(run/label/'result.json')['comparison']and comparison['qualified']and binding['original_qualification'],'unchanged all old qualifications and full comparison')
 return dict(P=values['P'],R=values['R'],frozen=case['frozen'][16],ce_gradient=case['readout_gradients'][2],parity_even=even)


def profile_binding(card,store,calls,worst,identity):
 p,rt=history(card['profile'],Binding(store,calls));row=S.read(p/'run/result.json')
 need(row['passed']and row['mode']=='profile'and row['attempted']==row['returned']==H.PROFILE_COUNTS and row['event_pairs']==2 and row['seconds']<60 and row['peak_rss_kib']<4*1024**2,'complete successful two-HVP profile')
 for n in FILES:
  if n!=CARD:need(rt['sources'][n]==identity[n],'same exact profiled closure '+n)
 need(rt['torch']==torch.__version__ and rt['numpy']==np.__version__,'same fresh profile libraries')
 need(H.forecast(row['times']['hvp'],row['return_bytes']['hvp'],worst,row['preforecast_bytes'])==row['forecast'],'recompute all measured maxima and exact retained-byte forecast')
 current=H.forecast(row['times']['hvp'],row['return_bytes']['hvp'],worst,store.used)
 need(current['seconds']<240 and current['bytes']<CAP,'fixed240s192MiB actual target admission');store.json('profile-reference.json',dict(profile=card['profile'],original_forecast=row['forecast'],current_forecast=current));return current


def main(args):
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ[k]=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source')
 card=S.read(ROOT/CARD);card_check(card,args.mode);need(args.out==(PROFILE if args.mode=='profile'else TARGET)/'run','exclusive worker root');source=N.execution();identity={n:N.entry(ROOT/n)for n in FILES}
 for n in FILES:need((ROOT/n).read_bytes()==N.git_bytes(source,n),'exact complete committed source')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);store=Store(args.out);calls=Calls(store,args.mode);out=dict(passed=False,mode=args.mode,cases=[])
 def stop(*_):raise TimeoutError('owned pair-block deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,calls.seconds)
 try:
  store.json('runtime.json',dict(source_commit=source,pid=os.getpid(),sources=identity,torch=torch.__version__,numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),dtype='float64',device='cpu'));store.json('run-card.json',card)
  for n in FILES:store.write('source/'+n,(ROOT/n).read_bytes())
  run,files,records,worst,pr,pf,pe=metadata(store,calls)
  if args.mode=='target':profile_binding(card,store,calls,worst,identity)
  parent,pin=N.PARENTS[args.mode];N.inventory(parent,pin,legacy=True)
  for seed in([28160]if args.mode=='profile'else[28161,28162]):
   theta,x,y,eta,checkpoint=N.select(parent,seed,50 if args.mode=='profile'else 600);problem=N.LMProblem(N.TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001)
   def hvp(state,direction):return calls.call('hvp',lambda:problem.hvp(torch.as_tensor(state,dtype=torch.float64),torch.as_tensor(direction,dtype=torch.float64)).detach().numpy().copy(),state,direction)
   if args.mode=='profile':
    calls.label='seed-28160-profile';store.json(calls.label+'/binding.json',checkpoint);u=np.random.default_rng(28166).standard_normal(15776);u/=np.linalg.norm(u)
    for _ in range(2):need(np.isfinite(hvp(theta,u)).all(),'finite measured full-size HVP')
   else:
    for role in('original','quarter_step'):
     label=f'seed-{seed}-{role}';case,rate,binding=P.load_case(run,files,records,seed,role,theta,eta);store.json(label+'/binding.json',dict(**binding,checkpoint=checkpoint,eta=rate,full_state=True));values={};old=baseline(pr,pf,pe,label,seed,role,case,rate,binding,checkpoint)
     for name,value in old.items():store.alias(label+'/baseline-'+name,value)
     for key in H.ORDERS:
      calls.label=label+'/'+key
      def save(trace,indices):store.alias(label+'/'+key,trace);store.json(label+'/'+key+'-order.json',dict(indices=indices))
      trace=H.propagate(case['reference'],case['direction'],rate,key,hvp,save,calls.check);values[key]=trace[-1]
     comparison=H.compare(case['direction'],old['P'],old['R'],values['U'],values['V'],old['frozen'],old['ce_gradient'],old['parity_even'],lambda n,v:store.alias(label+'/'+n,v))
     row=dict(seed=seed,rate_role=role,qualified=bool(comparison['qualified']and binding['original_qualification']),comparison=comparison,inherited_response_qualified=binding['original_qualification']);out['cases'].append(row);store.json(label+'/result.json',row)
  need(calls.attempted==calls.returned==calls.cap,'all128 or profile2 native calls exactly')
  if args.mode=='profile':
   out['preforecast_bytes']=store.used;bound=H.forecast(calls.times['hvp'],calls.return_bytes['hvp'],worst,store.used);out.update(forecast=bound,passed=bool(bound['seconds']<240 and bound['bytes']<CAP),target_arrays_unopened=True)
  else:out.update(passed=True,descriptive=H.descriptive(out['cases']),scope='Consumed full-state adjacent-pair-preserving directional contrast only. No projected model, eigenbasis attribution, operator-norm, forecast, training or geometry utility.')
 except BaseException as e:out.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-calls.started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted=calls.attempted,returned=calls.returned,times=calls.times,return_bytes=calls.return_bytes,event_pairs=calls.events);out['passed']=bool(out['passed']and out['seconds']<calls.seconds and out['peak_rss_kib']<4*1024**2);store.finish(out)
 print(json.dumps({k:out.get(k)for k in('passed','seconds','forecast','descriptive','error')}));return 0 if out['passed']else 2


def parser():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
