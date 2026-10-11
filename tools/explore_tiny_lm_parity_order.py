"""Fixed full-state parity-order pilot; separately admitted profile and target."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import numpy as np
import torch
import explore_tiny_lm_temporal_mean as N
import explore_tiny_lm_common_subspace as S
from neural_shrimp import tiny_lm_parity_order as H
need=H.need
BASE=Path('/data2/neural-shrimp-data/explore/tiny-lm-parity-order')
PROFILE=BASE/'attempt002-profile';TARGET=BASE/'attempt001-target'
CARD='docs/plans/tiny-lm-parity-order-inputs.json'
CAP=192*1024**2;RESERVE=4*1024**2
FAILED_PROFILE=dict(root=str(BASE/'attempt001-profile'),source='bdfdc963359d988bd807bb4dcf720f4391f70c4a',outer='8e51dcad6e0e9c39952cbc794533ee44852e110a3c04e608a6f4b2ab2aa831d9',worker='424b1210db90ba2c083a9f2331bd65fea0dcffa8f5bc97b98ad3178c27b884d7')
MEAN_AUDIT=dict(root='/data2/neural-shrimp-data/audits/tiny-lm-temporal-mean-independent/attempt001',source='8624111d26fc8e3ee916fe89c2f32fb5286139d2',outer='01efa208f3bd15e11ce2a969ff2216c10bc5f522ed2f7350321822a63c7d9eb1',worker='7d99e6c55de8ffa0bc1167af9106ffe32c548293f32c5ee01ddad5a5d2125e1b')
ORDER=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-ordering/attempt001-target',source='358a93ca3cc6434d5b8ea5c4989ad391fb471a0f',outer='eb3f0c4503a73ac21443fe56b88a3834ec4cf13ae98fc98d53d3e40b4e8fc686',worker='ce69af6c687cc66f6cc5118af7744734a8266cc233a6862e34e3d05ea60870ec')
ORDER_AUDIT=dict(root='/data2/neural-shrimp-data/audits/tiny-lm-ordering-independent/attempt001',source='f408bba83021bc5ea57e7fe5b5eb11f582fefb64',outer='13d4b20207e2d94052ab003d6b8488a07e68aca1e74d896736b6191bf5016f35',worker='b57d546bfa1aac86bcd4a2b232c30abf71dfc34f7e7ceee7c6d8da15b65068d5')
FILES=sorted(set(N.FILES+S.FILES+[CARD,'src/neural_shrimp/tiny_lm_parity_order.py','tools/explore_tiny_lm_parity_order.py','tools/launch_tiny_lm_parity_order.py','tests/test_tiny_lm_parity_order.py','tests/test_tiny_lm_parity_order_driver.py','docs/plans/tiny-lm-parity-order.md','docs/reviews/tiny-lm-common-subspace-outcome.json','docs/reviews/tiny-lm-parity-order-profile001.json']))
SETTINGS=dict(profile_counts=H.PROFILE_COUNTS,target_counts=H.TARGET_COUNTS,gates=H.GATES,orders={k:list(v)for k,v in H.ORDERS.items()},seeds=[28161,28162],roles=['original','quarter_step'],worker_seconds=dict(profile=60,target=300),rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=CAP)

class Store(N.Store):
 def write(self,name,raw,closing=False):
  need(self.used+len(raw)<=CAP-(0 if closing else RESERVE),'192MiB raw/4MiB closing reserve');p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)

class Calls(N.Calls):
 def __init__(self,store,mode):
  super().__init__(store,mode,60 if mode=='profile'else 300);self.cap=H.PROFILE_COUNTS if mode=='profile'else H.TARGET_COUNTS
  self.attempted=dict.fromkeys(self.cap,0);self.returned=dict.fromkeys(self.cap,0);self.times={k:[]for k in self.cap};self.return_bytes={k:[]for k in self.cap}

class Binding:
 def __init__(self,store,calls):self.store=store;self.calls=calls
 def tick(self):self.calls.check()
 def save(self,*a,**kw):self.store.json(*a,**kw)


def card_check(card,mode):
 need(card['schema']=='tiny-lm-parity-order-v1'and card['settings']==SETTINGS and card['mean_authority']==MEAN_AUDIT and card['ordering_authority']==ORDER_AUDIT,'literal fixed source design/authority')
 need(card['profile_root']==str(PROFILE)and card['target_root']==str(TARGET),'exclusive source paths')
 need(card['status']==('profile_admitted'if mode=='profile'else'target_admitted')and card['source_review'],'mode-specific reviewed committed admission')
 if mode=='target':need(isinstance(card['profile'],dict)and set(card['profile'])=={'root','source','outer','worker'}and card['profile']['root']==str(PROFILE)and all(card['profile'].values()),'literal completed profile pins')


def history(spec,w):
 root=Path(spec['root']);S.inventory(root,spec['outer'],w.tick);S.inventory(root/'run',spec['worker'],w.tick);rt=S.historical(root,spec['source']);return root,rt


def failed_profile(w):
 p=Path(FAILED_PROFILE['root']);S.inventory(p,FAILED_PROFILE['outer'],w.tick);S.inventory(p/'run',FAILED_PROFILE['worker'],w.tick);S.historical(p,FAILED_PROFILE['source'],2)
 row=S.read(p/'run/result.json');receipt=S.read(ROOT/'docs/reviews/tiny-lm-parity-order-profile001.json')
 need(receipt['execution']==FAILED_PROFILE['source']and receipt['outer']['sha256']==FAILED_PROFILE['outer']and receipt['worker']['sha256']==FAILED_PROFILE['worker']and receipt['result']==row,'literal retained original240s rejection')
 need(not row['passed']and row['attempted']==row['returned']=={'hvp':2}and row['event_pairs']==2 and row['target_arrays_unopened']and 'error'not in row and row['seconds']<60 and row['peak_rss_kib']<4*1024**2,'complete original measured profile')
 need(H.forecast(row['times']['hvp'],row['return_bytes']['hvp'],row['forecast']['old_max_hvp_seconds'],row['preforecast_bytes'])==row['forecast']and 240<row['forecast']['seconds']<300 and row['forecast']['bytes']<CAP,'original admission failed only time')
 w.save('retained-profile001.json',FAILED_PROFILE)


def metadata(store,calls):
 w=Binding(store,calls);failed_profile(w);oldcard=S.read(ROOT/S.CARD);need(oldcard['independent']==MEAN_AUDIT,'fixed already sealed mean authority');run,files,result=S.bind(w,oldcard);records=S.events(run,files);rt=S.read(run/'runtime.json')
 need(rt['torch']==torch.__version__ and rt['numpy']==np.__version__ and N.entry(ROOT/'src/neural_shrimp/tiny_lm.py')['sha256']==N.NATIVE_SHA,'same native and measured libraries')
 order,ort=history(ORDER,w);audit,_=history(ORDER_AUDIT,w);a=S.read(audit/'run/result.json');ib=S.read(audit/'run/input-binding.json')
 need(a['passed']and a['native_calls']==0 and a['retained_hvp_checked']==160 and a['rows']==16 and ib['execution']==ORDER['source'],'complete independent reversal audit')
 need(any(v['root']==ORDER['root']and v['manifest']==ORDER['outer']for v in ib['inventories']),'literal reverse producer in independent authority')
 need(ort['sources']['src/neural_shrimp/tiny_lm.py']['sha256']==N.NATIVE_SHA and ort['torch']==torch.__version__ and ort['numpy']==np.__version__,'same ordering native/runtime')
 oldprofile=S.read(N.PROFILE_ROOT/'run/result.json');times=[r['seconds']for r in records if r['operation']=='hvp']
 need(len(times)==1088 and len(oldprofile['times']['hvp'])==2,'complete actual1088 plus cold2 prior costs');worst=max(times+oldprofile['times']['hvp']+S.read(Path(FAILED_PROFILE['root'])/'run/result.json')['times']['hvp'])
 store.json('ordering-binding.json',dict(producer=ORDER,independent=ORDER_AUDIT,old_max_hvp_seconds=worst,actual_target_hvp_times=1088,cold_profile_hvp_times=2,retained_failed_parity_profile_hvp_times=2))
 # Retain all four abstentions as history without treating them as operator evidence.
 receipt=S.read(ROOT/'docs/reviews/tiny-lm-common-subspace-outcome.json');common=dict(root=receipt['root'],source=receipt['execution'],outer=receipt['outer']['sha256'],worker=receipt['worker']['sha256']);cr,_=history(common,w);co=S.read(cr/'run/result.json');need(co['passed']and co['qualified_cases']==0 and co['abstained_cases']==4 and co['native_calls']==0,'four unchanged leakage abstentions');store.json('retained-subspace-abstentions.json',common)
 return run,files,records,worst


def profile_binding(card,store,calls,worst,identity):
 p,rt=history(card['profile'],Binding(store,calls));row=S.read(p/'run/result.json')
 need(row['passed']and row['mode']=='profile'and row['attempted']==row['returned']==H.PROFILE_COUNTS and row['event_pairs']==2 and row['seconds']<60 and row['peak_rss_kib']<4*1024**2,'complete successful two-HVP profile')
 for n in FILES:
  if n!=CARD:need(rt['sources'][n]==identity[n],'same exact profiled closure '+n)
 need(rt['torch']==torch.__version__ and rt['numpy']==np.__version__,'same fresh profile libraries')
 need(H.forecast(row['times']['hvp'],row['return_bytes']['hvp'],worst,row['preforecast_bytes'])==row['forecast'],'recompute all measured maxima and exact retained-byte forecast')
 current=H.forecast(row['times']['hvp'],row['return_bytes']['hvp'],worst,store.used)
 need(current['seconds']<300 and current['bytes']<CAP,'fixed300s192MiB actual target admission');store.json('profile-reference.json',dict(profile=card['profile'],original_forecast=row['forecast'],current_forecast=current));return current


def load_case(run,files,records,seed,role,theta,eta):
 mean,actions,ordered,rate,binding=S.acquire_case(run,files,records,seed,role);del actions
 original,_=N.load_case(seed,theta,eta);label=f'seed-{seed}-{role}'
 if role=='original':case=original
 else:
  alias=lambda n:S.alias(run,label+'/'+n,files)
  case=dict(reference=alias('reference'),reference_gradients=alias('reference-gradients'),ordered=ordered,frozen=alias('frozen'),direction=original['direction'],readout_gradients=alias('readout-gradients'))
  need(np.array_equal(case['reference'][0],theta)and np.array_equal(case['reference'][1:],case['reference'][:-1]-rate*case['reference_gradients'][:-1]),'same full matched reference trajectory')
  need(np.array_equal(case['reference_gradients'][0],original['reference_gradients'][0])and np.array_equal(case['readout_gradients'],np.asarray([case['reference_gradients'][h]-.001*case['reference'][h]for h in(1,4,16)])),'same initial gradient and exact retained CE readout')
 need(binding['original_qualification']and rate==eta/(1 if role=='original'else 4)and np.array_equal(case['direction'],ordered[0]),'unchanged inherited1% qualifications and literal direction')
 for n in('reference','reference_gradients','ordered','frozen'):need(case[n].shape==(17,15776)and np.isfinite(case[n]).all(),'complete finite case '+n)
 return case,rate,binding


def original_reverse(seed,case,eta):
 root=Path(ORDER['root'])/'run';label=f'seed-{seed}-arm-256';folder=root/label/'direction-0-horizon-16';load=lambda p:np.load(p,allow_pickle=False)
 need(S.read(root/label/'inputs.json')==S.read(N.RESPONSE/'run'/label/'inputs.json'),'exact old reverse checkpoint/tokens/rate identity')
 trace=load(folder/'reverse-trace.npy');order=S.read(folder/'state-order.json');need(trace.shape==(17,15776)and trace.dtype==np.float64 and np.isfinite(trace).all()and np.array_equal(trace[0],case['direction'])and order['indices']==list(H.R)and order['operation_last']-order['operation_first']==15,'literal entire reverse trace')
 for i,index in enumerate(range(order['operation_first'],order['operation_last']+1)):
  pre=root/'operations'/f'hvp-{index:03d}';theta=load(Path(str(pre)+'-theta.npy'));vector=load(Path(str(pre)+'-vector.npy'));action=load(Path(str(pre)+'-return.npy'))
  need(np.array_equal(theta,case['reference'][15-i])and np.array_equal(vector,trace[i])and np.array_equal(trace[i+1],trace[i]-eta*action),'all16 exact reverse acquisition state/vector/action joins')
 return trace[-1]


def main(args):
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ[k]=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source')
 card=S.read(ROOT/CARD);card_check(card,args.mode);need(args.out==(PROFILE if args.mode=='profile'else TARGET)/'run','exclusive worker root');source=N.execution();identity={n:N.entry(ROOT/n)for n in FILES}
 for n in FILES:need((ROOT/n).read_bytes()==N.git_bytes(source,n),'exact complete committed source')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);store=Store(args.out);calls=Calls(store,args.mode);out=dict(passed=False,mode=args.mode,cases=[])
 def stop(*_):raise TimeoutError('owned parity-order deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,calls.seconds)
 try:
  store.json('runtime.json',dict(source_commit=source,pid=os.getpid(),sources=identity,torch=torch.__version__,numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),dtype='float64',device='cpu'));store.json('run-card.json',card)
  for n in FILES:store.write('source/'+n,(ROOT/n).read_bytes())
  run,files,records,worst=metadata(store,calls)
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
     label=f'seed-{seed}-{role}';case,rate,binding=load_case(run,files,records,seed,role,theta,eta);store.json(label+'/binding.json',dict(**binding,checkpoint=checkpoint,eta=rate,full_state=True));values={};traces={}
     for key in H.paths(role):
      calls.label=label+'/'+key
      def save(trace,indices):store.alias(label+'/'+key,trace);store.json(label+'/'+key+'-order.json',dict(indices=indices))
      trace=H.propagate(case['reference'],case['direction'],rate,key,hvp,save,calls.check);values[key]=trace[-1];traces[key]=trace
     if role=='original':values['R']=original_reverse(seed,case,rate)
     replay=H.replay(traces['P'],case['ordered']);store.json(label+'/chronological-replay.json',replay)
     comparison=H.compare(case['direction'],values['P'],values['R'],values['Q'],values['T'],case['frozen'][16],case['readout_gradients'][2],lambda n,v:store.alias(label+'/'+n,v))
     row=dict(seed=seed,rate_role=role,qualified=bool(replay['passed']and comparison['qualified']),chronological_replay=replay,comparison=comparison,inherited_response_qualified=binding['original_qualification']);out['cases'].append(row);store.json(label+'/result.json',row)
  need(calls.attempted==calls.returned==calls.cap,'all224 or profile2 native calls exactly')
  if args.mode=='profile':
   out['preforecast_bytes']=store.used;bound=H.forecast(calls.times['hvp'],calls.return_bytes['hvp'],worst,store.used);out.update(forecast=bound,passed=bool(bound['seconds']<300 and bound['bytes']<CAP),target_arrays_unopened=True)
  else:out.update(passed=True,descriptive_gate=H.descriptive_gate(out['cases']),scope='Consumed full-state directional parity-order contrast only. No projected model, eigenbasis attribution, operator-norm, forecast, training or geometry utility.')
 except BaseException as e:out.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-calls.started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted=calls.attempted,returned=calls.returned,times=calls.times,return_bytes=calls.return_bytes,event_pairs=calls.events);out['passed']=bool(out['passed']and out['seconds']<calls.seconds and out['peak_rss_kib']<4*1024**2);store.finish(out)
 print(json.dumps({k:out.get(k)for k in('passed','seconds','forecast','descriptive_gate','error')}));return 0 if out['passed']else 2


def parser():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
