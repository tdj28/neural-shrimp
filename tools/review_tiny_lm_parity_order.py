"""Independent full-state parity-order replay using saved actions and fsum."""
import hashlib,json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_tiny_lm_temporal_mean as M
B=M.B;need,read,entry,exact,array,dot,norm,close=M.need,M.read,M.entry,M.exact,M.array,M.dot,M.norm,M.close
ROOT=Path(__file__).resolve().parents[1];DATA=M.DATA
OUT=DATA/'audits/tiny-lm-parity-order-independent/attempt001/run'
TARGET=DATA/'explore/tiny-lm-parity-order/attempt001-target'
SPEC=dict(root=str(TARGET),source='e8030d33c47bb817b4ac9829951bbb6137315f57',outer='8e7b47c82b6c25db6873296c83e2a6589a1189e5dc8f3c3a8da2b7af2e69a012',worker='838bd1102ca3bcd005a10ace5c45bbb2e64902d12080203d1a1e7d5861dbb88f')
MEAN=dict(root=str(DATA/'audits/tiny-lm-temporal-mean-independent/attempt001'),source='8624111d26fc8e3ee916fe89c2f32fb5286139d2',outer='01efa208f3bd15e11ce2a969ff2216c10bc5f522ed2f7350321822a63c7d9eb1',worker='7d99e6c55de8ffa0bc1167af9106ffe32c548293f32c5ee01ddad5a5d2125e1b')
ORDER=dict(root=str(B.TARGET),source='358a93ca3cc6434d5b8ea5c4989ad391fb471a0f',outer='eb3f0c4503a73ac21443fe56b88a3834ec4cf13ae98fc98d53d3e40b4e8fc686',worker='ce69af6c687cc66f6cc5118af7744734a8266cc233a6862e34e3d05ea60870ec')
OA=dict(root=str(DATA/'audits/tiny-lm-ordering-independent/attempt001'),source='f408bba83021bc5ea57e7fe5b5eb11f582fefb64',outer='13d4b20207e2d94052ab003d6b8488a07e68aca1e74d896736b6191bf5016f35',worker='b57d546bfa1aac86bcd4a2b232c30abf71dfc34f7e7ceee7c6d8da15b65068d5')
THREADS=M.THREADS
GATES=dict(replay=1e-10,transpose=1e-8,decomposition=1e-12,material=.2)
ORDERS=dict(P=list(range(16)),R=list(range(15,-1,-1)),Q=list(range(0,16,2))+list(range(1,16,2)))
ORDERS['T']=list(reversed(ORDERS['Q']))
FILES=M.FILES+['tools/review_tiny_lm_parity_order.py','tools/launch_tiny_lm_parity_order_review.py','tests/test_tiny_lm_parity_order_review.py','docs/plans/tiny-lm-parity-order-independent.md']

def reduce(u,p,r,q,t,f,g):
 need(all(v.ndim==1 and v.shape==u.shape and np.isfinite(v).all()for v in(u,p,r,q,t,f,g)),'finite full common vectors')
 even=np.array([math.fsum([float(a),float(b),-float(c),-float(d)])/2 for a,b,c,d in zip(q,t,p,r,strict=True)])
 op=np.array([(float(a)-float(b))/2 for a,b in zip(p,r,strict=True)]);oq=np.array([(float(a)-float(b))/2 for a,b in zip(q,t,strict=True)])
 od=oq-op;delta=q-p;scale=max(norm(p),norm(r),norm(q),norm(t),1e-12)
 ve=norm(delta-even-od)/scale;ne,no,nd=norm(even),norm(od),norm(delta);cross=dot(even,od);se=abs(nd*nd-ne*ne-no*no-2*cross)/max(scale*scale,nd*nd,ne*ne+no*no)
 tr=[abs(dot(u,a-b))/max(norm(u)*max(norm(a),norm(b)),1e-12)for a,b in((p,r),(q,t))];np_=max(norm(p),1e-12)
 def measure(v):
  n=norm(v);return dict(gain=n,relative_to_ordered=norm(v-p)/np_,alignment=None if n==0 else dot(v,p)/(n*np_),training_ce_slope=dot(g,v))
 row=dict(qualified=max(tr)<=GATES['transpose']and ve<=GATES['decomposition']and se<=GATES['decomposition'],transpose_errors=tr,vector_identity_error=ve,squared_identity_error=se,even_norm=ne,even_relative_ordered=ne/np_,even_relative_frozen_error=ne/max(norm(p-f),1e-12),odd_difference_norm=no,chronological_odd_norm=norm(op),grouped_odd_norm=norm(oq),twice_cross_term=2*cross,grouped_minus_ordered_squared=nd*nd,even_training_ce_slope=dot(g,even),products={k:measure(v)for k,v in [('P',p),('R',r),('Q',q),('T',t)]},gates=GATES)
 return row,dict(parity_even=even,chronological_odd=op,grouped_odd=oq)

def derived(saved,ind,terms):
 bound=16*np.finfo(float).eps*np.sum(np.abs(np.asarray(terms)),axis=0)+np.finfo(float).tiny
 ratio=float(np.max(np.abs(saved-ind)/bound));need(ratio<=1,'independent derived-vector rounding envelope');return ratio

def bundle(spec,tick,exit_code=0):
 p=Path(spec['root']);inv=[B.inventory(p,spec['outer'],tick),B.inventory(p/'run',spec['worker'],tick)];rt=M.source(p/'run',spec['source']);la=read(p/'launcher-result.json');o=read(p/'run/result.json')
 need(la['source_commit']==spec['source']and la['exit_code']==exit_code and la['failure']is None and not la['owned_live_pids']and la['worker_pid_absent'],'owned historical completion')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'historical owners absent')
 return p,rt,o,inv

def metadata(tick):
 inv,mean_original=M.metadata(tick)
 target,rt,out,iv=bundle(SPEC,tick);inv+=iv
 mr,mrt,mo,iv=bundle(MEAN,tick);inv+=iv
 order,ort,oo,iv=bundle(ORDER,tick);inv+=iv
 ar,art,ao,iv=bundle(OA,tick);inv+=iv
 need(mo['passed']and mo['native_calls']==0 and mo['counts']==M.COUNTS and mo['rows']==12 and len(mo['cases'])==4,'complete independent mean/response authority')
 close(mo['cases'],mean_original['cases'])
 need(read(mr/'run/input-binding.json')['producer']==M.PRODUCER,'literal original mean authority')
 need(ao['passed']and ao['native_calls']==0 and ao['retained_hvp_checked']==160 and ao['rows']==16 and read(ar/'run/input-binding.json')['execution']==ORDER['source'],'complete independent reverse authority')
 ob=read(target/'run/ordering-binding.json');need(ob['producer']==ORDER and ob['independent']==OA,'exact ordering genealogy')
 ib=read(target/'run/input-binding.json');need(ib['producer']==M.PRODUCER and ib['independent']==MEAN and ib['event_pairs']==1308,'exact full mean genealogy')
 meanrt=read(M.TARGET/'run/runtime.json');need(meanrt['sources']['src/neural_shrimp/tiny_lm.py']==rt['sources']['src/neural_shrimp/tiny_lm.py'],'same qualified native model')
 for k in('torch','numpy'):need(meanrt[k]==rt[k],'same native runtime')
 for parentrt in(mrt,ort):
  if 'src/neural_shrimp/tiny_lm.py'in parentrt.get('sources',{}):need(parentrt['sources']['src/neural_shrimp/tiny_lm.py']==rt['sources']['src/neural_shrimp/tiny_lm.py'],'unchanged native code')
 card=read(target/'run/run-card.json');need(card==read(target/'run/source/docs/plans/tiny-lm-parity-order-inputs.json')and card['settings']['orders']==ORDERS and card['settings']['gates']==GATES and card['settings']['worker_seconds']==dict(profile=60,target=300),'literal executed fixed design')
 pref=read(target/'run/profile-reference.json');need(pref['profile']==card['profile'],'same measured admission')
 p,pr,po,iv=bundle(pref['profile'],tick);inv+=iv
 failed=read(target/'run/retained-profile001.json');fp,fr,fo,iv=bundle(failed,tick,2);inv+=iv
 need(po['passed']and po['attempted']==po['returned']=={'hvp':2}and fo['attempted']==fo['returned']=={'hvp':2}and not fo['passed']and fo['target_arrays_unopened'],'both complete two-HVP profiles')
 for a in(po,fo):
  max_time=max(a['times']['hvp']+[a['forecast']['old_max_hvp_seconds']]);close(a['forecast'],dict(seconds=2*224*max_time+30,bytes=2*528*(15776*8+128)+64*1024**2+a['preforecast_bytes'],effective_max_hvp_seconds=max_time,old_max_hvp_seconds=a['forecast']['old_max_hvp_seconds'],new_max_hvp_seconds=max(a['times']['hvp']),vectors=528));need(a['event_pairs']==2 and a['forecast']['bytes']<192*1024**2,'complete profile bytes')
 need(240<fo['forecast']['seconds']<300 and po['forecast']['seconds']<300 and pref['current_forecast']['seconds']==po['forecast']['seconds']and pref['current_forecast']['bytes']<192*1024**2,'preserved time-only amendment and actual admission')
 need(out['passed']and out['mode']=='target'and out['attempted']==out['returned']=={'hvp':224}and out['event_pairs']==224 and len(out['cases'])==4 and out['seconds']<300 and out['peak_rss_kib']<4*1024**2,'complete bounded full-state parity target')
 return inv,out,mo

def original_reverse(seed,u,ref,eta):
 folder=B.TARGET/'run'/f'seed-{seed}-arm-256'/'direction-0-horizon-16';tr=array(folder/'reverse-trace.npy',(17,15776));info=read(folder/'state-order.json')
 need(info['indices']==ORDERS['R']and info['operation_last']-info['operation_first']==15,'literal reverse16 history');exact(tr[0],u,'reverse starts same direction')
 for j,k in enumerate(range(info['operation_first'],info['operation_last']+1)):
  base=B.TARGET/'run/operations'/f'hvp-{k:03d}'
  x=array(Path(str(base)+'-theta.npy'),(15776,));v=array(Path(str(base)+'-vector.npy'),(15776,));h=array(Path(str(base)+'-return.npy'),(15776,));exact(x,ref[15-j],'reverse same state');exact(v,tr[j],'reverse same direction');exact(tr[j+1],tr[j]-eta*h,'reverse exact update')
 return tr[-1]

def audit(tick,save):
 inv,out,mean=metadata(tick);ev=M.Events(TARGET/'run',tick);old=M.Events(M.TARGET/'run',tick);cases=[];rounding=0
 for seed in(28161,28162):
  folder=M.RESPONSE/'run'/f'seed-{seed}-arm-256';md=read(folder/'inputs.json');eta=md['eta'];u=array(folder/'directions.npy',(2,15776))[0]
  for role in('original','quarter_step'):
   label=f'seed-{seed}-{role}';rate=eta/(1 if role=='original'else 4);load=lambda n,s:old.alias(label,n,s)
   if role=='original':
    ref=array(folder/'reference.npy',(17,15776));grad=array(folder/'reference-gradients.npy',(17,15776));ordered=array(folder/'direction-0-transported.npy',(17,15776));frozen=array(folder/'direction-0-frozen.npy',(17,15776));cg=array(folder/'readout-gradients.npy',(3,15776))
   else:ref=load('reference',(17,15776));grad=load('reference-gradients',(17,15776));ordered=load('ordered',(17,15776));frozen=load('frozen',(17,15776));cg=load('readout-gradients',(3,15776))
   exact(ref[1:],ref[:-1]-rate*grad[:-1],'same complete reference updates');exact(ordered[0],u,'literal ordered direction');exact(frozen[0],u,'literal frozen direction');exact(cg,np.asarray([grad[h]-.001*ref[h]for h in(1,4,16)]),'unchanged CE readout')
   mb=read(M.TARGET/'run'/label/'binding.json');bind=read(TARGET/'run'/label/'binding.json');need(bind['eta']==rate and bind['seed']==seed and bind['rate_role']==role and bind['full_state']and bind['original_qualification'],'literal current parity case');need(bind['checkpoint']=={k:mb[k]for k in('seed','arm','step','eta','chunk','state_float32_sha256','data_identity')},'exact checkpoint genealogy')
   mc=mean['cases'][len(cases)];need((mc['seed'],mc['rate_role'])==(seed,role)and all(r['mechanism_qualified']and r['training_ce_slope_resolved']for r in mc['rows']),'all inherited independent1percent gates')
   values={};traces={}
   for key in(('P','Q','T')if role=='original'else('P','Q','T','R')):
    tr=ev.alias(label,key,(17,15776));exact(tr[0],u,'same actual product direction');need(read(TARGET/'run'/label/(key+'-order.json'))==dict(indices=ORDERS[key]),'fixed actual index permutation')
    for j,state in enumerate(ORDERS[key]):exact(tr[j+1],tr[j]-rate*ev.call(label+'/'+key,'hvp',ref[state],tr[j]),'all exact retained native product updates')
    traces[key]=tr;values[key]=tr[-1]
   if role=='original':values['R']=original_reverse(seed,u,ref,rate)
   replay=dict(passed=True,rows=[dict(horizon=h,error=norm(traces['P'][h]-ordered[h])/max(norm(ordered[h]),1e-12))for h in(1,4,16)]);replay['passed']=all(r['error']<=1e-10 for r in replay['rows']);close(replay,read(TARGET/'run'/label/'chronological-replay.json'))
   cmp,vec=reduce(u,values['P'],values['R'],values['Q'],values['T'],frozen[16],cg[2])
   terms=dict(parity_even=[values[k]for k in('P','R','Q','T')],chronological_odd=[values[k]for k in('P','R')],grouped_odd=[values[k]for k in('Q','T')])
   for key,v in vec.items():rounding=max(rounding,derived(ev.alias(label,key,(15776,)),v,terms[key]))
   row=dict(seed=seed,rate_role=role,qualified=bool(replay['passed']and cmp['qualified']),chronological_replay=replay,comparison=cmp,inherited_response_qualified=True);saved=read(TARGET/'run'/label/'result.json');need(saved==out['cases'][len(cases)],'complete producer case ordering');close(row,saved);cases.append(row);save(label+'.json',row)
 need(ev.n==224 and ev.counts==dict(hvp=224,gradient=0,ce=0)and len(list((ev.run/'events').glob('*-input.json')))==len(list((ev.run/'events').glob('*-return.json')))==224,'all224 physical calls exhausted')
 gate=all(c['qualified']for c in cases)and all(c['comparison']['even_relative_ordered']>=.2 for c in cases if c['rate_role']=='original');need(gate==out['descriptive_gate'],'fixed two-original descriptive gate');save('input-binding.json',dict(producer=SPEC,mean=MEAN,ordering=ORDER,ordering_independent=OA,inventories=inv))
 return dict(passed=True,native_calls=0,retained_hvp_checked=224,old_reverse_hvp_checked=32,cases=cases,descriptive_gate=gate,maximum_derived_rounding_ratio=rounding,scope='Independent fsum full-state directional ordering contrast and all224 retained native updates plus32 old reverse associations. Shared native actions and inherited independent response authority. Consumed development linear-response intervention, not a new training trajectory or demonstrated utility.')
def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'120s1GiB audit cap')
 def stop(*_):raise TimeoutError('owned120s deadline')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,120)
 def write(name,raw,closing=False):
  nonlocal used
  need(used+len(raw)<(8 if closing else 6)*1024**2,'8MiB2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used+=len(raw)
 def save(name,value,closing=False):write(name,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=commit,pid=os.getpid(),numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),sources={n:entry(ROOT/n)for n in FILES}))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT),'complete committed own source');write('source/'+n,raw)
  t=time.monotonic();u=np.linspace(.1,1,15776);reduce(u,u,2*u,3*u,4*u,5*u,u);seconds=time.monotonic()-t;forecast=16*seconds+30;save('profile.json',dict(passed=forecast<120,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<120,'full-sized fsum/vector measured admission');result=audit(tick,save);tick()
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('result.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:result.get(k)for k in ('passed','seconds','error','descriptive_gate')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
