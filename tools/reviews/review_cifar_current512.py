"""Independent saved fresh-current512 basis and actual merit audit; zero native."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,json,math,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_cifar_nested512 as N
A=N.A;M=A.M;C=A.C;need=M.need;ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/cifar-current512-independent/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/fresh512002')
EXEC='4de18a1d43e0265980dbfe85372bffaa9407214d'
PINS=['6fec47a99f296f98d2098c9e4ca7f55a99433fbdee43be3343ef1fe79fe4c193','31463434567088f82976557de6d22a480aaae7679590b35736c4362b94362969']
PRIOR=Path('/data2/neural-shrimp-data/audits/cifar-nested512-independent/attempt001')
PRIOR_EXEC='f0adc44d3da85fc9ab463e5a5c0551481d19e0fa';PRIOR_PINS=['dc03a47c4482f4682505a54e6f0e993e4db71a044db96d5082565d552f57341d','c7d4dfa2b6ffe4d6d1556b773a794f574c315b86afa083573c7ae26a797630b1']
STATE_AUDIT=Path('/data2/neural-shrimp-data/audits/cifar-stationary-continuation/attempt002');STATE_AUDIT_PIN='8ca1ec27bd1a30a99a2b5a8e47fde0c96bcd73c9232e64b3b644742a690d83c8'
FILES=N.FILES[:4]+['tools/reviews/review_cifar_current512.py','tools/reviews/launch_cifar_current512_review.py','tests/reviews/test_cifar_current512_review.py','docs/plans/cifar-current512-independent.md']
SETTINGS=dict(columns=512,forcing_limit=.5,stationarity=1e-10,armijo=1e-4,trust_relative=.05,backtracks=13)


def events_bound(events,manifest):
 counts=dict(hvp=0,direct_gradient=0);calls={};rows=[];gate=None;trials=[];pos=0
 def item(v):
  name=v['file'];need(Path(name).name==name and manifest[name]==v['identity'],'complete native immutable file association');return name
 while pos<len(events):
  e=events[pos];kind=e['kind']
  if kind=='nested_reorthogonalization':
   need(gate is None and counts==dict(hvp=2+len(rows),direct_gradient=5)and e['from_column']==len(rows)and e['status']=='next_vector_qualified','fresh next-vector before its action');rows.append(e);pos+=1;continue
  if kind=='current_direction_gate':
   need(gate is None and counts==dict(hvp=514,direct_gradient=5)and len(rows)==511,'gate after complete fresh512 and actual action');gate={k:v for k,v in e.items()if k!='kind'};pos+=1;continue
  if kind=='current_merit_trial':
   need(gate is not None and e['index']==len(trials)and len(trials)<13 and not any(t['accepted']for t in trials),'ordered first-acceptance trials');trials.append({k:v for k,v in e.items()if k!='kind'});need(counts==dict(hvp=514,direct_gradient=5+sum(t['within_trust']for t in trials)),'one actual gradient only per eligible trust trial');pos+=1;continue
  b=events[pos:pos+4];op=e.get('operation');need(op in counts and len(b)==4 and [v['kind']for v in b]==['native_input','operation_started','operation_returned','native_output'],'complete native quartet')
  counts[op]+=1;idx=counts[op];need(all(v['operation']==op and v['index']==idx for v in b),'exact sequential action indices')
  need(math.isfinite(b[2]['seconds'])and b[2]['seconds']>=0,'finite action timing')
  if op=='hvp':need(gate is None and idx<=514 and (idx<=2 and counts['direct_gradient']==1 or 3<=idx<=513 and len(rows)==idx-2 and counts['direct_gradient']==5 or idx==514 and len(rows)==511 and counts['direct_gradient']==5),'actual action at prescribed basis stage')
  elif idx<=5:need(gate is None and (idx==1 and counts['hvp']==0 or idx>=2 and counts['hvp']==2),'prescribed initial and finite-difference gradients')
  else:need(gate is not None and idx==6+sum(t['within_trust']for t in trials),'actual current trial gradient')
  row=dict(state=item(e['state']),result=item(b[3]['result']))
  if op=='hvp':row['direction']=item(e['direction'])
  else:need('direction'not in e,'no gradient direction')
  calls[op,idx]=row;pos+=4
 need(gate is not None and len(rows)==511 and counts==dict(hvp=514,direct_gradient=5+sum(t['within_trust']for t in trials)),'complete current solve event stream')
 return counts,calls,rows,gate,trials


def acceptance(origin,x,g,p,pred,hp,trials,full):
 need(all(v.shape==x.shape and np.isfinite(v).all()for v in(origin,g,p,pred,hp)),'complete finite correction vectors')
 scale=max(1.,M.norm(origin));ng=M.norm(g);need(ng>0 and M.norm(x-origin)/scale<=.05,'positive gradient and original trust ball')
 force=M.norm(g+hp)/ng;err=M.norm(hp-pred)/max(ng,M.norm(hp),1e-12);slope=M.dot(g,hp)
 gate=dict(passed=bool(full and force<=.5 and err<=1e-8 and slope<0),full_basis_qualified=bool(full),true_forcing=force,physical_action_relative_error=err,slope=slope)
 need(gate['passed'],'fresh physical direction gate');rows=[];chosen=None;phi=M.dot(g,g)/2
 for i,pair in enumerate(trials):
  need(i<13 and chosen is None,'no trial past first acceptance');alpha=2.**(-i);y=x+alpha*p;distance=M.norm(y-origin)/scale;within=distance<=.05;row=dict(index=i,alpha=alpha,relative_origin_distance=distance,within_trust=within,accepted=False)
  if within:
   need(pair is not None,'every in-trust trial has actual native gradient');state,newg=pair;need(M.same(state,y)and newg.shape==x.shape and np.isfinite(newg).all(),'exact actual trial state and gradient');merit=M.dot(newg,newg)/2;bound=phi+1e-4*alpha*slope;row.update(merit=merit,armijo_bound=bound,accepted=bool(merit<=bound))
  else:need(pair is None,'no hidden out-of-trust gradient')
  rows.append(row)
  if row['accepted']:chosen=i
 need(chosen is not None or len(rows)==13,'complete first-acceptance stopping')
 y,newg=(x,g)if chosen is None else trials[chosen]
 return dict(status='numerical_correction_accepted'if chosen is not None else'backtracking_failed',direction_gate=gate,trials=rows,accepted_index=chosen,direction_applied=chosen is not None,relative_gradient_before=ng/max(1.,M.norm(x)),relative_gradient=M.norm(newg)/max(1.,M.norm(y)),gradient_norm=M.norm(newg),relative_origin_distance=M.norm(y-origin)/scale,stationarity_qualified=M.norm(newg)/max(1.,M.norm(y))<=1e-10),y,newg


def audit_source(root,execution,outer,worker,tick):
 M.inventory(root,outer,tick);M.inventory(root/'run',worker,tick);N.owner(root,execution);rt=M.read(root/'run/runtime.json')
 need(rt['source_commit']==execution,'literal independent execution')
 for n,pin in rt['files'].items():need(M.entry(root/'run/source'/n)==pin and hashlib.sha256(subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'complete exact independent source '+n)
 return M.read(root/'run/outcome.json')


def bind(tick):
 M.inventory(PARENT,PINS[0],tick);wi=M.inventory(PARENT/'run',PINS[1],tick);M.inventory(C.PARENT,C.PINS[0],tick);oldwi=M.inventory(C.PARENT/'run',C.PINS[1],tick)
 r=M.read(PARENT/'run/result.json');old=M.read(C.PARENT/'run/result.json');M.source(PARENT,r,EXEC);M.source(C.PARENT,old,C.EXEC);owner=N.owner(PARENT,EXEC);N.owner(C.PARENT,C.EXEC)
 need(r['status']=='current512_correction_complete'and r['settings']==SETTINGS and r['old_basis_reused']==0 and not r['accepted_orbit']and not r['rigorous_certificate'],'literal completed current-state numerical scope')
 need(r['elapsed_seconds']<1500 and owner['elapsed_seconds']<1515 and r['peak_rss_kib']<4*1024**2,'owned producer cap')
 for n,sha in old['source_sha256'].items():need(r['source_sha256'][n]==sha,'same original native code '+n)
 for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(r['runtime'][k]==old['runtime'][k],'same actual native runtime '+k)
 for k in('data_identity','eta','lam','model'):need(r['input_identity'][k]==old['input_identity'][k],'same physical objective '+k)
 need(M.read(PARENT/'run/input-identity.json')==r['input_identity'],'exact objective mirror')
 prior=audit_source(PRIOR,PRIOR_EXEC,*PRIOR_PINS,tick);need(prior['passed']and prior['native_calls']==0 and prior['recurrence_rows']==511 and prior['comparison']['strict_forcing_pass'],'completed historical method audit, no state transfer')
 M.inventory(STATE_AUDIT,STATE_AUDIT_PIN,tick);sa=M.read(STATE_AUDIT/'run/outcome.json');need(sa['passed']and sa['accepted_steps']==4 and sa['native_calls']==0,'independent original current-point authority')
 cb=M.read(PARENT/'run/current-binding.json');need(cb['current_parent']==dict(root=str(C.PARENT),source=C.EXEC,outer=C.PINS[0],worker=C.PINS[1])and cb['state']=='state-022.npy'and cb['gradient']==23 and cb['origin']=='global-original-midpoint.npy'and cb['old_basis_reused']==0,'exact current endpoint and global trust origin')
 need(cb['independent512']==dict(root=str(PRIOR),source=PRIOR_EXEC,outer=PRIOR_PINS[0],worker=PRIOR_PINS[1])and cb['positive512']==dict(root=str(N.PARENT),source=N.EXEC,outer=N.PINS[0],worker=N.PINS[1]),'literal nested method genealogy')
 es=[json.loads(v)for v in(PARENT/'run/events.jsonl').read_text().splitlines()];counts,calls,rows,gate,trials=events_bound(es,wi)
 need(counts==dict(hvp=514,direct_gradient=9)and r['operation_attempted']==r['operation_returned']==dict(action=0,j2vp=0,**counts)and r['operation_caps']==dict(action=0,j2vp=0,hvp=514,direct_gradient=18),'complete actual523 calls within original caps')
 need(all(c['state']=='state-000.npy'for (op,_),c in calls.items()if op=='hvp'),'every Hessian action at same current state')
 oe=[json.loads(v)for v in(C.PARENT/'run/events.jsonl').read_text().splitlines()];_,ocalls,_=C.bind_events(C.PARENT/'run',oe,oldwi,tick)
 need(ocalls['direct_gradient',23]['state']=='state-022.npy','old native current-point gradient authority');return r,calls,rows,gate,trials,ocalls,counts


def audit(binding,tick,save):
 r,calls,rows,gate,trials,ocalls,counts=binding;folder=PARENT/'run';load=lambda n:M.vector(folder/n);oldload=lambda n:M.vector(C.PARENT/'run'/n)
 x=load('initial-point.npy');g=load('initial-gradient.npy');origin=load('global-original-midpoint.npy')
 need(M.same(x,load('state-000.npy'))and M.same(x,oldload('state-022.npy'))and M.same(g,oldload(ocalls['direct_gradient',23]['result']))and M.same(origin,oldload('global-original-midpoint.npy')),'full same current state, gradient and original trust origin')
 need(M.same(g,load(calls['direct_gradient',1]['result']))and calls['direct_gradient',1]['state']=='state-000.npy','fresh current gradient equality')
 q=load(calls['hvp',1]['direction']);v=load(calls['hvp',2]['direction']);hq=load(calls['hvp',1]['result']);hv=load(calls['hvp',2]['result']);need(M.norm(q+g/M.norm(g))<1e-12,'actual first negative-gradient direction')
 ref=np.random.default_rng(28168).standard_normal(len(x));ref/=np.linalg.norm(ref);need(M.same(v,ref),'fixed unrelated qualification direction')
 sym=abs(M.dot(q,hv)-M.dot(v,hq))/max(M.norm(hq),M.norm(hv),1e-12);need(sym<=1e-10,'same-state Hessian symmetry qualification');fds=[]
 for i,factor in enumerate([1e-4,1e-5]):
  h=max(1.,float(np.linalg.norm(x)))*factor;cm=calls['direct_gradient',2+2*i];cp=calls['direct_gradient',3+2*i];need(M.same(load(cm['state']),x-h*q)and M.same(load(cp['state']),x+h*q),'exact qualified finite-difference states');fd=(load(cp['result'])-load(cm['result']))/(2*h);need(M.same(fd,load(f'qualification-fd-{i}.npy')),'complete finite-difference return');err=M.norm(hq-fd)/max(M.norm(hq),np.finfo(float).tiny);fds.append(dict(h=h,relative_error=err))
 need(fds[-1]['relative_error']<=5e-6,'actual finite-difference qualification');qual=dict(passed=True,symmetry_error=sym,finite_differences=fds);need(r['qualification']['passed']and abs(sym-r['qualification']['symmetry_error'])<1e-13,'same qualified near-zero symmetry on original action scale');M.compare(fds,r['qualification']['finite_differences'])
 V=np.empty((98666,512),order='F');W=np.empty_like(V)
 for j in range(512):
  c=calls['hvp',1 if j==0 else j+2];V[:,j]=load(c['direction']);W[:,j]=load(c['result']);tick()
 rec=N.recurrence(V,W,g,rows,tick);save('recurrence.json',dict(rows=rec,all512_columns=True))
 report=M.read(folder/'basis-report.json');need(report==r['basis']and report['prefix_columns']==1 and report['total_columns']==512 and report['new_basis_actions']==511 and report['status']=='maximum_columns'and report['completed_fixed_extension'],'complete fresh512 basis')
 coef=np.asarray(report['coefficients']);p,hp,gram,orth,rank,singular=A.assembly(V,W,coef);tick();own,_,ownrank,_=np.linalg.lstsq(W,-g,rcond=1e-12);tick();need(ownrank==rank==report['rank']==512,'full current action rank')
 ce=M.norm(own-coef)/max(1.,M.norm(own));need(ce<1e-9,'independent current least squares');stored=np.load(folder/'final-gram.npy',allow_pickle=False);need(stored.dtype==np.float64 and stored.shape==(512,512)and np.max(abs(stored-gram))<1e-13,'fresh full Gram');need(np.max(abs(singular-np.asarray(report['singular_values'])))/singular[0]<1e-11,'fresh action singular values')
 direction=load('candidate-direction.npy');pred=load('predicted-action.npy');final=calls['hvp',514];actual=load(final['result']);need(M.same(direction,load(final['direction'])),'literal fresh physical correction action');pe=M.norm(p-direction)/max(1e-12,M.norm(direction));he=M.norm(hp-pred)/max(1e-12,M.norm(pred));need(pe<1e-11 and he<1e-11,'independent full512 direction/action assembly')
 pairs=[];ng=5
 for t in trials:
  if t['within_trust']:ng+=1;c=calls['direct_gradient',ng];pairs.append((load(c['state']),load(c['result'])))
  else:pairs.append(None)
 answer,y,gy=acceptance(origin,x,g,direction,pred,actual,pairs,orth<=1e-8 and report['orthogonality_qualified']);M.compare(answer,r['correction']);M.compare(gate,answer['direction_gate']);M.compare(trials,answer['trials']);need(M.same(y,load('final-point.npy'))and M.same(gy,load('final-gradient.npy')),'literal first accepted final state')
 need(answer['accepted_index']==6 and len(answer['trials'])==7 and sum(not t['within_trust']for t in answer['trials'])==3 and not answer['stationarity_qualified'],'observed full trust/merit chain remains nonstationary')
 metrics=dict(qualification=qual,correction=answer,gram_inf_error=orth,numerical_action_rank=rank,least_squares_coefficient_error=ce,direction_assembly_error=pe,predicted_action_assembly_error=he,recurrence_rows=len(rec),merit_ratio=M.dot(gy,gy)/M.dot(g,g))
 save('input-binding.json',dict(producer=str(PARENT),source=EXEC,pins=PINS,current_parent=str(C.PARENT),current_parent_pins=C.PINS,independent_method=str(PRIOR),independent_method_pins=PRIOR_PINS,counts=counts,all_hessian_actions_at_current022=True));save('arithmetic.json',metrics)
 return dict(passed=True,native_calls=0,**metrics,scope='Independent saved binary64 full512 recurrence/Gram/least squares, actual current-state qualification, fsum physical forcing/trust/Armijo/first acceptance. Shares native autograd outputs and NumPy/BLAS/SVD libraries. No stationary root, eigenpair, training cycle or benefit theorem.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'300s4GiB budget')
 def stop(*_):raise TimeoutError('owned audit deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 def save(name,value,closing=False):
  nonlocal used
  b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<(8 if closing else 6)*1024**2,'8MiB2MiB reserve');q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 try:
  execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=execution,python=sys.version,numpy=np.__version__,pid=os.getpid(),files={n:M.entry(ROOT/n)for n in FILES},threads={k:os.environ[k]for k in THREADS}))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT),'literal committed own source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);used+=len(b)
  binding=bind(tick) # inventory/metadata only; scientific npy arrays unopened
  rng=np.random.default_rng(52513);V=np.asfortranarray(rng.standard_normal((98666,512))/np.sqrt(98666));W=np.asfortranarray(V*np.linspace(.5,2,98666)[:,None]);g=rng.standard_normal(98666);samples=[]
  for _ in range(3):
   t=time.monotonic();q,cs=N.pair_project(V,g);M.norm(q);M.norm(g);samples.append(time.monotonic()-t);tick()
  t=time.monotonic();answer=A.assembly(V,W,np.ones(512)/512);coef=np.linalg.lstsq(W,-g,rcond=1e-12);algebra=time.monotonic()-t;tick();forecast=time.monotonic()-start+2*(511*max(samples)+algebra)+30;passed=forecast<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024<4*1024**2
  save('profile.json',dict(passed=passed,rows=98666,columns=512,max_projection_seconds=max(samples),algebra_seconds=algebra,forecast_seconds=forecast,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,target_arrays_unopened=True));need(passed,'full-size measured admission');del V,W,g,q,cs,answer,coef;gc.collect()
  result=audit(binding,tick,save)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):M.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
