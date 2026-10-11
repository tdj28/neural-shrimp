"""Independent saved equal-action CIFAR comparison; no native model imports."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import hashlib,json,math,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_cifar_stationary_merit as M
import review_cifar_stationary_continuation as C
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/cifar-reorthogonalized-independent/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/reorthogonalized001')
EXEC='2195df22a1bcf8df7285175cca9abb4151cf906b'
PINS=['685caf2c84e977a79356782b60fbf607c083cc77d86616680fe74fe68bee6ed4','a49557ee2a69a86b8e9861de7a299510bf033c59bfd5dd9f72c3b2228b4e8994']
PRIOR=Path('/data2/neural-shrimp-data/audits/cifar-stationary-continuation/attempt002')
PRIOR_PIN='8ca1ec27bd1a30a99a2b5a8e47fde0c96bcd73c9232e64b3b644742a690d83c8'
FILES=['tools/reviews/review_cifar_stationary_merit.py','tools/reviews/review_cifar_stationary_continuation.py','tools/reviews/review_cifar_reorthogonalized.py','tools/reviews/launch_cifar_reorthogonalized_review.py','tests/reviews/test_cifar_reorthogonalized_review.py','docs/plans/cifar-reorthogonalized-independent.md']


def comparison(g,oldhp,hp,predicted,orth,actions):
 M.need(g.shape==oldhp.shape==hp.shape==predicted.shape and all(np.isfinite(x).all()for x in(g,oldhp,hp,predicted)),'full equal finite comparison vectors')
 ng=M.norm(g);M.need(ng>0,'positive gradient norm');base=M.norm(g+oldhp)/ng;M.need(base>0,'positive baseline forcing')
 force=M.norm(g+hp)/ng;error=M.norm(predicted-hp)/max(ng,M.norm(hp),1e-12);qualified=bool(actions==256 and orth<=1e-8 and error<=1e-8)
 return dict(qualified=qualified,baseline_forcing=base,true_forcing=force,forcing_ratio=force/base,physical_action_relative_error=error,material_improvement=bool(qualified and force<=.8*base),strict_forcing_pass=bool(qualified and force<=.5))


def assembly(V,W,c):
 M.need(V.shape==W.shape and V.ndim==2 and c.shape==(V.shape[1],)and all(np.isfinite(x).all()for x in(V,W,c)),'full finite basis and coefficients')
 p=np.zeros(V.shape[0]);hp=p.copy()
 for j,a in enumerate(c):p+=float(a)*V[:,j];hp+=float(a)*W[:,j]
 gram=V.T@V;error=max(math.fsum(abs(float(z))for z in row)for row in(gram-np.eye(len(c))))
 singular=np.linalg.svd(W,compute_uv=False);rank=int(np.count_nonzero(singular>1e-12*singular[0]))
 return p,hp,gram,error,rank,singular


def events_bound(events,manifest):
 pos=0;counts=dict(hvp=0,direct_gradient=0);calls={};extra=[]
 def item(v):
  name=v['file'];M.need(Path(name).name==name and manifest[name]==v['identity'],'native array in complete immutable inventory');return name
 while pos<len(events):
  e=events[pos]
  if e['kind']=='reorthogonalization':extra.append(e);pos+=1;continue
  b=events[pos:pos+4];op=e.get('operation');M.need(op in counts and len(b)==4 and [v['kind']for v in b]==['native_input','operation_started','operation_returned','native_output'],'complete ordered native quartet')
  counts[op]+=1;idx=counts[op];M.need(all(v['operation']==op and v['index']==idx for v in b),'unique sequential operation index')
  M.need(math.isfinite(b[2]['seconds'])and b[2]['seconds']>=0,'finite measured operation')
  row=dict(state=item(e['state']),result=item(b[3]['result']))
  if op=='hvp':row['direction']=item(e['direction'])
  else:M.need('direction'not in e,'gradient without direction')
  calls[op,idx]=row;pos+=4
 M.need(len(extra)==256 and [v['index']for v in extra]==list(range(256)),'all256 retained reorthogonalization rows')
 M.need(all(v['status']=='next_vector_qualified'for v in extra[:-1])and extra[-1]['status']=='fixed_action_cap','no early stopping or hidden trial')
 return counts,calls,extra


def audit(tick,save):
 M.inventory(PARENT,PINS[0],tick);inv=M.inventory(PARENT/'run',PINS[1],tick);M.inventory(C.PARENT,C.PINS[0],tick);oldinv=M.inventory(C.PARENT/'run',C.PINS[1],tick);M.inventory(PRIOR,PRIOR_PIN,tick)
 old=M.read(C.PARENT/'run/result.json');row=M.read(PARENT/'run/result.json');M.source(C.PARENT,old,C.EXEC);M.source(PARENT,row,EXEC)
 prior=M.read(PRIOR/'run/outcome.json');M.need(prior['passed']and prior['native_calls']==0 and prior['accepted_steps']==4,'prior independent physical forcing authority')
 owner=M.read(PARENT/'launcher-result.json');M.need(owner['source_commit']==EXEC and owner['exit_code']==0 and owner['scientific_complete']and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'completed owned producer')
 for k in('launcher_pid','owned_group_pid'):M.need(not Path('/proc',str(owner[k])).exists(),'actual owned process absent')
 M.need(owner['elapsed_seconds']<315 and row['elapsed_seconds']<300 and row['peak_rss_kib']<3*1024**2,'actual measured resource caps')
 for n,h in old['source_sha256'].items():M.need(row['source_sha256'][n]==h,'identical original native dependency '+n)
 for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):M.need(row['runtime'][k]==old['runtime'][k],'unchanged exact runtime '+k)
 for k in('data_identity','eta','lam','model'):M.need(row['input_identity'][k]==old['input_identity'][k],'same objective '+k)
 binding=M.read(PARENT/'run/comparison-binding.json');M.need(binding['operator']==dict(state='state-018.npy',gradient=19,first_hvp=774,true_hvp=1030)and binding['original_input_identity']==old['input_identity'],'fixed original operator')
 M.need(row['status']=='reorthogonalized_comparison_complete'and not row['direction_applied']and not row['accepted_orbit']and not row['rigorous_certificate'],'complete unchanged numerical scope')
 folder=PARENT/'run';load=lambda n:M.vector(folder/n);oldload=lambda n:M.vector(C.PARENT/'run'/n)
 e=[json.loads(v)for v in(folder/'events.jsonl').read_text().splitlines()];counts,calls,extra=events_bound(e,inv)
 M.need(counts==dict(hvp=257,direct_gradient=1)and row['operation_caps']==row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,**counts),'inclusive exact257HVP/1gradient')
 oe=[json.loads(v)for v in(C.PARENT/'run/events.jsonl').read_text().splitlines()];oc,oldcalls,_=C.bind_events(C.PARENT/'run',oe,oldinv,tick);M.need(oc==dict(hvp=1030,direct_gradient=23),'source-bound original event stream')
 M.need(all(v['state']=='state-000.npy'for v in calls.values()),'every action at identical point')
 M.need(M.same(load('state-000.npy'),oldload('state-018.npy')),'literal historical point, never finalstate022')
 g=load(calls['direct_gradient',1]['result']);M.need(M.same(g,oldload(oldcalls['direct_gradient',19]['result'])),'byte-exact old gradient')
 first=calls['hvp',1];orig=oldcalls['hvp',774]
 M.need(M.same(load(first['direction']),oldload(orig['direction']))and M.same(load(first['result']),oldload(orig['result'])),'byte-exact same first input and Hessian action')
 M.need(oldcalls['hvp',1030]['state']=='state-018.npy'and oldcalls['direct_gradient',19]['state']=='state-018.npy','same original current-state baseline')
 report=M.read(folder/'basis-report.json');M.need(report==row['basis']and report['actions']==256 and report['status']=='maximum_actions','complete basis report')
 V=np.column_stack([load(calls['hvp',j]['direction'])for j in range(1,257)]);W=np.column_stack([load(calls['hvp',j]['result'])for j in range(1,257)]);c=np.asarray(report['coefficients'],dtype=np.float64)
 tick();p,hp,gram,orth,rank,singular=assembly(V,W,c);tick()
 direction=load('candidate-direction.npy');prediction=load('predicted-action.npy');actual=load(calls['hvp',257]['result'])
 M.need(M.same(direction,load(calls['hvp',257]['direction'])),'literal fresh physical action direction')
 p_error=M.norm(p-direction)/max(M.norm(direction),1e-12);hp_error=M.norm(hp-prediction)/max(M.norm(prediction),1e-12);M.need(p_error<1e-11 and hp_error<1e-11,'independent sequential basis/action assembly')
 stored=np.load(folder/'final-gram.npy',allow_pickle=False);M.need(stored.dtype==np.float64 and stored.shape==(256,256)and np.max(np.abs(stored-gram))<1e-13,'recomputed full retained Gram')
 M.need(orth<=1e-8 and report['orthogonality_qualified']and rank==report['rank']==256,'full basis orthogonality and fixed-rcond numerical action rank')
 M.need(np.max(np.abs(singular-np.asarray(report['singular_values'])))/singular[0]<1e-11,'saved singular values reproduced')
 baseline=oldload(oldcalls['hvp',1030]['result']);answer=comparison(g,baseline,actual,prediction,orth,256);M.compare(answer,row['comparison']);M.close(answer['baseline_forcing'],prior['steps'][3]['forcing'],'independent accepted-unit baseline')
 M.need(answer['qualified']and not answer['material_improvement']and not answer['strict_forcing_pass'],'original20percent and0.5 negative gates retained')
 result=dict(comparison=answer,gram_inf_error=orth,numerical_action_rank=rank,direction_assembly_relative_error=p_error,predicted_action_assembly_relative_error=hp_error)
 save('input-binding.json',dict(producer=str(PARENT),producer_source=EXEC,producer_pins=PINS,prior=str(PRIOR),prior_pin=PRIOR_PIN,original_source=C.EXEC,original_pins=C.PINS,events=len(e),counts=counts,all_at_historical_state018=True));save('arithmetic.json',result)
 return dict(passed=True,native_calls=0,**result,scope='Independent fsum physical forcing comparison, exact same-state event/source binding, sequential vector assembly, full Gram and numerical SVD re-evaluation. Shares native autograd outputs and NumPy/BLAS/SVD libraries. No new derivatives, step, stationary point, full-Hessian spectral theorem or orbit certificate.')

need=M.need;entry=M.entry
def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'120s2GiB budget')
 def stop(*_):raise TimeoutError('saved audit deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,120)
 def save(name,value,closing=False):
  nonlocal used
  b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<4*1024**2-(0 if closing else 1024**2),'4MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 try:
  execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=execution,python=sys.version,numpy=np.__version__,pid=os.getpid(),threads={k:os.environ[k]for k in THREADS},files={n:entry(ROOT/n)for n in FILES}))
  for name in FILES:
   b=(ROOT/name).read_bytes();need(b==subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT),'committed own source');p=OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used+=len(b)
  # Full-dimensional assembly/Gram/SVD profile before target arrays.
  t=time.monotonic();rng=np.random.default_rng(72019);V=rng.standard_normal((98666,256))/np.sqrt(98666);W=V*np.linspace(.5,2,98666)[:,None];c=np.ones(256)/256;sample=assembly(V,W,c);g=np.ones(98666);z=comparison(g,-.2*g,-.4*g,-.4*g,.0,256);seconds=time.monotonic()-t
  save('profile.json',dict(seconds=seconds,forecast_seconds=3*seconds+20,rows=98666,columns=256,target_arrays_unopened=True));need(3*seconds+20<120,'measured full-size matrix and scalar gate');del V,W,c,sample,g,z
  result=audit(tick,save)
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
