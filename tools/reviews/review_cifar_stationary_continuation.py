"""Independent fsum acceptance audit of four saved Newton units; no model import."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import json,math,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_cifar_stationary_merit as M
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/cifar-stationary-continuation/attempt002/run')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/continuation001')
EXEC='59764281d2aaeeb7ed4f18da2044dc796afb4923'
PINS=['70194f724daad0bc1fb94812e233f2b27dc8eb32f3a8d560d6dd9d1f09a9dcea','8b2c8864d956974bc85e4f4d6a584ed53e6cdefffb79208945be31804868574a']
PRIOR=Path('/data2/neural-shrimp-data/audits/cifar-stationary-saved-merit/attempt001')
PRIOR_PIN='364f7147e20590d0299cf5174d3631d02c3f64434d0bfa7c0b4fd63a3f013297'
FILES=['tools/reviews/review_cifar_stationary_merit.py','tools/reviews/review_cifar_stationary_continuation.py','tools/reviews/launch_cifar_stationary_continuation_review.py','tests/reviews/test_cifar_stationary_continuation_review.py','docs/plans/cifar-stationary-continuation-independent.md']
SETTINGS=dict(max_steps=4,maxiter=256,rtol=1e-12,forcing_limit=.9,stationarity=1e-10,armijo=1e-4,trust_relative=.05)

def arithmetic(origin,x,g,p,hp,trials):
 M.need(origin.shape==x.shape==g.shape==p.shape==hp.shape and all(np.isfinite(v).all()for v in(origin,x,g,p,hp)),'equal finite full vectors')
 scale=max(1.,M.norm(origin));M.need(M.norm(x-origin)/scale<=.05,'same original trust ball')
 gg=M.dot(g,g);M.need(gg>0,'nonzero gradient');forcing=M.norm(g+hp)/math.sqrt(gg);slope=M.dot(g,hp);phi=.5*gg
 M.need(forcing<=.9 and slope<0,'independent true forcing and descent')
 rows=[];chosen=None
 for i,(y,newg)in enumerate(trials):
  M.need(chosen is None and i<13,'first acceptance only');alpha=2.**(-i)
  M.need(M.same(y,x+alpha*p)and newg.shape==x.shape and np.isfinite(newg).all(),'exact trial state/full finite gradient')
  distance=M.norm(y-origin)/scale;merit=.5*M.dot(newg,newg);bound=phi+1e-4*alpha*slope
  row=dict(index=i,alpha=alpha,relative_origin_distance=distance,within_trust=distance<=.05,merit=merit,armijo_bound=bound,accepted=distance<=.05 and merit<=bound);rows.append(row)
  if row['accepted']:chosen=i
 M.need(chosen is not None or len(rows)==13,'complete bounded stopping rule')
 y,newg=(x,g)if chosen is None else trials[chosen]
 return dict(forcing=forcing,slope=slope,trials=rows,accepted_index=chosen),dict(relative_gradient=M.norm(newg)/max(1.,M.norm(y)),relative_origin_distance=M.norm(y-origin)/scale,merit_ratio=M.dot(newg,newg)/gg)

def bind_events(folder,events,manifest,tick):
 """Authenticate complete calls in order; permit only recorded solver events between calls."""
 pos=0;counts=dict(hvp=0,direct_gradient=0);calls={};extra=[]
 def item(v):
  name=v['file'];M.need(Path(name).name==name and manifest[name]==v['identity'],'native file identity in full verified inventory');return name
 while pos<len(events):
  tick();e=events[pos]
  if e['kind']!='native_input':
   M.need(e['kind']in('stationary_iteration','stationary_linear','stationary_trial'),'no orphan native event');extra.append(e);pos+=1;continue
  b=events[pos:pos+4];op=e['operation'];M.need(op in counts and len(b)==4 and [x['kind']for x in b]==['native_input','operation_started','operation_returned','native_output'],'full native call quartet')
  counts[op]+=1;idx=counts[op];M.need(all(x['operation']==op and x['index']==idx for x in b),'exact sequential native indices')
  M.need(math.isfinite(b[2]['seconds'])and b[2]['seconds']>=0,'finite call duration')
  z=dict(state=item(e['state']),result=item(b[3]['result']))
  if op=='hvp':z['direction']=item(e['direction'])
  else:M.need('direction'not in e,'gradient has no direction')
  calls[op,idx]=z;pos+=4
 return counts,calls,extra

def fixed_input_names(inventory):
 names=('global-original-midpoint.npy','initial-point.npy','native-direct_gradient-0001-return.npy','stationary-point.npy','stationary-gradient.npy')
 M.need(all(n in inventory for n in names),'all literal saved input names present');return names

def audit(tick,save):
 M.inventory(PARENT,PINS[0],tick);inv=M.inventory(PARENT/'run',PINS[1],tick);fixed_input_names(inv)
 M.inventory(M.PARENT,M.PINS['parent'],tick);M.inventory(M.PARENT/'run',M.PINS['worker'],tick);M.inventory(PRIOR,PRIOR_PIN,tick)
 prior=M.read(PRIOR/'run/outcome.json');M.need(prior['passed']and prior['native_calls']==0,'prior acceptance audit passed')
 row=M.read(PARENT/'run/result.json');outer=M.read(PARENT/'launcher-result.json');old=M.read(M.PARENT/'run/result.json');M.source(PARENT,row,EXEC);M.source(M.PARENT,old,M.EXEC)
 M.need(row['source_commit']==row['review_commit']==EXEC and row['settings']==SETTINGS and row['status']=='stationary_continuation_complete'and row['comparison_qualified']and not row['accepted_orbit']and not row['rigorous_certificate']and row['mode']is None,'literal complete numerical scope')
 M.need(outer['source_commit']==EXEC and outer['exit_code']==0 and outer['scientific_complete']and not outer['owned_live_pids']and outer['worker_pid_absent']and outer['failure']is None,'completed owned producer')
 for pid in(outer['launcher_pid'],outer['owned_group_pid']):M.need(not Path('/proc',str(pid)).exists(),'actual owned process absent')
 M.need(row['elapsed_seconds']<400 and outer['elapsed_seconds']<420 and outer['maximum_group_rss_kib']<2*1024**2,'actual producer caps')
 ident=row['input_identity'];M.need(ident['parent_root']==str(M.PARENT)and ident['parent_execution']==M.EXEC and ident['parent_manifest']==M.PINS['parent'],'literal predecessor')
 for k in('data_identity','eta','lam','model'):M.need(ident[k]==old['input_identity'][k],'unchanged objective '+k)
 M.need(M.read(PARENT/'run/input-identity.json')==ident,'identity mirror')
 folder=PARENT/'run';load=lambda n:M.vector(folder/n);oldload=lambda n:M.vector(M.PARENT/'run'/n)
 origin=load('global-original-midpoint.npy');x=load('initial-point.npy');g=load('native-direct_gradient-0001-return.npy')
 M.need(M.same(origin,oldload('initial-point.npy'))and M.same(x,oldload('final-point.npy'))and M.same(g,oldload('final-gradient.npy')),'unchanged original trust origin and accepted starting point')
 events=[json.loads(z)for z in(folder/'events.jsonl').read_text().splitlines()];counts,calls,extra=bind_events(folder,events,inv,tick)
 M.need(counts==dict(hvp=1030,direct_gradient=23),'full actual call counts');M.need(row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,**counts),'reported full call counts')
 M.need(M.same(load(calls['direct_gradient',1]['state']),x)and M.same(load(calls['direct_gradient',1]['result']),g),'fresh current gradient exactly reproduced')
 steps=row['stationary']['steps'];M.need(len(steps)==4,'fixed four saved units');answers=[]
 for k,saved in enumerate(steps):
  tick();last=259+257*k;state=calls['hvp',last]['state'];M.need(M.same(load(state),x),'fresh exact state for accepted direction')
  M.need(all(calls['hvp',j]['state']==state for j in range(last-256,last+1)),'all256 linear actions and true direction at same current state')
  M.need(saved['step']==k and saved['iterations']==saved['linear_actions']==256 and saved['info']==256,'literal fixed solver budget')
  p=load(calls['hvp',last]['direction']);hp=load(calls['hvp',last]['result']);trials=[]
  M.need(len(saved['trials'])==4,'fixed observed four trial gradients')
  for j in range(4):
   call=calls['direct_gradient',8+4*k+j];trials.append((load(call['state']),load(call['result'])))
  z,stats=arithmetic(origin,x,g,p,hp,trials);M.compare(z,saved);M.need(z['accepted_index']==3,'first accepted dyadic trial');answers.append(dict(step=k,**z,**stats));x,g=trials[3]
  iterations=[e for e in extra if e['kind']=='stationary_iteration'and e['step']==k];M.need(len(iterations)==256 and [e['iteration']for e in iterations]==list(range(1,257)),'complete recorded solver iterations')
  linears=[e for e in extra if e['kind']=='stationary_linear'and e['step']==k];M.need(len(linears)==1,'one reported true direction');M.compare({a:saved[a]for a in('step','iterations','linear_actions','info','forcing','slope')},linears[0])
  es=[e for e in extra if e['kind']=='stationary_trial'and e['step']==k];M.need(len(es)==4,'complete trial records')
  for e,t in zip(es,z['trials']):M.compare(t,e)
 M.need(len(extra)==4*(256+1+4),'no extra solver/trial events')
 M.need(M.same(x,load('stationary-point.npy'))and M.same(g,load('stationary-gradient.npy')),'exact final accepted state and gradient')
 final=dict(relative_gradient=M.norm(g)/max(1.,M.norm(x)),relative_origin_distance=M.norm(x-origin)/max(1.,M.norm(origin)))
 M.compare(final,row['stationary']);M.need(not row['seed_qualified']and not row['stationary']['passed']and row['stationary']['status']=='maximum_steps'and final['relative_gradient']>1e-10,'stationarity remains unresolved')
 save('input-binding.json',dict(parent=str(PARENT),pins=PINS,source=EXEC,prior_audit=str(PRIOR),prior_pin=PRIOR_PIN,files=len(inv),events=len(events),counts=counts));save('arithmetic.json',dict(steps=answers,final=final))
 return dict(passed=True,native_calls=0,accepted_steps=4,steps=answers,final=final,scope='Independent saved binary64 acceptance arithmetic and exact current-state/original-origin binding. Shared native gradient and Hessian outputs. No independently computed derivatives, stationary root, eigenpair or orbit certificate.')

need=M.need;entry=M.entry
def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s1GiB budget')
 def stop(*_):raise TimeoutError('saved audit deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 def save(name,value,closing=False):
  nonlocal used
  b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<4*1024**2-(0 if closing else 1024**2),'4MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 try:
  execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=execution,python=sys.version,numpy=np.__version__,pid=os.getpid(),threads={k:os.environ[k]for k in THREADS},files={n:entry(ROOT/n)for n in FILES}))
  for name in FILES:
   b=(ROOT/name).read_bytes();need(b==subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT),'committed own source');p=OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used+=len(b)
  # Full-size unrelated fsum reducer admission before research arrays.
  t=time.monotonic();x=np.ones(98666);g=x*.01;p=-x*.01;hp=-g;sample=arithmetic(x,x,g,p,hp,[(x+p,g*.5)]);seconds=time.monotonic()-t;save('profile.json',dict(seconds=seconds,forecast_seconds=24*seconds+15,target_arrays_unopened=True));need(24*seconds+15<60,'measured full-size reducer gate');del x,g,p,hp,sample
  result=audit(tick,save)
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
