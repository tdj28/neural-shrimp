"""Bounded saved five-trial remainder diagnosis, no new model operations."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,io,json,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_cifar_curved512 as B
import cifar_ray_remainder as H
M=B.M;need=M.need
OUT=Path('/data2/neural-shrimp-data/audits/cifar-ray-remainder/attempt001/run')
AUDIT=dict(root='/data2/neural-shrimp-data/audits/cifar-curved512-independent/attempt001',source='1b4a629a326e85c4d36cf3279a23f1dd405cc24c',outer='c0bd3c49ed7bdcb6454e6bab07a542447c1d1fe0278cd0085de6239ea56cf0fd',worker='cf4218c3bd668b3ed31208b9755a17f84001033b39691a726ecbc72cf3bef2c5')
CARD='docs/plans/cifar-ray-remainder-inputs.json'
FILES=sorted(set(B.FILES+[CARD,'docs/plans/cifar-ray-remainder.md','tools/reviews/cifar_ray_remainder.py','tools/diagnose_cifar_ray_remainder.py','tools/launch_cifar_ray_remainder.py','tests/reviews/test_cifar_ray_remainder.py','tests/test_cifar_ray_remainder_driver.py']))
LIMITS=dict(seconds=60,rss_bytes=1024**3,as_bytes=3*1024**3,raw_bytes=32*1024**2,native_calls=0)
PRODUCER=dict(root=str(B.PARENT),source=B.EXEC,outer=B.PINS[0],worker=B.PINS[1])


def metadata(tick):
 binding=B.bind(tick)
 p=Path(AUDIT['root']);M.inventory(p,AUDIT['outer'],tick);M.inventory(p/'run',AUDIT['worker'],tick);B.N.owner(p,AUDIT['source'])
 rt=M.read(p/'run/runtime.json');need(rt['source_commit']==AUDIT['source']and not Path('/proc',str(rt['pid'])).exists(),'literal independent source and absent worker')
 actual={str(v.relative_to(p/'run/source'))for v in(p/'run/source').rglob('*')if v.is_file()};need(actual==set(rt['files']),'complete independent archived source')
 for n,pin in rt['files'].items():need(M.entry(p/'run/source'/n)==pin and hashlib.sha256(subprocess.check_output(['git','show',AUDIT['source']+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'historical independent source '+n);tick()
 need(rt['files']['tools/reviews/review_cifar_curved512.py']==M.entry(ROOT/'tools/reviews/review_cifar_curved512.py'),'literal separately audited parent binder')
 result=M.read(p/'run/outcome.json');ib=M.read(p/'run/input-binding.json')
 need(result['passed']and result['native_calls']==0 and result['qualification']['passed']and result['numerical_action_rank']==512 and result['gram_inf_error']<=1e-8 and result['correction']['accepted_index']==7 and not result['correction']['stationarity_qualified'],'complete independently qualified fresh512 and fixed first acceptance')
 need(ib['producer']==PRODUCER['root']and ib['source']==PRODUCER['source']and ib['pins']==B.PINS and ib['counts']==dict(hvp=514,direct_gradient=10)and ib['old_basis_reused']==0 and ib['all_hessian_actions_at_selected_curved_one_over32'],'literal same new-state independent authority')
 r,calls,rows,gate,trials,ocalls,counts=binding
 need(len(trials)==8 and [t['alpha']for t in trials if t['within_trust']]==list(H.ALPHAS)and [t['index']for t in trials if t['within_trust']]==list(range(3,8))and r['correction']['accepted_index']==7,'all five fixed acquired in-trust trials')
 return binding,dict(producer=PRODUCER,independent=AUDIT,settings=H.SETTINGS,counts=counts,all_five_consumed=True,fit_indices=list(H.FIT),check_indices=list(H.CHECK),target_arrays_unopened=True)


def target(binding,tick):
 r,calls,_,_,trials,_,_=binding
 folder=B.PARENT/'run';load=lambda n:M.vector(folder/n)
 x=load('initial-point.npy');g=load('initial-gradient.npy');p=load('candidate-direction.npy');hp=load(calls['hvp',514]['result']);origin=load('global-original-midpoint.npy')
 need(x.shape==(98666,)and all(v.shape==x.shape for v in(g,p,hp,origin)),'full actual98666 coordinates')
 need(M.same(x,load('state-000.npy'))and M.same(x,load(calls['direct_gradient',1]['state']))and M.same(g,load(calls['direct_gradient',1]['result']))and M.same(p,load(calls['hvp',514]['direction']))and calls['hvp',514]['state']=='state-000.npy','actual initial gradient and physical same-state Hp identity')
 ng=M.norm(g);scale=max(1.,M.norm(origin));need(M.norm(x-origin)/scale<=.05 and M.norm(g+hp)/ng<=.5 and M.dot(g,hp)<0,'original global trust and qualified physical forcing')
 ys=[];references=[]
 for i,a in enumerate(H.ALPHAS):
  c=calls['direct_gradient',i+6];state=load(c['state']);y=load(c['result']);need(M.same(state,x+a*p)and M.norm(state-origin)/scale<=.05,'literal saved alpha p state inside original global trust')
  trial=trials[i+3];need(trial['within_trust']and trial['alpha']==a,'fixed trial metadata identity');M.compare(M.dot(y,y)/2,trial['merit']);ys.append(y);references.append(dict(alpha=a,state=c['state'],gradient=c['result'],hvp=calls['hvp',514],trial_index=i+3));tick()
 report,arrays=H.diagnose(g,hp,ys,tick)
 need(M.same(ys[-1],load('final-gradient.npy'))and M.same(x+H.ALPHAS[-1]*p,load('final-point.npy')),'fixed original first-accepted endpoint retained')
 report.update(actual_source_references=references,source_native_calls_shared=dict(hvp=514,gradient=10),new_native_calls=0,independent_parent=AUDIT)
 return report,arrays


def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source')
 card=M.read(ROOT/CARD);need(card['status']=='reviewed'and card['limits']==LIMITS and card['producer']==PRODUCER and card['independent']==AUDIT and card['settings']==H.SETTINGS,'exact reviewed five-trial saved-only card')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,new_native_calls=0)
 def tick():need(time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s1GiB budget')
 def stop(*_):raise TimeoutError('owned diagnosis deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 def blob(name,b,closing=False):
  nonlocal used
  need(used+len(b)<(32 if closing else 30)*1024**2,'32MiB2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 def save(name,value,closing=False):blob(name,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,pid=os.getpid(),python=sys.version,numpy=np.__version__,threads={k:os.environ[k]for k in THREADS},sources={n:M.entry(ROOT/n)for n in FILES}))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'literal committed own source');blob('source/'+n,b)
  binding,authority=metadata(tick);save('input-binding.json',authority)
  rng=np.random.default_rng(52514);g=rng.standard_normal(98666);hp=rng.standard_normal(98666);k=rng.standard_normal(98666);ys=[g+a*hp+a*a*k for a in H.ALPHAS]
  t=time.monotonic();synthetic,arrays=H.diagnose(g,hp,ys,tick);elapsed=time.monotonic()-t;bio=io.BytesIO();np.savez(bio,**arrays);array_bytes=len(bio.getvalue());forecast=time.monotonic()-start+2*elapsed+15;raw_forecast=used+2*array_bytes+4*1024**2
  admitted=bool(synthetic['quadratic_sufficiency_qualified']and forecast<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+128*1024<1024**2 and raw_forecast<32*1024**2)
  save('profile.json',dict(passed=admitted,full_dimension=98666,five_trials=True,seconds=elapsed,forecast_seconds=forecast,array_bytes=array_bytes,forecast_raw_bytes=raw_forecast,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,target_arrays_unopened=True));need(admitted,'fresh fullsize measured admission');del g,hp,k,ys,synthetic,arrays,bio;gc.collect()
  report,arrays=target(binding,tick);bio=io.BytesIO();np.savez(bio,**arrays);blob('derived-vectors.npz',bio.getvalue());save('diagnosis.json',report);tick();result=dict(passed=True,new_native_calls=0,quadratic_sufficiency_qualified=report['quadratic_sufficiency_qualified'],rows=report['rows'],all_five_consumed_development=True)
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=bool(result['passed']and result['elapsed_seconds']<60 and result['peak_rss_kib']<1024**2);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):M.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':
 if sys.argv[1:]==['--source-import-check']:print('source import complete');raise SystemExit(0)
 need(not sys.argv[1:],'no unplanned arguments');raise SystemExit(main())
