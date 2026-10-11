"""Fixed saved current512 nonlinear gradient-merit diagnosis; no native calls."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import numpy as np
import review_cifar_current512 as C
import cifar_nonlinearity_math as H
M,need=C.M,H.need
OUT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/current512-nonlinearity001/run')
AUDIT=Path('/data2/neural-shrimp-data/audits/cifar-current512-independent/attempt001')
AUDIT_EXEC='eabcd27ca4a6c1c1007ef67f79fe41140c2a0a39';AUDIT_PINS=['10a4294cc1f88f921bb2d027ddf6b084afef0bf0b17430f264c3302678d16d40','2471e751e812c0dfc1bd2961c3246f98992b00cfd49e7cbe0df78f2bb4d5f7d1']
FILES=sorted(set(C.FILES+['tools/reviews/cifar_nonlinearity_math.py','tools/reviews/diagnose_cifar_nonlinearity.py','tools/reviews/launch_cifar_nonlinearity.py','tests/reviews/test_cifar_nonlinearity_math.py','tests/reviews/test_cifar_nonlinearity_driver.py','docs/plans/cifar-current512-nonlinearity.md']))


def bind(w):
 audit=C.audit_source(AUDIT,AUDIT_EXEC,*AUDIT_PINS,w.tick);ib=M.read(AUDIT/'run/input-binding.json');need(audit['passed']and audit['native_calls']==0 and audit['recurrence_rows']==511 and audit['numerical_action_rank']==512 and ib['producer']==str(C.PARENT)and ib['source']==C.EXEC and ib['pins']==C.PINS,'completed independent actual-current512 authority')
 binding=C.bind(w.tick);r,calls,rows,gate,trials,old,counts=binding;need(len(trials)==7 and sum(t['within_trust']for t in trials)==4 and trials[-1]['accepted']and not any(t['accepted']for t in trials[:-1]),'fixed complete seven-row actual history')
 w.save('input-binding.json',dict(producer=str(C.PARENT),source=C.EXEC,pins=C.PINS,independent=str(AUDIT),independent_source=AUDIT_EXEC,independent_pins=AUDIT_PINS,actual_counts=counts,expected_native_calls=0,scope='Same consumed current022 point only; no historical018 actions or new/current-final-state Hessian extrapolation.'))
 return r,calls,trials,audit


def target(w,binding):
 r,calls,trials,audit=binding;folder=C.PARENT/'run';load=lambda n:M.vector(folder/n);x=load('initial-point.npy');g=load('initial-gradient.npy');origin=load('global-original-midpoint.npy');p=load('candidate-direction.npy');final=calls['hvp',514];hp=load(final['result'])
 need(H.same(x,load('state-000.npy'))and H.same(g,load(calls['direct_gradient',1]['result']))and final['state']=='state-000.npy'and H.same(p,load(final['direction'])),'literal same-current physical action and gradient')
 pairs=[];count=5;selected=[]
 for row in trials:
  if row['within_trust']:
   count+=1;call=calls['direct_gradient',count];pairs.append((load(call['state']),load(call['result'])));selected.append(dict(index=row['index'],gradient_index=count,files=call))
  else:pairs.append(None)
  w.tick()
 result=H.analyze(origin,x,g,p,hp,pairs);w.tick();need(result['accepted_index']==6 and abs(result['true_forcing']-audit['correction']['direction_gate']['true_forcing'])<1e-12,'same independently qualified forcing/acceptance')
 for got,old in zip(result['rows'],trials,strict=True):
  need(all(got[k]==old[k]for k in('index','alpha','within_trust','accepted'))and abs(got['relative_origin_distance']-old['relative_origin_distance'])<1e-12,'every actual preserved trust/merit row')
 need(H.same(pairs[-1][0],load('final-point.npy'))and H.same(pairs[-1][1],load('final-gradient.npy')),'same actual accepted final state')
 w.save('selected-actions.json',dict(initial_gradient=calls['direct_gradient',1],physical_action=final,actual_gradients=selected));w.save('diagnosis.json',result)
 return dict(passed=True,native_calls=0,diagnosis=result,accepted_correction_unchanged=True,scope='Descriptive saved direction nonlinearity/trust diagnosis. No new trial, correction, derivative, stationary point or cycle; no inference of a rigorous remainder bound.')


def synthetic(n=98666):
 rng=np.random.default_rng(712813);x=np.zeros(n);x[0]=1.;g=rng.standard_normal(n)*1e-3;p=-.3*g/H.norm(g);hp=-g;pairs=[]
 for i in range(7):
  a=2.**(-i);pairs.append(None if i<3 else(x+a*p,g+a*hp+48*a*a*g))
 result=H.analyze(x,x,g,p,hp,pairs);need(result['accepted_index']==6 and all(abs(v['quadratic_remainder_coefficient']-48*H.norm(g))<1e-10 for v in result['rows']if v['nonlinear_observed']),'full-size known quadratic response/control')
 return result


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def tick(self):need(time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s1GiB diagnostic cap')
 def save(self,name,value,closing=False):
  raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(self.used+len(raw)<(8 if closing else 6)*1024**2,'8MiB2MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean exact nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned60s diagnostic')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),files={n:M.entry(ROOT/n)for n in FILES},python=platform.python_version(),numpy=np.__version__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'literal own committed source');need(w.used+len(raw)<6*1024**2,'bounded exact archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  binding=bind(w) # complete inventories/metadata only, scientific arrays unopened
  times=[]
  for i in range(2):
   t=time.monotonic();trial=synthetic();times.append(time.monotonic()-t);w.save(f'profile-{i}.json',trial);w.tick()
  forecast=time.monotonic()-w.start+4*max(times)+10;passed=forecast<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+128*1024<1024**2 and 4*w.used+2*1024**2<8*1024**2;w.save('profile.json',dict(passed=passed,dimension=98666,seconds=times,forecast_seconds=forecast,target_arrays_unopened=True));need(passed,'complete actual-size measured scalar profile');gc.collect();result=target(w,binding)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):M.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
