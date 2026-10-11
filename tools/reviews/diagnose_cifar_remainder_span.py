"""One fixed saved current022 nonlinear-image diagnostic; zero native calls."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,io,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import numpy as np
import diagnose_cifar_nonlinearity as N
import cifar_remainder_span_math as H
C,M,need=N.C,N.M,H.need
OUT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/remainder-span001/run')
CARD='docs/plans/cifar-remainder-span-inputs.json'
FILES=sorted(set(N.FILES+['tools/reviews/cifar_remainder_span_math.py','tools/reviews/diagnose_cifar_remainder_span.py','tools/reviews/launch_cifar_remainder_span.py','tests/reviews/test_cifar_remainder_span_math.py','tests/reviews/test_cifar_remainder_span_driver.py','docs/plans/cifar-remainder-span.md',CARD,'docs/reviews/cifar-current512-nonlinearity001.json']))
SETTINGS=dict(columns=512,dimension=98666,rank_rtol=1e-12,alphas=[1/8,1/16,1/32,1/64],reference_alpha=1/64,native_calls=0)
CAPS=dict(seconds=300,rss_bytes=4*1024**3,as_bytes=12*1024**3,raw_bytes=128*1024**2)


def bind(w,card):
 need(card['status']=='reviewed'and card['settings']==SETTINGS and card['caps']==CAPS,'fixed reviewed saved diagnostic');binding=N.bind(w);r,calls,trials,audit=binding;receipt=M.read(ROOT/'docs/reviews/cifar-current512-nonlinearity001.json');need(card['nonlinearity']==receipt,'literal prior complete nonlinear diagnosis');root=Path(receipt['root']);M.inventory(root,receipt['outer']['sha256'],w.tick);M.inventory(root/'run',receipt['worker']['sha256'],w.tick)
 old=M.read(root/'run/outcome.json');need(old['passed']and old['native_calls']==0 and old['diagnosis']['accepted_index']==6,'original unchanged nonlinear history');rt=M.read(root/'run/runtime.json');owner=M.read(root/'launcher-result.json');need(rt['source_commit']==receipt['execution']==owner['source_commit']and owner['exit_code']==0 and not owner['owned_live_pids']and owner['failure']is None,'source/owned prior diagnostic');
 for pid in[rt['pid'],owner['launcher_pid'],owner['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'prior owned processes absent')
 for n,pin in rt['files'].items():need(M.entry(root/'run/source'/n)==pin,'literal diagnostic source');need(hashlib.sha256(subprocess.check_output(['git','show',receipt['execution']+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'historical committed diagnostic source');w.tick()
 w.save('nonlinearity-authority.json',dict(receipt=receipt,settings=SETTINGS,caps=CAPS));return binding


def profile(w,n=98666,m=512):
 rng=np.random.default_rng(814527);w.save('profile-inputs.json',dict(seed=814527,shape=[n,m],recipe='V gaussian/sqrt(n), W=V*linspace(.5,2), gradient/p/Hp and all four quadratic trials in known image',native_calls=0,target_arrays_unopened=True));V=np.asfortranarray(rng.standard_normal((n,m))/np.sqrt(n));W=np.asfortranarray(V*np.linspace(.5,2,m));x=np.zeros(n);x[0]=1;g=.01*V[:,0];p=-.01*V[:,0];hp=-.01*W[:,0];b=.2*W[:,1];trials=[dict(alpha=a,state=x+a*p,gradient=g+a*hp+a*a*b)for a in SETTINGS['alphas']];before=time.monotonic();out=H.diagnose(V,W,x,g,p,hp,x,trials,rank_rtol=SETTINGS['rank_rtol'],tick=w.tick);elapsed=time.monotonic()-before;need(out['summary']['numerical_rank']==m and max(r['irreducible_fraction']for r in out['summary']['rows'])<1e-9,'full-size known quadratic image control');w.save('profile-summary.json',out['summary']);arrays=sum(v.nbytes+128 for row in out['vectors']for v in row.values());forecast=time.monotonic()-w.start+2*elapsed+60;raw=w.used+2*arrays+32*1024**2;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024;ans=dict(passed=forecast<300 and raw<128*1024**2 and rss<4*1024**2,seconds=elapsed,forecast_seconds=forecast,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True);w.save('profile.json',ans);need(ans['passed'],'full98666x512 complete measured admission');del V,W,trials,out;gc.collect()


def target(w,binding):
 _,calls,rows,audit=binding;folder=C.PARENT/'run';load=lambda n:M.vector(folder/n);x=load('initial-point.npy');g=load('initial-gradient.npy');p=load('candidate-direction.npy');hp=load(calls['hvp',514]['result']);origin=load('global-original-midpoint.npy')
 need(N.H.same(x,load('state-000.npy'))and N.H.same(g,load(calls['direct_gradient',1]['result']))and calls['hvp',514]['state']=='state-000.npy'and N.H.same(p,load(calls['hvp',514]['direction'])),'literal original current022 and actual direction/action')
 V=np.empty((98666,512),order='F');W=np.empty_like(V);indices=[1]+list(range(3,514));need(len(indices)==512,'literal all512 actions')
 for j,k in enumerate(indices):
  r=calls['hvp',k];need(r['state']=='state-000.npy','no cross-state Hessian action');V[:,j]=load(r['direction']);W[:,j]=load(r['result']);w.tick()
 trials=[];count=5;selected=[]
 for r in rows:
  if r['within_trust']:
   count+=1;call=calls['direct_gradient',count];trials.append(dict(alpha=r['alpha'],state=load(call['state']),gradient=load(call['result'])));selected.append(dict(trial_index=r['index'],gradient_index=count,files=call))
 need([r['alpha']for r in trials]==SETTINGS['alphas']and count==9,'all four original retained gradients');w.save('selected-actions.json',dict(hvp_indices=indices,physical_action=calls['hvp',514],actual_gradients=selected,initial_state='state-000.npy'))
 ans=H.diagnose(V,W,x,g,p,hp,origin,trials,rank_rtol=SETTINGS['rank_rtol'],tick=w.tick);need(ans['summary']['numerical_rank']==audit['numerical_action_rank']==512,'same independently qualified numerical image rank')
 for j,row in enumerate(ans['vectors']):
  for name,v in row.items():w.array(f'fits/{j:02d}-{name}.npy',v)
 w.save('diagnosis.json',ans['summary']);return dict(passed=True,native_calls=0,diagnosis=ans['summary'],original_best_endpoint_unchanged=True,scope='Saved same-current022 nonlinear-image representability and finite proposed accelerations only. No actual curved-path gradient, fresh HVP, correction application, improved merit, stationarity or orbit.')


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def tick(self):need(time.monotonic()-self.start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'300s4GiB saved diagnostic')
 def write(self,name,b,closing=False):
  need(self.used+len(b)<(128 if closing else 124)*1024**2,'128MiB4MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  self.used+=len(b)
 def save(self,name,value,closing=False):self.write(name,(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 def array(self,name,v):
  b=io.BytesIO();np.save(b,v,allow_pickle=False);self.write(name,b.getvalue());self.tick()


def main():
 resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned300s saved diagnostic')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),files={n:M.entry(ROOT/n)for n in FILES},python=platform.python_version(),numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),settings=SETTINGS,caps=CAPS))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'literal exact source');w.write('source/'+n,b)
  binding=bind(w,M.read(ROOT/CARD));profile(w);result=target(w,binding)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):M.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
