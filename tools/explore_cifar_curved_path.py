"""Fixed two-point same-current022 quadratic path with one actual acceleration action."""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
import gc,hashlib,io,json,math,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import explore_cifar_current512 as E
import review_cifar_current512 as A
import explore_cifar_regularized as R
from neural_shrimp.cifar_curved_path import evaluate,SETTINGS
B=E.B;need=E.need;read=E.read;entry=E.entry;Store=E.Store
OUT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/curved-path001')
CARD='docs/plans/cifar-curved-path-inputs.json'
CAPS=dict(action=0,j2vp=0,hvp=1,direct_gradient=2)
LIMITS=dict(seconds=120,rss_kib=4*1024**2,as_bytes=12*1024**3,raw_bytes=64*1024**2)
EXTRA=[CARD,'docs/plans/cifar-curved-path.md','tools/explore_cifar_curved_path.py','tools/launch_cifar_curved_path.py','src/neural_shrimp/cifar_curved_path.py','tests/test_cifar_curved_path.py','tests/test_cifar_curved_driver.py','docs/reviews/cifar-remainder-span001.json','docs/reviews/cifar-regularized001.json']
def install():
 R.install();local=[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_file()and Path(m.__file__).resolve().is_relative_to(ROOT)]
 B.SOURCES=tuple(sorted(set(B.SOURCES)|set(A.FILES)|set(EXTRA)|set(local)))

class Budget:
 def __init__(self,event):self.start=time.monotonic();self.event=event;self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0);self.stopped=False
 def check(self):need(not self.stopped and time.monotonic()-self.start<LIMITS['seconds']and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMITS['rss_kib'],'120s4GiB owned budget')
 def call(self,kind,fn,*a,**kw):
  self.check();need(self.attempted[kind]<CAPS[kind],'fixed native cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=fn(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v # Native stores complete return before checking elapsed budget.


def check_card(c):
 need(c['status']=='reviewed'and c['settings']==SETTINGS and c['limits']==LIMITS and c['operation_caps']==CAPS,'fixed reviewed quadratic path')
 for key,name in [('remainder','cifar-remainder-span001'),('native_cost','cifar-regularized001')]:need(c[key]==read(ROOT/f'docs/reviews/{name}.json'),'literal '+key+' authority')

def bind(c,b,store,inventory,runtime):
 check_card(c);binding=R.bind(read(ROOT/R.CARD),b,store,inventory,runtime)
 d=c['remainder'];root=Path(d['root']);need(str(root)=='/data2/neural-shrimp-data/explore/cifar/stationary-seed/remainder-span001','literal remainder source');A.M.inventory(root,d['outer']['sha256'],b.check);A.M.inventory(root/'run',d['worker']['sha256'],b.check)
 o=read(root/'run/outcome.json');rt=read(root/'run/runtime.json');owner=read(root/'launcher-result.json')
 need(o==d['outcome']and o['passed']and o['native_calls']==0 and o['diagnosis']['fixed_reference_alpha']==1/64,'fixed smallest-alpha acceleration')
 need(rt['source_commit']==owner['source_commit']==d['execution']and owner['exit_code']==0 and not owner['owned_live_pids']and owner['failure']is None,'sealed complete diagnostic')
 for pid in[rt['pid'],owner['launcher_pid'],owner['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual diagnostic owners absent')
 for n,v in rt['files'].items():need(entry(root/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',d['execution']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed exact diagnostic source');b.check()
 for n,v in d['artifacts'].items():need(entry(root/'run'/n)==v,'selected diagnostic file hash')
 r=c['native_cost'];rp=Path(r['root']);need(str(rp)=='/data2/neural-shrimp-data/explore/cifar/stationary-seed/regularized001','literal measured native source')
 for sub,pin in zip([rp,rp/'run'],r['inventories']):A.M.inventory(sub,pin['manifest']['sha256'],b.check)
 rr=read(rp/'run/result.json');need(rr==r['outcome'],'exact measured native outcome')
 for n,h in rr['source_sha256'].items():need(inventory[n]==h,'unchanged same-size native source '+n)
 for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(runtime[k]==rr['runtime'][k],'same measured native runtime '+k)
 owner=read(rp/'launcher-result.json');need(owner==r['launcher']and owner['exit_code']==0 and not owner['owned_live_pids']and owner['worker_pid_absent'],'completed native cost authority')
 for pid in[owner['launcher_pid'],owner['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual previous native owners absent')
 store.json('curved-binding.json',dict(card=c,target_arrays_unopened=True,shared_native_source_verified=True));return binding

def forecast(vector_seconds,maxima,elapsed,archive,peak):
 need(all(math.isfinite(v)and v>0 for v in[vector_seconds,*maxima.values()]),'positive measured costs')
 raw=32*(98666*8+128)+16*1024**2+archive
 seconds=elapsed+2*(vector_seconds+sum(CAPS[k]*maxima[k]for k in maxima))+30+raw/(50*1024**2)
 rss=peak+256*1024
 return dict(passed=seconds<LIMITS['seconds']and raw<LIMITS['raw_bytes']and rss<LIMITS['rss_kib'],forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,vector_seconds=vector_seconds,maximum_native_seconds=maxima,retained_vector_ceiling=32,target_arrays_unopened=True,native_profile_calls=0,cost_basis='Maximum sealed same-source same-model same-shape actual HVP/gradient time; full-size vector profile.',safety_factor=2)

def profile(c,b,store,archive):
 rng=np.random.default_rng(52526);x=rng.standard_normal(98666);g=x/100;p=-g;a=p/10
 start=time.monotonic()
 for alpha in SETTINGS['alphas']:
  y=x+alpha*p+alpha*alpha*a;float(np.linalg.norm(y));float(np.dot(g,p))
 elapsed=time.monotonic()-start;b.check();events=[]
 for folder in[A.PARENT,Path(c['native_cost']['root'])]:events.extend(json.loads(s)for s in(folder/'run/events.jsonl').read_text().splitlines())
 maxima={k:max(v['seconds']for v in events if v['kind']=='operation_returned'and v['operation']==k)for k in('hvp','direct_gradient')}
 peak=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,c['native_cost']['outcome']['peak_rss_kib'])
 z=forecast(elapsed,maxima,time.monotonic()-b.start,archive,peak);store.json('profile.json',z);need(z['passed'],'measured same-native admission before target arrays');return z

def run():
 install();B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();inventory=B.sources();runtime=B.environment();store=Store(OUT/'run',limit=LIMITS['raw_bytes'],reserve=4*1024**2);b=Budget(store.event)
 signal.signal(signal.SIGTERM,lambda *_:setattr(b,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(b,'stopped',True));signal.setitimer(signal.ITIMER_REAL,LIMITS['seconds'])
 result=dict(status='running',source_commit=source,source_sha256=inventory,runtime=runtime,settings=SETTINGS,accepted_orbit=False,rigorous_certificate=False,direction_applied=False)
 try:
  store.json('runtime-start.json',result);stream=io.BytesIO()
  with tarfile.open(fileobj=stream,mode='w:gz')as tar:
   for n,h in inventory.items():
    data=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==h,'literal committed own source');info=tarfile.TarInfo(n);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
  store.write('source.tar.gz',stream.getvalue());card=read(ROOT/CARD);binding=bind(card,b,store,inventory,runtime);result['profile']=profile(card,b,store,len(stream.getvalue()));old,calls,*_=binding
  folder=A.PARENT/'run';load=lambda n:A.M.vector(folder/n);x=load('state-000.npy');g=load(calls['direct_gradient',1]['result']);origin=load('global-original-midpoint.npy');p=load('candidate-direction.npy');hp=load(calls['hvp',514]['result']);pred=load('predicted-action.npy')
  need(B.byte_equal(x,A.M.vector(A.C.PARENT/'run/state-022.npy'))and B.byte_equal(x,load('initial-point.npy'))and B.byte_equal(g,load('initial-gradient.npy'))and calls['hvp',514]['state']=='state-000.npy'and B.byte_equal(p,load(calls['hvp',514]['direction'])),'exact current022 gradient and physical tangent action')
  dr=Path(card['remainder']['root'])/'run';acc=A.M.vector(dr/'fits/03-acceleration.npy');predacc=A.M.vector(dr/'fits/03-predicted_action.npy')
  for name,v in [('initial-point',x),('initial-gradient',g),('global-original-midpoint',origin),('candidate-direction',p),('original-action',hp),('original-predicted-action',pred),('acceleration',acc),('predicted-acceleration-action',predacc)]:store.array(name,v)
  arrays,model,_,_,identity=B.inputs()
  for k in('data_identity','eta','lam','model'):need(identity[k]==old['input_identity'][k],'same original training objective '+k)
  store.json('input-identity.json',identity);problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=b;native=E.E.Native(problem,store,b)
  actualacc=native.hvp(x,acc);correction,vectors=evaluate(x,g,origin,p,pred,hp,old['basis'],acc,predacc,actualacc,native.grad,event=store.event)
  baselineg=load('final-gradient.npy');baseline=A.M.dot(baselineg,baselineg)/A.M.dot(g,g)
  chosen=correction['accepted_index'];point,grad=(x,g)if chosen is None else vectors[chosen]
  store.array('final-point',point);store.array('final-gradient',grad);store.json('correction.json',correction)
  comparison=dict(baseline_alpha=1/64,baseline_merit_ratio=baseline,trial_merit_ratios=[r.get('merit_ratio')for r in correction['trials']],chosen_alpha=None if chosen is None else correction['trials'][chosen]['alpha'],chosen_merit_ratio=A.M.dot(grad,grad)/A.M.dot(g,g),improves_same_start_merit=bool(chosen is not None and correction['trials'][chosen]['merit_ratio']<baseline),scope='Adaptive development at exactly current022 with both fixed curved-path gradients. No stationary root, periodic orbit or test-set benefit follows.')
  store.json('comparison.json',comparison);need(b.attempted==b.returned and b.returned==dict(action=0,j2vp=0,hvp=1,direct_gradient=sum(r['admitted']for r in correction['trials'])),'all-inclusive complete native returns');b.check();result.update(status='curved_path_comparison_complete',correction=correction,comparison=comparison,input_identity=identity,direction_applied=chosen is not None)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-b.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=CAPS,operation_attempted=b.attempted,operation_returned=b.returned);store.finish(result)
 print(json.dumps({k:v for k,v in result.items()if k!='source_sha256'}));return 0 if result['status']=='curved_path_comparison_complete'else 2
if __name__=='__main__':
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');raise SystemExit(run())
