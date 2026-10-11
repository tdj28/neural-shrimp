"""One same-current022 saved-span regularized correction, actual nonlinear merit."""
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
from neural_shrimp.cifar_regularized_projection import propose
B=E.B;need=E.need;read=E.read;entry=E.entry;Store=E.Store
OUT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/regularized001')
CARD='docs/plans/cifar-regularized-inputs.json'
SETTINGS=dict(target=.49,rank_rtol=1e-12,current='state-022.npy',columns=512,origin='global-original-midpoint.npy',fresh_basis_actions=0)
CAPS=dict(action=0,j2vp=0,hvp=1,direct_gradient=14)
LIMITS=dict(seconds=300,rss_kib=4*1024**2,as_bytes=12*1024**3,raw_bytes=128*1024**2)
EXTRA=[CARD,'docs/plans/cifar-regularized-projection.md','tools/explore_cifar_regularized.py','tools/launch_cifar_regularized.py','src/neural_shrimp/cifar_regularized_projection.py','tests/test_cifar_regularized_projection.py','tests/test_cifar_regularized_driver.py','docs/reviews/cifar-current512-independent.json','docs/reviews/cifar-current512-nonlinearity001.json']
def install():
 E.install();local=[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_file()and Path(m.__file__).resolve().is_relative_to(ROOT)]
 B.SOURCES=tuple(sorted(set(B.SOURCES)|set(A.FILES)|set(EXTRA)|set(local)))

class Budget:
 def __init__(self,event):self.start=time.monotonic();self.event=event;self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0);self.stopped=False
 def check(self):need(not self.stopped and time.monotonic()-self.start<LIMITS['seconds']and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMITS['rss_kib'],'300s4GiB owned budget')
 def call(self,kind,fn,*a,**kw):
  self.check();need(self.attempted[kind]<CAPS[kind],'fixed native cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=fn(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v # Native stores complete return before checking elapsed budget.

def check_card(c):
 need(c['status']=='reviewed'and c['settings']==SETTINGS and c['limits']==LIMITS and c['operation_caps']==CAPS,'fixed reviewed same-current proposal')
 need(c['producer']==dict(root=str(A.PARENT),source=A.EXEC,outer=A.PINS[0],worker=A.PINS[1]),'literal complete current512 authority')
 need(c['independent']==read(ROOT/'docs/reviews/cifar-current512-independent.json'),'literal whole independent arithmetic receipt')
 need(c['diagnostic']==read(ROOT/'docs/reviews/cifar-current512-nonlinearity001.json'),'literal failed larger-step history')

def bind(c,b,store,inventory,runtime):
 check_card(c);binding=A.bind(b.check);old=binding[0]
 for n,h in old['source_sha256'].items():need(inventory[n]==h,'unchanged original native and current512 source '+n)
 for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(runtime[k]==old['runtime'][k],'same native runtime '+k)
 audit=c['independent'];root=Path(audit['root']);A.audit_source(root,audit['source'],audit['inventories'][0]['manifest']['sha256'],audit['inventories'][1]['manifest']['sha256'],b.check)
 ao=read(root/'run/outcome.json');need(ao['passed']and ao['recurrence_rows']==511 and ao['numerical_action_rank']==512 and ao['correction']['accepted_index']==6 and ao['native_calls']==0,'completed same-state qualified512 arithmetic')
 d=c['diagnostic'];dr=Path(d['root']);A.M.inventory(dr,d['outer']['sha256'],b.check);A.M.inventory(dr/'run',d['worker']['sha256'],b.check);need(read(dr/'run/outcome.json')['passed'],'complete saved nonlinear error diagnosis')
 store.json('input-binding.json',dict(card=c,all512_actions_same_original022=True,comparison_same_start=True,native_calls_before_profile=0));return binding

def forecasts(algebra,maxima,elapsed,archive,peak):
 need(all(math.isfinite(x)and x>0 for x in[algebra,*maxima.values()]),'finite positive complete measured maxima')
 raw=64*(98666*8+128)+32*1024**2+archive;seconds=elapsed+2*(algebra+sum(CAPS[k]*maxima[k]for k in maxima))+30+raw/(50*1024**2);rss=peak+768*1024
 return dict(passed=seconds<300 and raw<128*1024**2 and rss<4*1024**2,forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,algebra_seconds=algebra,maximum_native_seconds=maxima,retained_vector_ceiling=64,safety_factor=2,target_arrays_unopened=True)

def profile(b,store,archive):
 rng=np.random.default_rng(52524);n,m=98666,512;store.json('profile-input.json',dict(seed=52524,shape=[n,m],recipe='V=normal/sqrt(n),W=V*linspace(.5,2,512),g=V[:,0]; complete physical whitening, fullSVD,64step discrepancy solve and physical assembly',target_arrays_unopened=True))
 V=np.asfortranarray(rng.standard_normal((n,m))/np.sqrt(n));W=np.asfortranarray(V*np.linspace(.5,2,m));g=V[:,0].copy();t=time.monotonic();z=propose(V,W,g,tick=b.check);elapsed=time.monotonic()-t;b.check()
 events=[]
 for folder in[A.PARENT,A.C.PARENT,A.N.PARENT]:events.extend(json.loads(s)for s in(folder/'run/events.jsonl').read_text().splitlines())
 maxima={k:max(v['seconds']for v in events if v['kind']=='operation_returned'and v['operation']==k)for k in('hvp','direct_gradient')}
 result=forecasts(elapsed,maxima,time.monotonic()-b.start,archive,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);store.json('profile.json',result);need(result['passed'],'full-size measured admission before target arrays');del V,W,g,z;gc.collect();return result

def run():
 install();B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();inventory=B.sources();runtime=B.environment();store=Store(OUT/'run',limit=LIMITS['raw_bytes'],reserve=4*1024**2);b=Budget(store.event)
 signal.signal(signal.SIGTERM,lambda *_:setattr(b,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(b,'stopped',True));signal.setitimer(signal.ITIMER_REAL,300)
 result=dict(status='running',source_commit=source,source_sha256=inventory,runtime=runtime,settings=SETTINGS,accepted_orbit=False,rigorous_certificate=False,direction_applied=False)
 try:
  store.json('runtime-start.json',result);stream=io.BytesIO()
  with tarfile.open(fileobj=stream,mode='w:gz')as tar:
   for n,h in inventory.items():
    data=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==h,'literal committed own source');info=tarfile.TarInfo(n);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
  store.write('source.tar.gz',stream.getvalue());binding=bind(read(ROOT/CARD),b,store,inventory,runtime);result['profile']=profile(b,store,len(stream.getvalue()));old,calls,*_=binding
  folder=A.PARENT/'run';load=lambda n:A.M.vector(folder/n);x=load('state-000.npy');g=load(calls['direct_gradient',1]['result']);origin=load('global-original-midpoint.npy')
  need(B.byte_equal(x,A.M.vector(A.C.PARENT/'run/state-022.npy'))and B.byte_equal(x,load('initial-point.npy'))and B.byte_equal(g,load('initial-gradient.npy')),'same exact original022 current and gradient')
  V=np.empty((98666,512),order='F');W=np.empty_like(V)
  for j in range(512):
   b.check();row=calls['hvp',1 if j==0 else j+2];V[:,j]=load(row['direction']);W[:,j]=load(row['result'])
  z=propose(V,W,g,target=SETTINGS['target'],rank_rtol=SETTINGS['rank_rtol'],tick=b.check);p,pred=z['direction'],z['predicted_action'];store.json('proposal.json',z['summary']);store.array('coefficients',z['coefficients']);store.array('candidate-direction',p);store.array('predicted-action',pred);del V,W;gc.collect()
  arrays,model,_,_,identity=B.inputs()
  for k in('data_identity','eta','lam','model'):need(identity[k]==old['input_identity'][k],'same original training objective '+k)
  store.json('input-identity.json',identity)
  for n,v in [('initial-point',x),('initial-gradient',g),('global-original-midpoint',origin)]:store.array(n,v)
  problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=b;native=E.E.Native(problem,store,b)
  actualg=native.grad(x);need(B.byte_equal(actualg,g),'fresh byte-exact current gradient');hp=native.hvp(x,p)
  point,grad,correction=E.correct(x,g,origin,p,pred,hp,old['basis'],native.grad,event=store.event)
  store.array('final-point',point);store.array('final-gradient',grad);store.json('correction.json',correction)
  baselineg=load('final-gradient.npy');baseline=A.M.dot(baselineg,baselineg)/A.M.dot(g,g);ratio=A.M.dot(grad,grad)/A.M.dot(g,g)
  comparison=dict(baseline_alpha=1/64,baseline_merit_ratio=baseline,new_merit_ratio=ratio,new_alpha=None if correction['accepted_index']is None else 2.**(-correction['accepted_index']),improves_same_start_merit=bool(correction['direction_applied']and ratio<baseline),direction_norm_ratio=z['summary']['norm_ratio'],scope='Adaptive development comparison at exactly the same current022 point and original trust ball; no state transfer or stationary/cycle claim.')
  store.json('comparison.json',comparison);need(b.attempted==b.returned and b.returned==dict(action=0,j2vp=0,hvp=1,direct_gradient=1+sum(v['within_trust']for v in correction['trials'])),'complete all-inclusive actual calls');b.check();result.update(status='regularized_correction_complete',correction=correction,comparison=comparison,proposal=z['summary'],input_identity=identity,direction_applied=correction['direction_applied'])
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-b.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=CAPS,operation_attempted=b.attempted,operation_returned=b.returned);store.finish(result)
 print(json.dumps({k:v for k,v in result.items()if k not in('source_sha256','proposal')}));return 0 if result['status']=='regularized_correction_complete'else 2
if __name__=='__main__':
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');raise SystemExit(run())
