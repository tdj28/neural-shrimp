"""One fresh512 correction at the independently selected curved1/32 endpoint."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,io,json,math,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
import explore_cifar_nested512 as N
import explore_cifar_current512 as C
import audit_cifar_curved_saved as K
import explore_cifar_curved_path as P
from neural_shrimp.cifar_current_correction import correct,SETTINGS
from neural_shrimp.cifar_nested_krylov import extend
from neural_shrimp.cifar_reorthogonalized import orthogonalize,SETTINGS as KS
B=N.B;E=N.E;U=N.U;need=U.need;read=U.read;entry=U.entry;Store=N.Store
BASE=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/curved-fresh512001')
CARD='docs/plans/cifar-curved512-inputs.json'
CAPS=dict(action=0,j2vp=0,hvp=514,direct_gradient=18)
LIMITS=dict(seconds=1500,rss_kib=4*1024**2,as_bytes=12*1024**3,raw_bytes=1536*1024**2)
EXTRA=['tools/explore_cifar_curved512.py','tools/launch_cifar_curved512.py','tests/test_cifar_curved512_driver.py','docs/plans/cifar-curved512.md',CARD,'docs/reviews/cifar-curved-path-independent.json']
def install():
 P.install();B.SOURCES=tuple(sorted(set(B.SOURCES)|set(K.FILES)|set(EXTRA)))
class Budget:
 def __init__(self,event):self.start=time.monotonic();self.event=event;self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0);self.stopped=False
 def check(self):need(not self.stopped and time.monotonic()-self.start<LIMITS['seconds']and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMITS['rss_kib'],'owned1500s4GiB budget')
 def call(self,kind,fn,*a,**kw):
  self.check();need(self.attempted[kind]<CAPS[kind],'fixed native cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=fn(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v # Native stores its return before the next budget check.

def check_card(c):
 need(c['status']=='reviewed'and c['settings']==SETTINGS and c['limits']==LIMITS and c['operation_caps']==CAPS,'fixed reviewed curved-endpoint512 card')
 need(c['producer']==K.PRODUCER and c['independent']==read(ROOT/'docs/reviews/cifar-curved-path-independent.json'),'literal independently qualified curved endpoint')
 need(c['selected_alpha']==1/32 and c['selected_index']==0 and c['state']=='final-point.npy'and c['gradient']=='final-gradient.npy','fixed first admitted curved point')
 need(c['origin']=='global-original-midpoint.npy'and c['fresh_columns']==512 and c['old_basis_reused']==0,'global original5percent trust and fresh operator')

class Binding:
 def __init__(self,budget,store):self.budget=budget;self.store=store
 def tick(self):self.budget.check()
 def save(self,n,v):self.store.json('ancestry-'+n,v)

def bind(c,inventory,runtime,budget,store):
 check_card(c);w=Binding(budget,store);d=c['independent'];ar=Path(d['root']);K.inventory(ar,d['outer']['sha256'],w.tick);K.inventory(ar/'run',d['worker']['sha256'],w.tick)
 rt=read(ar/'run/runtime.json');ao=read(ar/'run/result.json');ab=read(ar/'run/input-binding.json');la=read(ar/'launcher-result.json')
 need(rt['source_commit']==la['source_commit']==d['execution']and ao==d['outcome']and ao['passed']and ao['native_calls']==0 and ao['arithmetic']['accepted_index']==0 and ao['arithmetic']['trials'][0]['alpha']==1/32 and ab['producer']==K.PRODUCER,'entire independent first-curved-endpoint authority')
 need(la['exit_code']==0 and la['failure']is None and not la['owned_live_pids']and la['worker_pid_absent'],'complete independent owned execution')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'actual prior owner absent')
 for n,e in rt['sources'].items():
  need(entry(ar/'run/source'/n)==e and hashlib.sha256(subprocess.check_output(['git','show',d['execution']+':'+n],cwd=ROOT)).hexdigest()==e['sha256'],'exact independent source '+n);budget.check()
 bound,ops,_,_=K.metadata(w);folder,files,old=bound[0]
 need(old['status']=='curved_path_comparison_complete'and old['direction_applied']and old['correction']['accepted_index']==0 and old['comparison']['chosen_alpha']==1/32,'literal original fixed first point')
 for n,h in old['source_sha256'].items():
  if n.startswith('src/') or n.startswith('tools/explore_cifar_current512'):
   need(inventory[n]==h,'unchanged native/helper source '+n)
 for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(runtime[k]==old['runtime'][k],'same measured native runtime '+k)
 events=[]
 for run,_,_ in bound:
  p=run/'events.jsonl'
  if p.exists():events.extend(json.loads(v)for v in p.read_text().splitlines())
 # Include the independently recorded actual cold floor, not just warm calls.
 floor=ao['cost_observation']['retained_upstream_hvp_seconds'];need(floor==1.1099920831620693,'literal retained cold HVP cost')
 events.append(dict(kind='operation_returned',operation='hvp',seconds=floor))
 store.json('curved-endpoint-binding.json',dict(producer=K.PRODUCER,independent=dict(root=d['root'],source=d['execution'],outer=d['outer'],worker=d['worker']),selected_alpha=1/32,selected_index=0,point=files['final-point.npy'],gradient=files['final-gradient.npy'],origin=files[c['origin']],actual_gradient=ops['direct_gradient',1],retained_cold_hvp_seconds=floor,old_basis_reused=0,target_arrays_unopened=True))
 return folder,ops,old,events


def forecasts(orth,algebra,maxima,elapsed,archive,peak):
 need(all(math.isfinite(v)and v>0 for v in[orth,algebra,*maxima.values()]),'positive finite measured costs')
 raw=1080*(98666*8+128)+128*1024**2+archive
 seconds=elapsed+2*(511*orth+algebra+sum(CAPS[k]*maxima[k]for k in('hvp','direct_gradient')))+30+raw/(50*1024**2)
 rss=peak+1024**2
 return dict(passed=seconds<LIMITS['seconds']and rss<LIMITS['rss_kib']and raw<LIMITS['raw_bytes'],forecast_seconds=seconds,forecast_raw_bytes=raw,forecast_peak_rss_kib=rss,max_projection_seconds=orth,algebra_seconds=algebra,maximum_native_seconds=maxima,safety_factor=2,raw_array_count_ceiling=1080,target_arrays_unopened=True)


def profile(events,budget,store,archive):
 rng=np.random.default_rng(52522);n,m=98666,512;store.json('profile-input.json',dict(seed=52522,shape=[n,m],recipe='full512 independent V/W; three maximum-width two-pass projections; fullGram/SVD/Vc/Wc',target_arrays_unopened=True))
 V=np.asfortranarray(rng.standard_normal((n,m))/np.sqrt(n));W=np.asfortranarray(np.linspace(.5,2,n)[:,None]*V);w=rng.standard_normal(n);times=[]
 for _ in range(3):
  t=time.monotonic();q,cs=orthogonalize(V,w);times.append(time.monotonic()-t);budget.check()
 t=time.monotonic();gram=V.T@V;coef,_,rank,sv=np.linalg.lstsq(W,-V[:,0],rcond=KS['svd_rcond']);p=V@coef;hp=W@coef;algebra=time.monotonic()-t;budget.check()
 maxima={k:max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']==k)for k in('hvp','direct_gradient')}
 answer=forecasts(max(times),algebra,maxima,time.monotonic()-budget.start,archive,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);answer.update(samples=times,synthetic_rank=int(rank));store.json('profile.json',answer);need(answer['passed'],'complete full-size admission before target arrays');del V,W,w,q,cs,gram,coef,sv,p,hp;gc.collect();return answer


def run(source):
 inventory=B.sources();runtime=B.environment();store=Store(BASE/'run',limit=LIMITS['raw_bytes'],reserve=4*1024**2);budget=Budget(store.event)
 signal.signal(signal.SIGTERM,lambda *_:setattr(budget,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(budget,'stopped',True));signal.setitimer(signal.ITIMER_REAL,LIMITS['seconds'])
 result=dict(status='running',source_commit=source,source_sha256=inventory,runtime=runtime,settings=SETTINGS,accepted_orbit=False,rigorous_certificate=False,direction_applied=False,old_basis_reused=0)
 try:
  need(source==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'exact execution source');card=read(ROOT/CARD);check_card(card);store.json('runtime-start.json',result);archive=io.BytesIO()
  with tarfile.open(fileobj=archive,mode='w:gz')as tar:
   for n in inventory:
    data=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==inventory[n],'literal own committed source');i=tarfile.TarInfo(n);i.size=len(data);i.mtime=0;tar.addfile(i,io.BytesIO(data))
  store.write('source.tar.gz',archive.getvalue());parent,index,old,events=bind(card,inventory,runtime,budget,store);result['profile']=profile(events,budget,store,len(archive.getvalue()))
  load=lambda n:U.vector(parent/n);x=load('final-point.npy');oldg=load('final-gradient.npy');origin=load('global-original-midpoint.npy')
  need(B.byte_equal(x,load(index['direct_gradient',1]['state']))and B.byte_equal(oldg,load(index['direct_gradient',1]['result'])),'literal independently selected curved1/32 state and gradient')
  need(float(np.linalg.norm(x-origin))/max(1.,float(np.linalg.norm(origin)))<=.05,'original5percent trust ball')
  arrays,model,_,_,identity=B.inputs()
  for k in('data_identity','eta','lam','model'):need(identity[k]==old['input_identity'][k],'unchanged objective '+k)
  store.json('input-identity.json',identity);store.array('initial-point',x);store.array('initial-gradient',oldg);store.array('global-original-midpoint',origin)
  problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;native=E.Native(problem,store,budget)
  g=native.grad(x);need(B.byte_equal(g,oldg),'byte-exact selected curved1/32 gradient');q=-g/np.linalg.norm(g);v=np.random.default_rng(28168).standard_normal(len(x));v/=np.linalg.norm(v)
  store.json('qualification-input.json',dict(state='initial-point.npy',random_seed=28168,steps=[1e-4,1e-5],relative_state_scale=max(1.,float(np.linalg.norm(x))),basis_first='negative_current_gradient'))
  hq,hv=native.hvp(x,q),native.hvp(x,v);sym=abs(float(np.dot(q,hv)-np.dot(v,hq)))/max(float(np.linalg.norm(hq)),float(np.linalg.norm(hv)),1e-12);need(sym<=1e-10,'fresh current Hessian pairing');fdrows=[]
  for i,h in enumerate(max(1.,float(np.linalg.norm(x)))*np.array([1e-4,1e-5])):
   gm,gp=native.grad(x-h*q),native.grad(x+h*q);fd=(gp-gm)/(2*h);store.array(f'qualification-fd-{i}',fd);fdrows.append(dict(h=float(h),relative_error=E.P.fd_error(hq,fd)))
  need(fdrows[-1]['relative_error']<=5e-6,'fresh current Hessian finite difference');result['qualification']=dict(passed=True,symmetry_error=sym,finite_differences=fdrows);store.json('qualification.json',result['qualification'])
  p,pred,gram,report=extend(g,q[:,None],hq[:,None],lambda w:native.hvp(x,w),max_columns=512,event=lambda k,v:store.event(dict(kind=k,**v)),tick=budget.check)
  for name,value in [('candidate-direction',p),('predicted-action',pred),('final-gram',gram)]:store.array(name,value)
  store.json('basis-report.json',report);actual=native.hvp(x,p)
  point,grad,answer=correct(x,g,origin,p,pred,actual,report,native.grad,event=store.event)
  store.array('final-point',point);store.array('final-gradient',grad);store.json('correction.json',answer)
  need(budget.attempted==budget.returned and budget.returned['hvp']==report['new_basis_actions']+3 and budget.returned['direct_gradient']==5+sum(t['within_trust']for t in answer['trials']),'complete fresh qualification/basis/final/trial operations')
  budget.check();result.update(status='curved512_correction_complete',basis=report,correction=answer,direction_applied=answer['direction_applied'],input_identity=identity)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-budget.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 print(json.dumps({k:v for k,v in result.items()if k not in('source_sha256','basis')}));return 0 if result['status']=='curved512_correction_complete'else 2

def main():
 resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');install();B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);return run(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
if __name__=='__main__':raise SystemExit(main())
