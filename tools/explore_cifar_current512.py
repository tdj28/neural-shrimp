"""One fresh512 current022 Newton correction; no historical Hessian transfer."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,io,json,math,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
import explore_cifar_nested512 as N
from neural_shrimp.cifar_current_correction import correct,SETTINGS
from neural_shrimp.cifar_nested_krylov import extend
from neural_shrimp.cifar_reorthogonalized import orthogonalize,SETTINGS as KS
B=N.B;E=N.E;U=N.U;need=U.need;read=U.read;entry=U.entry;Store=N.Store
BASE=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/fresh512001')
CARD='docs/plans/cifar-current512-inputs.json'
CAPS=dict(action=0,j2vp=0,hvp=514,direct_gradient=18)
LIMITS=dict(seconds=400,rss_kib=4*1024**2,as_bytes=12*1024**3,raw_bytes=1536*1024**2)
EXTRA=['tools/explore_cifar_current512.py','tools/launch_cifar_current512.py','src/neural_shrimp/cifar_current_correction.py','tests/test_cifar_current_correction.py','tests/test_cifar_current512_driver.py','docs/plans/cifar-current512.md',CARD,'docs/reviews/cifar-nested512001.json']
def install():N.install();B.SOURCES=tuple(dict.fromkeys(tuple(B.SOURCES)+tuple(EXTRA)))
class Budget:
 def __init__(self,event):self.start=time.monotonic();self.event=event;self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0);self.stopped=False
 def check(self):need(not self.stopped and time.monotonic()-self.start<LIMITS['seconds']and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<LIMITS['rss_kib'],'owned400s4GiB budget')
 def call(self,kind,fn,*a,**kw):
  self.check();need(self.attempted[kind]<CAPS[kind],'fixed native cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=fn(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));self.check();return v

def check_card(c):
 need(c['status']=='reviewed'and c['settings']==SETTINGS and c['limits']==LIMITS and c['operation_caps']==CAPS,'fixed reviewed current512 card')
 need(c['history']==read(ROOT/N.CARD)and c['state']=='state-022.npy'and c['gradient']==23,'exact current022 independent parent')
 need(c['origin']=='global-original-midpoint.npy'and c['fresh_columns']==512 and c['old_basis_reused']==0,'global origin and fresh operator only')
 need(isinstance(c['independent512'],dict)and set(c['independent512'])=={'root','source','outer','worker'},'sealed independent512 authority required')


def bind(c,inventory,runtime,budget,store):
 parent,index,old,*_=N.bind(c['history'],inventory,runtime,budget,store)
 p=c['positive512'];path=Path(p['root']);need(path==N.BASE,'literal completed larger-space comparison')
 U.inventory(path,p['outer'],budget.check);U.inventory(path/'run',p['worker'],budget.check);row=read(path/'run/result.json');U.source(path,row,p['source'])
 need(row['status']=='nested512_comparison_complete'and row['comparison']['qualified']and row['comparison']['material_improvement']and row['comparison']['strict_forcing_pass']and not row['direction_applied'],'qualified historical comparison only')
 for n,h in row['source_sha256'].items():need(inventory[n]==h,'unchanged nested/native source '+n)
 a=c['independent512'];ap=Path(a['root']);need(ap==Path('/data2/neural-shrimp-data/audits/cifar-nested512-independent/attempt001'),'literal512 independent audit')
 U.inventory(ap,a['outer'],budget.check);U.inventory(ap/'run',a['worker'],budget.check);ar=read(ap/'run/runtime.json');ao=read(ap/'run/outcome.json');ab=read(ap/'run/input-binding.json')
 need(ar['source_commit']==a['source']and ao['passed']and ao['native_calls']==0 and ao['recurrence_rows']==511 and ao['numerical_action_rank']==512 and all(ao['comparison'][k]for k in('qualified','material_improvement','strict_forcing_pass')),'independent complete512 arithmetic')
 need(ab['producer']==str(path)and ab['source']==p['source']and ab['pins']==[p['outer'],p['worker']]and ab['all_at_historical_state018'],'literal independent512 genealogy')
 for n,v in ar['files'].items():need(entry(ap/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',a['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'bound audit committed source')
 for pp,ss in [(path,p['source']),(ap,a['source'])]:
  owner=read(pp/'launcher-result.json');need(owner['source_commit']==ss and owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'completed owned predecessor')
  for k in('launcher_pid','owned_group_pid'):need(not Path('/proc',str(owner[k])).exists(),'actual predecessor gone')
 need(index['direct_gradient'][23]['state']=='state-022.npy'and all(v['state']!='state-022.npy'for v in index['hvp'].values()),'fresh current state has no old Hessian authority')
 events=[json.loads(x)for x in(parent/'run/events.jsonl').read_text().splitlines()]
 events += [json.loads(x)for x in(path/'run/events.jsonl').read_text().splitlines()]
 store.json('current-binding.json',dict(current_parent=c['history']['prior_comparison']['history']['producer'],state=c['state'],gradient=23,origin=c['origin'],positive512=p,independent512=a,old_basis_reused=0))
 return parent,index,old,events


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
  load=lambda n:U.vector(parent/'run'/n);x=load('state-022.npy');oldg=load(index['direct_gradient'][23]['output']);origin=load('global-original-midpoint.npy')
  need(B.byte_equal(x,load('stationary-point.npy'))and B.byte_equal(oldg,load('stationary-gradient.npy')),'literal independently accepted current022 endpoint')
  need(float(np.linalg.norm(x-origin))/max(1.,float(np.linalg.norm(origin)))<=.05,'original5percent trust ball')
  arrays,model,_,_,identity=B.inputs()
  for k in('data_identity','eta','lam','model'):need(identity[k]==old['input_identity'][k],'unchanged objective '+k)
  store.json('input-identity.json',identity);store.array('initial-point',x);store.array('initial-gradient',oldg);store.array('global-original-midpoint',origin)
  problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;native=E.Native(problem,store,budget)
  g=native.grad(x);need(B.byte_equal(g,oldg),'byte-exact current022 gradient');q=-g/np.linalg.norm(g);v=np.random.default_rng(28168).standard_normal(len(x));v/=np.linalg.norm(v)
  store.json('qualification-input.json',dict(state='state-022.npy',random_seed=28168,steps=[1e-4,1e-5],relative_state_scale=max(1.,float(np.linalg.norm(x))),basis_first='negative_current_gradient'))
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
  budget.check();result.update(status='current512_correction_complete',basis=report,correction=answer,direction_applied=answer['direction_applied'],input_identity=identity)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-budget.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 print(json.dumps({k:v for k,v in result.items()if k not in('source_sha256','basis')}));return 0 if result['status']=='current512_correction_complete'else 2

def main():
 resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');install();B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);return run(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
if __name__=='__main__':raise SystemExit(main())
