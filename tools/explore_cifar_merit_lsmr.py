"""Six bounded damped LSMR residual-merit steps at the consumed CIFAR state."""
import argparse,hashlib,inspect,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsmr
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import qualify_cifar_adjoint as C
B,D,A=C.B,C.D,C.A
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_residual_adjoint import residual_adjoint
from neural_shrimp.cifar_merit_lsmr import direction,trial
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/adjoint-pairing/attempt-001');PIN='7a11499aac75909d37e47d1f8acf3df717c6bcb0fc94319c810fd08e4b3af855';EXEC='ec1f5304207e45ff45010d7516e4d57641445e2b'
OUTER,INNER=6,200
CAPS=dict(action=OUTER*(INNER+2),adjoint=OUTER*(INNER+1),j2vp=OUTER*(INNER+2),hvp=OUTER*(4*INNER+6),direct_gradient=2+26*OUTER)
EXTRA=('src/neural_shrimp/cifar_merit_lsmr.py','tests/test_cifar_merit_lsmr.py','tools/explore_cifar_merit_lsmr.py','tools/launch_cifar_merit_lsmr.py','tests/test_cifar_merit_driver.py','docs/plans/cifar-merit-lsmr.md')
def install():C.install();B.SOURCES=tuple(B.SOURCES)+EXTRA
class Budget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=300.);self.attempted['adjoint']=0;self.returned['adjoint']=0
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def measured_gate(parent,events):
 B.require(parent['status']=='qualification_complete'and parent['qualification_passed']and len(parent['pairings'])==3 and all(x['passed']for x in parent['pairings']),'native transpose qualification')
 ts=[r['seconds']for r in parent['pairings']];B.require(all(len(t)==2 and all(math.isfinite(x)and x>0 for x in t)for t in ts),'native measured action times')
 grad=max(x['seconds']for x in events if x['kind']=='operation_returned'and x['operation']=='direct_gradient');startup=max(x['seconds']for x in events if x['kind']=='operation_returned'and x['operation']=='action')
 forecast=2*(startup+OUTER*((INNER+2)*(max(t[0]for t in ts)+max(t[1]for t in ts))+26*grad))+20
 B.require(math.isfinite(forecast)and forecast<300,'300s measured six-step LSMR gate')
 return dict(forecast_seconds=forecast,maximum_forward=max(t[0]for t in ts),maximum_adjoint=max(t[1]for t in ts),maximum_gradient=grad,startup_allowance=startup,outer_cap=OUTER,inner_cap=INNER,safety_factor=2,overhead_seconds=20)

def run(out,review):
 inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=255*1024**2);started=time.monotonic();budget=Budget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,accepted_orbit=False,spectrum_requested=False,comparison_qualified=False,method='damped LSMR residual-merit correction; actual Armijo and reduction-ratio acceptance; old Newton forcing reported separately',steps=[],accepted_steps=0)
 try:
  store.json('runtime-start.json',result);store.write('lsmr-source.py',inspect.getsource(lsmr).encode());A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC,'exact qualified operator source')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'unchanged qualified native source '+n)
  result['profile']=measured_gate(old,[json.loads(x)for x in(PARENT/'events.jsonl').read_text().splitlines()]);store.json('profile.json',result['profile'])
  def read(n):
   a=np.load(PARENT/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite qualified state');return a
  theta,one,two=[read('initial-'+n)for n in('theta','map1','map2')];arrays,basis,_,_,identity=B.inputs();B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same development map')
  result['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,parent_state_sha256=A.L.entry(PARENT/'initial-theta.npy')['sha256']);store.json('input-identity.json',result['input_identity']);store.array('initial-theta',theta)
  problem=B.CountedProblem(B.cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  def map_one(x):return problem.update(B.cifar.as_tensor(x),eta).numpy()
  first=map_one(theta);store.array('initial-map1',first);B.require(B.byte_equal(first,one),'native phase1 replay');second=map_one(first);store.array('initial-map2',second);B.require(B.byte_equal(second,two),'native phase2 replay')
  damp=.001;result['last_accepted_state']='initial-theta.npy';result['stop']='outer_budget_complete'
  for k in range(OUTER):
   budget.check();r=second-theta;closure=float(np.linalg.norm(r))/max(1.,float(np.linalg.norm(theta)));separation=float(np.linalg.norm(first-theta))/max(1.,float(np.linalg.norm(theta)))
   if closure<1e-8 and separation>=1e-7:result['stop']='closure_candidate_requires_separate_validation';break
   B.require(np.linalg.norm(r)>0 and separation>=1e-7,'nontrivial current residual/cycle separation');store.array(f'outer-{k}-residual',r)
   def forward(v):return budget.call('action',lambda:problem.jvp_two_step(B.cifar.as_tensor(theta),B.cifar.as_tensor(v),eta,theta_next=B.cifar.as_tensor(first)).numpy()-v)
   def adjoint(v):return budget.call('adjoint',lambda:residual_adjoint(problem,B.cifar.as_tensor(theta),B.cifar.as_tensor(first),B.cifar.as_tensor(v),eta).numpy())
   p,ap,linear=direction(forward,adjoint,r,damp,iterations=INNER,radius=.05*max(1.,float(np.linalg.norm(theta))),deadline=budget.deadline)
   store.array(f'outer-{k}-direction',p);store.array(f'outer-{k}-actual-action',ap);store.json(f'outer-{k}-linear.json',linear)
   step=dict(index=k,base_state=result['last_accepted_state'],linear=linear,trials=[],accepted_index=None);result['steps'].append(step)
   if not linear['usable_descent']:result['stop']='non_descent_direction';break
   for j in range(13):
    budget.check();alpha=2.**(-j);x=theta+alpha*p;y=map_one(x);z=map_one(y)
    for name,v in [('theta',x),('map1',y),('map2',z)]:store.array(f'outer-{k}-trial-{j}-'+name,v)
    metrics=trial(x,y,z,r,ap,alpha);metrics['index']=j;step['trials'].append(metrics);store.event(dict(kind='actual_merit_trial',outer=k,**metrics))
    if metrics['accepted_local_step']:
     theta,first,second=x,y,z;step['accepted_index']=j;result['accepted_steps']+=1;result['last_accepted_state']=f'outer-{k}-trial-{j}-theta.npy'
     if metrics['ratio']>.75:damp=max(1e-9,damp/4)
     elif metrics['ratio']<.25:damp=min(1.,damp*4)
     break
   if step['accepted_index']is None:damp=min(1.,damp*10)
   store.json(f'outer-{k}-step.json',step)
  budget.check();result.update(status='merit_pilot_complete',comparison_qualified=True,final_relative_residual=float(np.linalg.norm(second-theta))/max(1.,float(np.linalg.norm(theta))),final_relative_separation=float(np.linalg.norm(first-theta))/max(1.,float(np.linalg.norm(theta))),final_damp=damp)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
