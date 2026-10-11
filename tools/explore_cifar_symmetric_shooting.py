"""One native symmetric-shooting MINRES direction and qualified backtracking."""
import argparse,hashlib,inspect,json,math,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import minres
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_cifar_merit_lsmr as C
B,D,A=C.B,C.D,C.A
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_symmetric_shooting import shooting_action,direction
from neural_shrimp.cifar_linear_diagnostic import finite_vector
from neural_shrimp.cifar_merit_lsmr import trial
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/merit-lsmr/attempt-001');PIN='b5a555e1247f20dc2c2f1da62a70c7c256eb43d694386308388ab40ed8e799e6';EXEC='6a6362d3bdb0929655dd5e2202de6473fd308a37'
CAPS=dict(action=0,j2vp=0,hvp=816,direct_gradient=40)
EXTRA=('src/neural_shrimp/cifar_symmetric_shooting.py','tests/test_cifar_symmetric_shooting.py','tools/explore_cifar_symmetric_shooting.py','tools/launch_cifar_symmetric_shooting.py','tests/test_cifar_shooting_driver.py','docs/plans/cifar-symmetric-shooting.md')
def install():C.install();B.SOURCES=tuple(B.SOURCES)+EXTRA
class Budget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=300.)
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def measured_gate(events):
 maxima={k:max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']==k)for k in ['hvp','direct_gradient']}
 forecast=2*sum(CAPS[k]*v for k,v in maxima.items())+20
 B.require(all(math.isfinite(x)and x>0 for x in maxima.values())and forecast<300,'measured300s gate')
 return dict(forecast_seconds=forecast,maximum_operation_seconds=maxima,safety_factor=2,overhead_seconds=20)

def fd_error(actual,finite_difference):
 actual,finite_difference=[finite_vector(v)for v in(actual,finite_difference)]
 B.require(actual.shape==finite_difference.shape,'FDshape')
 return float(np.linalg.norm(actual-finite_difference)/max(np.linalg.norm(actual),np.finfo(float).tiny))

def run(out,review):
 inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=255*1024**2);started=time.monotonic();budget=Budget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,accepted_orbit=False,spectrum_requested=False,comparison_qualified=False,method='exact symmetric two-phase MINRES, true eliminated forcing, native two-step Armijo acceptance',trials=[],accepted_index=None)
 try:
  store.json('runtime-start.json',result);store.write('minres-source.py',inspect.getsource(minres).encode());A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='merit_pilot_complete'and old['accepted_steps']==6 and old['last_accepted_state']=='outer-5-trial-0-theta.npy','fixed consumed parent')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'unchanged native source '+n)
  result['profile']=measured_gate([json.loads(x)for x in(PARENT/'events.jsonl').read_text().splitlines()]);store.json('profile.json',result['profile'])
  def read(n):
   a=np.load(PARENT/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite consumed state');return a
  theta,one,two=[read('outer-5-trial-0-'+n)for n in('theta','map1','map2')];arrays,basis,_,_,identity=B.inputs();B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same development map')
  result['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,parent_state_sha256=A.L.entry(PARENT/old['last_accepted_state'])['sha256']);store.json('input-identity.json',result['input_identity']);store.array('initial-theta',theta)
  problem=B.CountedProblem(B.cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  def map_one(x):return problem.update(B.cifar.as_tensor(x),eta).numpy()
  first=map_one(theta);store.array('initial-map1',first);B.require(B.byte_equal(first,one),'native phase1 replay');second=map_one(first);store.array('initial-map2',second);B.require(B.byte_equal(second,two),'native phase2 replay')
  def jx(v):return v-eta*problem.hvp(B.cifar.as_tensor(theta),B.cifar.as_tensor(v)).numpy()
  def jy(v):return v-eta*problem.hvp(B.cifar.as_tensor(first),B.cifar.as_tensor(v)).numpy()
  n=len(theta);pairs=[]
  for k in range(2):
   u,v=C.C.vectors(2*n,k+11);au=shooting_action(jx,jy,u);av=shooting_action(jx,jy,v);row=C.C.pairing(u,v,au,av);pairs.append(row)
   for name,value in [('u',u),('v',v),('Au',au),('Av',av)]:store.array(f'pair-{k}-'+name,value)
   B.require(row['passed'],'native symmetric pair')
  v,_=C.C.vectors(2*n,19);av=shooting_action(jx,jy,v);store.array('fd-vector',v);store.array('fd-action',av);errors=[]
  scale=max(1.,float(np.linalg.norm(np.r_[theta,first])))
  for k,h in enumerate([scale*1e-3,scale*1e-4,scale*1e-5]):
   vals=[]
   for sign in [-1,1]:
    x,y=theta+sign*h*v[:n],first+sign*h*v[n:];f,g=map_one(x),map_one(y);value=np.r_[f-y,g-x];vals.append(value);store.array(f'fd-{k}-{sign}-residual',value)
   diff=(vals[1]-vals[0])/(2*h);error=fd_error(av,diff);errors.append(dict(h=h,relative_error=error));store.array(f'fd-{k}-difference',diff)
  B.require(errors[-1]['relative_error']<=5e-6,'independent native central difference')
  result['qualification']=dict(pairings=pairs,finite_differences=errors,fd_threshold=5e-6,passed=True);store.json('qualification.json',result['qualification'])
  r=second-theta;store.array('initial-residual',r);p,ap,z,az,linear=direction(jx,jy,r,iterations=400,deadline=budget.deadline)
  for name,value in [('direction',p),('actual-action',ap),('shooting-direction',z),('shooting-action',az)]:store.array(name,value)
  result['linear']=linear;store.json('linear.json',linear);result['stop']='linear_forcing_or_descent_failed';result['last_accepted_state']='initial-theta.npy'
  if linear['forcing_pass']and linear['usable_descent']:
   result['stop']='backtracking_exhausted'
   for j in range(13):
    budget.check();alpha=2.**(-j);x=theta+alpha*p;y=map_one(x);w=map_one(y)
    for name,value in [('theta',x),('map1',y),('map2',w)]:store.array(f'trial-{j}-'+name,value)
    metrics=trial(x,y,w,r,ap,alpha);metrics['index']=j;result['trials'].append(metrics)
    if metrics['accepted_local_step']:theta,first,second=x,y,w;result.update(accepted_index=j,last_accepted_state=f'trial-{j}-theta.npy',stop='one_step_accepted');break
  budget.check();result.update(status='shooting_pilot_complete',comparison_qualified=True,final_relative_residual=float(np.linalg.norm(second-theta))/max(1.,float(np.linalg.norm(theta))),final_relative_separation=float(np.linalg.norm(first-theta))/max(1.,float(np.linalg.norm(theta))))
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
