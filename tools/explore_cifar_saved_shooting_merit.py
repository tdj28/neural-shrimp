"""Test one retained direction using the previously qualified residual-merit rule.

This is a disclosed change of step strategy; the failed Newton-forcing attempt
remains failed. No new solve, data, spectrum, orbit acceptance or holdout.
"""
import argparse,hashlib,json,math,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_cifar_symmetric_shooting_extended as C
B,D,A=C.B,C.D,C.A
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_merit_lsmr import dot,trial
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/symmetric-shooting/attempt-002');PIN='6cbaa8aa34330a870c5cb135787325e7601a75c6fd91d8c663900a744e3ec661'
EXEC='1e8d9abf07381e9bce200fe4ac77d7c29743abb1'
CAPS=dict(action=0,j2vp=0,hvp=2,direct_gradient=28)
EXTRA=('tools/explore_cifar_saved_shooting_merit.py','tools/launch_cifar_saved_shooting_merit.py','tests/test_cifar_saved_shooting_merit.py','docs/plans/cifar-saved-shooting-merit.md')
def install():C.install();B.SOURCES=tuple(B.SOURCES)+EXTRA
class Budget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=90.)
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def gate(theta,r,p,ap):
 scale=max(1.,float(np.linalg.norm(theta)));forcing=math.sqrt(dot(r+ap,r+ap)/dot(r,r));slope=dot(r,ap);pred=(dot(r,r)-dot(r+ap,r+ap))/2;size=float(np.linalg.norm(p))/scale
 return dict(true_forcing=forcing,old_newton_forcing_pass=bool(forcing<=.5),true_slope=slope,full_predicted_reduction=pred,relative_direction_norm=size,passed=bool(slope<0 and pred>0 and size<=.1))

def run(out,review):
 inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);store=Store(out,limit=255*1024**2);started=time.monotonic();budget=Budget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=B.environment(),accepted_orbit=False,spectrum_requested=False,comparison_qualified=False,method='one saved MINRES direction; actual residual-merit strategy after failed Newton forcing, no new Krylov solve',trials=[],accepted_index=None)
 try:
  store.json('runtime-start.json',result);A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='shooting_pilot_complete'and old['linear']['iterations']==1600 and old['accepted_index']is None and old['qualification']['passed'],'fixed retained direction')
  for name,h in old['source_sha256'].items():B.require(inventory[name]==h and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+name],cwd=ROOT)).hexdigest()==h,'unchanged parent source '+name)
  events=[json.loads(x)for x in(PARENT/'events.jsonl').read_text().splitlines()];maximum={k:max(x['seconds']for x in events if x['kind']=='operation_returned'and x['operation']==k)for k in ['hvp','direct_gradient']};forecast=2*sum(CAPS[k]*v for k,v in maximum.items())+20;B.require(math.isfinite(forecast)and forecast<90,'measured90s gate');result['profile']=dict(maximum_seconds=maximum,forecast_seconds=forecast)
  def read(n):
   a=np.load(PARENT/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite same dimension');return a
  theta,one,two,p,saved_ap=[read(n)for n in ['initial-theta','initial-map1','initial-map2','direction','actual-action']];arrays,basis,_,_,identity=B.inputs();B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same development map');result['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC);store.json('input-identity.json',result['input_identity'])
  problem=B.CountedProblem(B.cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  def map_one(x):return problem.update(B.cifar.as_tensor(x),eta).numpy()
  first=map_one(theta);second=map_one(first);B.require(B.byte_equal(first,one)and B.byte_equal(second,two),'exact native phase replay')
  jx=p-eta*problem.hvp(B.cifar.as_tensor(theta),B.cifar.as_tensor(p)).numpy();ap=jx-eta*problem.hvp(B.cifar.as_tensor(first),B.cifar.as_tensor(jx)).numpy()-p;B.require(B.byte_equal(ap,saved_ap),'exact physical retained action replay')
  r=second-theta;metrics=gate(theta,r,p,ap);B.require(metrics['passed'],'bounded predicted descent direction');result['direction_gate']=metrics;result['stop']='backtracking_exhausted';result['last_accepted_state']='initial-theta.npy'
  for name,v in [('initial-theta',theta),('initial-map1',first),('initial-map2',second),('residual',r),('direction',p),('actual-action',ap)]:store.array(name,v)
  for j in range(13):
   budget.check();alpha=2.**(-j);x=theta+alpha*p;y=map_one(x);z=map_one(y)
   for name,v in [('theta',x),('map1',y),('map2',z)]:store.array(f'trial-{j}-'+name,v)
   row=trial(x,y,z,r,ap,alpha);row['index']=j;result['trials'].append(row)
   if row['accepted_local_step']:theta,first,second=x,y,z;result.update(accepted_index=j,last_accepted_state=f'trial-{j}-theta.npy',stop='saved_direction_merit_step_accepted');break
  budget.check();result.update(status='saved_merit_complete',comparison_qualified=True,final_relative_residual=float(np.linalg.norm(second-theta))/max(1.,float(np.linalg.norm(theta))),final_relative_separation=float(np.linalg.norm(first-theta))/max(1.,float(np.linalg.norm(theta))))
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)))
