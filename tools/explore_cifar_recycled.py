"""One fixed recycled-direction test at saved accepted CIFAR state."""
import argparse,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
import explore_cifar_damped as D
B,A=D.B,D.A
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_damped import ALPHAS,trial_metrics
from neural_shrimp.cifar_recycled import fit,actual_gate
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/damped-inexact/attempt-001')
PIN='a17661b9f202574d675bf0d1c4dcdd2ab8d5f80abda9502c349d159eddbe7c1a'
EXEC='87156e9ea3aa178405ecc4d23d1332e9c756f3cf'
CAPS=dict(action=3,j2vp=3,hvp=6,direct_gradient=28)
EXTRA=('src/neural_shrimp/cifar_recycled.py','tools/explore_cifar_recycled.py','tools/launch_cifar_recycled.py','tests/test_cifar_recycled.py','docs/plans/cifar-recycled-implementation.md')
def install():D.install();B.SOURCES=tuple(B.SOURCES)+EXTRA
class Budget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=45.)
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def measured_gate(events):
 def times(k):
  ts=[e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']==k];B.require(ts and all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in ts),'finite timings');return ts
 actions=times('action');grad=times('direct_gradient');B.require(len(actions)==804 and len(grad)==14,'bound prior counts');projected=2*(3*max(actions)+grad[0]+27*max(grad[1:]))+10;B.require(projected<45,'45s gate');return dict(projected_seconds=projected,maximum_action_seconds=max(actions),first_gradient_seconds=grad[0],maximum_later_gradient_seconds=max(grad[1:]),scope='resource only')
def run(out,review):
 inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=63*1024**2);started=time.monotonic();budget=Budget(store.event)
 r=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,accepted_orbit=False,applied_to_continuation=False,spectrum_requested=False,comparison_qualified=False,accepted_steps=0,last_accepted_state='initial-theta.npy',trials=[],method='fixed three-direction residual minimization; no Krylov solve')
 try:
  store.json('runtime-start.json',r);A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='diagnostic_complete'and old['accepted_steps']==1 and old['stop']=='true_forcing_or_descent_failed'and old['last_accepted_state']=='newton-0-trial-5-theta.npy'and old['steps'][1]['base_state']==old['last_accepted_state'],'fixed parent')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'parent source '+n)
  A.L.verify_comparison(A.RAW,A.PIN);r['profile']=measured_gate([json.loads(x)for x in(PARENT/'events.jsonl').read_text().splitlines()]);store.json('profile.json',r['profile'])
  def read(n,root=PARENT):
   a=np.load(root/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite native input');return a
  theta,one,two=[read('newton-0-trial-5-'+n)for n in('theta','map1','map2')];base=read('initial-theta');p0=read('newton-0-direction');rb=read('initial-residual');ap0=read('newton-0-applied-direction');B.require(B.byte_equal(theta,base+p0/32),'parent theta bytes');phi=D.dot(rb,rb)/2;slope=D.dot(rb,ap0);merit=D.dot(two-theta,two-theta)/2;forcing=math.sqrt(D.dot(ap0+rb,ap0+rb)/D.dot(rb,rb));sep=math.sqrt(D.dot(one-theta,one-theta))/max(1.,math.sqrt(D.dot(theta,theta)));B.require(all(math.isfinite(v)for v in(forcing,slope,phi,merit,sep))and forcing<=.5 and slope<0 and merit<phi and merit<=phi+1e-4*slope/32 and sep>=1e-7,'parent true forcing/Armijo/primitive')
  directions=np.column_stack([read('direction',A.RAW),p0,read('newton-1-direction')]);store.array('historical-directions',directions);arrays,basis,_,_,identity=B.inputs();B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same exact fixed map');r['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,old400_root=str(A.RAW),old400_manifest=A.PIN,parent_state_sha256=A.L.entry(PARENT/'newton-0-trial-5-theta.npy')['sha256']);store.json('input-identity.json',r['input_identity']);store.array('initial-theta',theta)
  problem=B.CountedProblem(B.cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  def map_one(x):return problem.update(B.cifar.as_tensor(x),eta).numpy()
  first=map_one(theta);store.array('initial-map1',first);B.require(B.byte_equal(first,one),'map1 parity');second=map_one(first);store.array('initial-map2',second);B.require(B.byte_equal(second,two),'map2 parity');res=second-theta;store.array('initial-residual',res);B.require(B.byte_equal(read('newton-1-linear-error'),read('newton-1-applied-direction')+res),'saved current error identity')
  def action(v):return budget.call('action',lambda:problem.jvp_two_step(B.cifar.as_tensor(theta),B.cifar.as_tensor(v),eta,theta_next=B.cifar.as_tensor(first)).numpy()-v)
  actions=[]
  for i in range(2):
   value=action(directions[:,i]);store.array(f'fresh-historical-action-{i}',value);actions.append(value)
  actions.append(read('newton-1-applied-direction'));matrix=np.column_stack(actions);store.array('historical-actions-at-current-state',matrix)
  def emit(kind,values):
   for k,v in values.items():
    if isinstance(v,np.ndarray):store.array('fit-'+kind+'-'+k,v)
   store.json('fit-'+kind+'.json',{k:v for k,v in values.items()if not isinstance(v,np.ndarray)})
  proposal,rank=fit(directions,matrix,res,emit);r['rank']=rank;r['stop']='rank_or_condition_unqualified'
  if proposal is not None:
   p=proposal['direction'];ap=action(p);store.array('combined-direction',p);store.array('actual-combined-action',ap);store.array('actual-combined-error',ap+res);r['linear']=actual_gate(ap,proposal['predicted_action'],res);store.json('actual-linear.json',r['linear']);r['stop']='combined_true_linear_gate_failed'
   if r['linear']['qualified']:
    phi=float(res@res)/2;slope=r['linear']['merit_directional_derivative'];r['stop']='all_thirteen_trials_failed'
    for i,alpha in enumerate(ALPHAS):
     budget.check();trial=theta+alpha*p;store.array(f'trial-{i}-theta',trial);a=map_one(trial);store.array(f'trial-{i}-map1',a);b=map_one(a);store.array(f'trial-{i}-map2',b);store.array(f'trial-{i}-residual',b-trial);metric=trial_metrics(trial,a,b,alpha,phi,slope);metric['index']=i;r['trials'].append(metric);store.event(dict(metric,kind='trial_result'))
     if metric['accepted_local_step']:
      store.json('accepted.json',dict(index=i,alpha=alpha,state=f'trial-{i}-theta.npy',metrics=metric));r.update(accepted_steps=1,last_accepted_state=f'trial-{i}-theta.npy',stop='one_local_step_accepted');theta,first,second=trial,a,b;break
  budget.check();r.update(status='diagnostic_complete',comparison_qualified=True,final_relative_residual=float(np.linalg.norm(second-theta))/max(1.,float(np.linalg.norm(theta))),final_relative_separation=float(np.linalg.norm(first-theta))/max(1.,float(np.linalg.norm(theta))))
 except BaseException as e:r.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:r.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(r)
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
