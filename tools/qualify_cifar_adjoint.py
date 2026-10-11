"""Bounded actual native two-step derivative/adjoint pairing qualification."""
import argparse,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import explore_cifar_recycled as C
B,D,A=C.B,C.D,C.A
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_residual_adjoint import residual_adjoint
PARENT=C.PARENT;PIN=C.PIN;EXEC=C.EXEC
CAPS=dict(action=5,adjoint=5,j2vp=5,hvp=20,direct_gradient=2)
EXTRA=('src/neural_shrimp/cifar_residual_adjoint.py','tests/test_cifar_residual_adjoint.py','tools/qualify_cifar_adjoint.py','tools/launch_cifar_adjoint.py','tests/test_cifar_adjoint_pairing.py','docs/plans/cifar-adjoint-pairing.md')
def install():C.install();B.SOURCES=tuple(B.SOURCES)+EXTRA
class Budget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=30.);self.attempted['adjoint']=0;self.returned['adjoint']=0
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def vectors(n,k):
 u=np.sin(np.arange(n,dtype=np.float64)*(.13+.017*k)+.3*k);v=np.cos(np.arange(n,dtype=np.float64)*(.19+.023*k)+.4*k)
 return u/np.linalg.norm(u),v/np.linalg.norm(v)
def pairing(u,v,au,atv):
 arrays=[np.asarray(x)for x in (u,v,au,atv)];B.require(all(x.shape==arrays[0].shape and x.ndim==1 and np.isfinite(x).all()for x in arrays),'finite pairing vectors')
 left=D.dot(v,au);right=D.dot(u,atv);scale=max(float(np.linalg.norm(v)*np.linalg.norm(au)),float(np.linalg.norm(u)*np.linalg.norm(atv)),np.finfo(float).tiny)
 gap=abs(left-right)/scale
 return dict(left=left,right=right,cauchy_scale=scale,scaled_error=gap,threshold=5e-11,passed=bool(gap<=5e-11 and scale>1e-10))
def cost(times,grad):
 B.require(len(times)==2 and all(len(t)==2 and all(math.isfinite(x)and x>0 for x in t)for t in times)and math.isfinite(grad)and grad>0,'measured full-size synthetic operations')
 forecast=2*(2*grad+3*sum(max(t[k]for t in times)for k in range(2)))+8
 B.require(forecast<30,'30s measured native pairing gate');return forecast

def run(out,review):
 inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=31*1024**2);started=time.monotonic();budget=Budget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,accepted_orbit=False,correction_attempted=False,qualification_passed=False,method='three fixed native forward/adjoint pairings after full-size synthetic profile')
 try:
  store.json('runtime-start.json',result);A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='diagnostic_complete'and old['accepted_steps']==1 and old['last_accepted_state']=='newton-0-trial-5-theta.npy','fixed consumed parent')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'same parent executable '+n)
  events=[json.loads(s)for s in(PARENT/'events.jsonl').read_text().splitlines()];grad=max(e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']=='direct_gradient')
  model=B.cifar.MLP(3072,32,10);theta=np.sin(np.arange(model.n_parameters)*.01)*.1;x=np.cos(np.arange(512*3072,dtype=float).reshape(512,3072)*.0001);y=np.eye(10)[np.arange(512)%10]
  synthetic=B.CountedProblem(model,x,y,.01);synthetic.budget=budget
  def actions(problem,theta,first,eta,u,v):
   t=time.monotonic();au=budget.call('action',lambda:problem.jvp_two_step(B.cifar.as_tensor(theta),B.cifar.as_tensor(u),eta,theta_next=B.cifar.as_tensor(first)).numpy()-u);a=time.monotonic()-t
   t=time.monotonic();atv=budget.call('adjoint',lambda:residual_adjoint(problem,B.cifar.as_tensor(theta),B.cifar.as_tensor(first),B.cifar.as_tensor(v),eta).numpy());b=time.monotonic()-t
   return au,atv,(a,b)
  times=[];profile=[]
  for k in range(2):
   u,v=vectors(model.n_parameters,k);au,atv,t=actions(synthetic,theta,theta+.01,.008,u,v);row=pairing(u,v,au,atv);B.require(row['passed'],'synthetic pairing');times.append(t);profile.append(row)
  forecast=cost(times,grad);store.json('profile.json',dict(times=times,prior_gradient_maximum=grad,forecast_seconds=forecast,pairings=profile,synthetic=dict(theta='0.1*sin(arange(p)*.01)',second='theta+.01',x='cos(arange(512*3072).reshape(512,3072)*.0001)',y='eye(10)[arange(512)%10]',eta=.008,lam=.01)))
  del synthetic,x,y,theta,au,atv,u,v
  def read(n):
   a=np.load(PARENT/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'native finite input');return a
  theta,one,two=[read('newton-0-trial-5-'+n)for n in('theta','map1','map2')];base=read('initial-theta');p=read('newton-0-direction');rb=read('initial-residual');ap=read('newton-0-applied-direction')
  B.require(B.byte_equal(theta,base+p/32),'accepted parent state bytes');phi=D.dot(rb,rb)/2;slope=D.dot(rb,ap);merit=D.dot(two-theta,two-theta)/2;forcing=math.sqrt(D.dot(ap+rb,ap+rb)/D.dot(rb,rb));sep=float(np.linalg.norm(one-theta))/max(1.,float(np.linalg.norm(theta)))
  B.require(forcing<=.5 and slope<0 and merit<phi and merit<=phi+1e-4*slope/32 and sep>=1e-7,'original parent actual forcing/Armijo/primitive')
  arrays,basis,_,_,identity=B.inputs();B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same development map')
  result['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,parent_state_sha256=A.L.entry(PARENT/'newton-0-trial-5-theta.npy')['sha256']);store.json('input-identity.json',result['input_identity']);store.array('initial-theta',theta)
  problem=B.CountedProblem(model,arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  first=problem.update(B.cifar.as_tensor(theta),eta).numpy();store.array('initial-map1',first);B.require(B.byte_equal(first,one),'native map1 replay');second=problem.update(B.cifar.as_tensor(first),eta).numpy();store.array('initial-map2',second);B.require(B.byte_equal(second,two),'native map2 replay')
  rows=[]
  for k in range(3):
   u,v=vectors(model.n_parameters,k+2);au,atv,t=actions(problem,theta,first,eta,u,v)
   for name,value in [('u',u),('v',v),('Au',au),('ATv',atv)]:store.array(f'pair-{k}-'+name,value)
   row=pairing(u,v,au,atv);row.update(index=k,seconds=t);store.json(f'pair-{k}.json',row);rows.append(row);B.require(row['passed'],'actual native forward/adjoint pairing')
  B.require(budget.attempted==budget.returned==CAPS,'complete fixed native/profile operation accounting')
  budget.check();result.update(status='qualification_complete',qualification_passed=True,pairings=rows,final_relative_residual=float(np.linalg.norm(second-theta))/max(1.,float(np.linalg.norm(theta))),profile_forecast_seconds=forecast)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
