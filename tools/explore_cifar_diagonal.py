"""One measured right-diagonally-preconditioned400-space CIFAR correction."""
import argparse,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
import explore_cifar_recycled as C
B,D,A=C.B,C.D,C.A
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_linear_diagnostic import solve_direction
from neural_shrimp.cifar_damped import ALPHAS,trial_metrics
from neural_shrimp.cifar_diagonal import loss_hessian_diagonal,scaling
PARENT=C.PARENT;PIN=C.PIN;EXEC=C.EXEC
CAPS=dict(action=402,j2vp=402,hvp=804,direct_gradient=28,analytic_diagonal=4)
EXTRA=('src/neural_shrimp/cifar_diagonal.py','tools/explore_cifar_diagonal.py','tools/launch_cifar_diagonal.py','tests/test_cifar_diagonal.py','docs/plans/cifar-diagonal-implementation.md')
def install():C.install();B.SOURCES=tuple(B.SOURCES)+EXTRA
class Budget(BaseBudget):
 def __init__(self,event):
  super().__init__(event,seconds=75.);self.attempted['analytic_diagonal']=0;self.returned['analytic_diagonal']=0
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def run(out,review):
 inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=127*1024**2);started=time.monotonic();budget=Budget(store.event)
 r=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,accepted_orbit=False,applied_to_continuation=False,spectrum_requested=False,comparison_qualified=False,accepted_steps=0,last_accepted_state='initial-theta.npy',trials=[],method='one400 right positive diagonal preconditioner; true unscaled residual gates')
 try:
  store.json('runtime-start.json',r);A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='diagnostic_complete'and old['accepted_steps']==1 and old['last_accepted_state']=='newton-0-trial-5-theta.npy','fixed parent')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'parent source '+n)
  A.L.verify_comparison(A.RAW,A.PIN);old400=json.loads((A.RAW/'result.json').read_text());events=[json.loads(x)for x in(PARENT/'events.jsonl').read_text().splitlines()];gt=[e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']=='direct_gradient'];B.require(len(gt)==14 and all(math.isfinite(t)and t>=0 for t in gt),'profile timings')
  def diagonal(theta,x,y,model,lam,tag):
   begin=time.monotonic();value,aux=budget.call('analytic_diagonal',loss_hessian_diagonal,theta,x,y,model.d,model.width,model.k,lam);store.array(tag+'-diagonal',value)
   for n,a in aux.items():store.array(tag+'-'+n,a)
   B.require(np.isfinite(value).all()and all(np.isfinite(a).all()for a in aux.values()),'finite analytic return');return value,time.monotonic()-begin
  model=B.cifar.MLP(3072,32,10);synthetic_theta=np.sin(np.arange(model.n_parameters)*.01)*.1;synthetic_x=np.cos(np.arange(512*3072,dtype=float).reshape(512,3072)*.0001);synthetic_y=np.eye(10)[np.arange(512)%10]
  store.json('profile-inputs.json',dict(model=dict(d=3072,width=32,k=10),n=512,theta='0.1*sin(arange(n_parameters)*.01)',x='cos(arange(512*3072).reshape(512,3072)*.0001)',y='eye(10)[arange(512)%10]',second_theta='theta+.01',lam=.01,scope='synthetic cost only; no target arrays opened'))
  times=[]
  for i in range(2):_,t=diagonal(synthetic_theta+i*.01,synthetic_x,synthetic_y,model,.01,'profile-'+str(i));times.append(t)
  measured400=max(old400['elapsed_seconds'],*(s['linear']['seconds']for s in old['steps']));projection=2*(measured400+2*max(times)+gt[0]+27*max(gt[1:]))+10;B.require(math.isfinite(projection)and projection<75,'75s resource gate');r['profile']=dict(analytic_times=times,prior400_seconds=measured400,projected_seconds=projection,safety_factor=2,overhead=10);store.json('profile.json',r['profile']);del synthetic_theta,synthetic_x,synthetic_y
  def read(n):
   a=np.load(PARENT/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite native input');return a
  theta,one,two=[read('newton-0-trial-5-'+n)for n in('theta','map1','map2')];base=read('initial-theta');previous=read('newton-0-direction');rb=read('initial-residual');ap0=read('newton-0-applied-direction');B.require(B.byte_equal(theta,base+previous/32),'parent theta bytes');phi=D.dot(rb,rb)/2;slope=D.dot(rb,ap0);merit=D.dot(two-theta,two-theta)/2;forcing=math.sqrt(D.dot(ap0+rb,ap0+rb)/D.dot(rb,rb));sep=math.sqrt(D.dot(one-theta,one-theta))/max(1.,math.sqrt(D.dot(theta,theta)));B.require(all(math.isfinite(v)for v in(forcing,slope,phi,merit,sep))and forcing<=.5 and slope<0 and merit<phi and merit<=phi+1e-4*slope/32 and sep>=1e-7,'parent forcing/Armijo/primitive')
  arrays,basis,_,_,identity=B.inputs();B.require(basis['model']==dict(d=3072,width=32,k=10)and arrays['x'].shape==(512,3072),'profile target dimensions');B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same exact map');r['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,parent_state_sha256=A.L.entry(PARENT/'newton-0-trial-5-theta.npy')['sha256']);store.json('input-identity.json',r['input_identity']);store.array('initial-theta',theta)
  problem=B.CountedProblem(model,arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  def map_one(x):return problem.update(B.cifar.as_tensor(x),eta).numpy()
  first=map_one(theta);store.array('initial-map1',first);B.require(B.byte_equal(first,one),'map1 parity');second=map_one(first);store.array('initial-map2',second);B.require(B.byte_equal(second,two),'map2 parity');res=second-theta;store.array('initial-residual',res)
  d0,_=diagonal(theta,arrays['x'],arrays['y'],model,basis['lam'],'phase0');d1,_=diagonal(first,arrays['x'],arrays['y'],model,basis['lam'],'phase1');inverse,proxy,sc= scaling(d0,d1,eta,basis['lam']);store.array('right-inverse-scaling',inverse);store.array('approximate-residual-diagonal',proxy);store.json('scaling.json',sc);r['scaling']=sc;final=[False]
  def scaled_action(v):
   physical=inverse*v;value=budget.call('action',lambda:problem.jvp_two_step(B.cifar.as_tensor(theta),B.cifar.as_tensor(physical),eta,theta_next=B.cifar.as_tensor(first)).numpy()-physical)
   if final[0]:store.array('actual-direction-action',value);store.array('actual-linear-error',value+res)
   return value
  def save_direction(y,info):store.array('solver-direction',y);store.array('physical-direction',inverse*y);store.json('returned-direction.json',dict(info=info));final[0]=True
  y,ap,error,lin=solve_direction(scaled_action,res,restart=400,cycles=1,rtol=1e-10,forcing_limit=.5,max_matvec=402,explicit_zero_start=True,deadline=budget.deadline,event=store.event,save_direction=save_direction);p=inverse*y;store.json('linear.json',lin);r['linear']=lin;r['stop']='true_forcing_or_descent_failed'
  if lin['forcing_test_pass']and lin['descent_direction']:
   phi=float(res@res)/2;slope=lin['merit_directional_derivative'];r['stop']='all_thirteen_trials_failed'
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
