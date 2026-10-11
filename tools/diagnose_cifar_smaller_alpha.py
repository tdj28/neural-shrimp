"""Nine fixed smaller damping trials on one retained CIFAR400 direction."""
import argparse,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
import diagnose_cifar_large_space as L
B=L.B
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
RAW=Path('/data2/neural-shrimp-data/explore/cifar/large-space/attempt-002')
PIN='814d5ebb4b4422c971eeb48425435080ce21f58aa9d2fa83c44f200cdcbb0f3b'
EXEC='bcba5eea358640fb3f8b6672f7fdbe88aef298c6'
ALPHAS=tuple(2.**(-k)for k in range(4,13))
EXTRA=('tools/diagnose_cifar_smaller_alpha.py','tools/launch_cifar_smaller_alpha.py','tests/test_cifar_smaller_alpha.py','docs/plans/cifar-smaller-alpha.md')

def install():
 L.install();B.SOURCES=tuple(B.SOURCES)+EXTRA

class SmallBudget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=45.)
 def call(self,kind,function,*args,**kwargs):
  self.check();B.require(kind=='direct_gradient'and self.attempted[kind]<20,'only20directgradients; no HVP/JVP/action')
  self.attempted[kind]+=1;started=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*args,**kwargs);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-started));return v

def reductions(res,ap):
 B.require(res.shape==ap.shape and res.ndim==1 and np.isfinite(res).all()and np.isfinite(ap).all(),'finite native R/Ap')
 rr=float(res@res);ratio=float(np.linalg.norm(ap+res)/np.sqrt(rr));slope=float(res@ap)
 B.require(all(np.isfinite(v)for v in(rr,ratio,slope))and rr>0 and ratio<=.5 and slope<0,'retained forcing and descent')
 return dict(initial_merit=.5*rr,true_forcing_ratio=ratio,slope=slope)

def metrics(theta,first,second,alpha,red):
 B.require(alpha in ALPHAS,'fixed smaller alpha only');res=second-theta
 B.require(theta.shape==first.shape==second.shape and theta.ndim==1 and all(np.isfinite(v).all()for v in(theta,first,second)),'finite trial')
 norm=float(np.linalg.norm(res));scale=max(1.,float(np.linalg.norm(theta)));merit=.5*float(res@res);sep=float(np.linalg.norm(first-theta))/scale;upper=red['initial_merit']+1e-4*alpha*red['slope'];relative=norm/scale
 B.require(all(np.isfinite(x)for x in(norm,scale,merit,sep,upper,relative)),'finite metrics')
 return dict(alpha=alpha,merit=merit,merit_ratio=merit/red['initial_merit'],armijo_upper=upper,armijo_pass=merit<=upper,relative_residual=relative,relative_separation=sep,closure_screen=relative<1e-8,primitive_screen=sep>=1e-7,inexact_step_candidate=bool(red['true_forcing_ratio']<=.5 and red['slope']<0 and merit<=upper),accepted_orbit=False,applied_to_continuation=False)

def measured_gate(events):
 ts=[e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']=='direct_gradient']
 B.require(len(ts)==11 and all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in ts),'11finite recorded gradients')
 estimate=2*20*max(ts)+5;B.require(estimate<45,'measured45s admission');return dict(max_prior_gradient_seconds=max(ts),projection_seconds=estimate,gradient_cap=20,scope='conservative timing admission, not convergence prediction')

def run(out,review_commit):
 inventory=B.sources();B.require(len(review_commit)==40,'review commit');subprocess.run(['git','cat-file','-e',review_commit+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=63*1024**2);started=time.monotonic();budget=SmallBudget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review_commit,runtime=runtime,settings=dict(alphas=ALPHAS,forcing_limit=.5,armijo=1e-4,closure_tolerance=1e-8,primitive_separation=1e-7,internal_seconds=45,gradient_cap=20,store_bytes=63*1024**2),trials=[],accepted_orbit=False,applied_to_continuation=False,spectrum_requested=False,comparison_qualified=False,new_linear_solves=0)
 try:
  store.json('runtime-start.json',result);L.verify_comparison(RAW,PIN);old=json.loads((RAW/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='diagnostic_complete'and old['comparison_qualified'],'completed400source')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'same source '+n)
  events=[json.loads(x)for x in(RAW/'events.jsonl').read_text().splitlines()];result['profile']=measured_gate(events);store.json('profile.json',result['profile'])
  arrays,basis,saved,_,identity=B.inputs();theta0=np.load(RAW/'initial-theta.npy',allow_pickle=False);res=np.load(RAW/'initial-residual.npy',allow_pickle=False);step=np.load(RAW/'direction.npy',allow_pickle=False);ap=np.load(RAW/'applied-direction.npy',allow_pickle=False);e=np.load(RAW/'true-linear-residual.npy',allow_pickle=False)
  B.require(B.byte_equal(theta0,arrays['theta'])and B.byte_equal(res,saved['initial-residual'])and B.byte_equal(ap+res,e),'same400state/residual/error');B.require(step.shape==theta0.shape and step.dtype==theta0.dtype and np.isfinite(step).all(),'saved direction');red=reductions(res,ap)
  B.require(np.isclose(red['true_forcing_ratio'],old['linear']['true_forcing_ratio'],rtol=1e-13,atol=0)and np.isclose(red['slope'],old['linear']['merit_directional_derivative'],rtol=1e-13,atol=0),'same400reductions')
  identity.update(direction_root=str(RAW),direction_manifest=PIN,direction_execution=EXEC,direction_sha256=L.entry(RAW/'direction.npy')['sha256']);result.update(input_identity=identity,reused_linear=red);store.json('input-identity.json',identity)
  for n,v in [('initial-theta',theta0),('initial-residual',res),('direction',step),('applied-direction',ap)]:store.array(n,v)
  problem=B.CountedProblem(B.cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;theta=B.cifar.as_tensor(theta0);eta=basis['eta']
  one=problem.update(theta,eta);B.replay(store,'initial-map1',one.numpy(),saved);two=problem.update(one,eta);B.replay(store,'initial-map2',two.numpy(),saved);B.require(B.byte_equal((two-theta).numpy(),res),'exact two-step residual parity')
  for i,alpha in enumerate(ALPHAS):
   budget.check();trial=theta0+alpha*step;store.array(f'trial-{i}-theta',trial);B.require(np.isfinite(trial).all(),'finite trial input');t=B.cifar.as_tensor(trial);a=problem.update(t,eta);store.array(f'trial-{i}-map1',a.numpy());b=problem.update(a,eta);store.array(f'trial-{i}-map2',b.numpy());store.array(f'trial-{i}-residual',(b-t).numpy());row=metrics(trial,a.numpy(),b.numpy(),alpha,red);row['index']=i;result['trials'].append(row);store.event(dict(kind='trial_result',**row))
  budget.check();result.update(status='diagnostic_complete',comparison_qualified=True,candidate_trial_indices=[r['index']for r in result['trials']if r['inexact_step_candidate']])
 except BaseException as ex:result.update(status='failed_or_interrupted',error=type(ex).__name__+': '+str(ex));raise
 finally:result.update(elapsed_seconds=time.monotonic()-started,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);r=run(a.out,a.review_commit);print(json.dumps(r));return 0
if __name__=='__main__':raise SystemExit(main())
