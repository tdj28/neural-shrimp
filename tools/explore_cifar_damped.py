"""One two-direction damped inexact corrector; no orbit/spectrum acceptance."""
import argparse,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
import diagnose_cifar_smaller_alpha as A
B=A.B
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_damped import correct
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/smaller-alpha/attempt-001')
PIN='1e83906ec3a735f5ce8ee30a0791176dd4c1d23d49c8c5b73a12d87bc941105a'
EXEC='9b3181712912fbd0011afe75d7f9fb5d6930d803'
CAPS=dict(action=804,j2vp=804,hvp=1608,direct_gradient=54)
EXTRA=('src/neural_shrimp/cifar_damped.py','tools/explore_cifar_damped.py','tools/launch_cifar_damped.py','tests/test_cifar_damped.py','docs/plans/cifar-damped-implementation.md')
def install():A.install();B.SOURCES=tuple(B.SOURCES)+EXTRA
class Budget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=150.)
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'operation cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
def parent_gate(old,read):
 theta,r,p,ap=[read(n)for n in ('initial-theta','initial-residual','direction','applied-direction')];chosen,one,two=[read('trial-1-'+n)for n in ('theta','map1','map2')]
 B.require(B.byte_equal(chosen,theta+p/32),'exact declared parent');phi=dot(r,r)/2;slope=dot(r,ap);forcing=math.sqrt(dot(ap+r,ap+r)/dot(r,r));merit=dot(two-chosen,two-chosen)/2;sep=math.sqrt(dot(one-chosen,one-chosen))/max(1.,math.sqrt(dot(chosen,chosen)))
 B.require(all(math.isfinite(v)for v in(phi,slope,forcing,merit,sep))and forcing<=.5 and slope<0 and merit<phi and merit<=phi+1e-4*slope/32 and sep>=1e-7,'independent parent forcing/Armijo/primitive gate')
 B.require(old['candidate_trial_indices'][0]==1 and old['trials'][1]['alpha']==1/32,'largest fixed passing parent')
 return chosen,one,two,dict(alpha=1/32,forcing=forcing,merit_ratio=merit/phi,slope=slope,separation=sep)

def measured_gate(old400,events):
 ts=[e['seconds']for e in events if e['kind']=='operation_returned'and e['operation']=='direct_gradient'];B.require(len(ts)==20 and all(type(t)in(int,float)and math.isfinite(t)and t>=0 for t in ts),'20finite parent timings')
 elapsed=old400['elapsed_seconds'];B.require(type(elapsed)in(int,float)and math.isfinite(elapsed)and elapsed>0,'prior400time');projected=2*(2*elapsed+ts[0]+53*max(ts[1:]))+10;B.require(projected<150,'150s admission');return dict(projected_seconds=projected,prior400_complete_seconds=elapsed,first_gradient_seconds=ts[0],maximum_later_gradient_seconds=max(ts[1:]),scope='measured cost screen only; no convergence guarantee')

def run(out,review):
 inventory=B.sources();B.require(len(review)==40,'review commit');subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=127*1024**2);started=time.monotonic();budget=Budget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,accepted_orbit=False,applied_to_continuation=False,spectrum_requested=False,comparison_qualified=False,method='two-direction damped inexact Newton; no old-state prefix control',steps=[],accepted_steps=0)
 try:
  store.json('runtime-start.json',result);A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='diagnostic_complete'and old['comparison_qualified'],'complete parent')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'parent source '+n)
  A.L.verify_comparison(A.RAW,A.PIN);old400=json.loads((A.RAW/'result.json').read_text());events=[json.loads(x)for x in(PARENT/'events.jsonl').read_text().splitlines()];result['profile']=measured_gate(old400,events);store.json('profile.json',result['profile'])
  def read(n):
   a=np.load(PARENT/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite parent arrays');return a
  theta,one,two,admission=parent_gate(old,read);arrays,basis,_,_,identity=B.inputs();B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same fixed map')
  result.update(parent_admission=admission,input_identity=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,parent_state_sha256=A.L.entry(PARENT/'trial-1-theta.npy')['sha256']),operation_caps=CAPS);store.json('input-identity.json',result['input_identity']);store.json('parent-admission.json',admission);store.array('initial-theta',theta)
  problem=B.CountedProblem(B.cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  def map_one(x):return problem.update(B.cifar.as_tensor(x),eta).numpy()
  first=map_one(theta);store.array('initial-map1',first);B.require(B.byte_equal(first,one),'parent map1 byte replay');second=map_one(first);store.array('initial-map2',second);B.require(B.byte_equal(second,two),'parent map2 byte replay');store.array('initial-residual',second-theta)
  def action(theta,first,v):return budget.call('action',lambda:problem.jvp_two_step(B.cifar.as_tensor(theta),B.cifar.as_tensor(v),eta,theta_next=B.cifar.as_tensor(first)).numpy()-v)
  correct(theta,first,second,map_one,action,budget,store,result);budget.check();result.update(status='diagnostic_complete',comparison_qualified=True)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:result.update(elapsed_seconds=time.monotonic()-started,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
