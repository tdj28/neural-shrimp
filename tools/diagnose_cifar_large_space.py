"""One disclosed restart400 decision, preserving the earlier200-space result."""
import hashlib,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
import diagnose_cifar_restart as B
from neural_shrimp.cifar_restart_diagnostic import Budget as OriginalBudget,Prefix as OriginalPrefix
BASE=Path('/data2/neural-shrimp-data/explore/cifar/restart-sensitivity/attempt-001')
PIN='5f9f77bb2926581c828e5528e02e475858ca626fc9df49be7a6f234eae4b1e96'
EXECUTION='6fea29cda810ebb72c234dbd0fc1533bbca4e5cb'
CAPS=dict(action=402,j2vp=403,hvp=807,direct_gradient=11)
SETTINGS=dict(B.SETTINGS,restart=400,max_matvec=402,internal_seconds=90.,operation_caps=CAPS,prefix_length=200)
EXTRA=('tools/diagnose_cifar_large_space.py','tools/launch_cifar_large_space.py','tests/test_cifar_large_space.py','docs/plans/cifar-large-space-decision.md')

def measured_gate(result,events):
 B.require(result['source_commit']==EXECUTION and result['status']=='diagnostic_complete' and result['comparison_qualified'] and not result['accepted_orbit'],'completed prior200 diagnostic')
 B.require(result['settings']['restart']==200 and result['settings']['cycles']==1 and result['linear']['inner_iterations']==200 and result['linear']['matvec_calls']==202,'fixed prior solve')
 rows=[r for r in events if r['kind']=='operation_returned' and r['operation']=='action']
 B.require(len(rows)==202 and all(type(r['seconds']) in(int,float) and math.isfinite(r['seconds']) and r['seconds']>=0 for r in rows),'all prior action timings')
 matvec=sum(r['seconds'] for r in rows);total=result['elapsed_seconds'];solve=result['linear']['seconds']
 B.require(all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in(total,solve)) and matvec<=solve<=total,'finite prior cost partition')
 estimate=matvec*(402/202)+(solve-matvec)*(80200/20100)+(total-solve)
 projected=2*estimate+5
 B.require(projected<90,'measured larger-space admission')
 return {'estimated_seconds':estimate,'safety_projected_seconds':projected,'basis_bytes':401*98666*8,'scope':'cost estimate only, no convergence prediction;90s actual worker cap'}

class Budget400(OriginalBudget):
 def __init__(self,event):super().__init__(event,seconds=90.)
 def call(self,kind,function,*args,**kwargs):
  self.check();B.require(self.attempted[kind]<CAPS[kind],kind+' operation cap');self.attempted[kind]+=1;started=time.monotonic()
  self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]))
  value=function(*args,**kwargs);self.returned[kind]+=1
  self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-started));return value

class Prefix400(OriginalPrefix):
 def __call__(self,row):
  self.event(row)
  if row['kind']=='gmres_inner':
   i=len(self.values);self.values.append(row['estimated_relative_residual'])
   B.require(row['iteration']==i+1 and len(self.values)<=400,'callback index/cap')
   if i<200:B.require(np.isclose(self.values[-1],self.baseline[i],rtol=1e-10,atol=1e-12),'first200 prefix mismatch')
 def complete(self):B.require(len(self.values)>=200,'fewer than200 matched callbacks; comparison unqualified')

def install():
 original_inputs=B.inputs
 B.SOURCES=tuple(B.SOURCES)+EXTRA;B.SETTINGS=SETTINGS;B.Budget=Budget400;B.Prefix=Prefix400
 def inputs():
  arrays,basis,saved,old,identity=original_inputs();B.verify_frozen(BASE,PIN)
  result=json.loads((BASE/'result.json').read_text());linear=json.loads((BASE/'linear.json').read_text());events=[json.loads(s) for s in(BASE/'events.jsonl').read_text().splitlines()]
  current_sources=B.sources()
  for name,digest in result['source_sha256'].items():
   B.require(hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT)).hexdigest()==digest and current_sources[name]==digest,'unchanged prior executable '+name)
  B.require(result['input_identity']['state_sha256']==identity['state_sha256'] and result['input_identity']['data_identity']==identity['data_identity'],'same development state/data')
  B.require(B.byte_equal(np.load(BASE/'initial-theta.npy',allow_pickle=False),arrays['theta']),'same exact prior state')
  B.require(result['linear']==linear,'prior linear record identity')
  identity.update(comparison200_root=str(BASE),comparison200_manifest=PIN,comparison200_execution=EXECUTION,larger_space_cost=measured_gate(result,events))
  return arrays,basis,saved,linear,identity
 B.inputs=inputs

if __name__=='__main__':
 install();B.main()
