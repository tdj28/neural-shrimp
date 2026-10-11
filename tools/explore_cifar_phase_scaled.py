"""One fixed phase-balanced native MINRES correction at the latest accepted state."""
import argparse,hashlib,inspect,json,math,os,resource,signal,subprocess,sys,time,tarfile,io
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import minres
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import explore_cifar_saved_shooting_merit as C
import explore_cifar_merit_lsmr as Methods
B,D,A=C.B,C.D,C.A
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
from neural_shrimp.cifar_symmetric_shooting import shooting_action
from neural_shrimp.cifar_phase_scaling import direction,physical,dual,scales
from neural_shrimp.cifar_linear_diagnostic import finite_vector
from neural_shrimp.cifar_merit_lsmr import trial
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/saved-shooting-merit/attempt-001');PIN='d036dabfc82130137c4cdaad5249c836de2f960b82497c9740f4610b04668004';EXEC='56efc51180599bcfc31b306a4f9283d450f2675b'
DIAGNOSTIC=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/diagnostic002');DIAG_PIN='77226408ede357e368a08a195b2a66ca42ad5976bf5e80d3de590f5e3aa5962a'
CAPS=dict(action=0,j2vp=0,hvp=818,direct_gradient=40)
EXTRA=('src/neural_shrimp/cifar_phase_scaling.py','tools/explore_cifar_phase_scaled.py','tools/launch_cifar_phase_scaled.py','tests/test_cifar_phase_scaling.py','tests/test_cifar_phase_scaled_driver.py','docs/plans/cifar-phase-scaled-corrector.md','docs/plans/cifar-phase-native-sources.json')
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

def verify_diagnostic(root,pin,tick):
 B.require(A.L.entry(root/'manifest.json')['sha256']==pin,'literal completed diagnostic')
 files=json.loads((root/'manifest.json').read_text())['files']
 B.require(set(files)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'complete diagnostic nested inventory')
 for n,e in files.items():
  tick();p=root/n;B.require(not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and not p.stat().st_mode&0o222 and A.L.entry(p)==e,'exact immutable diagnostic')
 row=json.loads((root/'launcher-result.json').read_text());B.require(row['exit_code']==0 and not row['owned_live_pids']and row['worker_pid_absent'],'completed diagnostic ownership')
 for pid in (row['launcher_pid'],row['owned_group_pid']):B.require(not Path('/proc',str(pid)).exists(),'diagnostic exact owned pid absent')
 return row

def run(out,review):
 inventory=B.sources();subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True);runtime=B.environment();store=Store(out,limit=255*1024**2);started=time.monotonic();budget=Budget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,accepted_orbit=False,spectrum_requested=False,comparison_qualified=False,method='fixed positive mean/alternating congruence, 400 MINRES iterations, fresh physical forcing and native Armijo',trials=[],accepted_index=None)
 try:
  store.json('runtime-start.json',result);store.write('minres-source.py',inspect.getsource(minres).encode());A.L.verify_comparison(PARENT,PIN);old=json.loads((PARENT/'result.json').read_text());B.require(old['source_commit']==EXEC and old['status']=='saved_merit_complete'and old['accepted_index']==4 and old['last_accepted_state']=='trial-4-theta.npy','fixed consumed parent')
  for n,d in old['source_sha256'].items():B.require(inventory[n]==d and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d,'unchanged native source '+n)
  # Source-authenticated completed physical-HVP timings, before target arrays.
  measured=Path('/data2/neural-shrimp-data/explore/cifar/symmetric-shooting/attempt-002');A.L.verify_comparison(measured,'6cbaa8aa34330a870c5cb135787325e7601a75c6fd91d8c663900a744e3ec661');mr=json.loads((measured/'result.json').read_text());B.require(mr['source_commit']=='1e8d9abf07381e9bce200fe4ac77d7c29743abb1'and mr['qualification']['passed']and mr['linear']['iterations']==1600,'completed same-operator measured reference')
  result['profile']=measured_gate([json.loads(x)for x in(measured/'events.jsonl').read_text().splitlines()]);store.json('profile.json',result['profile'])
  verify_diagnostic(DIAGNOSTIC,DIAG_PIN,budget.check);dr=json.loads((DIAGNOSTIC/'run/outcome.json').read_text());B.require(dr['passed']and dr['native_calls']==0 and dr['current_state']['entry']==A.L.entry(PARENT/'trial-4-theta.npy'),'completed saved diagnosis/current state')
  # Complete source archive is retained before native acquisition.
  archive=io.BytesIO()
  with tarfile.open(fileobj=archive,mode='w:gz')as tar:
   for n in inventory:
    data=(ROOT/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
  store.write('source.tar.gz',archive.getvalue())
  def read(n):
   a=np.load(PARENT/(n+'.npy'),allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'finite consumed state');return a
  theta,one,two=[read('trial-4-'+n)for n in('theta','map1','map2')];arrays,basis,_,_,identity=B.inputs();B.require(identity['data_identity']==old['input_identity']['data_identity']and identity['eta']==old['input_identity']['eta']and identity['lam']==old['input_identity']['lam'],'same development map')
  B.require(identity['model']==old['input_identity']['model']and identity['eta']==.00827753 and identity['lam']==.01,'exact same98666D map')
  result['input_identity']=dict(identity,parent_root=str(PARENT),parent_manifest=PIN,parent_execution=EXEC,parent_state_sha256=A.L.entry(PARENT/old['last_accepted_state'])['sha256']);store.json('input-identity.json',result['input_identity']);store.array('initial-theta',theta)
  problem=B.CountedProblem(B.cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam']);problem.budget=budget;eta=basis['eta']
  def map_one(x):return problem.update(B.cifar.as_tensor(x),eta).numpy()
  first=map_one(theta);store.array('initial-map1',first);B.require(B.byte_equal(first,one),'native phase1 replay');second=map_one(first);store.array('initial-map2',second);B.require(B.byte_equal(second,two),'native phase2 replay')
  def jx(v):return v-eta*problem.hvp(B.cifar.as_tensor(theta),B.cifar.as_tensor(v)).numpy()
  def jy(v):return v-eta*problem.hvp(B.cifar.as_tensor(first),B.cifar.as_tensor(v)).numpy()
  n=len(theta);pairs=[]
  for k in range(2):
   u,v=Methods.C.vectors(2*n,k+11);weights,_=scales(eta,basis['lam']);pu,pv=physical(u,weights),physical(v,weights);au=dual(shooting_action(jx,jy,pu),weights);av=dual(shooting_action(jx,jy,pv),weights);row=Methods.C.pairing(u,v,au,av);pairs.append(row)
   for name,value in [('u',u),('v',v),('Au',au),('Av',av),('physical-u',pu),('physical-v',pv)]:store.array(f'pair-{k}-'+name,value)
   B.require(row['passed'],'actual native scaled symmetric pair')
  v,_=Methods.C.vectors(2*n,19);av=shooting_action(jx,jy,v);store.array('fd-vector',v);store.array('fd-action',av);errors=[]
  scale=max(1.,float(np.linalg.norm(np.r_[theta,first])))
  for k,h in enumerate([scale*1e-3,scale*1e-4,scale*1e-5]):
   vals=[]
   for sign in [-1,1]:
    x,y=theta+sign*h*v[:n],first+sign*h*v[n:];f,g=map_one(x),map_one(y);value=np.r_[f-y,g-x];vals.append(value);store.array(f'fd-{k}-{sign}-residual',value)
   diff=(vals[1]-vals[0])/(2*h);error=fd_error(av,diff);errors.append(dict(h=h,relative_error=error));store.array(f'fd-{k}-difference',diff)
  B.require(errors[-1]['relative_error']<=5e-6,'independent native central difference')
  result['qualification']=dict(pairings=pairs,finite_differences=errors,fd_threshold=5e-6,passed=True);store.json('qualification.json',result['qualification'])
  result['current_profile']=measured_gate([json.loads(x)for x in(out/'events.jsonl').read_text().splitlines()]);store.json('current-profile.json',result['current_profile'])
  r=second-theta;store.array('initial-residual',r);p,ap,z,az,q,linear=direction(jx,jy,r,eta,basis['lam'],iterations=400,deadline=budget.deadline)
  for name,value in [('direction',p),('actual-action',ap),('shooting-direction',z),('shooting-action',az),('scaled-direction',q)]:store.array(name,value)
  result['linear']=linear;store.json('linear.json',linear);result['stop']='linear_forcing_or_descent_failed';result['last_accepted_state']='initial-theta.npy'
  if linear['forcing_pass']and linear['usable_descent']:
   result['stop']='backtracking_exhausted'
   for j in range(13):
    budget.check();alpha=2.**(-j);x=theta+alpha*p;y=map_one(x);w=map_one(y)
    for name,value in [('theta',x),('map1',y),('map2',w)]:store.array(f'trial-{j}-'+name,value)
    metrics=trial(x,y,w,r,ap,alpha);metrics['index']=j;result['trials'].append(metrics)
    if metrics['accepted_local_step']:theta,first,second=x,y,w;result.update(accepted_index=j,last_accepted_state=f'trial-{j}-theta.npy',stop='one_step_accepted');break
  budget.check();result.update(status='phase_scaled_pilot_complete',comparison_qualified=True,final_relative_residual=float(np.linalg.norm(second-theta))/max(1.,float(np.linalg.norm(theta))),final_relative_separation=float(np.linalg.norm(first-theta))/max(1.,float(np.linalg.norm(theta))))
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:result.update(elapsed_seconds=time.monotonic()-started,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--review-commit',required=True);a=p.parse_args();install();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(a.out,a.review_commit)));return 0
if __name__=='__main__':raise SystemExit(main())
