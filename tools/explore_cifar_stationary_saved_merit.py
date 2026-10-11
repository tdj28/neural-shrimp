"""One exact-source saved stationary direction with fresh actual merit trials."""
import argparse,hashlib,io,json,math,resource,signal,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import explore_cifar_stationary as S
from neural_shrimp.cifar_stationary_merit import SETTINGS,direction_gate,trial_gate
from neural_shrimp.cifar_restart_diagnostic import Budget as BaseBudget,Store,request_stop
B=S.B
BASE=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/saved-merit001')
PARENT=S.BASE;PIN='36fb7d130b05c4392b32b583567aa7090da70223b79b5463f82e77e1cf4879be';EXEC='ebc18b48e0c42c793e66b4048379a6740313e10e'
CAPS=dict(action=0,j2vp=0,hvp=1,direct_gradient=14)
EXTRA=('src/neural_shrimp/cifar_stationary_merit.py','tools/explore_cifar_stationary_saved_merit.py','tools/launch_cifar_stationary_saved_merit.py','tests/test_cifar_stationary_saved_merit.py','docs/plans/cifar-stationary-saved-merit.md','docs/reviews/cifar-stationary001.json')
def install():S.install();B.SOURCES=tuple(dict.fromkeys(tuple(B.SOURCES)+EXTRA))
class Budget(BaseBudget):
 def __init__(self,event):super().__init__(event,seconds=90.)
 def call(self,kind,function,*a,**kw):
  self.check();B.require(self.attempted[kind]<CAPS[kind],'fixed saved-merit cap '+kind);self.attempted[kind]+=1;t=time.monotonic();self.event(dict(kind='operation_started',operation=kind,index=self.attempted[kind]));v=function(*a,**kw);self.returned[kind]+=1;self.event(dict(kind='operation_returned',operation=kind,index=self.returned[kind],seconds=time.monotonic()-t));return v

def check_parent(old):
 steps=old.get('stationary',{}).get('steps',[])
 B.require(old['source_commit']==EXEC and old['status']=='stationary_seed_complete'and old['comparison_qualified']and old['qualification']['passed']and not old['accepted_orbit']and not old['rigorous_certificate']and not old['seed_qualified']and old['mode']is None and old['stationary']['status']=='true_forcing_failed'and not old['stationary']['passed'],'exact negative stationary seed')
 B.require(len(steps)==1 and steps[0]['linear_actions']==256 and steps[0]['iterations']==256 and steps[0]['trials']==[]and steps[0]['accepted_index']is None and old['operation_attempted']==old['operation_returned']==dict(action=0,j2vp=0,hvp=259,direct_gradient=7),'complete unchanged failed linear attempt')
def run(out,review):
 B.require(out==BASE/'run','exclusive fixed output');inventory=B.sources();runtime=B.environment();store=Store(out,limit=128*1024**2,reserve=4*1024**2);start=time.monotonic();budget=Budget(store.event)
 result=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=inventory,review_commit=review,runtime=runtime,settings=SETTINGS,accepted_orbit=False,rigorous_certificate=False,comparison_qualified=False,accepted_index=None,trials=[])
 try:
  B.require(result['source_commit']==review,'exact reviewed source');store.json('runtime-start.json',result);S.P.verify_diagnostic(PARENT,PIN,budget.check);old=json.loads((PARENT/'run/result.json').read_text());check_parent(old)
  for k in ('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):B.require(runtime[k]==old['runtime'][k],'same measured runtime '+k)
  for n,h in old['source_sha256'].items():B.require(inventory[n]==h==hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest(),'same original source '+n)
  with tarfile.open(PARENT/'run/source.tar.gz','r:gz')as tar:
   members=tar.getmembers();B.require({m.name for m in members}==set(old['source_sha256'])and all(m.isfile()and m.size<2*1024**2 for m in members),'complete source archive')
   for m in members:B.require(hashlib.sha256(tar.extractfile(m).read()).hexdigest()==inventory[m.name],'archived original source')
  ev=[json.loads(v)for v in(PARENT/'run/events.jsonl').read_text().splitlines()];mx={k:max(v['seconds']for v in ev if v['kind']=='operation_returned'and v['operation']==k)for k in ('hvp','direct_gradient')};forecast=2*sum(CAPS[k]*v for k,v in mx.items())+20;B.require(math.isfinite(forecast)and forecast<90,'complete measured90s gate');result['profile']=dict(maximum_seconds=mx,forecast_seconds=forecast);store.json('profile.json',result['profile'])
  archive=io.BytesIO()
  with tarfile.open(fileobj=archive,mode='w:gz')as tar:
   for n in inventory:
    data=(ROOT/n).read_bytes();m=tarfile.TarInfo(n);m.size=len(data);m.mtime=0;tar.addfile(m,io.BytesIO(data))
  store.write('source.tar.gz',archive.getvalue())
  def load(n):
   a=np.load(PARENT/'run'/n,allow_pickle=False);B.require(a.shape==(98666,)and a.dtype==np.float64 and np.isfinite(a).all(),'full finite stored vector');return a
  x=load('initial-midpoint.npy');g=load('stationary-gradient.npy');p=load('native-hvp-0259-vector.npy');hp=load('native-hvp-0259-return.npy')
  B.require(np.array_equal(x,load('stationary-point.npy'))and np.array_equal(g,load('native-direct_gradient-0001-return.npy')),'unmodified source state/gradient')
  pairs={}
  for v in ev:
   if v['kind']=='native_input'and(v['operation'],v['index'])in [('hvp',259),('direct_gradient',1)]:pairs[(v['operation'],v['index'])]=v
  B.require(len(pairs)==2,'exact original native inputs')
  for v in pairs.values():B.require(np.array_equal(load(v['state']['file']),x),'same full native evaluation state')
  B.require(pairs[('hvp',259)]['direction']['file']=='native-hvp-0259-vector.npy','saved final direct action')
  arrays,model,_,_,identity=B.inputs()
  for k in ('data_identity','eta','lam','model'):B.require(identity[k]==old['input_identity'][k],'same original objective '+k)
  result['input_identity']=dict(identity,parent=str(PARENT),parent_manifest=PIN,parent_source=EXEC);store.json('input-identity.json',result['input_identity'])
  problem=B.CountedProblem(B.cifar.MLP(**model['model']),arrays['x'],arrays['y'],model['lam']);problem.budget=budget;native=S.Native(problem,store,budget)
  fresh_g=native.grad(x);fresh_hp=native.hvp(x,p);B.require(B.byte_equal(fresh_g,g)and B.byte_equal(fresh_hp,hp),'byte-exact original gradient and direction-action replay')
  gate=direction_gate(x,g,p,hp);B.require(gate['passed']and not gate['old_forcing_pass'],'new disclosed merit direction, prior gate remains failed');B.require(abs(gate['forcing']-old['stationary']['steps'][0]['forcing'])<1e-12,'same saved true forcing');result['direction_gate']=gate
  for n,v in [('initial-point',x),('initial-gradient',g),('saved-direction',p),('saved-action',hp)]:store.array(n,v)
  chosen=x;chosen_g=g
  for j in range(SETTINGS['trials']):
   budget.check();alpha=2.**(-j);candidate=x+alpha*p;distance=float(np.linalg.norm(candidate-x))/max(1.,float(np.linalg.norm(x)));row=dict(index=j,alpha=alpha,relative_origin_distance=distance,within_trust=distance<=SETTINGS['trust_relative'],accepted=False)
   if row['within_trust']:
    newg=native.grad(candidate);row.update(trial_gate(x,candidate,newg,alpha,gate));store.array(f'trial-{j:02d}-point',candidate);store.array(f'trial-{j:02d}-gradient',newg)
   result['trials'].append(row);store.event(dict(kind='stationary_saved_merit_trial',**row))
   if row['accepted']:result['accepted_index']=j;chosen=candidate;chosen_g=newg;break
  store.array('final-point',chosen);store.array('final-gradient',chosen_g);budget.check();result.update(status='stationary_saved_merit_complete',comparison_qualified=True,initial_relative_gradient=float(np.linalg.norm(g))/max(1.,float(np.linalg.norm(x))),final_relative_gradient=float(np.linalg.norm(chosen_g))/max(1.,float(np.linalg.norm(chosen))),merit_ratio=float(np.dot(chosen_g,chosen_g)/np.dot(g,g)),stationarity_qualified=bool(float(np.linalg.norm(chosen_g))/max(1.,float(np.linalg.norm(chosen)))<=SETTINGS['stationarity']),scope='One numerical gradient-merit seed step only. No eigenpair, optimizer-rate run or training-orbit acceptance.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));raise
 finally:result.update(elapsed_seconds=time.monotonic()-start,operation_caps=CAPS,operation_attempted=budget.attempted,operation_returned=budget.returned);store.finish(result)
 return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--review-commit',required=True);v=a.parse_args();install();signal.signal(signal.SIGTERM,request_stop);B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1);print(json.dumps(run(v.out,v.review_commit)))
