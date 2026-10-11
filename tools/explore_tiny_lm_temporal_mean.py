"""One source-pinned retrospective LM temporal-mean unit; exact saved directions."""
import os
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS: os.environ[key] = '1'
import argparse, ast, hashlib, io, json, resource, signal, subprocess, sys, time, traceback
from pathlib import Path
# Apply the declared worker address-space limit before loading NumPy/Torch.
# The lightweight supervisor intentionally starts with a512MiB soft limit.
if __name__ == '__main__': resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from neural_shrimp.tiny_lm import TinyTransformer, LMProblem
from neural_shrimp import tiny_lm_temporal_mean as T
need = T.need
DATA = Path('/data2/neural-shrimp-data')
BASE = DATA/'explore/tiny-lm-temporal-mean'
PROFILE_ROOT=BASE/'attempt002-profile'
TARGET_ROOT=BASE/'attempt001-target'
PROFILE_SOURCE='adc6bbe153a05f8e47dcd10d2a03ed38972cd377'
PROFILE_OUTER='7b46a95ca74d33c4b88ab2f279a5882aea1ec09ed67ff41de3a03ac96242e587'
PROFILE_WORKER='199c1d3f794ca630d4e2e369400da44a5ad9b7aae138cea38a73c6120b07eb57'
CARD = 'docs/plans/tiny-lm-temporal-mean-inputs.json'
PARENTS = {
 'profile': (DATA/'explore/tiny-lm-broader-data/attempt-001-profile','ce56de837203182187912654b9ac0200096366a15d850ed7e917fb903837d2cd'),
 'target': (DATA/'explore/tiny-lm-broader-data/attempt-002-collection','2ec1948521fe50a05a81b0ddec2e3e741cb25745c58f887868e3c5db43e2413d')}
RESPONSE = DATA/'explore/tiny-lm-response/attempt001-target'
SNAPSHOT = DATA/'explore/tiny-lm-two-snapshot/attempt001'
HISTORY = [
 (RESPONSE,'fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5','081c5dcc779c3b2f60d0a3102275f0fa7609efbf'),
 (SNAPSHOT,'c4ca66df6fdc7d848b98e3d232b0e1b99055a86130e2aae8fe52548ca5f1a91b','a86bb1e3a3332814f3db0040eda9ff1841891a89'),
 (DATA/'audits/tiny-lm-response-independent/attempt001','296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948','5a117145c0a0e7d91cb7a9bd96b8f8e49bc514d2'),
 (DATA/'audits/tiny-lm-two-snapshot-independent/attempt001','7378d61f7e7b50789156f0195407ea302be9093d71612ece6514ead2673ab74d','14296f7d67e4f10ee0d04d625543bc44e02abac0')]
NATIVE_SHA = '27508573066c416322fe697471ada13ebd1c99618a82ae0e1115672a3c94f7e8'
FILES = ['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/tiny_lm.py','src/neural_shrimp/tiny_lm_temporal_mean.py',
 'tools/explore_tiny_lm_temporal_mean.py','tools/launch_tiny_lm_temporal_mean.py',
 'tests/test_tiny_lm_temporal_mean.py','tests/test_tiny_lm_temporal_mean_driver.py',
 'docs/plans/tiny-lm-temporal-mean.md','docs/reviews/tiny-lm-temporal-mean-profile001-failure.json','docs/reviews/tiny-lm-temporal-mean-profile002.json']
CAP = 512*1024**2
RESERVE = 4*1024**2


def read(p): return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024**2): h.update(b)
 return h.hexdigest()
def entry(p): return dict(bytes=p.stat().st_size,sha256=sha(p))
def canonical(x): return (json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def git_bytes(source,name): return subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
def execution(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def inventory(root,pin,legacy=False):
 need(not root.is_symlink() and sha(root/'manifest.json')==pin,'literal manifest '+str(root))
 files=read(root/'manifest.json')['files']
 if isinstance(files,list): files={r['path']:{k:r[k] for k in ('bytes','sha256')} for r in files}
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
 need(actual==set(files),'complete immutable inventory')
 for name,value in files.items():
  p=root/name
  need(not Path(name).is_absolute() and '..' not in Path(name).parts and not p.is_symlink(),'safe immutable path')
  need(entry(p)=={k:value[k] for k in ('bytes','sha256')} and not p.stat().st_mode&0o222,'immutable exact file '+name)
 need(not (root/'manifest.json').stat().st_mode&0o222,'immutable manifest')
 if not legacy: need(not root.stat().st_mode&0o222,'immutable root')
 return files


def history(root,pin,source):
 inventory(root,pin)
 runtime=read(root/'run/runtime.json');need(runtime['source_commit']==source,'exact historical execution')
 identities=runtime.get('source_identity',runtime.get('sources',runtime.get('source_sha256')))
 need(isinstance(identities,dict) and identities,'complete historical source map')
 for name,value in identities.items():
  raw=git_bytes(source,name);expected=value['sha256'] if isinstance(value,dict) else value
  need(hashlib.sha256(raw).hexdigest()==expected,'historical source '+name)
  if (root/'run/source').exists(): need((root/'run/source'/name).read_bytes()==raw,'exact archived historical source')
 endpath=root/('launcher-result.json' if (root/'launcher-result.json').exists() else 'launch-end.json')
 end=read(endpath)
 code=end.get('exit_code',end.get('code'));need(code==0,'successful historical exit')
 if endpath.name=='launch-end.json':need(end['owned_group_empty'] and end['child_pid_absent'],'completed independent audit owner')
 else:need(end['owned_live_pids']==[],'completed producer owner')
 for pid in [runtime.get('pid'),end.get('launcher_pid'),end.get('owned_group_pid'),end.get('child_pid')]:
  if pid is not None: need(not Path('/proc',str(pid)).exists(),'historical owner absent')
 if (root/'owned-process.json').exists():
  record=read(root/'owned-process.json')
  for pid in record.values():
   if type(pid) is int: need(not Path('/proc',str(pid)).exists(),'historical owned process absent')
 result=read(root/'run/result.json');need(result['passed'],'historical arithmetic/response passed')
 return runtime,result


class Store:
 def __init__(self,root): self.root=root;root.mkdir(parents=True,exist_ok=False);self.used=0;self.objects={}
 def write(self,name,raw,closing=False):
  need(self.used+len(raw)<=CAP-(0 if closing else RESERVE),'512MiB raw/4MiB closing reserve')
  p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def json(self,name,x,closing=False): self.write(name,canonical(x),closing)
 def array(self,value):
  # Complete returned values, including NaN/Inf, are retained before validation.
  a=np.asarray(value,dtype=np.float64);buf=io.BytesIO();np.save(buf,a,allow_pickle=False);raw=buf.getvalue();key=hashlib.sha256(raw).hexdigest()
  pointer=dict(path='objects/'+key+'.npy',bytes=len(raw),sha256=key,shape=list(a.shape),dtype=str(a.dtype))
  if key not in self.objects:self.write(pointer['path'],raw);self.objects[key]=pointer
  return pointer
 def alias(self,name,value):
  a=np.asarray(value,dtype=np.float64)
  # Lossless row references avoid re-saving native vectors as dense trajectories.
  obj=dict(rows=[self.array(row) for row in a],shape=list(a.shape)) if a.ndim==2 and a.shape[1]==15776 else dict(array=self.array(a))
  self.json('aliases/'+name+'.json',obj)
 def finish(self,out):
  self.json('result.json',out,True)
  self.json('manifest.json',dict(files={str(p.relative_to(self.root)):entry(p) for p in self.root.rglob('*') if p.is_file()}),True)


class Calls:
 def __init__(self,store,mode,seconds):
  self.store=store;self.started=time.monotonic();self.seconds=seconds
  self.cap=T.PROFILE_COUNTS if mode=='profile' else T.TARGET_COUNTS
  self.attempted=dict.fromkeys(self.cap,0);self.returned=dict.fromkeys(self.cap,0)
  self.times={k:[] for k in self.cap};self.return_bytes={k:[] for k in self.cap};self.events=0;self.label='unassigned'
 def check(self):
  need(time.monotonic()-self.started<self.seconds,'worker time cap')
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'4GiB worker RSS')
 def call(self,name,fun,state,direction=None):
  self.check();need(self.attempted[name]<self.cap[name],'fixed native operation cap')
  self.attempted[name]+=1;self.events+=1;sequence=self.events;start=time.monotonic()
  row=dict(sequence=sequence,operation=name,index=self.attempted[name],case=self.label,state=self.store.array(state))
  if direction is not None:row['direction']=self.store.array(direction)
  self.store.json(f'events/{sequence:05d}-input.json',row)
  value=fun();pointer=self.store.array(value)  # before finite/shape/update validation
  row.update(returned=pointer,seconds=time.monotonic()-start)
  self.store.json(f'events/{sequence:05d}-return.json',row)
  self.times[name].append(time.monotonic()-start);self.return_bytes[name].append(pointer['bytes']);self.returned[name]+=1
  self.check();return value


def select(parent,seed,step):
 folder=parent/f'seed-{seed}-arm-256';old=read(folder/'result.json')
 need(old['seed']==seed and old['arm']==256 and old['outcome']=='completed','literal completed checkpoint')
 path=folder/f'states-{step//50*50:06d}-{step//50*50+49:06d}.npy';states=np.load(path,allow_pickle=False)
 need(states.shape==(50,15776) and states.dtype==np.float32 and np.isfinite(states).all(),'exact native state chunk')
 theta=states[step%50].astype(np.float64);data=read(parent/'data.json')
 need(data['data_identity_sha256']=='a65b88c04f7de18031005672a345af8eaf929ff75cf67e1d0c401b02c7f6f6aa' and len(data['vocabulary'])==66,'same consumed data')
 with np.load(parent/'data-tokens.npz',allow_pickle=False) as z:x,y=z['expanded_X'].copy(),z['expanded_Y'].copy()
 expected=np.asarray(data['roles']['expanded']['window_token_ids'],dtype=np.int64)
 need(x.shape==y.shape==(256,32) and x.dtype==y.dtype==np.int64 and np.array_equal(x,expected[:,:-1]) and np.array_equal(y,expected[:,1:]),'exact training-only tokens')
 eta=old['eta'];need(type(eta) is float and np.isfinite(eta) and eta>0,'original rate')
 return theta,x,y,eta,dict(seed=seed,arm=256,step=step,eta=eta,chunk=entry(path),state_float32_sha256=hashlib.sha256(states[step%50].tobytes()).hexdigest(),data_identity=data['data_identity_sha256'])


def align_snapshot(direction,first,values):
 need(values.shape==(16,len(direction)) and np.array_equal(values[0],first),'literal saved h1 through h16 prefix')
 return np.concatenate((direction[None,:],values),axis=0)


def load_case(seed,theta,eta):
 folder=RESPONSE/'run'/f'seed-{seed}-arm-256'
 def a(name):return np.load(folder/(name+'.npy'),allow_pickle=False)
 old=read(folder/'outcome.json');inputs=read(folder/'inputs.json')
 reference,gradients,directions=a('reference'),a('reference-gradients'),a('directions')
 need(reference.shape==gradients.shape==(17,15776) and directions.shape==(2,15776),'complete original state/gradient/directions')
 need(inputs['eta']==eta and np.array_equal(reference[0],theta) and np.array_equal(a('initial'),theta),'literal consumed checkpoint and physical rate')
 need(np.array_equal(reference[1:],reference[:-1]-eta*gradients[:-1]),'exact original reference updates')
 u=directions[0];need(np.array_equal(u,gradients[0]/np.linalg.norm(gradients[0])),'exact observed normalized gradient direction')
 branches=[]
 for k in range(2):
  row={}
  for label in ('plus','minus'):
   row[label]=a(f'direction-0-epsilon-{k}-{label}')
   values=a(f'direction-0-epsilon-{k}-{label}-ce');need(values.shape==(3,2) and np.array_equal(values[:,0],T.HORIZONS),'all original readout indices')
   row['ce_'+label]={int(h):float(v) for h,v in values}
  branches.append(row)
 cegrad=a('readout-gradients');need(cegrad.shape==(3,15776) and np.array_equal(cegrad,np.asarray([gradients[h]-.001*reference[h] for h in T.HORIZONS])),'counted original CE gradients')
 case=dict(reference=reference,reference_gradients=gradients,direction=u,frozen=a('direction-0-frozen'),ordered=a('direction-0-transported'),branches=branches,epsilons=old['epsilon_values'],readout_gradients=cegrad)
 need(case['epsilons']==[e*max(1.,float(np.linalg.norm(theta))) for e in (1e-4,5e-5)],'exact old perturbation amplitudes')
 extra={name:np.load(SNAPSHOT/'run'/f'seed-{seed}-arm-256'/f'direction-0-{file}.npy',allow_pickle=False) for name,file in [('two_snapshot','alternating'),('latest_J1','latest')]}
 # Original saved rows are h1..h16, so prepend the authenticated h0 direction.
 extra={name:align_snapshot(u,case['ordered'][1],values) for name,values in extra.items()}
 return case,extra


def card_check(card,mode):
 need(card['schema']=='tiny-lm-temporal-mean-v1' and card['target_counts']==T.TARGET_COUNTS and card['profile_counts']==T.PROFILE_COUNTS,'fixed operation card')
 need(card['seeds']==[28161,28162] and card['arm']==256 and card['step']==600 and card['control_rate_factor']==.25 and card['horizons']==list(T.HORIZONS),'fixed development design')
 need(card['worker_seconds']==dict(profile=60,target=900) and card['worker_rss_bytes']==4*1024**3 and card['worker_as_bytes']==8*1024**3 and card['worker_raw_bytes']==CAP,'fixed resource card')
 need(card['profile_root']==str(PROFILE_ROOT) and card['target_root']==str(TARGET_ROOT),'exclusive paths')
 need(card['status'] in ('source_prepared','profile_admitted','target_admitted'),'explicit card status')
 if mode=='profile':need(card['status']=='profile_admitted' and card['source_review'],'source-cleared profile admission')
 else:need(card['status']=='target_admitted' and card['profile_outer'] and card['profile_worker'] and card['source_review'],'separate measured target admission')


def preserved_profile_failure(store):
 root=BASE/'attempt001-profile';pin='4760072e1ecaf3487b215656a98b80dc54a37cad2b87fbdcc7270173b7af8c0b'
 files=inventory(root,pin);out=read(root/'launcher-result.json')
 need(out['source_commit']=='a717415c7852e2e4303b089e581acad52ed3d0b4' and out['exit_code']==1 and out['owned_live_pids']==[] and out['worker_pid_absent'],'literal failed profile owner')
 need(not(root/'run').exists() and 'failed to map segment' in (root/'process.log').read_text(),'zero-native import failure')
 need((root/'launcher.py').read_bytes()==git_bytes(out['source_commit'],'tools/launch_tiny_lm_temporal_mean.py'),'original failed launcher source')
 for pid in (out['launcher_pid'],out['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'failed profile owners absent')
 store.json('preserved-profile001-failure.json',dict(root=str(root),manifest=pin,source=out['source_commit'],files=files,native_calls=0))


def native_function_identity(raw):
 tree=ast.parse(raw)
 names={'Store','Calls','select','grad','hvp','ce'}
 found={n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
 need(set(found)==names,'complete profiled acquisition/retention function inventory')
 return found


def profile_binding(card,identity):
 root=PROFILE_ROOT
 need(card['profile_outer']==PROFILE_OUTER and card['profile_worker']==PROFILE_WORKER,'literal completed002 measured profile')
 inventory(root,PROFILE_OUTER);inventory(root/'run',PROFILE_WORKER)
 prior=read(root/'run/result.json');runtime=read(root/'run/runtime.json');owner=read(root/'launcher-result.json')
 need(runtime['source_commit']==owner['source_commit']==PROFILE_SOURCE,'exact completed measured execution')
 need(not prior['passed'] and 'error' not in prior and prior['mode']=='profile' and prior['attempted']==prior['returned']==T.PROFILE_COUNTS and prior['event_pairs']==6,'complete finite profile rejected only old time forecast')
 for name,value in runtime['sources'].items():
  raw=git_bytes(PROFILE_SOURCE,name)
  need((root/'run/source'/name).read_bytes()==raw and hashlib.sha256(raw).hexdigest()==value['sha256'],'complete measured source archive')
 for name in ['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/tiny_lm.py','src/neural_shrimp/tiny_lm_temporal_mean.py']:
  need(runtime['sources'][name]==identity[name],'unchanged profiled native/math module')
 worker='tools/explore_tiny_lm_temporal_mean.py'
 need(native_function_identity((root/'run/source'/worker).read_bytes())==native_function_identity((ROOT/worker).read_bytes()),'same exact measured acquisition/retention functions')
 need(runtime['torch']==torch.__version__ and runtime['numpy']==np.__version__,'exact measured runtime')
 need(owner['exit_code']==2 and not owner['owned_live_pids'] and owner['worker_pid_absent'],'retained measured time-rejection owner')
 for pid in (owner['launcher_pid'],owner['owned_group_pid'],runtime['pid']):need(not Path('/proc',str(pid)).exists(),'profile owners absent')
 bound=T.forecast(prior['times'],prior['return_bytes'])
 need(bound==prior['forecast'] and 600<bound['seconds']<900 and bound['bytes']<CAP-RESERVE and prior['seconds']<60 and prior['peak_rss_kib']<4*1024**2,'unchanged cold maxima fit disclosed900s time-only amendment')
 return bound


def main(args):
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ[k]=='1' for k in THREADS),'CPU1nice10')
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source')
 source=execution();identity={n:entry(ROOT/n) for n in FILES}
 for n in FILES:need((ROOT/n).read_bytes()==git_bytes(source,n),'committed source')
 need(identity['src/neural_shrimp/tiny_lm.py']['sha256']==NATIVE_SHA,'unchanged native model')
 card=read(ROOT/CARD);need((ROOT/CARD).read_bytes()==git_bytes(source,CARD),'committed run card');card_check(card,args.mode)
 need(args.out==(PROFILE_ROOT if args.mode=='profile' else TARGET_ROOT)/'run','exclusive worker path')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 store=Store(args.out);calls=Calls(store,args.mode,60 if args.mode=='profile' else 900);out=dict(passed=False,mode=args.mode,native_scope='consumed development only',cases=[])
 def stop(*_):raise TimeoutError('owned temporal-mean deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,calls.seconds)
 try:
  store.json('runtime.json',dict(source_commit=source,sources=identity,pid=os.getpid(),torch=torch.__version__,numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),dtype='float64',device='cpu'))
  store.json('run-card.json',card)
  for n in FILES:store.write('source/'+n,(ROOT/n).read_bytes())
  if args.mode=='target':store.json('profile-reference.json',dict(outer=card['profile_outer'],worker=card['profile_worker'],forecast=profile_binding(card,identity)))
  preserved_profile_failure(store)
  parent,pin=PARENTS[args.mode];inventory(parent,pin,legacy=True)
  if args.mode=='target':
   histories=[history(*h) for h in HISTORY]
   need(all(rt['torch']==torch.__version__ and rt['numpy']==np.__version__ for rt,_ in histories[:2]),'original response/comparator runtime')
   need(histories[0][0]['source_identity']['src/neural_shrimp/tiny_lm.py']['sha256']==NATIVE_SHA and histories[1][0]['sources']['src/neural_shrimp/tiny_lm.py']['sha256']==NATIVE_SHA,'same historical native model')
   need(histories[0][1]['operation_attempted']==histories[0][1]['operation_returned']==dict(gradient=580,hvp=256,ce=96),'complete original response')
   need(histories[1][1]['hvp_attempted']==histories[1][1]['hvp_returned']==240 and not histories[1][1]['candidate_gate'],'retained failed two-snapshot candidate')
   need(not histories[3][1]['candidate_gate'] and histories[3][1]['actions_recomputed']==240,'independently retained failed candidate')
  store.json('input-binding.json',dict(parent=dict(root=str(parent),manifest=pin),history=[] if args.mode=='profile' else [dict(root=str(r),manifest=p,source=s) for r,p,s in HISTORY],scope='Training tokens only; original direction and checkpoints consumed; no assessment or fresh confirmation.'))
  for seed in ([28160] if args.mode=='profile' else [28161,28162]):
   theta,x,y,eta,binding=select(parent,seed,50 if args.mode=='profile' else 600)
   problem=LMProblem(TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001)
   def tensor(v):return torch.as_tensor(v,dtype=torch.float64)
   def grad(v):return calls.call('gradient',lambda:problem.grad(tensor(v)).detach().numpy().copy(),v)
   def hvp(v,w):return calls.call('hvp',lambda:problem.hvp(tensor(v),tensor(w)).detach().numpy().copy(),v,w)
   def ce(v):return calls.call('ce',lambda:float(problem.cross_entropy(tensor(v))),v)
   calls.label=f'seed-{seed}-profile';store.json(calls.label+'/binding.json',binding)
   if args.mode=='profile':
    direction=np.random.default_rng(28166).standard_normal(15776);direction/=np.linalg.norm(direction)
    for _ in range(2):
     for fun in (lambda:grad(theta),lambda:hvp(theta,direction),lambda:ce(theta)):
      value=fun();need(np.isfinite(value).all(),'finite full-size measured return')
   else:
    original,extra=load_case(seed,theta,eta)
    for role,rate in [('original',eta),('quarter_step',eta/4)]:
     calls.label=f'seed-{seed}-{role}';label=calls.label;save=lambda n,v:store.alias(label+'/'+n,v)
     store.json(label+'/binding.json',dict(**binding,rate_role=role,actual_eta=rate,direction_sha256=hashlib.sha256(original['direction'].tobytes()).hexdigest(),original_direction_preserved=True))
     if role=='original':case=original
     else:
      case=T.acquire_control(theta,original['direction'],rate,.001,grad,hvp,ce,save,calls.check)
      need(np.array_equal(case['reference'][0],original['reference'][0]) and np.array_equal(case['direction'],original['direction']) and np.array_equal(case['reference_gradients'][0],original['reference_gradients'][0]),'same state/direction and deterministic initial gradient control')
     mean=T.mean_power(case['reference'],case['direction'],rate,hvp,save,calls.check)
     rows=T.measurements(case,mean,extra if role=='original' else None)
     result=dict(seed=seed,rate_role=role,eta=rate,rows=rows,small_step_monotone=case.get('small_step_monotone'),objective_increments=case.get('objective_increments'))
     store.json(label+'/result.json',result);out['cases'].append(result)
  need(calls.attempted==calls.returned==calls.cap,'all exact declared operations completed')
  if args.mode=='profile':
   bound=T.forecast(calls.times,calls.return_bytes);out.update(forecast=bound,passed=bool(bound['seconds']<900 and bound['bytes']<CAP-RESERVE))
  else:out.update(passed=True,descriptive_gate=T.descriptive_gate(out['cases']),scope='Retrospective temporal-mean sufficiency only; no causal forecast, commutator isolation, utility or shrimp claim.')
 except BaseException as e:out.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-calls.started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted=calls.attempted,returned=calls.returned,times=calls.times,return_bytes=calls.return_bytes,event_pairs=calls.events)
  out['passed']=bool(out['passed'] and out['seconds']<calls.seconds and out['peak_rss_kib']<4*1024**2);store.finish(out)
 print(json.dumps({k:out.get(k) for k in ('passed','seconds','forecast','descriptive_gate','error')}));return 0 if out['passed'] else 2


def parser():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
